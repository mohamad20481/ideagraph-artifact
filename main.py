"""
main.py - CLI entry point for IdeaGraph.

Usage:
    python main.py --topic "large language model reasoning" --budget 2.0 --iterations 20
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text
from rich import print as rprint

import config
from pipeline import IdeaGraphPipeline

console = Console()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="IdeaGraph: Multi-agent LLM research ideation with DAG knowledge graphs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py --topic \"few-shot learning for NLP\"\n"
            "  python main.py --topic \"protein structure prediction\" --budget 3.0 --iterations 15\n"
        ),
    )
    parser.add_argument(
        "--topic",
        required=True,
        help="Research topic to explore (e.g., 'graph neural networks for drug discovery')",
    )
    parser.add_argument(
        "--budget",
        type=float,
        default=2.0,
        metavar="USD",
        help="Approximate API budget in USD (default: 2.0)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=20,
        metavar="N",
        help="Maximum ideation loop iterations (default: 20)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(config.OUTPUT_DIR / "results.json"),
        metavar="PATH",
        help="Output JSON file path (default: output/results.json)",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="",
        choices=["deepseek", "openai", "groq", "gemini", ""],
        help="LLM provider (default: from .env or deepseek)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="",
        help="Model name override (default: provider's default model)",
    )
    return parser


def display_results(results: dict) -> None:
    """Render a summary of results to the console."""
    console.print()
    console.print(Panel.fit(
        f"[bold green]IdeaGraph Complete[/bold green]\n"
        f"Topic: [cyan]{results['topic']}[/cyan]\n"
        f"Coverage: [yellow]{results['coverage']:.1%}[/yellow] of QD archive filled\n"
        f"Ideas generated: [yellow]{len(results['ideas'])}[/yellow]",
        title="Results",
    ))

    # QD Archive grid
    archive_data = results.get("archive", {})
    grid = archive_data.get("grid", [])
    novelty_labels = archive_data.get("novelty_labels", ["incremental", "moderate", "substantial"])

    if grid:
        table = Table(title="QD Archive (quality scores)", show_lines=True)
        table.add_column("Methodology", style="bold cyan", min_width=22)
        for label in novelty_labels:
            table.add_column(label.capitalize(), justify="center", min_width=12)

        for row_data in grid:
            method = row_data.get("methodology", "").replace("_", " ").title()
            cells = row_data.get("cells", [])
            row_vals = []
            for cell in cells:
                q = cell.get("quality")
                if q is None:
                    row_vals.append("[dim]empty[/dim]")
                else:
                    color = "green" if q >= 0.6 else "yellow" if q >= 0.4 else "red"
                    row_vals.append(f"[{color}]{q:.3f}[/{color}]")
            table.add_row(method, *row_vals)

        console.print(table)

    # Ideas summary
    ideas = results.get("ideas", [])
    if ideas:
        console.print()
        console.print(Panel.fit(
            "[bold]Top Ideas[/bold]", subtitle=f"{len(ideas)} total"
        ))
        for i, idea in enumerate(ideas[:5], 1):
            q = idea.get("quality_score", 0.0)
            color = "green" if q >= 0.6 else "yellow" if q >= 0.4 else "red"
            console.print(
                f"  {i}. [{color}]q={q:.2f}[/{color}] [bold]{idea.get('title', 'Untitled')}[/bold]"
            )
            console.print(f"     {idea.get('methodology_type', '?')} / {idea.get('novelty_level', '?')}")
        if len(ideas) > 5:
            console.print(f"  … and {len(ideas) - 5} more (see output file)")

    # Stats
    stats = results.get("stats", {})
    if stats:
        console.print()
        console.print(
            f"[dim]Iterations: {stats.get('iterations', '?')} | "
            f"Elapsed: {stats.get('elapsed_seconds', '?')}s[/dim]"
        )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Apply CLI provider/model overrides
    if args.provider:
        config.PROVIDER = args.provider
    if args.model:
        config.MODEL = args.model
    elif args.provider:
        # Use provider's default model when provider changed but model not specified
        config.MODEL = config._DEFAULT_MODELS.get(args.provider, config.MODEL)

    # Header
    console.print(Panel.fit(
        "[bold blue]IdeaGraph[/bold blue] — Multi-agent LLM Research Ideation\n"
        f"[dim]Provider: {config.PROVIDER} | Model: {config.MODEL}[/dim]",
        subtitle="Starting …",
    ))
    console.print(f"Topic: [cyan]{args.topic}[/cyan]")
    console.print(f"Budget: [yellow]${args.budget:.2f}[/yellow] | Iterations: [yellow]{args.iterations}[/yellow]")
    console.print()

    # Progress display
    log_lines = []

    def on_progress(msg: str) -> None:
        log_lines.append(msg)
        # Print each update; use carriage-return style for spinner feel
        console.print(f"  [dim]{msg}[/dim]")

    pipeline = IdeaGraphPipeline()

    try:
        results = pipeline.run(
            topic=args.topic,
            budget_usd=args.budget,
            max_iterations=args.iterations,
            on_progress=on_progress,
        )
    except KeyboardInterrupt:
        console.print("\n[red]Interrupted by user.[/red]")
        sys.exit(1)
    except Exception as exc:
        console.print(f"\n[red]Pipeline failed: {exc}[/red]")
        raise

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    console.print(f"\n[green]Results saved to:[/green] {output_path}")
    display_results(results)


if __name__ == "__main__":
    main()
