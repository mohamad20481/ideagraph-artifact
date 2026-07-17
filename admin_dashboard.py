"""
admin_dashboard.py - Admin analytics dashboard for IdeaGraph operators.

Shows: revenue, costs, active users, churn, top topics, system health.
Only accessible to admin users (user_id == 1 or configurable).
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

try:
    import db
except ImportError:
    db = None  # type: ignore


ADMIN_USER_IDS = {1}  # Override via ADMIN_USER_IDS env var (comma-separated)
_admin_env = os.getenv("ADMIN_USER_IDS", "")
if _admin_env:
    ADMIN_USER_IDS = {int(x.strip()) for x in _admin_env.split(",") if x.strip().isdigit()}


def is_admin(user_id: int) -> bool:
    return user_id in ADMIN_USER_IDS


def get_admin_stats() -> Dict[str, Any]:
    """Aggregate platform-wide statistics for the admin dashboard."""
    if not db:
        return {}

    try:
        with db._lock:
            conn = db._get_conn()
            try:
                # User metrics
                total_users = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
                today = datetime.now().strftime("%Y-%m-%d")
                week_ago = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
                month_ago = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

                new_users_7d = conn.execute(
                    "SELECT COUNT(*) FROM users WHERE created_at >= ?", (week_ago,)
                ).fetchone()[0]
                new_users_30d = conn.execute(
                    "SELECT COUNT(*) FROM users WHERE created_at >= ?", (month_ago,)
                ).fetchone()[0]

                # Run metrics
                total_runs = conn.execute("SELECT COUNT(*) FROM results").fetchone()[0]
                runs_7d = conn.execute(
                    "SELECT COUNT(*) FROM results WHERE created_at >= ?", (week_ago,)
                ).fetchone()[0]
                runs_30d = conn.execute(
                    "SELECT COUNT(*) FROM results WHERE created_at >= ?", (month_ago,)
                ).fetchone()[0]

                # Idea metrics
                total_ideas = conn.execute(
                    "SELECT COALESCE(SUM(ideas_count), 0) FROM results"
                ).fetchone()[0]

                # Cost metrics
                total_cost = 0.0
                cost_7d = 0.0
                try:
                    total_cost = conn.execute(
                        "SELECT COALESCE(SUM(cost_usd), 0) FROM cost_tracking_log"
                    ).fetchone()[0]
                    cost_7d = conn.execute(
                        "SELECT COALESCE(SUM(cost_usd), 0) FROM cost_tracking_log "
                        "WHERE timestamp >= ?", (week_ago,)
                    ).fetchone()[0]
                except Exception:
                    pass

                # Revenue (from subscriptions)
                tier_counts = {}
                try:
                    rows = conn.execute(
                        "SELECT tier, COUNT(*) as cnt FROM subscriptions "
                        "WHERE status = 'active' GROUP BY tier"
                    ).fetchall()
                    tier_counts = {r["tier"]: r["cnt"] for r in rows}
                except Exception:
                    pass

                # Revenue estimate
                tier_prices = {"free": 0, "pro": 15, "team": 49, "enterprise": 299}
                monthly_revenue = sum(
                    tier_prices.get(tier, 0) * count
                    for tier, count in tier_counts.items()
                )

                # Top topics
                top_topics = []
                try:
                    rows = conn.execute(
                        "SELECT topic, COUNT(*) as cnt, AVG(coverage) as avg_cov "
                        "FROM results GROUP BY topic ORDER BY cnt DESC LIMIT 10"
                    ).fetchall()
                    top_topics = [dict(r) for r in rows]
                except Exception:
                    pass

                # Sharing metrics
                total_shares = conn.execute(
                    "SELECT COUNT(*) FROM share_tokens"
                ).fetchone()[0]
                total_views = conn.execute(
                    "SELECT COALESCE(SUM(views), 0) FROM share_tokens"
                ).fetchone()[0]
                total_likes = conn.execute(
                    "SELECT COALESCE(SUM(likes), 0) FROM share_tokens"
                ).fetchone()[0]

                # Active users (ran pipeline in last 7 days)
                active_7d = conn.execute(
                    "SELECT COUNT(DISTINCT user_id) FROM results WHERE created_at >= ?",
                    (week_ago,),
                ).fetchone()[0]

                # Avg run time
                avg_runtime = 0.0
                try:
                    row = conn.execute(
                        "SELECT AVG(elapsed_seconds) FROM results WHERE elapsed_seconds > 0"
                    ).fetchone()
                    avg_runtime = row[0] or 0
                except Exception:
                    pass

                return {
                    "total_users": total_users,
                    "new_users_7d": new_users_7d,
                    "new_users_30d": new_users_30d,
                    "active_users_7d": active_7d,
                    "total_runs": total_runs,
                    "runs_7d": runs_7d,
                    "runs_30d": runs_30d,
                    "total_ideas": total_ideas,
                    "total_cost_usd": round(total_cost, 2),
                    "cost_7d_usd": round(cost_7d, 2),
                    "tier_counts": tier_counts,
                    "monthly_revenue_usd": monthly_revenue,
                    "top_topics": top_topics,
                    "total_shares": total_shares,
                    "total_views": total_views,
                    "total_likes": total_likes,
                    "avg_runtime_seconds": round(avg_runtime, 1),
                }
            finally:
                conn.close()
    except Exception as e:
        return {"error": str(e)}


def render_admin_dashboard(st_module) -> None:
    """Render the admin analytics dashboard in Streamlit."""
    st = st_module
    st.title("Admin Dashboard")
    st.caption("Platform analytics, pipeline mechanics & operational metrics")

    # Six top-level tabs: live platform stats, the educational pipeline
    # simulator, the federated-MAP-Elites A/B, a runtime LLM-provider
    # control panel, a feature-toggle panel for operator on/off switches,
    # and a Visual Rendering panel for the FLUX / Nano-Banana image API
    # key + model + endpoint. Each toggle/panel mutates the config module
    # at runtime and optionally persists to .env.
    (tab_stats, tab_sim, tab_pop, tab_llm, tab_toggles, tab_visual,
     tab_billing, tab_runs) = st.tabs([
        "📊 Platform Stats",
        "🎬 Pipeline Simulator",
        "🌐 Population (Federation)",
        "🔌 LLM Provider",
        "🎚️ Feature Toggles",
        "🎨 Visual Rendering",
        "💳 Billing",
        "🚦 Active Runs",
    ])

    with tab_stats:
        _render_stats(st)

    with tab_sim:
        try:
            from pipeline_simulator import render_pipeline_simulator
            render_pipeline_simulator(st)
        except ImportError as e:
            st.error(f"pipeline_simulator unavailable: {e}")
        except Exception as e:
            st.error(f"Simulator error: {e}")

    with tab_pop:
        try:
            _render_population_panel(st)
        except ImportError as e:
            st.error(f"federated_diversity unavailable: {e}")
        except Exception as e:
            st.error(f"Population simulator error: {e}")

    with tab_llm:
        try:
            _render_llm_provider_panel(st)
        except Exception as e:
            st.error(f"LLM provider panel error: {e}")

    with tab_toggles:
        try:
            _render_feature_toggles_panel(st)
        except Exception as e:
            st.error(f"Feature toggles panel error: {e}")

    with tab_visual:
        try:
            _render_visual_rendering_panel(st)
        except Exception as e:
            st.error(f"Visual rendering panel error: {e}")

    with tab_billing:
        try:
            _render_billing_panel(st)
        except Exception as e:
            st.error(f"Billing panel error: {e}")

    with tab_runs:
        try:
            _render_active_runs_panel(st)
        except Exception as e:
            st.error(f"Active runs panel error: {e}")


def _render_active_runs_panel(st) -> None:
    """Admin Active Runs tab: view + force-release stuck concurrency
    slots / quota reservations.

    Why this exists:
        ConcurrencyGuard + QuotaEnforcer store in-flight runs as
        in-memory counters in production_optimization.py. If a worker
        thread releases via the main-thread queue path (the original
        design) and the user closes the tab mid-run before the queue
        drains, the slot stays held forever — the user starts seeing
        "You already have 3 runs in progress (max 3). Wait for one to
        complete." even though nothing is running.

        The fix in app.py (try/finally in _run_pipeline_thread +
        _run_scientist_thread) makes that leak much rarer, but if any
        slot DOES get stuck — e.g. from a worker thread that hard-
        crashed or a code path we haven't wrapped yet — this panel
        gives you a one-click recovery without restarting Streamlit.
    """
    from production_optimization import (
        get_concurrency_guard, get_quota_enforcer,
    )

    st.markdown("### 🚦 Active runs")
    st.caption(
        "In-flight pipeline / scientist runs by user. Use **🗑️ Release "
        "slots** to free a stuck user's reservations without restarting "
        "the server. Counters live in process memory only (lost on "
        "Streamlit restart) — they're never persisted to db."
    )

    guard = get_concurrency_guard()
    quota = get_quota_enforcer()
    guard_snap = guard.snapshot()
    quota_snap = quota.snapshot()
    guard_stats = guard.stats()

    # ── Global stats row ────────────────────────────────────────────────
    sc1, sc2, sc3, sc4 = st.columns(4)
    sc1.metric(
        "Global active",
        f"{guard_stats['global_active']}/{guard_stats['global_max']}",
    )
    sc2.metric("Users with runs", guard_stats["users_active"])
    sc3.metric("Per-user max", guard.per_user_max)
    sc4.metric("Quota reserved (users)", len(quota_snap))

    # ── Combined per-user table ─────────────────────────────────────────
    all_uids = sorted(set(guard_snap) | set(quota_snap))
    if not all_uids:
        st.success(
            "✅ No active or stuck runs. Everyone has 0 slots held."
        )
    else:
        # Resolve usernames so the table is readable.
        try:
            import db as _db
            with _db._lock:
                conn = _db._get_conn()
                try:
                    rows = conn.execute(
                        "SELECT id, username FROM users "
                        f"WHERE id IN ({','.join('?' * len(all_uids))})",
                        all_uids,
                    ).fetchall()
                    usernames = {int(r["id"]): r["username"] for r in rows}
                finally:
                    conn.close()
        except Exception:
            usernames = {}

        st.markdown("**Users with held slots:**")
        for uid in all_uids:
            uname = usernames.get(uid, f"user#{uid}")
            g_slots = int(guard_snap.get(uid, 0))
            q_slots = int(quota_snap.get(uid, 0))
            row = st.columns([3, 1, 1, 2])
            row[0].markdown(
                f"**#{uid}** {uname}  "
                f"<span style='color:#94a3b8;font-size:11px'>"
                f"concurrency={g_slots} · quota={q_slots}</span>",
                unsafe_allow_html=True,
            )
            row[1].metric("Conc.", g_slots)
            row[2].metric("Quota", q_slots)
            if row[3].button(
                f"🗑️ Release slots",
                key=f"runs_release_{uid}",
                use_container_width=True,
                help=(
                    "Zero this user's slot counters. They'll be able to "
                    "click Run immediately. No effect on completed runs."
                ),
            ):
                n_g = guard.reset_user(int(uid))
                n_q = quota.reset_user(int(uid))
                st.success(
                    f"✅ Released {n_g} concurrency slot(s) + "
                    f"{n_q} quota reservation(s) for user #{uid}."
                )
                st.rerun()

    st.divider()

    # ── Nuclear option ──────────────────────────────────────────────────
    st.markdown("#### ⚠️ Reset all")
    st.caption(
        "Zero EVERY user's slot counters. Useful when something "
        "unexpected has happened and you want a known-good baseline. "
        "Does not affect any saved results, subscriptions, or pipeline "
        "data — only the in-memory in-flight counters."
    )
    if st.button(
        "🚨 Reset ALL active runs", key="runs_reset_all",
        type="primary",
        help="Wipes ConcurrencyGuard + QuotaEnforcer in-flight state",
    ):
        n_g = guard.clear()
        n_q = quota.clear()
        st.success(
            f"✅ Released {n_g} concurrency slot(s) + "
            f"{n_q} quota reservation(s) across all users."
        )
        st.rerun()


def _render_population_panel(st) -> None:
    """A/B comparison: independent runs vs federated MAP-Elites."""
    from federated_diversity import (
        compare_populations,
        cell_saturation_grid,
        coverage_distribution,
    )

    st.markdown("### 🌐 Population-Scale Homogenization Study")
    st.caption(
        "If 10,000 researchers run IdeaGraph independently, do their archives "
        "collapse onto the same modes? This panel runs N synthetic users, "
        "measures aggregate Div-Pair (average pairwise Jaccard distance "
        "across users' cell sets), and demonstrates the **federated MAP-Elites** "
        "mechanism: each user broadcasts a privacy-preserving hash of their "
        "occupied cells; the global census penalizes generators that target "
        "globally-saturated cells."
    )

    c1, c2, c3, c4 = st.columns(4)
    n_users = c1.slider(
        "Users (N)", min_value=5, max_value=100, value=30, step=5,
        key="pop_n_users",
        help="More users = stronger homogenization signal but slower run.",
    )
    ideas_per_user = c2.slider(
        "Ideas / user", min_value=4, max_value=24, value=12, step=2,
        key="pop_n_ideas",
    )
    fed_threshold = c3.slider(
        "Federation threshold", min_value=0.10, max_value=0.60,
        value=0.30, step=0.05, key="pop_fed_threshold",
        help="Cells occupied by < this fraction of users incur no penalty.",
    )
    fed_strength = c4.slider(
        "Federation strength", min_value=0.30, max_value=1.00,
        value=0.85, step=0.05, key="pop_fed_strength",
        help="Maximum penalty applied to fully-saturated cells.",
    )

    if st.button("▶ Run A/B comparison",
                  type="primary", use_container_width=True,
                  key="pop_run_btn"):
        with st.spinner(
            f"Simulating {n_users} users twice (independent vs federated)…"
        ):
            results = compare_populations(
                n_users=int(n_users),
                ideas_per_user=int(ideas_per_user),
                seed=7,
                federation_threshold=float(fed_threshold),
                federation_strength=float(fed_strength),
            )
        st.session_state["_pop_results"] = results

    results = st.session_state.get("_pop_results")
    if not results:
        st.info(
            "Click **▶ Run A/B comparison** to run the population study. "
            "Both arms (independent + federated) use the same seed and topic "
            "pool, so the only difference is whether users see the global "
            "census."
        )
        return

    indep = results["independent"]
    fed = results["federated"]

    # ── Headline metrics ────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("**Aggregate diversity metrics**")
    h1, h2, h3, h4 = st.columns(4)
    delta_dp = fed.div_pair - indep.div_pair
    delta_h = fed.homogenization - indep.homogenization
    h1.metric(
        "Div-Pair (independent)",
        f"{indep.div_pair:.3f}",
        help="Average pairwise Jaccard distance between users' cell sets — "
              "higher = more diverse population.",
    )
    h2.metric(
        "Div-Pair (federated)",
        f"{fed.div_pair:.3f}",
        delta=f"{delta_dp:+.3f}",
        help="Same metric with federated MAP-Elites enabled.",
    )
    h3.metric(
        "Homogenization (independent)",
        f"{indep.homogenization:.3f}",
        help="Gini coefficient over per-cell user counts. 0 = perfectly "
              "uniform; 1 = everyone in one cell.",
    )
    h4.metric(
        "Homogenization (federated)",
        f"{fed.homogenization:.3f}",
        delta=f"{delta_h:+.3f}",
        delta_color="inverse",  # lower is better here
    )

    # ── Verdict line ─────────────────────────────────────────────────────────
    if delta_dp > 0.005:
        verdict = (
            f"✅ **Federation helps.** With **N={indep.n_users}** users, "
            f"federated MAP-Elites raised aggregate Div-Pair by "
            f"**{delta_dp:+.3f}** ({delta_dp/max(0.001, indep.div_pair)*100:+.1f}%) "
            f"and reduced the homogenization Gini by **{abs(delta_h):.3f}**."
        )
        st.success(verdict)
    elif delta_dp < -0.005:
        st.warning(
            f"⚠️ Federation reduced Div-Pair by {abs(delta_dp):.3f}. "
            "Try lowering the threshold or strength."
        )
    else:
        st.info(
            "↔️ Federation effect is within noise at this N. Try increasing "
            "N or ideas-per-user."
        )

    # ── Saturation heatmaps side-by-side ────────────────────────────────────
    st.markdown("---")
    st.markdown("**Cell-saturation grids** — fraction of users covering each cell")
    try:
        import plotly.graph_objects as go
        from models.idea import METHODOLOGY_TYPES, NOVELTY_LEVELS

        def _make_heatmap(z, title):
            fig = go.Figure(data=go.Heatmap(
                z=z,
                x=[n.title() for n in NOVELTY_LEVELS],
                y=[m.replace("_", " ").title() for m in METHODOLOGY_TYPES],
                colorscale=[[0, "#f0fdf4"], [0.5, "#fde68a"], [1, "#dc2626"]],
                zmin=0, zmax=1,
                text=[[f"{v*100:.0f}%" for v in row] for row in z],
                texttemplate="%{text}",
                hovertemplate="<b>%{y}</b> × <b>%{x}</b><br>"
                              "%{z:.2%} of users<extra></extra>",
                colorbar=dict(thickness=10, len=0.7),
            ))
            fig.update_layout(
                title=dict(text=title, font=dict(size=13)),
                height=360, margin=dict(l=140, r=20, t=40, b=20),
                xaxis=dict(side="top"),
                yaxis=dict(autorange="reversed"),
                plot_bgcolor="rgba(0,0,0,0)",
            )
            return fig

        z_indep = cell_saturation_grid(indep.census)
        z_fed = cell_saturation_grid(fed.census)
        col_a, col_b = st.columns(2)
        with col_a:
            st.plotly_chart(
                _make_heatmap(z_indep, "Independent (homogenization risk)"),
                use_container_width=True, key="pop_heat_indep",
            )
        with col_b:
            st.plotly_chart(
                _make_heatmap(z_fed, "Federated MAP-Elites"),
                use_container_width=True, key="pop_heat_fed",
            )
        st.caption(
            "Red cells are 'crowded' — most users land there. Federation "
            "should spread occupancy more evenly across the grid (less red, "
            "more uniform color)."
        )
    except Exception as e:
        st.caption(f"Heatmap unavailable: {e}")

    # ── Per-user coverage distribution ──────────────────────────────────────
    st.markdown("---")
    st.markdown("**Per-user coverage distribution**")
    try:
        import plotly.graph_objects as go
        cov_indep = sorted(coverage_distribution(indep.archives), reverse=True)
        cov_fed = sorted(coverage_distribution(fed.archives), reverse=True)
        xs = list(range(1, len(cov_indep) + 1))
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=xs, y=cov_indep, mode="lines+markers", name="Independent",
            line=dict(color="#94a3b8", width=2.5),
            marker=dict(size=6),
        ))
        fig.add_trace(go.Scatter(
            x=xs, y=cov_fed, mode="lines+markers", name="Federated",
            line=dict(color="#0ea5e9", width=2.5),
            marker=dict(size=6),
        ))
        fig.update_layout(
            height=300,
            xaxis_title="User rank (sorted by coverage, desc)",
            yaxis_title="Cells occupied / 21",
            yaxis=dict(range=[0, 1]),
            margin=dict(l=40, r=20, t=10, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                          xanchor="right", x=1),
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True, key="pop_cov_chart")
        st.caption(
            f"Independent: mean per-user coverage = "
            f"**{indep.mean_user_coverage*100:.0f}%**. "
            f"Federated: **{fed.mean_user_coverage*100:.0f}%**. "
            "Individual coverage is largely unchanged — federation "
            "redistributes WHERE users explore, not whether they explore."
        )
    except Exception as e:
        st.caption(f"Coverage chart unavailable: {e}")

    # ── How the mechanism works ─────────────────────────────────────────────
    with st.expander("📋 How the federated mechanism works", expanded=False):
        st.markdown(
            "1. **Per-user broadcast.** Each user computes a "
            "`CellHashBroadcast` — a set of `sha256(cell, quality_bucket)` "
            "12-char hashes. No titles, methods, or probe scores are shared.\n"
            "2. **Global census.** A `GlobalCellCensus` aggregates incoming "
            "broadcasts and keeps a per-cell user count.\n"
            "3. **Saturation penalty.** A new user's pipeline reads the "
            "census and builds `federated_penalty_fn(census, threshold, "
            "strength)`. Cells whose user-fraction exceeds `threshold` "
            "get a penalty rising linearly to `strength` at fraction = 1.0.\n"
            "4. **Biased cell selection.** Inside `fake_pipeline_run`, "
            "the methodology × novelty cell weights are multiplied by "
            "`(1 - penalty(cell))` before sampling. Saturated cells almost "
            "vanish from the distribution; under-explored cells dominate.\n"
            "5. **The user still owns their archive.** Federation only "
            "affects WHERE the user generates next — once an idea is "
            "generated and probed, it lives entirely in that user's local "
            "archive. The mechanism is exactly the kind of "
            "**privacy-preserving aggregation** that scales to 10K+ users."
        )


# ─────────────────────────────────────────────────────────────────────────────
# LLM Provider control panel
# ─────────────────────────────────────────────────────────────────────────────

_PROVIDER_META = {
    "deepseek":  ("DeepSeek",  "🐋"),
    "openai":    ("OpenAI",    "🟢"),
    "groq":      ("Groq (Llama fast inference)", "⚡"),
    "gemini":    ("Gemini",    "✨"),
    "azure":     ("Azure",     "🔷"),
    "anthropic": ("Anthropic Claude", "🧠"),
    "kimi":      ("Kimi (Moonshot)", "🌙"),
    # xAI Grok — DIFFERENT from Groq above. "xai" provider name keeps
    # the two unambiguous in the dropdown ordering + key resolver.
    "xai":       ("xAI Grok (Elon's Grok)", "🚀"),
}


def _api_key_for(cfg, provider: str) -> str:
    return {
        "deepseek":  getattr(cfg, "DEEPSEEK_API_KEY", ""),
        "openai":    getattr(cfg, "OPENAI_API_KEY", ""),
        "groq":      getattr(cfg, "GROQ_API_KEY", ""),
        "gemini":    getattr(cfg, "GEMINI_API_KEY", ""),
        "azure":     getattr(cfg, "AZURE_API_KEY", ""),
        "anthropic": getattr(cfg, "ANTHROPIC_API_KEY", ""),
        "kimi":      getattr(cfg, "KIMI_API_KEY", ""),
        # xAI Grok shares the GROK_API_KEY env var with the image-gen
        # provider since both auth against the same xAI account.
        "xai":       getattr(cfg, "GROK_API_KEY", ""),
    }.get(provider, "")


def _persist_env_updates(updates: Dict[str, str]) -> Optional[str]:
    """Apply a {key: value} dict of updates to .env in place.

    Preserves comments + ordering. Existing keys are updated; missing
    keys are appended at the end. Returns None on success, an error
    string on failure. This is the GENERIC version used by both
    _persist_to_env (provider/model switch) and the API-key persistence
    path on the admin panel.

    Empty values are still written verbatim — the caller is responsible
    for not wiping a key by passing "".
    """
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    try:
        if os.path.exists(env_path):
            with open(env_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        else:
            lines = []
        seen = {k: False for k in updates}
        out: List[str] = []
        for line in lines:
            stripped = line.lstrip()
            if stripped.startswith("#") or "=" not in stripped:
                out.append(line)
                continue
            key = stripped.split("=", 1)[0].strip()
            if key in updates:
                out.append(f"{key}={updates[key]}\n")
                seen[key] = True
            else:
                out.append(line)
        for k, was_seen in seen.items():
            if not was_seen:
                if out and not out[-1].endswith("\n"):
                    out[-1] = out[-1] + "\n"
                out.append(f"{k}={updates[k]}\n")
        with open(env_path, "w", encoding="utf-8") as f:
            f.writelines(out)
        return None
    except Exception as e:
        return str(e)


def _persist_to_env(provider: str, model: str) -> Optional[str]:
    """Update IDEAGRAPH_PROVIDER and IDEAGRAPH_MODEL in .env.

    Thin wrapper around _persist_env_updates kept for backward compat
    with existing call sites.
    """
    return _persist_env_updates({
        "IDEAGRAPH_PROVIDER": provider,
        "IDEAGRAPH_MODEL": model,
    })


# Maps each LLM provider to the canonical .env key + the matching
# attribute on the `config` module. Used by the admin API-key-input
# widget to know what env-var to update and where to reflect the new
# key at runtime.
_PROVIDER_KEY_NAMES: Dict[str, Tuple[str, str]] = {
    # provider → (env_var_name, config_attr_name)
    "deepseek":  ("DEEPSEEK_API_KEY",  "DEEPSEEK_API_KEY"),
    "openai":    ("OPENAI_API_KEY",    "OPENAI_API_KEY"),
    "groq":      ("GROQ_API_KEY",      "GROQ_API_KEY"),
    "gemini":    ("GEMINI_API_KEY",    "GEMINI_API_KEY"),
    "azure":     ("AZURE_API_KEY",     "AZURE_API_KEY"),
    "anthropic": ("ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY"),
    "kimi":      ("KIMI_API_KEY",      "KIMI_API_KEY"),
    # xAI Grok shares the GROK_API_KEY env var with the image-gen
    # provider (both use the same xAI account secret).
    "xai":       ("GROK_API_KEY",      "GROK_API_KEY"),
}


def _test_provider_api_key(provider: str, candidate_key: str) -> Dict[str, Any]:
    """Live-ping the provider with a tiny request to verify a candidate
    key works BEFORE the operator commits it via Apply.

    Returns {"ok": bool, "status_code": int|None, "message": str}.
    Never raises — network errors fold into ok=False.

    The probe uses each provider's smallest no-output endpoint:
      - DeepSeek / Kimi / Groq / OpenAI / Azure → /v1/models (cheap)
      - Anthropic → /v1/models (Anthropic also has this)
      - Gemini    → /v1beta/models (Google AI Studio)
    """
    if not (candidate_key or "").strip():
        return {"ok": False, "status_code": None,
                "message": "Empty key — nothing to test."}
    try:
        import requests
    except ImportError:
        return {"ok": False, "status_code": None,
                "message": "requests library not installed."}

    # Per-provider probe configuration.
    probes = {
        "deepseek":  ("https://api.deepseek.com/v1/models",
                      {"Authorization": f"Bearer {candidate_key}"}),
        "openai":    ("https://api.openai.com/v1/models",
                      {"Authorization": f"Bearer {candidate_key}"}),
        "groq":      ("https://api.groq.com/openai/v1/models",
                      {"Authorization": f"Bearer {candidate_key}"}),
        "anthropic": ("https://api.anthropic.com/v1/models",
                      {"x-api-key": candidate_key,
                       "anthropic-version": "2023-06-01"}),
        "gemini":    (f"https://generativelanguage.googleapis.com/v1beta/models?key={candidate_key}",
                      {}),
        "kimi":      ("https://api.moonshot.cn/v1/models",
                      {"Authorization": f"Bearer {candidate_key}"}),
        # xAI Grok — OpenAI-compatible API. Must NOT route to groq.com.
        "xai":       ("https://api.x.ai/v1/models",
                      {"Authorization": f"Bearer {candidate_key}"}),
    }
    probe = probes.get(provider)
    if not probe:
        return {"ok": False, "status_code": None,
                "message": f"No probe configured for provider {provider!r}."}
    url, headers = probe
    try:
        r = requests.get(url, headers=headers, timeout=10.0)
    except Exception as e:
        return {"ok": False, "status_code": None,
                "message": f"network error: {type(e).__name__}: {str(e)[:150]}"}
    if r.status_code == 200:
        return {"ok": True, "status_code": 200,
                "message": "✅ Key works — auth + endpoint reachable."}
    if r.status_code == 401:
        return {"ok": False, "status_code": 401,
                "message": "❌ HTTP 401 — key is invalid / expired / wrong account."}
    if r.status_code == 403:
        return {"ok": False, "status_code": 403,
                "message": "❌ HTTP 403 — key valid but lacks permission for this endpoint."}
    if r.status_code == 429:
        return {"ok": False, "status_code": 429,
                "message": "⚠️ HTTP 429 rate-limited — key likely works, try again in a moment."}
    return {"ok": False, "status_code": r.status_code,
            "message": f"HTTP {r.status_code}: {r.text[:200]}"}


def _reset_provider_key_to_env(provider: str) -> Dict[str, Any]:
    """Recovery helper: reload the provider's API key from .env, dropping
    any in-process runtime override. Useful when the operator typed a
    bad key into the admin panel and wants to revert without restarting
    Streamlit.

    Reads .env file directly (bypasses dotenv's "first-wins" cache) so
    the on-disk value is what's loaded.
    """
    entry = _PROVIDER_KEY_NAMES.get(provider)
    if not entry:
        return {"ok": False, "error": f"Unknown provider: {provider!r}"}
    env_var, cfg_attr = entry
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    env_value = ""
    try:
        if os.path.exists(env_path):
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    stripped = line.lstrip()
                    if stripped.startswith("#") or "=" not in stripped:
                        continue
                    k, _, v = stripped.partition("=")
                    if k.strip() == env_var:
                        env_value = v.strip().rstrip("\n").strip('"').strip("'")
                        break
    except Exception as e:
        return {"ok": False, "error": f"Couldn't read .env: {e}"}

    # Restore os.environ + cfg attribute to the .env value, then drop
    # the cached client so the next call rebuilds with the .env key.
    try:
        os.environ[env_var] = env_value
        import config as _cfg
        setattr(_cfg, cfg_attr, env_value)
    except Exception as e:
        return {"ok": False, "error": f"runtime restore failed: {e}"}
    try:
        from claude_provider import get_claude_client
        get_claude_client(reload=True)
    except Exception:
        pass
    return {
        "ok": True,
        "env_key_suffix": env_value[-5:] if env_value else "(empty)",
        "had_value": bool(env_value),
    }


def _set_provider_api_key(
    provider: str,
    new_key: str,
    persist_env: bool = False,
    force_clear: bool = False,
) -> Dict[str, Any]:
    """Set a provider's API key at runtime (cfg attribute) and
    optionally persist to .env. Returns a dict describing what happened
    for the UI to show: {ok, runtime_set, persisted, env_error,
    client_reloaded, client_reload_error}.

    Also drops any cached OpenAI-compat / Anthropic client so the next
    LLM call rebuilds with the new key — without this, the cached
    client would keep using the stale key until Streamlit restart.

    Empty-key safety: passing `new_key=""` is rejected unless the
    caller sets `force_clear=True`. Without this guard, an accidental
    empty submission would silently wipe a working key from both
    `cfg.*_API_KEY` and (if persist_env) the .env file.
    """
    out: Dict[str, Any] = {
        "ok": False, "runtime_set": False,
        "persisted": False, "env_error": None,
        "client_reloaded": False, "client_reload_error": None,
    }
    entry = _PROVIDER_KEY_NAMES.get(provider)
    if not entry:
        out["env_error"] = f"Unknown provider: {provider!r}"
        return out
    # Empty-key safety: reject unless caller explicitly opted in. Otherwise
    # a user who accidentally clears the input field and clicks Apply
    # would lose their key without warning.
    if not (new_key or "").strip() and not force_clear:
        out["env_error"] = (
            "Empty API key — refusing to wipe existing. "
            "Pass force_clear=True if you really want to clear it."
        )
        return out
    env_var, cfg_attr = entry
    # 1. Runtime set: write BOTH os.environ (so os.getenv-based readers
    # see it without a restart) AND the config module attribute (so
    # codepaths that imported the value at module-load see the new key).
    try:
        os.environ[env_var] = new_key or ""
        import config as _cfg
        setattr(_cfg, cfg_attr, new_key or "")
        out["runtime_set"] = True
    except Exception as e:
        out["env_error"] = f"runtime set failed: {e}"
        return out

    # 2. Optional persist to .env.
    if persist_env:
        err = _persist_env_updates({env_var: new_key or ""})
        if err:
            out["env_error"] = err
        else:
            out["persisted"] = True

    # 3. Invalidate cached clients so the new key takes effect on the
    # next LLM call (no Streamlit restart needed). Capture any reload
    # error in the result so the caller can surface it — silent pass
    # would hide stale-client state if the new key is malformed and
    # the singleton refuses to rebuild.
    try:
        from claude_provider import get_claude_client
        get_claude_client(reload=True)
        out["client_reloaded"] = True
    except Exception as e:
        out["client_reload_error"] = str(e)[:200]
        out["client_reloaded"] = False

    out["ok"] = True
    return out


def _render_llm_provider_panel(st) -> None:
    """Operator-level control panel for the active LLM provider/model.

    Mutates `config.PROVIDER` and `config.MODEL` at runtime — base_agent reads
    these on every call. Optionally persists the choice to .env so it survives
    a restart. Also refreshes the cached Anthropic Claude singleton if the
    new active provider is Anthropic.
    """
    import config as cfg

    st.markdown("### 🔌 Runtime LLM Provider")
    st.caption(
        "Operator-only switch for the active provider/model. The change takes "
        "effect immediately for new pipeline runs (no restart). Tick "
        "**Persist to .env** to make it survive restarts."
    )
    # Cache-bust marker — if you don't see this line, hard-refresh
    # (Ctrl+F5) to clear the stale JS bundle.
    st.caption(
        f"🔄 Build marker: panel-v2 · "
        f"{len(cfg.SUPPORTED_PROVIDERS)} providers registered "
        f"({', '.join(cfg.SUPPORTED_PROVIDERS)})."
    )
    # ── Diagnostic: explicit DeepSeek check ─────────────────────────────
    # Some browsers don't render the 🐋 emoji and DeepSeek may appear as
    # a blank/invisible row in the dropdown. This explicit indicator
    # confirms DeepSeek IS registered + which dropdown position it's at.
    if "deepseek" in cfg.SUPPORTED_PROVIDERS:
        _ds_idx = cfg.SUPPORTED_PROVIDERS.index("deepseek")
        _ds_key_set = bool(getattr(cfg, "DEEPSEEK_API_KEY", ""))
        st.info(
            f"🐋 **DeepSeek** is registered (dropdown position #{_ds_idx + 1}) — "
            f"API key: {'✓ set' if _ds_key_set else '✗ MISSING'}. "
            f"If you don't see a 🐋 emoji in the dropdown, your "
            f"browser/OS doesn't render it — look for the **DeepSeek** "
            f"text at position #{_ds_idx + 1}."
        )
    else:
        st.error(
            "🐋 DeepSeek is NOT in SUPPORTED_PROVIDERS. This is a real "
            "config bug. Check config.py:_DEFAULT_MODELS."
        )

    # ── Current active state ───────────────────────────────────────────────
    active_provider = (cfg.PROVIDER or "").lower()
    active_model = cfg.MODEL or ""
    a1, a2, a3 = st.columns(3)
    _emoji = _PROVIDER_META.get(active_provider, ("?", "•"))[1]
    _name = _PROVIDER_META.get(active_provider, (active_provider.title(), ""))[0]
    a1.metric("Active provider", f"{_emoji} {_name}")
    a2.metric("Active model", active_model or "—")
    rates = cfg.COST_RATES.get(active_provider, {"input": 0.0, "output": 0.0})
    a3.metric(
        "Rate (in/out / M-tok)",
        f"${rates.get('input', 0):.2f} / ${rates.get('output', 0):.2f}",
    )

    st.divider()

    # ── Provider/key status overview ───────────────────────────────────────
    st.markdown("**Configured providers** (key presence only — no live ping)")
    rows: List[Dict[str, Any]] = []
    for p in cfg.SUPPORTED_PROVIDERS:
        key = _api_key_for(cfg, p)
        meta = _PROVIDER_META.get(p, (p.title(), "•"))
        r = cfg.COST_RATES.get(p, {"input": 0.0, "output": 0.0})
        rows.append({
            "Provider": f"{meta[1]} {meta[0]}",
            "Default model": cfg._DEFAULT_MODELS.get(p, ""),
            "API key": "✓ set" if key else "— missing",
            "Input $/M": f"{r.get('input', 0):.2f}",
            "Output $/M": f"{r.get('output', 0):.2f}",
            "Active": "● yes" if p == active_provider else "",
        })
    try:
        st.dataframe(rows, hide_index=True, use_container_width=True)
    except Exception:
        for row in rows:
            st.write(row)

    st.divider()

    # ── Switch form ────────────────────────────────────────────────────────
    st.markdown("**Switch active provider**")

    def _label(p: str) -> str:
        meta = _PROVIDER_META.get(p, (p.title(), "•"))
        has_key = bool(_api_key_for(cfg, p))
        # Put NAME first so unrenderable emojis (e.g. 🐋 on older Windows
        # fonts) can't accidentally hide an entry by rendering as a blank.
        return f"{meta[0]} {meta[1]} {'✓' if has_key else '(no key)'}"

    try:
        idx = cfg.SUPPORTED_PROVIDERS.index(active_provider)
    except ValueError:
        idx = 0

    new_provider = st.selectbox(
        "Provider",
        options=cfg.SUPPORTED_PROVIDERS,
        index=idx,
        format_func=_label,
        key="admin_llm_provider",
    )

    default_model = cfg._DEFAULT_MODELS.get(new_provider, "")

    if new_provider == "anthropic":
        try:
            from claude_provider import AVAILABLE_MODELS as CLAUDE_MODELS
            labels = {
                "claude-opus-4-7":   "Opus 4.7 — Premium ($15/$75 per M)",
                "claude-sonnet-4-6": "Sonnet 4.6 — Balanced ($3/$15)",
                "claude-haiku-4-5":  "Haiku 4.5 — Fast & Cheap ($1/$5)",
            }
            seed = active_model if (active_provider == "anthropic"
                                       and active_model in CLAUDE_MODELS) \
                else default_model
            cidx = CLAUDE_MODELS.index(seed) if seed in CLAUDE_MODELS else 1
            new_model = st.selectbox(
                "Claude model",
                options=CLAUDE_MODELS,
                index=cidx,
                format_func=lambda m: labels.get(m, m),
                key="admin_llm_claude_model",
            )
        except Exception:
            new_model = st.text_input(
                "Model", value=default_model, key="admin_llm_model_text_fb",
            )
    elif new_provider == "gemini":
        # Curated Gemini model dropdown — sourced from config.GEMINI_KNOWN_MODELS.
        # Includes Gemini 3, 2.5, 2.0, 1.5 LLMs plus Deep Research Pro
        # Preview and Antigravity. Custom… option for unlisted preview names.
        _gemini_models = list(
            getattr(cfg, "GEMINI_KNOWN_MODELS", []) or []
        ) or [default_model]
        _CUSTOM = "✏️ Custom…"
        _seed = (active_model if active_provider == "gemini"
                                          else default_model)
        _options = list(_gemini_models) + [_CUSTOM]
        if _seed in _gemini_models:
            _idx = _gemini_models.index(_seed)
        elif _seed:
            _idx = len(_gemini_models)  # → Custom
        else:
            _idx = 0
        # Map model names to human labels (only the notable ones).
        _gemini_labels = {
            "gemini-3.1-pro":                    "Gemini 3.1 Pro — newest flagship",
            "gemini-3-pro":                      "Gemini 3 Pro — flagship",
            "gemini-3-pro-preview":              "Gemini 3 Pro (preview)",
            "gemini-2.5-pro":                    "Gemini 2.5 Pro — strong reasoning",
            "gemini-2.5-flash":                  "Gemini 2.5 Flash — fast + cheap",
            "gemini-2.5-flash-lite":             "Gemini 2.5 Flash Lite — cheapest 2.5",
            "gemini-2.0-flash":                  "Gemini 2.0 Flash — solid default",
            "gemini-deep-research-pro-preview":  "Deep Research Pro (preview) — research agent",
            "gemini-antigravity-preview":        "Antigravity (preview) — coding agent",
            "gemini-1.5-pro":                    "Gemini 1.5 Pro (legacy)",
            "gemini-1.5-flash":                  "Gemini 1.5 Flash (legacy)",
        }
        _chosen = st.selectbox(
            "Gemini model",
            options=_options,
            index=_idx,
            format_func=lambda m: (
                _CUSTOM if m == _CUSTOM
                else _gemini_labels.get(m, m)
            ),
            key="admin_llm_gemini_model",
            help="Curated list of Gemini family models. Pick **Custom…** "
                  "to type any model name (e.g. a brand-new preview not "
                  "yet in this list).",
        )
        if _chosen == _CUSTOM:
            new_model = st.text_input(
                "Custom Gemini model name",
                value=(_seed if _seed not in _gemini_models else ""),
                key="admin_llm_gemini_model_custom",
                placeholder="e.g. gemini-3.1-flash-preview",
            )
        else:
            new_model = _chosen
        # Helpful pointer for unfamiliar variants.
        if new_model and any(tag in new_model for tag in (
            "deep-research", "antigravity",
        )):
            st.info(
                "ℹ️ This is a **specialized variant** (research / coding "
                "agent). It may require beta access or a different "
                "endpoint than standard chat models. If you hit 404, "
                "use **🎨 Visual Rendering → 📋 List models** to "
                "confirm what your key has access to."
            )
    elif new_provider == "xai":
        # Curated xAI Grok dropdown — sourced from config.XAI_KNOWN_MODELS.
        # Mirrors Anthropic/Gemini/DeepSeek pattern. "xai" is DISTINCT
        # from "groq" — clarified in the friendly labels so users can't
        # confuse the two providers.
        _xai_models = list(
            getattr(cfg, "XAI_KNOWN_MODELS", []) or []
        ) or [default_model]
        _CUSTOM = "✏️ Custom…"
        _seed = (active_model if active_provider == "xai"
                                          else default_model)
        _options = list(_xai_models) + [_CUSTOM]
        if _seed in _xai_models:
            _idx = _xai_models.index(_seed)
        elif _seed:
            _idx = len(_xai_models)
        else:
            _idx = 0
        _xai_labels = {
            # DeepSeek V4 via xAI — most accounts have these
            "deepseek-v4-pro":   "DeepSeek V4 Pro — full quality (works on most xai- keys)",
            "deepseek-v4-flash": "DeepSeek V4 Flash — faster / cheaper",
            # Native Grok — requires higher xAI tier
            "grok-2-latest":     "Grok-2 Latest — needs Grok-tier xAI account",
            "grok-3":            "🆕 Grok-3 — flagship (beta access required)",
            "grok-3-mini":       "Grok-3 Mini — smaller / cheaper",
            "grok-2-1212":       "Grok-2 Dec 2024 — pinned snapshot",
            "grok-2":            "Grok-2 — generic alias",
            "grok-vision-beta":  "Grok Vision Beta — multimodal",
            "grok-beta":         "Grok Beta (legacy)",
        }
        _chosen = st.selectbox(
            "xAI Grok model",
            options=_options,
            index=_idx,
            format_func=lambda m: (
                _CUSTOM if m == _CUSTOM
                else _xai_labels.get(m, m)
            ),
            key="admin_llm_xai_model",
            help="Curated list of xAI Grok chat models. Pick "
                  "**Custom…** to type a new preview alias.",
        )
        if _chosen == _CUSTOM:
            new_model = st.text_input(
                "Custom xAI Grok model name",
                value=(_seed if _seed not in _xai_models else ""),
                key="admin_llm_xai_model_custom",
                placeholder="e.g. grok-3-fast-preview",
            )
        else:
            new_model = _chosen
        # Cost hint for grok-3 beta access.
        if new_model and "grok-3" in new_model:
            st.info(
                "🆕 **Grok-3** is in beta — the xAI API may return "
                "**HTTP 403 / 404** if your account doesn't have access. "
                "Fall back to `grok-2-latest` (production) if you hit "
                "auth errors."
            )

    elif new_provider == "deepseek":
        # Curated DeepSeek model dropdown — sourced from
        # config.DEEPSEEK_KNOWN_MODELS. Mirrors the Gemini UX so users
        # can pick between deepseek-chat (V3.x) and deepseek-reasoner
        # (R1.x) without typing the model name. Custom… for any new
        # preview alias DeepSeek ships before this list is updated.
        _ds_models = list(
            getattr(cfg, "DEEPSEEK_KNOWN_MODELS", []) or []
        ) or [default_model]
        _CUSTOM = "✏️ Custom…"
        _seed = (active_model if active_provider == "deepseek"
                                          else default_model)
        _options = list(_ds_models) + [_CUSTOM]
        if _seed in _ds_models:
            _idx = _ds_models.index(_seed)
        elif _seed:
            _idx = len(_ds_models)
        else:
            _idx = 0
        _ds_labels = {
            "deepseek-chat":     "DeepSeek Chat — auto (always latest production)",
            "deepseek-reasoner": "DeepSeek Reasoner (R1.x) — chain-of-thought",
            "deepseek-v4":       "🆕 DeepSeek V4 — next-gen flagship (may 404 if pre-GA)",
            "deepseek-v3.2":     "DeepSeek V3.2 — non-reasoning, pinned",
            "deepseek-v3.1":     "DeepSeek V3.1 (legacy)",
            "deepseek-v3":       "DeepSeek V3 (legacy)",
            "deepseek-r1.1":     "DeepSeek R1.1 — reasoning, pinned",
            "deepseek-r1":       "DeepSeek R1 (legacy reasoning)",
            "deepseek-coder":    "DeepSeek Coder (legacy — may 404)",
        }
        _chosen = st.selectbox(
            "DeepSeek model",
            options=_options,
            index=_idx,
            format_func=lambda m: (
                _CUSTOM if m == _CUSTOM
                else _ds_labels.get(m, m)
            ),
            key="admin_llm_deepseek_model",
            help="Curated list of DeepSeek family models. Pick "
                  "**Custom…** to type any model name (e.g. a brand-"
                  "new preview alias not yet in this list).",
        )
        if _chosen == _CUSTOM:
            new_model = st.text_input(
                "Custom DeepSeek model name",
                value=(_seed if _seed not in _ds_models else ""),
                key="admin_llm_deepseek_model_custom",
                placeholder="e.g. deepseek-v3.3-preview",
            )
        else:
            new_model = _chosen
        # Cost hint — DeepSeek Reasoner has different pricing than Chat.
        if new_model and "reasoner" in new_model.lower() or "r1" in (new_model or "").lower():
            st.info(
                "ℹ️ **Reasoner** models use chain-of-thought and produce "
                "long visible reasoning traces — much higher output-"
                "token cost than chat. Budget-cap-sensitive runs should "
                "use `deepseek-chat` unless reasoning quality matters."
            )
    else:
        seed = active_model if active_provider == new_provider else default_model
        new_model = st.text_input(
            "Model", value=seed or default_model, key="admin_llm_model_text",
        )

    has_key_for_new = bool(_api_key_for(cfg, new_provider))
    if not has_key_for_new:
        st.warning(
            f"No API key for **{new_provider}**. The switch will apply, "
            f"but calls will fail until you add the matching key in .env."
        )

    # ── deepseek-v4 pre-GA warning ──────────────────────────────────────
    # The next-gen DeepSeek V4 model name is pre-released here for users
    # who want to opt in early. The DeepSeek API may return 404 if V4 is
    # not yet generally available on their account — fall back to
    # deepseek-chat in that case.
    if new_provider == "deepseek" and new_model == "deepseek-v4":
        st.warning(
            "🆕 **DeepSeek V4** is the next-gen flagship — the API may "
            "return **HTTP 404** if it's not yet generally available "
            "on your account. If you hit 404, fall back to "
            "`deepseek-chat` (which auto-routes to the current "
            "production model)."
        )

    # ── API key input (new in panel-v3) ─────────────────────────────────
    # Lets the operator paste / update a provider key directly without
    # editing .env. Optionally persists to .env via the same path the
    # provider/model switch uses. The widget is generic over any of the
    # 7 supported providers — pre-filled with whatever is currently
    # configured. type="password" masks the input so screen sharing
    # during demos doesn't leak the key.
    _existing_key = _api_key_for(cfg, new_provider)
    _key_placeholder = {
        "deepseek":  "sk-...  (DeepSeek)",
        "openai":    "sk-...  (OpenAI)",
        "anthropic": "sk-ant-... (Anthropic Claude)",
        "groq":      "gsk_...  (Groq)",
        "gemini":    "AIza...  (Google AI Studio — NOT gen-lang-client-...)",
        "azure":     "Azure resource key",
        "kimi":      "sk-...  (Moonshot / Kimi)",
    }.get(new_provider, "Paste API key")
    new_api_key = st.text_input(
        f"{new_provider.title()} API key",
        value=_existing_key,
        type="password",
        key=f"admin_llm_api_key_{new_provider}",
        placeholder=_key_placeholder,
        help=(
            "Paste the secret. Leave unchanged to keep the current key. "
            "Use **🧪 Test connection** below to verify before clicking "
            "Apply — a bad key here causes 'No ideas generated' "
            "(HTTP 401) on the next pipeline run. "
            "Tick **Persist to .env** to save permanently — otherwise "
            "the change is in-process only and resets on Streamlit restart."
        ),
    )

    # ── Recovery + verification helpers (test key before commit,
    # ── reset to .env if you typed a bad key) ───────────────────────────
    kc1, kc2 = st.columns([1, 1])
    if kc1.button(
        "🧪 Test connection",
        key=f"admin_llm_test_key_{new_provider}",
        use_container_width=True,
        help="Ping the provider's /models endpoint with the key in the "
              "field above. Catches typos / expired keys BEFORE you "
              "click Apply (so you don't trap yourself with a bad "
              "runtime override).",
    ):
        with st.spinner(f"Testing {new_provider} key…"):
            result = _test_provider_api_key(new_provider, new_api_key)
        if result.get("ok"):
            st.success(result["message"])
        else:
            st.error(result["message"])
    if kc2.button(
        "↩️ Reset to .env value",
        key=f"admin_llm_reset_key_{new_provider}",
        use_container_width=True,
        help="Drop any in-process runtime key override and reload the "
              "value from .env. Useful if you typed a bad key earlier "
              "and want to revert without restarting Streamlit.",
    ):
        result = _reset_provider_key_to_env(new_provider)
        if result.get("ok"):
            if result["had_value"]:
                st.success(
                    f"↩️ {new_provider} key restored from .env "
                    f"(ends in `…{result['env_key_suffix']}`). "
                    f"Hard-refresh (Ctrl+Shift+R) so the input field "
                    f"shows the restored value."
                )
            else:
                st.warning(
                    f".env has no {new_provider} key — runtime override "
                    f"cleared. Either paste a key above and Apply, or "
                    f"add it to .env."
                )
        else:
            st.error(result.get("error", "Reset failed."))

    persist = st.checkbox(
        "Persist to .env (survives restart)",
        value=False,
        key="admin_llm_persist",
    )

    c1, c2 = st.columns([1, 1])
    if c1.button("Apply now", type="primary", use_container_width=True,
                 key="admin_llm_apply"):
        prev_provider, prev_model = cfg.PROVIDER, cfg.MODEL
        cfg.PROVIDER = new_provider
        cfg.MODEL = (new_model or "").strip() or default_model

        # ── API key update (only if the user changed it) ────────────────
        # Diff-check against the value pulled at render time so we
        # don't redundantly call _set_provider_api_key when the field
        # was untouched — that path invalidates the cached client,
        # which is expensive on Anthropic.
        _key_changed = bool(new_api_key) and new_api_key != _existing_key
        _key_result: Optional[Dict[str, Any]] = None
        _client_reload_error: Optional[str] = None
        if _key_changed:
            _key_result = _set_provider_api_key(
                provider=new_provider,
                new_key=new_api_key,
                persist_env=bool(persist),
            )
            # If _set_provider_api_key already reloaded the client, we
            # don't need to reload again below — note its outcome.
            if _key_result and _key_result.get("client_reload_error"):
                _client_reload_error = _key_result["client_reload_error"]

        # Refresh cached Anthropic client so it picks up the new
        # model/key. Skip if _set_provider_api_key above already
        # reloaded — avoids the double-invalidation regression the
        # v6 adversarial reviewer flagged. Only reload here if the
        # provider/model changed AND the key path didn't already do it.
        _provider_or_model_changed = (
            new_provider != prev_provider
            or cfg.MODEL != prev_model
        )
        if _provider_or_model_changed and not _key_changed:
            try:
                from claude_provider import get_claude_client
                get_claude_client(reload=True)
            except Exception as _reload_e:
                _client_reload_error = str(_reload_e)[:200]

        msg_lines = [
            f"Switched **{prev_provider}/{prev_model}** → "
            f"**{cfg.PROVIDER}/{cfg.MODEL}**."
        ]
        if persist:
            err = _persist_to_env(cfg.PROVIDER, cfg.MODEL)
            if err:
                msg_lines.append(f"⚠️ .env write failed: {err}")
            else:
                msg_lines.append("Saved provider/model to `.env`.")

        # ── Report on the API-key update (if any) ───────────────────────
        if _key_result is not None:
            if _key_result.get("ok"):
                if _key_result.get("persisted"):
                    msg_lines.append(
                        f"🔑 **{new_provider}** API key updated + "
                        f"saved to `.env`."
                    )
                else:
                    msg_lines.append(
                        f"🔑 **{new_provider}** API key updated for "
                        f"this session (NOT persisted — tick **Persist "
                        f"to .env** to make it survive a restart)."
                    )
            err = _key_result.get("env_error")
            if err:
                msg_lines.append(f"⚠️ API-key update issue: {err}")

        # ── Surface client-reload failure ──────────────────────────────
        # If the cached Claude client refused to rebuild (e.g. malformed
        # new key), say so explicitly — silently swallowing this hides
        # the fact that the next LLM call may still use the stale key.
        if _client_reload_error:
            msg_lines.append(
                f"⚠️ Claude client reload failed: {_client_reload_error} — "
                f"the cached client may still be using the OLD key. "
                f"Click **Reload Anthropic client** to force a rebuild "
                f"after fixing the key."
            )

        st.success("\n\n".join(msg_lines))
        st.session_state["_admin_llm_last_apply"] = {
            "provider": cfg.PROVIDER, "model": cfg.MODEL,
            "persisted": bool(persist),
            "key_changed": bool(_key_changed),
        }

    if c2.button("Reload Anthropic client", use_container_width=True,
                 key="admin_llm_reload_claude",
                 help="Re-read ANTHROPIC_API_KEY/MODEL/BASE_URL and rebuild "
                       "the cached Claude singleton."):
        try:
            from claude_provider import get_claude_client
            client = get_claude_client(reload=True)
            if client and client.is_configured:
                st.success(f"Claude client reloaded (model={client.model}).")
            else:
                st.warning("Claude client reloaded but is not configured "
                            "(missing ANTHROPIC_API_KEY).")
        except Exception as e:
            st.error(f"Reload failed: {e}")

    last = st.session_state.get("_admin_llm_last_apply")
    if last:
        st.caption(
            f"Last apply (this session): {last['provider']} / {last['model']} "
            f"{'(persisted)' if last['persisted'] else '(session-only)'}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Feature toggles panel
# ─────────────────────────────────────────────────────────────────────────────
#
# Generic registry of operator-settable on/off switches. Each toggle has a
# `cfg_attr` (the attribute on the `config` module), an `env_key` (the
# environment variable that backs it), a default, a label and a description.
# Adding a new toggle is one entry in FEATURE_TOGGLES + one config line.

FEATURE_TOGGLES: List[Dict[str, Any]] = [
    {
        "cfg_attr":    "ENABLE_CORPUS_ANCHORED_NOVELTY",
        "env_key":     "IDEAGRAPH_CORPUS_ANCHORED_NOVELTY",
        "default":     True,
        "label":       "🛰️ Corpus-anchored novelty (Novelty Lab mode Q)",
        "description": (
            "Operationalized novelty: scores each candidate against a "
            "reference corpus (embedding distance to nearest neighbor). "
            "When OFF, the `corpus_anchored` mode is hidden from the "
            "Novelty Lab radio."
        ),
    },
    {
        "cfg_attr":    "ENABLE_VISUAL_RENDERING",
        "env_key":     "IDEAGRAPH_VISUAL_RENDERING",
        "default":     True,
        "label":       "🎨 Visual abstract rendering (FLUX / Nano Banana)",
        "description": (
            "Adds a Visual Abstract panel to each idea card that "
            "generates a paper-figure-style illustration via the FLUX "
            "image API. When OFF, the panel is hidden and no requests "
            "are made. Requires NANO_BANANA_API_KEY in .env (also "
            "settable in the LLM Provider tab below)."
        ),
    },
]


def _persist_toggles_to_env(updates: Dict[str, bool]) -> Optional[str]:
    """Update arbitrary `IDEAGRAPH_*` boolean keys in .env.

    `updates` maps env-var name → bool. Each pair becomes a `KEY=true`
    or `KEY=false` line. Preserves comments and unrelated keys, appends
    missing keys, creates the file if absent. Returns an error string
    on failure or None on success.
    """
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    try:
        if os.path.exists(env_path):
            with open(env_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        else:
            lines = []
        norm = {k: ("true" if v else "false") for k, v in updates.items()}
        seen = {k: False for k in norm}
        out: List[str] = []
        for line in lines:
            stripped = line.lstrip()
            if stripped.startswith("#") or "=" not in stripped:
                out.append(line)
                continue
            key = stripped.split("=", 1)[0].strip()
            if key in norm:
                out.append(f"{key}={norm[key]}\n")
                seen[key] = True
            else:
                out.append(line)
        for k, was_seen in seen.items():
            if not was_seen:
                if out and not out[-1].endswith("\n"):
                    out[-1] = out[-1] + "\n"
                out.append(f"{k}={norm[k]}\n")
        with open(env_path, "w", encoding="utf-8") as f:
            f.writelines(out)
        return None
    except Exception as e:
        return str(e)


def _render_feature_toggles_panel(st) -> None:
    """Operator panel for runtime feature toggles.

    Each toggle's current value is read from the `config` module. Flipping
    a toggle mutates that attribute live (so subsequent renders see the
    new value) and optionally writes the change to .env for persistence
    across restarts.
    """
    import config as cfg

    st.markdown("### 🎚️ Feature Toggles")
    st.caption(
        "Operator-only on/off switches. Changes take effect immediately "
        "for new pipeline runs and Novelty Lab renders. Tick **Persist "
        "to .env** to make a change survive restarts."
    )

    # ── Current state table ───────────────────────────────────────────────
    st.markdown("**Current state**")
    rows: List[Dict[str, Any]] = []
    for toggle in FEATURE_TOGGLES:
        current = bool(getattr(cfg, toggle["cfg_attr"], toggle["default"]))
        rows.append({
            "Toggle":   toggle["label"],
            "State":    "● ON" if current else "○ OFF",
            "Default":  "ON" if toggle["default"] else "OFF",
            "Env var":  toggle["env_key"],
        })
    try:
        st.dataframe(rows, hide_index=True, use_container_width=True)
    except Exception:
        for r in rows:
            st.write(r)

    st.divider()

    # ── Edit form ─────────────────────────────────────────────────────────
    st.markdown("**Flip toggles**")
    pending: Dict[str, bool] = {}
    for toggle in FEATURE_TOGGLES:
        attr = toggle["cfg_attr"]
        current = bool(getattr(cfg, attr, toggle["default"]))
        new_value = st.toggle(
            toggle["label"],
            value=current,
            key=f"admin_toggle_{attr}",
            help=toggle["description"],
        )
        st.caption(toggle["description"])
        pending[attr] = new_value

    persist = st.checkbox(
        "Persist to .env (survives restart)",
        value=False,
        key="admin_toggles_persist",
    )

    c1, c2 = st.columns([1, 1])
    if c1.button("Apply now", type="primary", use_container_width=True,
                 key="admin_toggles_apply"):
        changes: List[str] = []
        env_updates: Dict[str, bool] = {}
        for toggle in FEATURE_TOGGLES:
            attr = toggle["cfg_attr"]
            new_value = pending[attr]
            old_value = bool(getattr(cfg, attr, toggle["default"]))
            if new_value != old_value:
                setattr(cfg, attr, new_value)
                changes.append(
                    f"`{toggle['label']}`: "
                    f"{'ON' if old_value else 'OFF'} → "
                    f"{'ON' if new_value else 'OFF'}"
                )
            env_updates[toggle["env_key"]] = new_value

        if not changes and not persist:
            st.info("No changes.")
        else:
            msg_lines = []
            if changes:
                msg_lines.append("Applied changes:")
                msg_lines.extend(f"- {c}" for c in changes)
            else:
                msg_lines.append("No runtime changes (toggles already at "
                                   "the requested values).")
            if persist:
                err = _persist_toggles_to_env(env_updates)
                if err:
                    msg_lines.append(f"⚠️ .env write failed: {err}")
                else:
                    msg_lines.append("Saved to `.env`.")
            st.success("\n\n".join(msg_lines))

    if c2.button("Reset to defaults", use_container_width=True,
                 key="admin_toggles_reset"):
        for toggle in FEATURE_TOGGLES:
            setattr(cfg, toggle["cfg_attr"], toggle["default"])
        st.success("All toggles reset to defaults (session-only — "
                    "tick **Persist to .env** and Apply to save).")


# ─────────────────────────────────────────────────────────────────────────────
# Visual Rendering panel (FLUX / Nano-Banana image API key + config)
# ─────────────────────────────────────────────────────────────────────────────


def _persist_visual_to_env(
    api_key: str, model: str, endpoint: str, provider: str = "",
) -> Optional[str]:
    """Update NANO_BANANA_API_KEY / _MODEL / _ENDPOINT / _PROVIDER in .env.
    Returns error string on failure, or None on success."""
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    try:
        if os.path.exists(env_path):
            with open(env_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        else:
            lines = []
        updates = {
            "NANO_BANANA_PROVIDER": provider or "flux_bfl",
            "NANO_BANANA_API_KEY": api_key,
            "NANO_BANANA_MODEL":   model,
            "NANO_BANANA_ENDPOINT": endpoint,
        }
        seen = {k: False for k in updates}
        out: List[str] = []
        for line in lines:
            stripped = line.lstrip()
            if stripped.startswith("#") or "=" not in stripped:
                out.append(line)
                continue
            key = stripped.split("=", 1)[0].strip()
            if key in updates:
                out.append(f"{key}={updates[key]}\n")
                seen[key] = True
            else:
                out.append(line)
        for k, was_seen in seen.items():
            if not was_seen:
                if out and not out[-1].endswith("\n"):
                    out[-1] = out[-1] + "\n"
                out.append(f"{k}={updates[k]}\n")
        with open(env_path, "w", encoding="utf-8") as f:
            f.writelines(out)
        return None
    except Exception as e:
        return str(e)


def _render_visual_rendering_panel(st) -> None:
    """Operator panel for the visual-abstract (FLUX / Nano Banana) API.

    Sets:
      - NANO_BANANA_API_KEY  (the actual secret)
      - NANO_BANANA_MODEL    (default flux-pro-1.0)
      - NANO_BANANA_ENDPOINT (default https://api.bfl.ml/v1)

    Same pattern as the LLM Provider tab: mutates `config` at runtime,
    optionally writes back to .env so the change survives restarts.
    """
    import config as cfg

    st.markdown("### 🎨 Visual Rendering — FLUX / Nano Banana")
    st.caption(
        "Configure the image-generation API used by the 🎨 Visual "
        "Abstract panel on each idea card. Changes take effect "
        "immediately for the next idea you render. Tick **Persist "
        "to .env** to keep the change across restarts."
    )

    # Pull the provider registry + defaults from the renderer module so
    # we don't duplicate the catalog. If the import fails (renderer not
    # installed), fall back to a hardcoded minimal list.
    try:
        from ideagraph_image_renderer import (
            PROVIDER_REGISTRY, PROVIDER_DEFAULTS,
        )
        _provider_keys = list(PROVIDER_REGISTRY.keys())
    except Exception:
        PROVIDER_DEFAULTS = {
            "flux_bfl": {"model": "flux-pro-1.0",
                          "endpoint": "https://api.bfl.ml/v1"},
            "gemini_imagen": {
                "model": "imagen-3.0-generate-002",
                "endpoint": "https://generativelanguage.googleapis.com/v1beta",
            },
        }
        _provider_keys = list(PROVIDER_DEFAULTS.keys())

    # ── Current state ──────────────────────────────────────────────────────
    active_provider = (
        getattr(cfg, "NANO_BANANA_PROVIDER", "") or "flux_bfl"
    ).lower()
    active_key = (getattr(cfg, "NANO_BANANA_API_KEY", "") or "").strip()
    active_model = (
        getattr(cfg, "NANO_BANANA_MODEL", "")
        or PROVIDER_DEFAULTS.get(active_provider, {}).get("model", "")
    )
    active_endpoint = (
        getattr(cfg, "NANO_BANANA_ENDPOINT", "")
        or PROVIDER_DEFAULTS.get(active_provider, {}).get("endpoint", "")
    )

    a1, a2, a3, a4 = st.columns(4)
    _provider_emoji = {
        "flux_bfl": "🟠", "gemini_imagen": "🟢",
        "gemini_flash_image": "🟡", "veo": "🎬", "grok": "🤖",
    }.get(active_provider, "•")
    a1.metric("Provider", f"{_provider_emoji} {active_provider}")
    if active_key:
        a2.metric(
            "API key",
            f"✓ set ({len(active_key)} chars)",
            help=f"Last 4 chars: …{active_key[-4:]}",
        )
    else:
        a2.metric("API key", "— not set")
    a3.metric("Model", active_model)
    a4.metric(
        "Endpoint",
        active_endpoint.replace("https://", "").rstrip("/")[:24],
        help=active_endpoint,
    )

    # Feature toggle gate — if visual rendering is OFF, surface a warning.
    if not bool(getattr(cfg, "ENABLE_VISUAL_RENDERING", True)):
        st.warning(
            "🎚️ **Visual rendering is currently DISABLED** (in the "
            "Feature Toggles tab). Even after saving a key here, the "
            "🎨 panel won't appear on idea cards until you re-enable "
            "the toggle."
        )

    st.divider()

    # ── Video models (Veo) — informational only, not yet a provider ────
    try:
        from ideagraph_image_renderer import (
            VEO_VIDEO_MODELS, VEO_INFO_MESSAGE,
        )
        _veo_available = True
    except ImportError:
        _veo_available = False
    if _veo_available:
        with st.expander(
            "🎬 Video models (Veo) — informational",
            expanded=False,
        ):
            st.info(VEO_INFO_MESSAGE)
            st.markdown("**Known Veo models** (use directly via the "
                          "long-running operation API):")
            for vm in VEO_VIDEO_MODELS:
                st.markdown(f"- `{vm}`")

    # ── Where to get a key ─────────────────────────────────────────────────
    with st.expander("Where to get an API key", expanded=not active_key):
        st.markdown(
            """
- **BlackForest Labs (FLUX official)** — https://api.bfl.ml
  - Format: `bfl-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`
  - Pay-as-you-go, ~$0.05/image at FLUX-pro-1.0
  - **Recommended default** — the renderer's `FluxBFLProvider` targets this endpoint.
- **Nano Banana gateway** — https://nanobang.com
  - Compatible with the same endpoint shape (POST + poll)
- **Runway ML** — https://www.runwayml.com
  - Different endpoint shape — requires a custom `RunwayProvider` subclass
    (see INTEGRATION_GUIDE.md, "Custom providers" section).
- **Google AI Studio (Gemini Image / Imagen)** — https://aistudio.google.com/apikey
  - Format: `AIzaSy...` (39 chars, starts with `AIza`)
  - **Free tier**: text-only models. Image generation usually returns 404.
  - **Paid tier** (buy credit at https://aistudio.google.com/usage):
    - Imagen 4 / 3 (provider = `gemini_imagen`, endpoint `:predict`)
    - Gemini 2.5 Flash Image (provider = `gemini_flash_image`, endpoint `:generateContent`)
  - `gen-lang-client-*` is a client identifier, **not** an API key.
  - Pricing: ~$0.02–$0.04/image for Imagen 4, ~$0.039/image for Gemini Flash Image.
            """
        )

    # ── Edit form ──────────────────────────────────────────────────────────
    st.markdown("**Set or update the provider + key**")

    # Provider dropdown — switching this auto-fills model + endpoint
    # with the per-provider default (via the on_change reset).
    _labels = {
        "flux_bfl": "🟠 FLUX via BlackForest Labs (api.bfl.ml)",
        "gemini_imagen": (
            "🟢 Google AI Studio (paid) — Imagen 3 / 4 "
            "(image, ~$0.02-0.06/img)"
        ),
        "gemini_flash_image": (
            "🟡 Google AI Studio (paid) — Gemini Flash Image "
            "(\"Nano Banana\", image)"
        ),
        "veo": (
            "🎬 Google AI Studio (paid) — Veo 3 "
            "(VIDEO clips, ~$0.30-0.75/clip)"
        ),
        "grok": (
            "🤖 xAI Grok — grok-2-image "
            "(image only; xAI has no public video API)"
        ),
    }
    _idx = _provider_keys.index(active_provider) \
        if active_provider in _provider_keys else 0
    new_provider = st.selectbox(
        "Provider",
        options=_provider_keys,
        index=_idx,
        format_func=lambda k: _labels.get(k, k),
        key="admin_visual_provider",
        help="The image-generation backend. Switching here also "
              "suggests sensible model + endpoint defaults below.",
    )
    # When the dropdown selection differs from the currently-saved
    # provider, the user is mid-switch — pre-fill the model + endpoint
    # with the new provider's defaults so they don't have to type them.
    _switching = (new_provider != active_provider)
    _defaults_for_new = PROVIDER_DEFAULTS.get(new_provider, {})
    _default_model = _defaults_for_new.get("model", "")
    _default_endpoint = _defaults_for_new.get("endpoint", "")
    if _switching:
        st.info(
            f"ℹ️ Switching provider to **{new_provider}**. The fields "
            f"below show its defaults — adjust if needed, then click "
            f"**Apply now**."
        )

    # Provider-specific access caveats.
    if new_provider == "gemini_imagen":
        st.info(
            "💡 **Google AI Studio (paid) — Imagen.** With paid credit, "
            "try **`imagen-4.0-generate-001`** first (newest GA) — if "
            "404, fall back to `imagen-3.0-generate-002` then "
            "`imagen-3.0-fast-generate-001`. Use **📋 List models** "
            "below to confirm exact names available on your key."
        )
    elif new_provider == "gemini_flash_image":
        st.info(
            "💡 **Google AI Studio (paid) — Gemini Flash Image.** Uses "
            "the `:generateContent` endpoint. Try **`gemini-2.5-flash-image`** "
            "first; if 404, the older `gemini-2.0-flash-preview-image-"
            "generation` is the most reliable fallback. Use **📋 List "
            "models** below if both fail."
        )
    elif new_provider == "veo":
        st.warning(
            "🎬 **Veo 3 generates VIDEO, not still images.** Each clip "
            "is 4-8 seconds, ~$0.30-0.75 per generation, takes 30-120s. "
            "Requires paid AI Studio credit. The 🎨 Visual Abstract "
            "panel on each idea card will now produce a **video** "
            "instead of an image. To go back to image generation, "
            "switch the provider above."
        )
    elif new_provider == "grok":
        st.info(
            "🤖 **xAI Grok (image).** OpenAI-compatible endpoint at "
            "`https://api.x.ai/v1/images/generations`. Key format "
            "`xai-…`. Image-only — for video the Visual Simulation tab "
            "keeps using the **veo** provider regardless of what's "
            "set here. The key resolves from `GROK_API_KEY` (or "
            "`XAI_API_KEY`) in `.env` and takes precedence over the "
            "generic `NANO_BANANA_API_KEY` chain."
        )

    new_key = st.text_input(
        "API key",
        value=active_key,
        type="password",
        key="admin_visual_api_key",
        placeholder=(
            "bfl-…  (FLUX)"
            if new_provider == "flux_bfl"
            else "xai-…  (xAI Grok)"
            if new_provider == "grok"
            else "AIza…  (Google AI Studio, NOT gen-lang-client-…)"
        ),
        help="Paste the actual secret. For Google: get an `AIza…` key "
              "from https://aistudio.google.com/apikey — the "
              "`gen-lang-client-…` value is a client identifier, "
              "not a key.",
    )
    # Model selector — known variants in a dropdown, plus a "Custom…"
    # option that reveals a text input for unlisted model names.
    _known_models: List[str] = (
        _defaults_for_new.get("known_models") or []
    )
    _CUSTOM_SENTINEL = "✏️ Custom…"
    _current_model_value = _default_model if _switching else active_model
    # Build options: known models first, then custom-sentinel.
    _model_options = list(_known_models) + [_CUSTOM_SENTINEL]
    # Pre-select the current model if it's in the known list; otherwise
    # default to Custom so the user can see their freeform value.
    if _current_model_value in _known_models:
        _model_idx = _known_models.index(_current_model_value)
    elif _current_model_value:
        # User has a model name not in the known list — surface it via Custom.
        _model_idx = len(_known_models)
    else:
        _model_idx = 0

    _chosen_model = st.selectbox(
        "Model",
        options=_model_options,
        index=_model_idx,
        key=f"admin_visual_model_select_{new_provider}",
        help=(
            "Pick a known model for this provider. Choose **Custom…** "
            "to type any model name (e.g. for previews not in this list "
            "yet)."
        ),
    )
    if _chosen_model == _CUSTOM_SENTINEL:
        new_model = st.text_input(
            "Custom model name",
            value=(_current_model_value
                    if _current_model_value not in _known_models else ""),
            key=f"admin_visual_model_custom_{new_provider}",
            placeholder=(
                "flux-pro-1.1-ultra" if new_provider == "flux_bfl"
                else "imagen-4.0-generate-preview-06-06"
                if new_provider == "gemini_imagen"
                else "gemini-2.5-flash-image-preview"
            ),
        )
    else:
        new_model = _chosen_model
    new_endpoint = st.text_input(
        "Endpoint",
        value=(_default_endpoint if _switching else active_endpoint),
        key=f"admin_visual_endpoint_{new_provider}",  # remount on switch
        help=(
            "Override only if using Runway / Replicate / self-hosted FLUX."
            if new_provider == "flux_bfl"
            else "Google's Generative Language API v1beta base URL."
        ),
    )

    # ── Quick sanity check on key format vs provider ───────────────────────
    if new_key:
        looks_like_flux = new_key.startswith("bfl") or len(new_key) >= 40
        looks_like_google = new_key.startswith("AIza")
        looks_like_clientid = new_key.startswith("gen-lang-client-")

        if looks_like_clientid:
            st.error(
                "⚠️ `gen-lang-client-…` is a Google **client identifier**, "
                "not an API key. The actual API key (from "
                "https://aistudio.google.com/apikey) starts with `AIza…`. "
                "Saving this will produce 401/403 errors."
            )
        elif new_provider == "flux_bfl" and looks_like_google:
            st.warning(
                "⚠️ The provider is set to **FLUX (api.bfl.ml)** but the "
                "key starts with `AIza…` (Google AI Studio format). "
                "Switch the provider to **gemini_imagen** above, or paste "
                "a BFL key (`bfl-…`)."
            )
        elif new_provider == "gemini_imagen" and looks_like_flux \
                and not looks_like_google:
            st.warning(
                "⚠️ The provider is set to **Google Imagen** but the "
                "key doesn't look like an `AIza…` Google AI Studio key. "
                "Either switch the provider to **flux_bfl** above, or "
                "paste an `AIza…` key from "
                "https://aistudio.google.com/apikey."
            )
        elif not (looks_like_flux or looks_like_google):
            st.info(
                f"ℹ️ Unrecognized key format ({len(new_key)} chars). "
                "FLUX/BFL keys are usually ≥40 chars; Google AI Studio "
                "keys start with `AIza`. Saving anyway — generation may "
                "fail with a 401/403 if the key isn't valid for the "
                "configured provider."
            )

    persist = st.checkbox(
        "Persist to .env (survives restart)",
        value=True,
        key="admin_visual_persist",
    )

    c1, c2 = st.columns([1, 1])
    if c1.button(
        "Apply now",
        type="primary",
        use_container_width=True,
        key="admin_visual_apply",
    ):
        # Update the live config first. Fall back to per-provider
        # defaults when a field was left blank.
        _defaults = PROVIDER_DEFAULTS.get(new_provider, {})
        cfg.NANO_BANANA_PROVIDER = new_provider
        cfg.NANO_BANANA_API_KEY = (new_key or "").strip()
        cfg.NANO_BANANA_MODEL = (
            (new_model or "").strip() or _defaults.get("model", "")
        )
        cfg.NANO_BANANA_ENDPOINT = (
            (new_endpoint or "").strip() or _defaults.get("endpoint", "")
        )
        # Mirror to the env vars so anything reading os.getenv at call
        # time (the renderer's key resolution path) sees the new value.
        os.environ["NANO_BANANA_PROVIDER"] = cfg.NANO_BANANA_PROVIDER
        os.environ["NANO_BANANA_API_KEY"] = cfg.NANO_BANANA_API_KEY
        os.environ["NANO_BANANA_MODEL"] = cfg.NANO_BANANA_MODEL
        os.environ["NANO_BANANA_ENDPOINT"] = cfg.NANO_BANANA_ENDPOINT

        msg_lines = [
            f"Applied: provider=**{cfg.NANO_BANANA_PROVIDER}**, "
            f"model=**{cfg.NANO_BANANA_MODEL}**, "
            f"endpoint=`{cfg.NANO_BANANA_ENDPOINT}`, "
            f"key={'(set)' if cfg.NANO_BANANA_API_KEY else '(empty)'}.",
        ]
        if persist:
            err = _persist_visual_to_env(
                cfg.NANO_BANANA_API_KEY,
                cfg.NANO_BANANA_MODEL,
                cfg.NANO_BANANA_ENDPOINT,
                provider=cfg.NANO_BANANA_PROVIDER,
            )
            if err:
                msg_lines.append(f"⚠️ .env write failed: {err}")
            else:
                msg_lines.append("Saved to `.env`.")
        st.success("\n\n".join(msg_lines))

    # ── Diagnostic: list models the user's key can actually access ────────
    # Google gates image-gen models per account. A user can hit 404 on
    # several Imagen / Flash-Image model names with a perfectly valid
    # API key. Calling ListModels reveals exactly what's enabled.
    if new_provider in ("gemini_imagen", "gemini_flash_image", "veo"):
        if st.button(
            "📋 List models my key can access (free, no image generated)",
            use_container_width=True,
            key="admin_visual_list_models",
            help="Calls Google's ListModels endpoint to enumerate the "
                  "models this API key has access to. Image-generation "
                  "candidates appear at the top.",
            disabled=not (active_key or new_key),
        ):
            try:
                from ideagraph_image_renderer import list_gemini_models
                _key_to_use = (new_key or active_key).strip()
                _ep_to_use = (
                    (new_endpoint or "").strip()
                    or active_endpoint
                    or "https://generativelanguage.googleapis.com/v1beta"
                )
                with st.spinner("Calling Google ListModels…"):
                    _list = list_gemini_models(
                        _key_to_use, endpoint=_ep_to_use,
                    )
                st.session_state["_admin_visual_listed"] = _list
            except Exception as _e:
                st.session_state["_admin_visual_listed"] = {
                    "error": f"{type(_e).__name__}: {_e}",
                }

        _listed = st.session_state.get("_admin_visual_listed")
        if _listed:
            if "error" in _listed:
                st.error(
                    f"❌ ListModels failed: {_listed['error']}\n\n"
                    "If you got HTTP 403/404 here, the key likely "
                    "doesn't have access to the Generative Language "
                    "API. Enable it at https://console.cloud.google.com"
                    "/apis/library/generativelanguage.googleapis.com."
                )
            else:
                models = _listed.get("models", [])
                img_models = [m for m in models if m["supports_image_gen"]]
                st.success(
                    f"✅ ListModels: your key can see "
                    f"**{_listed['count']}** model(s) total · "
                    f"**{len(img_models)}** look like image-generation."
                )
                if img_models:
                    st.markdown("**🎨 Image-generation candidates** "
                                "(click one to fill it into the Model field)")
                    for m in img_models[:10]:
                        if st.button(
                            f"📌 `{m['name']}` — "
                            f"methods: {', '.join(m['generation_methods'])}",
                            key=f"admin_visual_pick_{m['name']}",
                            use_container_width=True,
                        ):
                            # Stash in pending so the rerun picks it up
                            # via the model selectbox + Custom option.
                            st.session_state[
                                f"admin_visual_model_select_{new_provider}"
                            ] = "✏️ Custom…"
                            st.session_state[
                                f"admin_visual_model_custom_{new_provider}"
                            ] = m["name"]
                            st.rerun()
                else:
                    st.warning(
                        "⚠️ Your key sees other models but **none look "
                        "like image generation**. Likely reasons:\n\n"
                        "1. **Image-gen models require paid Vertex AI** "
                        "(not just an AI Studio key). Add billing at "
                        "https://console.cloud.google.com/billing.\n"
                        "2. Image-gen access may be **gated by region**.\n"
                        "3. Try **🟠 FLUX via BlackForest Labs** instead "
                        "— pay-as-you-go (~$0.05/img) with no gating."
                    )
                with st.expander(
                    f"All {_listed['count']} models visible to your key",
                    expanded=False,
                ):
                    for m in models:
                        st.caption(
                            f"`{m['name']}` — "
                            f"{m.get('display_name', '')} "
                            f"({', '.join(m['generation_methods'])})"
                        )
                if st.button(
                    "🗑️ Hide model list",
                    key="admin_visual_hide_list",
                    use_container_width=True,
                ):
                    st.session_state.pop("_admin_visual_listed", None)
                    st.rerun()

    if c2.button(
        "Test with a tiny prompt",
        use_container_width=True,
        key="admin_visual_test",
        help="Sends a single image-generation request to verify the key + "
              "endpoint work. Costs one image-worth of credit.",
        disabled=not active_key,
    ):
        try:
            from ideagraph_image_renderer import NanoBananaImageRenderer
            renderer = NanoBananaImageRenderer(
                api_key=cfg.NANO_BANANA_API_KEY,
                model=cfg.NANO_BANANA_MODEL,
                endpoint=cfg.NANO_BANANA_ENDPOINT,
                provider_name=cfg.NANO_BANANA_PROVIDER,
            )
            with st.spinner(
                f"Calling {renderer.provider.name} "
                f"(model={cfg.NANO_BANANA_MODEL})…"
            ):
                visual = renderer.render(
                    {
                        "title": "API connectivity test",
                        "method": "Render a simple test pattern.",
                        "methodology_type": "system_design",
                    },
                    force=True,  # bypass cache so we actually hit the API
                )
            if visual.success:
                st.success(
                    f"✅ Test passed in {visual.attempts} attempt(s). "
                    f"Provider: `{visual.provider}`, model: `{visual.model}`."
                )
                # Render either as a video or as an image based on the
                # media_type the renderer stamped.
                _media_src = visual.cached_path or visual.image_url
                if _media_src:
                    try:
                        if getattr(visual, "is_video", False):
                            st.video(_media_src)
                            st.caption("Test clip")
                        else:
                            st.image(
                                visual.cached_path,
                                caption="Test image",
                                width=200,
                            )
                    except Exception:
                        pass
            else:
                st.error(
                    f"❌ Test failed: {visual.error}\n\n"
                    f"Provider: `{visual.provider}`, model: "
                    f"`{visual.model}`, attempts: {visual.attempts}."
                )
                # 404 from a Google provider almost always means "wrong
                # model name for your tier". Automatically pull
                # ListModels so the user sees what actually works
                # without having to click another button.
                _is_google = visual.provider in (
                    "gemini_imagen", "gemini_flash_image", "veo",
                )
                _is_404 = "404" in (visual.error or "")
                if _is_google and _is_404:
                    st.info(
                        "🔍 Auto-running **ListModels** to find a model "
                        "name that works with your key…"
                    )
                    try:
                        from ideagraph_image_renderer import (
                            list_gemini_models,
                        )
                        _list = list_gemini_models(
                            cfg.NANO_BANANA_API_KEY,
                            endpoint=cfg.NANO_BANANA_ENDPOINT
                            or "https://generativelanguage.googleapis.com/v1beta",
                        )
                        st.session_state["_admin_visual_listed"] = _list
                        if "error" in _list:
                            st.error(
                                f"ListModels also failed: "
                                f"{_list['error']}"
                            )
                        else:
                            _img_candidates = [
                                m for m in _list.get("models", [])
                                if m["supports_image_gen"]
                            ]
                            if _img_candidates:
                                st.success(
                                    f"✅ Found **{len(_img_candidates)}** "
                                    f"image-generation model(s) your key "
                                    f"CAN access. Pick one:"
                                )
                                # Render each as a clickable button that
                                # auto-fills the Model field.
                                for m in _img_candidates[:10]:
                                    if st.button(
                                        f"📌 Use `{m['name']}`",
                                        key=f"admin_visual_pick_after_test_{m['name']}",
                                        use_container_width=True,
                                    ):
                                        st.session_state[
                                            f"admin_visual_model_select_{new_provider}"
                                        ] = "✏️ Custom…"
                                        st.session_state[
                                            f"admin_visual_model_custom_{new_provider}"
                                        ] = m["name"]
                                        st.rerun()
                            else:
                                st.warning(
                                    "⚠️ ListModels succeeded but found "
                                    "**zero image-generation models** "
                                    "for your key. Your AI Studio tier "
                                    "may not yet have image-gen enabled "
                                    "for this region/account. Consider "
                                    "**🟠 FLUX via BlackForest Labs** "
                                    "(api.bfl.ml) — no Google gating, "
                                    "pay-as-you-go from $0.05/image."
                                )
                    except Exception as _le:
                        st.warning(
                            f"Auto-ListModels failed: {_le}. Click "
                            f"**📋 List models** manually above."
                        )
        except Exception as e:
            st.error(f"Test crashed: {type(e).__name__}: {e}")


def _render_stats(st) -> None:
    """The original platform-stats panel, factored out for tabbing."""
    stats = get_admin_stats()
    if not stats or stats.get("error"):
        st.error(f"Failed to load stats: {stats.get('error', 'Unknown')}")
        return

    # ── Key metrics row ──────────────────────────────────────────────────────
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total Users", stats["total_users"], delta=f"+{stats['new_users_7d']} this week")
    m2.metric("Active (7d)", stats["active_users_7d"])
    m3.metric("Total Runs", stats["total_runs"], delta=f"+{stats['runs_7d']} this week")
    m4.metric("Total Ideas", stats["total_ideas"])
    m5.metric("Monthly Revenue", f"${stats['monthly_revenue_usd']}")

    st.divider()

    # ── Financial row ────────────────────────────────────────────────────────
    f1, f2, f3, f4 = st.columns(4)
    f1.metric("Total LLM Cost", f"${stats['total_cost_usd']:.2f}")
    f2.metric("Cost (7d)", f"${stats['cost_7d_usd']:.2f}")
    margin = stats["monthly_revenue_usd"] - stats["total_cost_usd"]
    f3.metric("Gross Margin", f"${margin:.0f}/mo")
    f4.metric("Avg Runtime", f"{stats['avg_runtime_seconds']:.0f}s")

    st.divider()

    # ── Tier breakdown ───────────────────────────────────────────────────────
    st.subheader("Subscription Tiers")
    tiers = stats.get("tier_counts", {})
    if tiers:
        tier_prices = {"free": 0, "pro": 15, "team": 49, "enterprise": 299}
        for tier, count in sorted(tiers.items()):
            price = tier_prices.get(tier, 0)
            st.markdown(f"**{tier.title()}**: {count} users (${price * count}/mo)")
    else:
        st.caption("No subscription data yet.")

    # ── Sharing metrics ──────────────────────────────────────────────────────
    st.subheader("Community & Sharing")
    s1, s2, s3 = st.columns(3)
    s1.metric("Shared Ideas", stats["total_shares"])
    s2.metric("Total Views", stats["total_views"])
    s3.metric("Total Likes", stats["total_likes"])

    # ── Top topics ───────────────────────────────────────────────────────────
    st.subheader("Top Topics (by run count)")
    for t in stats.get("top_topics", [])[:10]:
        st.markdown(f"- **{t['topic'][:50]}** — {t['cnt']} runs, avg coverage {t.get('avg_cov', 0):.0%}")

    # ── System health ────────────────────────────────────────────────────────
    st.subheader("System Health")
    try:
        from production_optimization import health_snapshot
        health = health_snapshot()
        st.json(health)
    except Exception:
        st.caption("Production optimization module not available.")


# ── Billing admin panel ─────────────────────────────────────────────────────

def _render_billing_panel(st) -> None:
    """Admin Billing tab: plan catalog overview + Stripe configuration +
    per-user tier override.

    Sections:
      1. Plan catalog at a glance — sanity-check what each tier unlocks.
      2. Stripe configuration & connection test (key status, price IDs,
         live round-trip to Stripe API).
      3. Distribution — how many users on each tier (free-tier conversion
         funnel insight).
      4. Per-user override — set any user's tier and optionally reset
         their monthly run counter (grant comps to testers).
    """
    import billing

    st.markdown("### 💳 Plan catalog")
    st.caption(
        "Reference view of every tier and what it unlocks. Edit "
        "`billing.py` to change prices, features, or quotas."
    )
    cols = st.columns(len(billing.PLANS))
    for col, tier in zip(cols, billing.PLANS.keys()):
        plan = billing.PLANS[tier]
        with col:
            st.markdown(
                f"**{plan.label}**  \n"
                f"`${plan.price_usd_monthly:.0f}/mo`"
            )
            quota = (
                "Unlimited" if plan.monthly_run_limit < 0
                else f"{plan.monthly_run_limit}/mo"
            )
            st.caption(f"Quota: {quota}")
            st.caption(f"Unlocks: {len(plan.unlocks)} features")
            with st.expander("Features", expanded=False):
                for f in plan.features:
                    st.markdown(f"- {f}")

    st.divider()

    # ── Stripe configuration ───────────────────────────────────────────
    _render_stripe_configuration_panel(st)

    st.divider()

    # ── Tier distribution ──────────────────────────────────────────────
    st.markdown("### 📊 Tier distribution")
    try:
        import db as _db_mod
        with _db_mod._lock:
            conn = _db_mod._get_conn()
            try:
                rows = conn.execute(
                    "SELECT COALESCE(s.tier, 'free') AS tier, "
                    "       COUNT(*) AS n "
                    "  FROM users u "
                    "  LEFT JOIN subscriptions s ON s.user_id = u.id "
                    " GROUP BY COALESCE(s.tier, 'free')"
                ).fetchall()
            finally:
                conn.close()
        counts = {r["tier"]: r["n"] for r in rows}
        if counts:
            dist_cols = st.columns(len(billing.PLANS))
            for col, tier in zip(dist_cols, billing.PLANS.keys()):
                with col:
                    n = counts.get(tier, 0)
                    st.metric(billing.PLANS[tier].label, n)
        else:
            st.info("No users registered yet.")
    except Exception as e:
        st.caption(f"Couldn't compute tier distribution: {e}")

    st.divider()

    # ── Per-user override (provided by billing module) ─────────────────
    billing.render_admin_plan_override(st)


# ── Stripe configuration sub-panel ──────────────────────────────────────────

def _render_stripe_configuration_panel(st) -> None:
    """Live Stripe config status, price-ID assignments, and a one-click
    connection test. Mutates os.environ at runtime when keys are set
    inline — does NOT persist to .env (operator must edit .env to make
    keys survive a restart). The inline path is for emergency comp work.
    """
    st.markdown("### 💳 Stripe configuration")
    try:
        import stripe_integration as _si
    except Exception as e:
        st.error(f"stripe_integration import failed: {e}")
        return

    diag = _si.test_connection()

    # ── Status row ──────────────────────────────────────────────────────
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric(
        "SDK installed",
        "✅ Yes" if diag["sdk_installed"] else "❌ No",
    )
    col_b.metric(
        "API key",
        "✅ Set" if diag["key_present"] else "❌ Missing",
    )
    col_c.metric(
        "Mode",
        diag["mode"].title() if diag["mode"] != "unknown" else "—",
    )
    col_d.metric(
        "Connection",
        "✅ OK" if diag["ok"] else "❌ Failed",
    )

    if diag.get("error"):
        st.warning(f"Stripe API error: {diag['error']}")
    if diag.get("account_id"):
        st.caption(f"Connected to Stripe account: `{diag['account_id']}`")

    # ── Price IDs status ─────────────────────────────────────────────────
    st.markdown("**Price IDs** (set in .env)")
    px_a, px_b, px_c = st.columns(3)
    for col, tier in zip((px_a, px_b, px_c), ("pro", "team", "enterprise")):
        with col:
            pid = _si.PRICE_IDS.get(tier) or ""
            ok = bool(pid)
            label = "✅" if ok else "❌"
            st.markdown(
                f"**{label} {tier.title()}** — `{pid or '(unset)'}`"
            )

    # ── Quick-test card hint (test mode only) ───────────────────────────
    if diag.get("mode") == "test":
        st.info(
            "🧪 **Test mode** — use card `4242 4242 4242 4242`, any "
            "future date, any CVC, any postal code. No real charge."
        )
    elif diag.get("mode") == "live":
        st.warning(
            "⚠️ **Live mode** — real charges. Make sure you've tested "
            "the flow end-to-end in test mode first."
        )
    else:
        st.info(
            "Key prefix not recognized. Stripe keys are formatted "
            "`sk_test_...` (test) or `sk_live_...` (live)."
        )

    # ── Setup help ──────────────────────────────────────────────────────
    with st.expander("📖 How to set up Stripe", expanded=False):
        st.markdown(
            """
1. **Sign up at [stripe.com](https://stripe.com)** (free, no card required).
2. From the dashboard, copy your **test mode** keys:
   - Secret key (`sk_test_…`) — keep this server-side only
   - Publishable key (`pk_test_…`) — safe to expose
3. Create three products at **[dashboard.stripe.com/test/products](https://dashboard.stripe.com/test/products)**:
   - **Pro** — Recurring, $15/month
   - **Team** — Recurring, $49/month
   - **Enterprise** — Recurring, $299/month
4. For each product, copy the **Price ID** (looks like `price_1A2b3C…`).
5. Edit your `.env` and add:
   ```
   STRIPE_SECRET_KEY=sk_test_...
   STRIPE_PUBLIC_KEY=pk_test_...
   STRIPE_PRICE_PRO=price_...
   STRIPE_PRICE_TEAM=price_...
   STRIPE_PRICE_ENTERPRISE=price_...
   APP_BASE_URL=http://localhost:8510
   ```
6. Restart Streamlit.
7. Come back here and click **🔄 Re-test connection**. Expect `✅ OK`.

**Going live:**
- Activate your Stripe account (business details, bank info).
- Replace `sk_test_…` / `pk_test_…` with `sk_live_…` / `pk_live_…`.
- Recreate the products in live mode and update the price IDs.
- Update `APP_BASE_URL` to your public HTTPS URL.
            """
        )

    # ── Manual re-test button ───────────────────────────────────────────
    if st.button(
        "🔄 Re-test Stripe connection", key="stripe_retest_btn",
    ):
        # Force a fresh diagnostic (recomputes account.retrieve).
        diag2 = _si.test_connection()
        if diag2["ok"]:
            st.success(
                f"✅ Connected to Stripe ({diag2['mode']} mode, "
                f"account `{diag2['account_id']}`)."
            )
        else:
            st.error(
                f"❌ Connection failed: {diag2.get('error', 'unknown error')}"
            )
