"""Sequential pipeline runs with process isolation via subprocess."""
import subprocess, sys, os, json, time

TOPICS = [
    # Agents with LLMs - different angles
    ("retrieval augmented generation and tool augmented language models", "agents_rag"),
    ("LLM planning and reasoning with code generation", "agents_plan"),
    # Generative AI - different angles  
    ("stable diffusion image generation and text to image synthesis", "genai_diffusion"),
    ("generative adversarial networks for image and video synthesis", "genai_gan"),
    # GNN - different angles
    ("graph attention networks and message passing neural networks", "gnn_attention"),
    ("knowledge graph embedding and graph representation learning", "gnn_knowledge"),
    # Social Norms - different angles
    ("computational social science agent based modeling and simulation", "social_abm"),
    ("cooperative artificial intelligence and multi-agent communication", "social_coop"),
]

total_new = 0
for i, (topic, tag) in enumerate(TOPICS):
    print(f"\n{'='*60}")
    print(f"[{i+1}/{len(TOPICS)}] {tag}: {topic}")
    print(f"{'='*60}")
    
    result = subprocess.run(
        [sys.executable, "run_single.py", topic, tag],
        capture_output=True, text=True, timeout=600,
        cwd=os.path.dirname(os.path.abspath(__file__))
    )
    
    # Print output
    for line in result.stdout.strip().split('\n'):
        print(f"  {line}")
    if result.stderr:
        for line in result.stderr.strip().split('\n')[-5:]:
            print(f"  ERR: {line}")
    
    # Check result
    outfile = f"output/results_{tag}.json"
    if os.path.exists(outfile):
        with open(outfile) as f:
            r = json.load(f)
        n = len(r.get('ideas', []))
        if n > 0:
            total_new += n
            print(f"  >>> {n} ideas saved!")
        else:
            print(f"  >>> 0 ideas (will retry)")
    
    # Wait between runs to respect Semantic Scholar rate limit
    if i < len(TOPICS) - 1:
        print("  Waiting 5s for API rate limits...")
        time.sleep(5)

print(f"\n{'='*60}")
print(f"SEQUENTIAL BATCH DONE: {total_new} new ideas")

# Final account summary
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import db
db.init_db()
uid = db.login_user('anon-user', 'Muha9999!')
results = db.get_user_results(uid)
grand = sum(r['ideas_count'] for r in results)
print(f"anon-user TOTAL: {len(results)} runs, {grand} ideas")
print(f"{'='*60}")
