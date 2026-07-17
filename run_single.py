"""Run a single pipeline and save to anon-user account. Called by run_batch2.py."""
import json, time, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

topic = sys.argv[1]
tag = sys.argv[2]

import config as cfg
cfg.PROVIDER = 'azure'
cfg.MODEL = 'DeepSeek-V3.2-Speciale'

import db
db.init_db()
user_id = db.login_user('anon-user', 'Muha9999!')

print(f"[{tag}] Starting: {topic[:70]}...")
start = time.time()

from pipeline import IdeaGraphPipeline
pipeline = IdeaGraphPipeline()
results = pipeline.run(
    topic=topic,
    budget_usd=5.0,
    max_iterations=50,
    on_progress=lambda msg: print(f"  {msg}"),
)

elapsed = time.time() - start
ideas = results.get('ideas', [])
coverage = results.get('coverage', 0)
print(f"[{tag}] RESULT: {len(ideas)} ideas, {coverage:.1%} coverage, {elapsed:.0f}s")

if ideas:
    rid = db.save_result(user_id, topic, coverage, len(ideas), results)
    print(f"[{tag}] SAVED: result_id={rid}")
else:
    print(f"[{tag}] SKIPPED (0 ideas)")

outfile = f"output/results_{tag}.json"
with open(outfile, 'w') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"[{tag}] DONE: {len(ideas)} ideas")
