"""Batch runner: 8 pipeline runs in sequence, saving each to anon-user account."""
import json, time, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config as cfg
cfg.PROVIDER = 'azure'
cfg.MODEL = 'DeepSeek-V3.2-Speciale'

import db
db.init_db()
user_id = db.login_user('anon-user', 'Muha9999!')
print(f"Logged in as anon-user (user_id={user_id})")

TOPICS = [
    # Agents with LLMs - 2 more sub-topics
    ("LLM agent tool use, function calling, API orchestration, and retrieval augmented generation for autonomous task completion", "agents_llm_2"),
    ("multi-agent debate, self-reflection, chain-of-thought verification, and collaborative reasoning in large language models", "agents_llm_3"),
    # Generative AI - 2 more sub-topics
    ("text-to-image generation, controllable diffusion models, image editing, inpainting, and style transfer with neural networks", "gen_ai_2"),
    ("video generation, 3D content synthesis, neural radiance fields, and generative models for audio and music", "gen_ai_3"),
    # GNN - 2 more sub-topics
    ("graph transformers, heterogeneous graph learning, temporal graph networks, and scalable graph neural network architectures", "gnn_2"),
    ("link prediction, node classification, graph generation, and graph neural networks for combinatorial optimization", "gnn_3"),
    # Social Norms - 2 more sub-topics
    ("multi-agent reinforcement learning social dilemmas trust reputation and collective decision making in artificial societies", "social_2"),
    ("language model alignment value learning cultural transmission opinion dynamics and emergent behavior in agent populations", "social_3"),
]

total_ideas = 0
for i, (topic, tag) in enumerate(TOPICS):
    print(f"\n{'='*60}")
    print(f"Run {i+1}/{len(TOPICS)}: {tag}")
    print(f"Topic: {topic[:80]}...")
    print(f"{'='*60}")
    
    # Reset module-level caches between runs
    try:
        from agents.base_agent import _TOKEN_USAGE
        _TOKEN_USAGE.clear()
    except:
        pass
    
    start = time.time()
    try:
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
        print(f"\n  RESULT: {len(ideas)} ideas, {coverage:.1%} coverage, {elapsed:.0f}s")
        
        if ideas:
            rid = db.save_result(user_id, topic, coverage, len(ideas), results)
            print(f"  SAVED to DB: result_id={rid}")
            total_ideas += len(ideas)
        else:
            print(f"  SKIPPED (0 ideas)")
            
        # Save to file too
        outfile = f"output/results_{tag}.json"
        with open(outfile, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
            
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()

print(f"\n{'='*60}")
print(f"BATCH COMPLETE: {total_ideas} new ideas generated and saved")
print(f"{'='*60}")

# Final summary
all_results = db.get_user_results(user_id)
grand_total = sum(r['ideas_count'] for r in all_results)
print(f"\nanon-user total: {len(all_results)} runs, {grand_total} ideas")
