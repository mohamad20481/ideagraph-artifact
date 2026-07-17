"""Run Phase 2: Debate Arena on top 24 ideas."""
import json, sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config as cfg
cfg.PROVIDER = 'azure'
cfg.MODEL = 'DeepSeek-V3.2-Speciale'

import db
db.init_db()

from models.idea import Idea
from agents.debate_arena import DebateArena
from agents.agent_memory import AgentMemoryManager

uid = db.login_user('anon-user', 'Muha9999!')
print(f"Logged in as anon-user (uid={uid})")

# Load top 24 ideas
with open('output/top24_for_debate.json') as f:
    top24_dicts = json.load(f)

print(f"\nLoaded {len(top24_dicts)} ideas for debate")
print(f"Quality range: {top24_dicts[-1]['quality_score']:.3f} - {top24_dicts[0]['quality_score']:.3f}")

# Convert to Idea objects
ideas = []
for d in top24_dicts:
    idea = Idea(
        title=d.get("title", ""),
        motivation=d.get("motivation", ""),
        method=d.get("method", ""),
        hypothesis=d.get("hypothesis", ""),
        resources=d.get("resources", ""),
        expected_outcome=d.get("expected_outcome", ""),
        risk_assessment=d.get("risk_assessment", ""),
        source_strategy=d.get("source_strategy", ""),
        methodology_type=d.get("methodology_type"),
        novelty_level=d.get("novelty_level"),
        quality_score=d.get("quality_score", 0),
        probe_scores=d.get("probe_scores", {}),
        probe_passed=d.get("probe_passed", True),
    )
    ideas.append(idea)

# Initialize memory manager and debate arena
memory_mgr = AgentMemoryManager(user_id=uid)
arena = DebateArena(memory_manager=memory_mgr)

print(f"\n{'='*60}")
print(f"PHASE 2: DEBATE ARENA — {len(ideas)} ideas in tournament")
print(f"{'='*60}\n")

start = time.time()
tournament = arena.run_tournament(
    ideas=ideas,
    domain="AI research",
    on_progress=lambda msg: print(msg),
)
elapsed = time.time() - start

print(f"\n{'='*60}")
print(f"TOURNAMENT COMPLETE in {elapsed:.0f}s")
print(f"Champion: {tournament.champion_title}")
print(f"Total matches: {len(tournament.all_matches)}")
print(f"Rounds: {len(tournament.bracket)}")
print(f"{'='*60}")

# Save tournament to DB
tournament_dict = tournament.to_dict()
debate_id = db.save_debate(
    user_id=uid,
    result_id=None,
    tournament_dict=tournament_dict,
    winner_title=tournament.champion_title,
    rounds_count=len(tournament.bracket),
)
print(f"\nSaved debate to DB: debate_id={debate_id}")

# Save full transcript to file
with open('output/debate_transcript.json', 'w') as f:
    json.dump(tournament_dict, f, indent=2, ensure_ascii=False)
print(f"Saved transcript to output/debate_transcript.json")

# Print match results summary
print(f"\n{'='*60}")
print("MATCH RESULTS:")
print(f"{'='*60}")
for round_idx, round_matches in enumerate(tournament.bracket):
    print(f"\n--- Round {round_idx + 1} ---")
    for m in round_matches:
        title_a = m.idea_a['title'][:40]
        title_b = m.idea_b['title'][:40]
        winner = title_a if m.winner_side == 'a' else title_b
        score_a = m.judge_verdict.get('score_a', '?')
        score_b = m.judge_verdict.get('score_b', '?')
        print(f"  {title_a} vs {title_b}")
        print(f"    Winner: {winner[:50]} (A={score_a}, B={score_b})")
        reasoning = m.judge_verdict.get('reasoning', '')[:100]
        print(f"    Reasoning: {reasoning}")
