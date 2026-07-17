"""
Tests for UX features: LaTeX export, admin dashboard, user feedback, retry logic.
"""
import pytest


class TestLatexExport:
    def test_generates_compilable_latex(self):
        from export import export_ideas_latex
        ideas = [
            {
                "title": "GNN for Drug Discovery",
                "motivation": "Current methods are slow",
                "method": "Use message passing with attention",
                "hypothesis": "Will improve accuracy by 10%",
                "resources": "4x A100 GPUs",
                "expected_outcome": "State of the art on ZINC",
                "risk_assessment": "May overfit on small datasets",
                "quality_score": 0.72,
                "methodology_type": "empirical_study",
                "novelty_level": "moderate",
                "probe_scores": {"code": 0.8, "dataset": 0.7, "novelty": 0.6},
            },
        ]
        tex = export_ideas_latex(ideas, "Drug Discovery", stats={"coverage": 0.5, "iterations": 10, "quality_mean": 0.65})
        assert r"\documentclass" in tex
        assert r"\begin{document}" in tex
        assert r"\end{document}" in tex
        assert "GNN for Drug Discovery" in tex
        assert r"\maketitle" in tex
        assert r"\tableofcontents" in tex

    def test_escapes_special_chars(self):
        from export import export_ideas_latex
        ideas = [{"title": "Using $100 & 50% of #data", "method": "test_method",
                   "motivation": "", "hypothesis": "", "resources": "",
                   "expected_outcome": "", "risk_assessment": "",
                   "quality_score": 0.5, "probe_scores": {}}]
        tex = export_ideas_latex(ideas, "Test & $pecial")
        assert "\\$" in tex
        assert "\\&" in tex
        assert "\\%" in tex
        assert "\\#" in tex

    def test_includes_bibliography(self):
        from export import export_ideas_latex
        ideas = [{"title": "Test", "method": "m", "motivation": "", "hypothesis": "",
                   "resources": "", "expected_outcome": "", "risk_assessment": "",
                   "quality_score": 0.5, "probe_scores": {}}]
        papers = [{"title": "Paper One", "year": "2024", "authors": [{"name": "Smith"}]}]
        tex = export_ideas_latex(ideas, "Test", dag_papers=papers)
        assert r"\begin{thebibliography}" in tex
        assert "Smith" in tex
        assert "Paper One" in tex

    def test_handles_empty_ideas(self):
        from export import export_ideas_latex
        tex = export_ideas_latex([], "Empty Topic")
        assert r"\begin{document}" in tex
        assert "0 research ideas" in tex


class TestAdminDashboard:
    def test_is_admin_recognizes_configured_ids(self):
        from admin_dashboard import is_admin, ADMIN_USER_IDS
        # At least one admin must be configured
        assert len(ADMIN_USER_IDS) >= 1
        for uid in ADMIN_USER_IDS:
            assert is_admin(uid) is True
        # Random non-admin id should not be recognized
        non_admin = max(ADMIN_USER_IDS) + 9999
        assert is_admin(non_admin) is False

    def test_get_admin_stats_returns_dict(self):
        from admin_dashboard import get_admin_stats
        stats = get_admin_stats()
        assert isinstance(stats, dict)


class TestUserFeedback:
    @pytest.fixture
    def fresh_db(self, tmp_path):
        import db as _db
        orig_path = _db._DB_PATH
        orig_dir = _db._DB_DIR
        _db._DB_PATH = str(tmp_path / "test.db")
        _db._DB_DIR = str(tmp_path)
        _db._conn_local = __import__("threading").local()
        try:
            _db.init_db()
            yield _db
        finally:
            _db._DB_PATH = orig_path
            _db._DB_DIR = orig_dir
            _db._conn_local = __import__("threading").local()

    def test_save_and_get_feedback(self, fresh_db):
        uid = fresh_db.register_user("feedbacktest", "StrongPassword123")
        fresh_db.save_idea_feedback(uid, "Idea A", "useful")
        fresh_db.save_idea_feedback(uid, "Idea B", "not_useful")
        fb = fresh_db.get_idea_feedback(uid)
        assert fb["Idea A"] == "useful"
        assert fb["Idea B"] == "not_useful"

    def test_feedback_upsert(self, fresh_db):
        uid = fresh_db.register_user("upserttest", "StrongPassword123")
        fresh_db.save_idea_feedback(uid, "Idea X", "useful")
        fresh_db.save_idea_feedback(uid, "Idea X", "not_useful")
        fb = fresh_db.get_idea_feedback(uid)
        assert fb["Idea X"] == "not_useful"  # updated, not duplicated

    def test_empty_feedback(self, fresh_db):
        uid = fresh_db.register_user("emptytest", "StrongPassword123")
        fb = fresh_db.get_idea_feedback(uid)
        assert fb == {}
