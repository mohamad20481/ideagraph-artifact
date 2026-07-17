"""Tests for billing.py — plan catalog, get/set plan, has_feature, UI helpers."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import billing


# ── Plan catalog shape ──────────────────────────────────────────────────────

def test_all_required_tiers_present():
    """The 4 canonical tiers exist with the expected ids."""
    assert set(billing.PLANS.keys()) == {"free", "pro", "team", "enterprise"}


def test_default_tier_is_free():
    assert billing.DEFAULT_TIER == "free"
    assert billing.DEFAULT_TIER in billing.PLANS


def test_plan_fields_well_formed():
    """Every plan has the required fields and consistent types."""
    for tier, plan in billing.PLANS.items():
        assert plan.tier == tier
        assert plan.label
        assert plan.tagline
        assert isinstance(plan.price_usd_monthly, (int, float))
        assert plan.price_usd_monthly >= 0
        assert isinstance(plan.features, list) and plan.features
        assert isinstance(plan.unlocks, list)
        assert plan.cta_label


def test_prices_monotonic_by_tier_order():
    """free < pro < team < enterprise — needed for upgrade-only CTAs."""
    prices = [
        billing.PLANS["free"].price_usd_monthly,
        billing.PLANS["pro"].price_usd_monthly,
        billing.PLANS["team"].price_usd_monthly,
        billing.PLANS["enterprise"].price_usd_monthly,
    ]
    assert prices == sorted(prices)
    assert prices[0] == 0  # free is free


def test_quotas_grow_with_tier():
    """Higher tiers get more (or unlimited) runs. -1 = unlimited."""
    free_q = billing.PLANS["free"].monthly_run_limit
    pro_q = billing.PLANS["pro"].monthly_run_limit
    team_q = billing.PLANS["team"].monthly_run_limit
    ent_q = billing.PLANS["enterprise"].monthly_run_limit
    assert free_q > 0
    assert pro_q > free_q
    assert team_q > pro_q
    assert ent_q == -1  # unlimited


def test_unlocks_only_reference_known_feature_ids():
    """Every unlock id is in FEATURE_IDS — typos here = silent breakage."""
    for tier, plan in billing.PLANS.items():
        for fid in plan.unlocks:
            assert fid in billing.FEATURE_IDS, (
                f"{tier} unlocks unknown feature {fid!r}"
            )


def test_higher_tiers_include_lower_tier_unlocks():
    """pro ⊇ free, team ⊇ pro, enterprise ⊇ team — no regressions on upgrade."""
    f = set(billing.PLANS["free"].unlocks)
    p = set(billing.PLANS["pro"].unlocks)
    t = set(billing.PLANS["team"].unlocks)
    e = set(billing.PLANS["enterprise"].unlocks)
    assert f <= p, f"pro missing free unlocks: {f - p}"
    assert p <= t, f"team missing pro unlocks: {p - t}"
    assert t <= e, f"enterprise missing team unlocks: {t - e}"


# ── get_plan ────────────────────────────────────────────────────────────────

def test_get_plan_no_user_returns_free():
    info = billing.get_plan(None)
    assert info["tier"] == "free"
    assert info["_plan"].tier == "free"
    assert info["runs_this_month"] == 0


def test_get_plan_reads_from_db():
    """get_plan delegates to db.get_user_subscription and attaches _plan."""
    with patch.object(billing, "db") as mock_db, \
         patch.object(billing, "_HAS_DB", True):
        mock_db.get_user_subscription.return_value = {
            "tier": "pro",
            "runs_this_month": 17,
            "stripe_customer_id": "cus_abc",
        }
        info = billing.get_plan(42)
        mock_db.get_user_subscription.assert_called_once_with(42)
        assert info["tier"] == "pro"
        assert info["runs_this_month"] == 17
        assert info["_plan"].tier == "pro"
        assert info["stripe_customer_id"] == "cus_abc"


def test_get_plan_swallows_db_errors():
    """A DB blow-up shouldn't crash the UI — fall back to free."""
    with patch.object(billing, "db") as mock_db, \
         patch.object(billing, "_HAS_DB", True):
        mock_db.get_user_subscription.side_effect = RuntimeError("db down")
        info = billing.get_plan(42)
        assert info["tier"] == "free"


def test_get_plan_unknown_tier_falls_back_to_free():
    """If the DB has a tier we don't know about, render as free (no crash)."""
    with patch.object(billing, "db") as mock_db, \
         patch.object(billing, "_HAS_DB", True):
        mock_db.get_user_subscription.return_value = {
            "tier": "platinum",  # not in PLANS
            "runs_this_month": 0,
        }
        info = billing.get_plan(99)
        assert info["_plan"].tier == "free"


# ── set_plan ────────────────────────────────────────────────────────────────

def test_set_plan_unknown_tier_raises():
    with pytest.raises(ValueError, match="unknown tier"):
        billing.set_plan(1, "platinum")


def test_set_plan_no_user_id_raises():
    with pytest.raises(ValueError, match="user_id"):
        billing.set_plan(0, "pro")


def test_set_plan_persists_to_db():
    with patch.object(billing, "db") as mock_db, \
         patch.object(billing, "_HAS_DB", True):
        billing.set_plan(5, "team", persist=True)
        mock_db.update_subscription.assert_called_once_with(5, tier="team")


def test_set_plan_persist_false_skips_db():
    with patch.object(billing, "db") as mock_db, \
         patch.object(billing, "_HAS_DB", True):
        billing.set_plan(5, "team", persist=False)
        mock_db.update_subscription.assert_not_called()


def test_set_plan_db_failure_raises_runtime_error():
    with patch.object(billing, "db") as mock_db, \
         patch.object(billing, "_HAS_DB", True):
        mock_db.update_subscription.side_effect = RuntimeError("locked")
        with pytest.raises(RuntimeError, match="failed to persist"):
            billing.set_plan(5, "pro")


# ── has_feature ─────────────────────────────────────────────────────────────

def test_has_feature_unknown_feature_is_permissive():
    """Unknown / untracked features default to True — adding has_feature()
    in new code shouldn't accidentally lock users out."""
    assert billing.has_feature(None, "totally_made_up_feature") is True


def test_has_feature_empty_feature_is_permissive():
    assert billing.has_feature(None, "") is True


def test_has_feature_free_gates_pro_features():
    """A free user should NOT have multi_panel_figures (pro+)."""
    with patch.object(billing, "db") as mock_db, \
         patch.object(billing, "_HAS_DB", True):
        mock_db.get_user_subscription.return_value = {
            "tier": "free", "runs_this_month": 0,
        }
        assert billing.has_feature(1, "multi_panel_figures") is False
        assert billing.has_feature(1, "video_generation") is False


def test_has_feature_pro_unlocks_pro_tier_features():
    with patch.object(billing, "db") as mock_db, \
         patch.object(billing, "_HAS_DB", True):
        mock_db.get_user_subscription.return_value = {
            "tier": "pro", "runs_this_month": 0,
        }
        assert billing.has_feature(1, "multi_panel_figures") is True
        assert billing.has_feature(1, "n_sample_variants") is True
        # but not team-tier features
        assert billing.has_feature(1, "video_generation") is False


def test_has_feature_enterprise_unlocks_everything():
    with patch.object(billing, "db") as mock_db, \
         patch.object(billing, "_HAS_DB", True):
        mock_db.get_user_subscription.return_value = {
            "tier": "enterprise", "runs_this_month": 0,
        }
        for fid in billing.FEATURE_IDS:
            assert billing.has_feature(1, fid) is True, (
                f"enterprise should unlock {fid}"
            )


# ── runs_remaining ──────────────────────────────────────────────────────────

def test_runs_remaining_basic_math():
    with patch.object(billing, "db") as mock_db, \
         patch.object(billing, "_HAS_DB", True):
        mock_db.get_user_subscription.return_value = {
            "tier": "free", "runs_this_month": 3,
        }
        # free is 5/mo → 5 - 3 = 2 left
        assert billing.runs_remaining(1) == 2


def test_runs_remaining_clamps_at_zero():
    with patch.object(billing, "db") as mock_db, \
         patch.object(billing, "_HAS_DB", True):
        mock_db.get_user_subscription.return_value = {
            "tier": "free", "runs_this_month": 999,
        }
        assert billing.runs_remaining(1) == 0


def test_runs_remaining_unlimited_for_enterprise():
    with patch.object(billing, "db") as mock_db, \
         patch.object(billing, "_HAS_DB", True):
        mock_db.get_user_subscription.return_value = {
            "tier": "enterprise", "runs_this_month": 10000,
        }
        assert billing.runs_remaining(1) == -1


# ── plan_features / monthly_run_limit ───────────────────────────────────────

def test_plan_features_returns_list():
    f = billing.plan_features("pro")
    assert isinstance(f, list) and f
    # unknown tier → free
    assert billing.plan_features("nope") == billing.plan_features("free")


def test_monthly_run_limit_lookup():
    assert billing.monthly_run_limit("free") == 5
    assert billing.monthly_run_limit("enterprise") == -1
    assert billing.monthly_run_limit("unknown") == billing.monthly_run_limit("free")


# ── start_upgrade_flow ──────────────────────────────────────────────────────

def test_start_upgrade_flow_unknown_tier():
    out = billing.start_upgrade_flow(1, "platinum")
    assert "error" in out


def test_start_upgrade_flow_known_tier_with_stripe_unconfigured():
    """With no Stripe keys set, the flow returns a 'not configured'
    message and a None checkout_url. (Real Stripe integration is tested
    in tests/test_stripe_payments.py with mocked SDK calls.)"""
    import stripe_integration as si
    with patch.object(si, "STRIPE_SECRET_KEY", ""):
        out = billing.start_upgrade_flow(1, "pro")
    assert out["target_tier"] == "pro"
    assert out["target_price"] == billing.PLANS["pro"].price_usd_monthly
    assert out["checkout_url"] is None
    assert "not configured" in out["message"].lower()


# ── UI render helpers (smoke tests — verify they don't crash) ───────────────

def _make_st_stub():
    """A MagicMock that returns itself from chainable methods so things like
    `with st.expander(...):` and `st.columns(N)` work without exploding."""
    stub = MagicMock()
    # st.columns(N) → list of N MagicMocks (column ctx managers)
    stub.columns.side_effect = lambda n, **kw: [MagicMock() for _ in range(n)]
    # Make .expander() and .container() context-manage cleanly.
    stub.expander.return_value.__enter__ = MagicMock(return_value=stub)
    stub.expander.return_value.__exit__ = MagicMock(return_value=None)
    stub.container.return_value.__enter__ = MagicMock(return_value=stub)
    stub.container.return_value.__exit__ = MagicMock(return_value=None)
    # Buttons return False (no click) by default.
    stub.button.return_value = False
    stub.checkbox.return_value = False
    return stub


def test_render_plan_card_smoke_no_user():
    """Anonymous user (no user_id) → renders free plan card without crash."""
    st = _make_st_stub()
    billing.render_plan_card(st, None)
    # At least the header markdown and the "What's included" header rendered.
    assert st.markdown.called


def test_render_plan_card_smoke_pro_user():
    st = _make_st_stub()
    with patch.object(billing, "db") as mock_db, \
         patch.object(billing, "_HAS_DB", True):
        mock_db.get_user_subscription.return_value = {
            "tier": "pro", "runs_this_month": 12,
        }
        billing.render_plan_card(st, 1)
    assert st.markdown.called


def test_render_plan_card_enterprise_shows_success():
    """Enterprise user gets a 'highest tier' confirmation, no upgrade CTAs."""
    st = _make_st_stub()
    with patch.object(billing, "db") as mock_db, \
         patch.object(billing, "_HAS_DB", True):
        mock_db.get_user_subscription.return_value = {
            "tier": "enterprise", "runs_this_month": 50,
        }
        billing.render_plan_card(st, 1)
    assert st.success.called


def test_render_admin_plan_override_no_users():
    """Empty users table → info banner, no crash."""
    st = _make_st_stub()
    with patch("db._lock"), \
         patch("db._get_conn") as mock_conn:
        cur = MagicMock()
        cur.fetchall.return_value = []
        mock_conn.return_value.execute.return_value = cur
        billing.render_admin_plan_override(st)
    assert st.info.called or st.error.called or st.markdown.called
