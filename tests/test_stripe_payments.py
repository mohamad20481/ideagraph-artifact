"""Tests for the Stripe payment flow.

Covers stripe_integration.verify_and_apply_checkout, create_checkout_session,
test_connection, create_customer_portal_session, reconcile_stripe_subscriptions,
and the billing.start_upgrade_flow rewiring that calls into Stripe.

All Stripe SDK calls are mocked — no network, no real Stripe account
required. Tests do not consume your Stripe rate limit.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ── verify_and_apply_checkout ──────────────────────────────────────────────

def _mock_session(payment_status="paid", status="complete",
                  metadata=None, customer="cus_abc",
                  subscription="sub_xyz", client_reference_id=None):
    """Build an object that quacks like a Stripe Checkout Session.

    The real SDK returns a StripeObject that supports both attribute
    access (.id, .payment_status) AND mapping access (.get(...)). Our
    code uses both, so we mock with a dict-like MagicMock.
    """
    s = MagicMock()
    s.get = lambda k, default=None: {
        "payment_status": payment_status,
        "status": status,
        "metadata": metadata or {},
        "customer": customer,
        "subscription": subscription,
        "client_reference_id": client_reference_id,
    }.get(k, default)
    s.payment_status = payment_status
    s.status = status
    return s


def test_verify_and_apply_checkout_not_configured():
    """No Stripe key → returns ok=False without crashing."""
    import stripe_integration as si
    with patch.object(si, "HAS_STRIPE", True), \
         patch.object(si, "STRIPE_SECRET_KEY", ""):
        result = si.verify_and_apply_checkout("cs_test_abc")
    assert result["ok"] is False
    assert "not configured" in result["error"].lower()


def test_verify_and_apply_checkout_invalid_session_id():
    """Garbage session_id rejected before hitting Stripe."""
    import stripe_integration as si
    with patch.object(si, "HAS_STRIPE", True), \
         patch.object(si, "STRIPE_SECRET_KEY", "sk_test_123"):
        result = si.verify_and_apply_checkout("not_a_session")
    assert result["ok"] is False
    assert "invalid session" in result["error"].lower()


def test_verify_and_apply_checkout_unpaid_session():
    """Stripe says payment isn't complete → don't apply the plan."""
    import stripe_integration as si
    mock_session = _mock_session(payment_status="unpaid", status="open")
    with patch.object(si, "HAS_STRIPE", True), \
         patch.object(si, "STRIPE_SECRET_KEY", "sk_test_123"), \
         patch.object(si, "stripe") as mock_stripe:
        mock_stripe.checkout.Session.retrieve.return_value = mock_session
        result = si.verify_and_apply_checkout("cs_test_abc")
    assert result["ok"] is False
    assert "not paid" in result["error"].lower()


def test_verify_and_apply_checkout_happy_path():
    """Paid session → persists tier to db via update_subscription."""
    import stripe_integration as si
    mock_session = _mock_session(
        payment_status="paid", status="complete",
        metadata={"user_id": "42", "tier": "pro"},
        customer="cus_test123",
        subscription="sub_test456",
    )
    with patch.object(si, "HAS_STRIPE", True), \
         patch.object(si, "STRIPE_SECRET_KEY", "sk_test_123"), \
         patch.object(si, "stripe") as mock_stripe, \
         patch.object(si, "db") as mock_db:
        mock_stripe.checkout.Session.retrieve.return_value = mock_session
        result = si.verify_and_apply_checkout("cs_test_abc")
    assert result["ok"] is True
    assert result["user_id"] == 42
    assert result["tier"] == "pro"
    mock_db.update_subscription.assert_called_once_with(
        user_id=42, tier="pro",
        stripe_customer_id="cus_test123",
        stripe_subscription_id="sub_test456",
        status="active",
    )


def test_verify_and_apply_checkout_unknown_user_id():
    """Session with no user_id in metadata or client_reference_id → fail."""
    import stripe_integration as si
    mock_session = _mock_session(
        payment_status="paid", status="complete",
        metadata={"tier": "pro"},  # no user_id
        client_reference_id=None,
    )
    with patch.object(si, "HAS_STRIPE", True), \
         patch.object(si, "STRIPE_SECRET_KEY", "sk_test_123"), \
         patch.object(si, "stripe") as mock_stripe:
        mock_stripe.checkout.Session.retrieve.return_value = mock_session
        result = si.verify_and_apply_checkout("cs_test_abc")
    assert result["ok"] is False
    assert "user_id" in result["error"]


def test_verify_and_apply_checkout_uses_client_reference_id_fallback():
    """If metadata.user_id is missing, fall back to client_reference_id."""
    import stripe_integration as si
    mock_session = _mock_session(
        payment_status="paid", status="complete",
        metadata={"tier": "team"},
        client_reference_id="99",
    )
    with patch.object(si, "HAS_STRIPE", True), \
         patch.object(si, "STRIPE_SECRET_KEY", "sk_test_123"), \
         patch.object(si, "stripe") as mock_stripe, \
         patch.object(si, "db") as mock_db:
        mock_stripe.checkout.Session.retrieve.return_value = mock_session
        result = si.verify_and_apply_checkout("cs_test_abc")
    assert result["ok"] is True
    assert result["user_id"] == 99
    assert result["tier"] == "team"


def test_verify_and_apply_checkout_unknown_tier_rejected():
    """Session with a tier we don't recognize → don't write garbage to db."""
    import stripe_integration as si
    mock_session = _mock_session(
        payment_status="paid", status="complete",
        metadata={"user_id": "1", "tier": "platinum"},
    )
    with patch.object(si, "HAS_STRIPE", True), \
         patch.object(si, "STRIPE_SECRET_KEY", "sk_test_123"), \
         patch.object(si, "stripe") as mock_stripe, \
         patch.object(si, "db") as mock_db:
        mock_stripe.checkout.Session.retrieve.return_value = mock_session
        result = si.verify_and_apply_checkout("cs_test_abc")
    assert result["ok"] is False
    assert "unknown tier" in result["error"].lower()
    mock_db.update_subscription.assert_not_called()


# ── create_checkout_session ─────────────────────────────────────────────────

def test_create_checkout_session_unconfigured_returns_none():
    import stripe_integration as si
    with patch.object(si, "HAS_STRIPE", True), \
         patch.object(si, "STRIPE_SECRET_KEY", ""):
        assert si.create_checkout_session(1, "pro", "x@example.com") is None


def test_create_checkout_session_missing_price_id_returns_none():
    import stripe_integration as si
    with patch.object(si, "HAS_STRIPE", True), \
         patch.object(si, "STRIPE_SECRET_KEY", "sk_test_123"), \
         patch.object(si, "PRICE_IDS", {"pro": "", "team": "", "enterprise": ""}):
        assert si.create_checkout_session(1, "pro", "x@example.com") is None


def test_create_checkout_session_happy_path_passes_through_url():
    import stripe_integration as si
    with patch.object(si, "HAS_STRIPE", True), \
         patch.object(si, "STRIPE_SECRET_KEY", "sk_test_123"), \
         patch.object(si, "PRICE_IDS", {"pro": "price_pro_test"}), \
         patch.object(si, "stripe") as mock_stripe:
        fake = MagicMock()
        fake.url = "https://checkout.stripe.com/c/pay/cs_test_xyz"
        mock_stripe.checkout.Session.create.return_value = fake
        url = si.create_checkout_session(7, "pro", "user@example.com")
    assert url == "https://checkout.stripe.com/c/pay/cs_test_xyz"
    # Verify the call included client_reference_id (so we can recover user
    # ID even if metadata is lost).
    kwargs = mock_stripe.checkout.Session.create.call_args.kwargs
    assert kwargs["client_reference_id"] == "7"
    assert kwargs["metadata"] == {"user_id": "7", "tier": "pro"}
    # success_url includes the {CHECKOUT_SESSION_ID} placeholder.
    assert "{CHECKOUT_SESSION_ID}" in kwargs["success_url"]


# ── test_connection ─────────────────────────────────────────────────────────

def test_test_connection_no_sdk():
    import stripe_integration as si
    with patch.object(si, "HAS_STRIPE", False):
        result = si.test_connection()
    assert result["sdk_installed"] is False
    assert result["ok"] is False
    assert "sdk not installed" in result["error"].lower()


def test_test_connection_no_key():
    import stripe_integration as si
    with patch.object(si, "HAS_STRIPE", True), \
         patch.object(si, "STRIPE_SECRET_KEY", ""):
        result = si.test_connection()
    assert result["ok"] is False
    assert "not set" in result["error"].lower()


def test_test_connection_detects_test_mode():
    import stripe_integration as si
    with patch.object(si, "HAS_STRIPE", True), \
         patch.object(si, "STRIPE_SECRET_KEY", "sk_test_123"), \
         patch.object(si, "stripe") as mock_stripe:
        mock_acct = MagicMock()
        mock_acct.get = lambda k, d=None: {"id": "acct_test_999"}.get(k, d)
        mock_stripe.Account.retrieve.return_value = mock_acct
        result = si.test_connection()
    assert result["mode"] == "test"
    assert result["ok"] is True
    assert result["account_id"] == "acct_test_999"


def test_test_connection_detects_live_mode():
    import stripe_integration as si
    with patch.object(si, "HAS_STRIPE", True), \
         patch.object(si, "STRIPE_SECRET_KEY", "sk_live_xyz"), \
         patch.object(si, "stripe") as mock_stripe:
        mock_acct = MagicMock()
        mock_acct.get = lambda k, d=None: {"id": "acct_live_111"}.get(k, d)
        mock_stripe.Account.retrieve.return_value = mock_acct
        result = si.test_connection()
    assert result["mode"] == "live"
    assert result["ok"] is True


def test_test_connection_bad_key_surfaces_error():
    import stripe_integration as si
    with patch.object(si, "HAS_STRIPE", True), \
         patch.object(si, "STRIPE_SECRET_KEY", "sk_test_invalid"), \
         patch.object(si, "stripe") as mock_stripe:
        mock_stripe.Account.retrieve.side_effect = RuntimeError(
            "Invalid API Key provided"
        )
        result = si.test_connection()
    assert result["ok"] is False
    assert "invalid api key" in result["error"].lower()


# ── create_customer_portal_session ─────────────────────────────────────────

def test_create_customer_portal_session_no_stripe_customer_returns_none():
    """Free user with no Stripe customer record → can't open portal."""
    import stripe_integration as si
    with patch.object(si, "HAS_STRIPE", True), \
         patch.object(si, "STRIPE_SECRET_KEY", "sk_test_123"), \
         patch.object(si, "db") as mock_db:
        mock_db.get_user_subscription.return_value = {
            "tier": "free", "stripe_customer_id": "",
        }
        assert si.create_customer_portal_session(1) is None


def test_create_customer_portal_session_happy_path():
    import stripe_integration as si
    with patch.object(si, "HAS_STRIPE", True), \
         patch.object(si, "STRIPE_SECRET_KEY", "sk_test_123"), \
         patch.object(si, "db") as mock_db, \
         patch.object(si, "stripe") as mock_stripe:
        mock_db.get_user_subscription.return_value = {
            "tier": "pro", "stripe_customer_id": "cus_abc",
        }
        fake = MagicMock()
        fake.url = "https://billing.stripe.com/p/session/test_xyz"
        mock_stripe.billing_portal.Session.create.return_value = fake
        url = si.create_customer_portal_session(1)
    assert url == "https://billing.stripe.com/p/session/test_xyz"


# ── reconcile_stripe_subscriptions ──────────────────────────────────────────

def test_reconcile_no_subscription_on_record():
    import stripe_integration as si
    with patch.object(si, "HAS_STRIPE", True), \
         patch.object(si, "STRIPE_SECRET_KEY", "sk_test_123"), \
         patch.object(si, "db") as mock_db:
        mock_db.get_user_subscription.return_value = {"tier": "free"}
        result = si.reconcile_stripe_subscriptions(1)
    assert result["ok"] is False
    assert "no stripe subscription" in result["error"].lower()


def test_reconcile_canceled_subscription_downgrades_to_free():
    import stripe_integration as si
    with patch.object(si, "HAS_STRIPE", True), \
         patch.object(si, "STRIPE_SECRET_KEY", "sk_test_123"), \
         patch.object(si, "db") as mock_db, \
         patch.object(si, "stripe") as mock_stripe:
        mock_db.get_user_subscription.return_value = {
            "tier": "pro", "stripe_subscription_id": "sub_xyz",
        }
        mock_sub = MagicMock()
        mock_sub.get = lambda k, d=None: {"status": "canceled"}.get(k, d)
        mock_sub.status = "canceled"
        mock_stripe.Subscription.retrieve.return_value = mock_sub
        result = si.reconcile_stripe_subscriptions(1)
    assert result["ok"] is True
    assert result["tier"] == "free"
    mock_db.update_subscription.assert_called_once_with(
        user_id=1, tier="free", status="canceled",
    )


def test_reconcile_active_subscription_keeps_tier():
    import stripe_integration as si
    with patch.object(si, "HAS_STRIPE", True), \
         patch.object(si, "STRIPE_SECRET_KEY", "sk_test_123"), \
         patch.object(si, "db") as mock_db, \
         patch.object(si, "stripe") as mock_stripe:
        mock_db.get_user_subscription.return_value = {
            "tier": "team", "stripe_subscription_id": "sub_xyz",
        }
        mock_sub = MagicMock()
        mock_sub.get = lambda k, d=None: {"status": "active"}.get(k, d)
        mock_sub.status = "active"
        mock_stripe.Subscription.retrieve.return_value = mock_sub
        result = si.reconcile_stripe_subscriptions(1)
    assert result["ok"] is True
    assert result["tier"] == "team"


# ── billing.start_upgrade_flow rewiring ─────────────────────────────────────

def test_start_upgrade_flow_enterprise_returns_contact_sales():
    """Enterprise stays contact-sales — no Checkout session attempted."""
    import billing
    out = billing.start_upgrade_flow(1, "enterprise")
    assert out["checkout_url"] is None
    assert "contact-sales" in out["message"].lower() or \
           "contact sales" in out["message"].lower()


def test_start_upgrade_flow_stripe_unconfigured():
    import billing
    import stripe_integration as si
    with patch.object(si, "HAS_STRIPE", True), \
         patch.object(si, "STRIPE_SECRET_KEY", ""):
        out = billing.start_upgrade_flow(1, "pro")
    assert out["checkout_url"] is None
    assert "not configured" in out["message"].lower()


def test_start_upgrade_flow_missing_price_id():
    import billing
    import stripe_integration as si
    with patch.object(si, "HAS_STRIPE", True), \
         patch.object(si, "STRIPE_SECRET_KEY", "sk_test_123"), \
         patch.object(si, "PRICE_IDS", {"pro": "", "team": "", "enterprise": ""}):
        out = billing.start_upgrade_flow(1, "pro")
    assert out["checkout_url"] is None
    assert "price id" in out["message"].lower()


def test_start_upgrade_flow_happy_path_returns_checkout_url():
    import billing
    import stripe_integration as si
    with patch.object(si, "HAS_STRIPE", True), \
         patch.object(si, "STRIPE_SECRET_KEY", "sk_test_123"), \
         patch.object(si, "PRICE_IDS", {"pro": "price_pro_test"}), \
         patch.object(si, "stripe") as mock_stripe:
        fake = MagicMock()
        fake.url = "https://checkout.stripe.com/c/pay/cs_test_xyz"
        mock_stripe.checkout.Session.create.return_value = fake
        out = billing.start_upgrade_flow(7, "pro", customer_email="a@b.com")
    assert out["checkout_url"] == "https://checkout.stripe.com/c/pay/cs_test_xyz"
    assert out["target_tier"] == "pro"


def test_start_upgrade_flow_stripe_returns_no_url():
    """create_checkout_session returned None (Stripe rejected) → surface error."""
    import billing
    import stripe_integration as si
    with patch.object(si, "HAS_STRIPE", True), \
         patch.object(si, "STRIPE_SECRET_KEY", "sk_test_123"), \
         patch.object(si, "PRICE_IDS", {"pro": "price_pro_test"}), \
         patch.object(si, "stripe") as mock_stripe:
        mock_stripe.checkout.Session.create.side_effect = RuntimeError("400 bad")
        out = billing.start_upgrade_flow(7, "pro")
    assert out["checkout_url"] is None
    assert "rejected" in out["message"].lower()
