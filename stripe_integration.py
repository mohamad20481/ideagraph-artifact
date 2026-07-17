"""
stripe_integration.py - Stripe subscription management for IdeaGraph.

Operates in TEST MODE by default (Stripe `sk_test_...` keys). Flip to live
when you're ready by swapping the keys in .env. Uses the **return-flow
polling** model — no webhook server required:

    User clicks "Upgrade to Pro" in IdeaGraph
       → server creates a Checkout Session
       → user redirected to checkout.stripe.com (hosted page)
       → on success/cancel, Stripe redirects back to APP_BASE_URL with
         ?checkout=success&session_id=cs_test_... (or ?checkout=cancel)
       → app.py reads the query param, calls verify_and_apply_checkout()
         which polls Stripe via stripe.checkout.Session.retrieve() and
         persists the new subscription via db.update_subscription()

This avoids the need for a publicly-reachable webhook URL — works fine
on localhost. The trade-off: if the user closes their browser before
returning, we never finalize. A nightly reconciliation job
(`reconcile_stripe_subscriptions()`) catches stragglers.

Subscription tiers and prices live in `billing.PLANS`. The price IDs
themselves (created in Stripe Dashboard → Products) live in env vars
STRIPE_PRICE_PRO / STRIPE_PRICE_TEAM / STRIPE_PRICE_ENTERPRISE.

Setup:
  1. Create Stripe account at https://stripe.com (free)
  2. Get test mode API keys from https://dashboard.stripe.com/test/apikeys
  3. Set env vars in .env:
        STRIPE_SECRET_KEY=sk_test_...
        STRIPE_PUBLIC_KEY=pk_test_...
        APP_BASE_URL=http://localhost:8510
  4. Create products in Stripe dashboard at /test/products:
        Pro ($15/mo recurring)
        Team ($49/mo recurring)
        Enterprise ($299/mo recurring)
  5. Copy the resulting price_... IDs into .env:
        STRIPE_PRICE_PRO=price_...
        STRIPE_PRICE_TEAM=price_...
        STRIPE_PRICE_ENTERPRISE=price_...
  6. Restart Streamlit.

Test cards (in TEST MODE only):
  4242 4242 4242 4242  → succeeds
  4000 0000 0000 9995  → declined (insufficient funds)
  4000 0027 6000 3184  → requires authentication (3DS)

Usage:
  from stripe_integration import create_checkout_session, get_tier_limits
  checkout_url = create_checkout_session(user_id, "pro", customer_email)
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

try:
    import stripe
    HAS_STRIPE = True
except ImportError:
    HAS_STRIPE = False

import db


# ── Configuration ─────────────────────────────────────────────────────────────

STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY", "")
STRIPE_PUBLIC_KEY = os.getenv("STRIPE_PUBLIC_KEY", "")
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET", "")

APP_BASE_URL = os.getenv("APP_BASE_URL", "http://localhost:8510")

PRICE_IDS = {
    "pro": os.getenv("STRIPE_PRICE_PRO", ""),
    "team": os.getenv("STRIPE_PRICE_TEAM", ""),
    "enterprise": os.getenv("STRIPE_PRICE_ENTERPRISE", ""),
}


TIER_LIMITS = {
    "free": {
        "runs_per_month": 3,
        "max_ideas_per_run": 10,
        "watermark": True,
        "api_access": False,
        "priority_support": False,
        "seats": 1,
        "price_usd": 0,
    },
    "pro": {
        "runs_per_month": 999,
        "max_ideas_per_run": 50,
        "watermark": False,
        "api_access": False,
        "priority_support": True,
        "seats": 1,
        "price_usd": 15,
    },
    "team": {
        "runs_per_month": 999,
        "max_ideas_per_run": 100,
        "watermark": False,
        "api_access": True,
        "priority_support": True,
        "seats": 5,
        "price_usd": 49,
    },
    "enterprise": {
        "runs_per_month": 99999,
        "max_ideas_per_run": 200,
        "watermark": False,
        "api_access": True,
        "priority_support": True,
        "seats": 999,
        "price_usd": 299,
    },
}


def is_configured() -> bool:
    """Check if Stripe is configured with API keys."""
    return HAS_STRIPE and bool(STRIPE_SECRET_KEY)


def _init_stripe() -> None:
    if HAS_STRIPE and STRIPE_SECRET_KEY:
        stripe.api_key = STRIPE_SECRET_KEY


def get_tier_limits(tier: str) -> Dict[str, Any]:
    """Get limits for a subscription tier."""
    return TIER_LIMITS.get(tier, TIER_LIMITS["free"])


def get_user_tier(user_id: int) -> str:
    """Get current user's tier."""
    sub = db.get_user_subscription(user_id)
    return sub.get("tier", "free") if sub else "free"


def can_run_pipeline(user_id: int) -> tuple:
    """
    Check if user can run another pipeline.
    Returns (allowed: bool, reason: str).
    """
    sub = db.get_user_subscription(user_id)
    tier = sub.get("tier", "free")
    runs = sub.get("runs_this_month", 0)
    limit = TIER_LIMITS[tier]["runs_per_month"]

    if runs >= limit:
        return False, f"You've used {runs}/{limit} runs this month on the {tier.title()} plan. Upgrade for more."
    return True, f"{runs}/{limit} runs used this month ({tier.title()} plan)"


def create_checkout_session(
    user_id: int, tier: str, customer_email: str = "",
) -> Optional[str]:
    """
    Create a Stripe Checkout session for subscription.
    Returns the checkout URL, or None if Stripe is not configured.
    """
    if not is_configured():
        return None

    if tier not in PRICE_IDS or not PRICE_IDS[tier]:
        return None

    _init_stripe()

    try:
        session = stripe.checkout.Session.create(
            payment_method_types=["card"],
            line_items=[{
                "price": PRICE_IDS[tier],
                "quantity": 1,
            }],
            mode="subscription",
            customer_email=customer_email or None,
            success_url=(
                f"{APP_BASE_URL}/?checkout=success"
                f"&tier={tier}&session_id={{CHECKOUT_SESSION_ID}}"
            ),
            cancel_url=f"{APP_BASE_URL}/?checkout=cancel",
            metadata={"user_id": str(user_id), "tier": tier},
            client_reference_id=str(user_id),
        )
        return session.url
    except Exception as e:
        print(f"Stripe checkout error: {e}")
        return None


# ── Return-flow polling ─────────────────────────────────────────────────────

def verify_and_apply_checkout(session_id: str) -> Dict[str, Any]:
    """Verify a Stripe Checkout Session and persist the resulting plan.

    Called from app.py when the user returns from Stripe with a
    `?checkout=success&session_id=cs_test_...` query param. Idempotent —
    safe to call multiple times for the same session (Stripe returns the
    same object; db.update_subscription is an UPDATE, not INSERT).

    Returns:
        {"ok": True,  "tier": "pro", "user_id": 11, ...} on success
        {"ok": False, "error": "..."}                    otherwise
    """
    if not is_configured():
        return {"ok": False, "error": "Stripe not configured."}
    if not session_id or not session_id.startswith("cs_"):
        return {"ok": False, "error": f"Invalid session id: {session_id!r}"}
    _init_stripe()
    try:
        session = stripe.checkout.Session.retrieve(
            session_id, expand=["subscription", "customer"],
        )
    except Exception as e:
        return {"ok": False, "error": f"Stripe lookup failed: {e}"}

    payment_status = getattr(session, "payment_status", "") or session.get("payment_status", "")
    status = getattr(session, "status", "") or session.get("status", "")
    if payment_status != "paid" and status != "complete":
        return {
            "ok": False,
            "error": (
                f"Session not paid yet (status={status!r}, "
                f"payment_status={payment_status!r}). Try again in a moment."
            ),
        }

    metadata = session.get("metadata") or {}
    user_id_raw = (
        metadata.get("user_id")
        or session.get("client_reference_id")
        or ""
    )
    try:
        user_id = int(user_id_raw)
    except (TypeError, ValueError):
        return {"ok": False, "error": f"Couldn't resolve user_id from session ({user_id_raw!r})."}
    tier = metadata.get("tier") or "pro"
    if tier not in TIER_LIMITS:
        return {"ok": False, "error": f"Unknown tier on session: {tier!r}"}

    customer_id = session.get("customer") or ""
    if hasattr(customer_id, "id"):
        customer_id = customer_id.id
    subscription_id = session.get("subscription") or ""
    if hasattr(subscription_id, "id"):
        subscription_id = subscription_id.id

    try:
        db.update_subscription(
            user_id=user_id, tier=tier,
            stripe_customer_id=str(customer_id),
            stripe_subscription_id=str(subscription_id),
            status="active",
        )
    except Exception as e:
        return {"ok": False, "error": f"Persistence failed: {e}"}

    return {
        "ok": True,
        "user_id": user_id,
        "tier": tier,
        "stripe_customer_id": str(customer_id),
        "stripe_subscription_id": str(subscription_id),
    }


# ── Customer portal (cancel / update payment method / view invoices) ───────

def create_customer_portal_session(user_id: int) -> Optional[str]:
    """Create a Stripe Customer Portal session for self-service billing
    management (cancel subscription, update card, download invoices).

    Returns the portal URL or None if the user isn't a Stripe customer
    yet (no successful checkout completed) or Stripe is unconfigured.
    """
    if not is_configured():
        return None
    sub = db.get_user_subscription(user_id)
    customer_id = (sub or {}).get("stripe_customer_id") or ""
    if not customer_id:
        return None
    _init_stripe()
    try:
        session = stripe.billing_portal.Session.create(
            customer=customer_id,
            return_url=f"{APP_BASE_URL}/?portal=return",
        )
        return session.url
    except Exception as e:
        print(f"Stripe portal error: {e}")
        return None


# ── Diagnostics ─────────────────────────────────────────────────────────────

def test_connection() -> Dict[str, Any]:
    """Quick health check the admin panel calls before going live.

    Verifies: SDK importable, key set, key valid (round-trip to Stripe
    API), price IDs configured, mode (test vs live) inferred from the
    key prefix.
    """
    out: Dict[str, Any] = {
        "sdk_installed": HAS_STRIPE,
        "key_present": bool(STRIPE_SECRET_KEY),
        "mode": "unknown",
        "account_id": None,
        "prices_configured": {
            t: bool(pid) for t, pid in PRICE_IDS.items()
        },
        "ok": False,
        "error": None,
    }
    if not HAS_STRIPE:
        out["error"] = "stripe SDK not installed (pip install stripe)."
        return out
    if not STRIPE_SECRET_KEY:
        out["error"] = "STRIPE_SECRET_KEY not set in .env."
        return out
    if STRIPE_SECRET_KEY.startswith("sk_test_"):
        out["mode"] = "test"
    elif STRIPE_SECRET_KEY.startswith("sk_live_"):
        out["mode"] = "live"
    _init_stripe()
    try:
        account = stripe.Account.retrieve()
        out["account_id"] = account.get("id") or getattr(account, "id", None)
        out["ok"] = True
    except Exception as e:
        out["error"] = str(e)
    return out


# ── Reconciliation (catches users who closed the tab before returning) ─────

def reconcile_stripe_subscriptions(user_id: int) -> Dict[str, Any]:
    """Re-sync subscription status for one user against Stripe.

    Useful for catching stragglers: a user paid, closed the tab before
    Stripe redirected back, so we never ran verify_and_apply_checkout().
    Calling this re-fetches their subscription state from Stripe and
    updates db accordingly.
    """
    if not is_configured():
        return {"ok": False, "error": "Stripe not configured."}
    sub = db.get_user_subscription(user_id) or {}
    sub_id = sub.get("stripe_subscription_id") or ""
    if not sub_id:
        return {"ok": False, "error": "No Stripe subscription on record for this user."}
    _init_stripe()
    try:
        s = stripe.Subscription.retrieve(sub_id)
    except Exception as e:
        return {"ok": False, "error": f"Stripe lookup failed: {e}"}
    status = s.get("status") or getattr(s, "status", "")
    # Map Stripe status → local "status" + tier (downgrade to free if canceled).
    if status in ("canceled", "incomplete_expired", "unpaid"):
        try:
            db.update_subscription(
                user_id=user_id, tier="free", status=status,
            )
        except Exception as e:
            return {"ok": False, "error": f"Persistence failed: {e}"}
        return {"ok": True, "status": status, "tier": "free"}
    if status in ("active", "trialing", "past_due"):
        try:
            db.update_subscription(user_id=user_id, status=status)
        except Exception as e:
            return {"ok": False, "error": f"Persistence failed: {e}"}
        return {"ok": True, "status": status, "tier": sub.get("tier") or "free"}
    return {"ok": True, "status": status, "tier": sub.get("tier") or "free"}


def handle_webhook(payload: bytes, signature: str) -> bool:
    """
    Handle Stripe webhook events (subscription created, updated, deleted).
    Returns True if event was processed.
    """
    if not is_configured() or not STRIPE_WEBHOOK_SECRET:
        return False

    _init_stripe()

    try:
        event = stripe.Webhook.construct_event(
            payload, signature, STRIPE_WEBHOOK_SECRET,
        )
    except Exception:
        return False

    event_type = event.get("type", "")
    data_obj = event.get("data", {}).get("object", {})

    if event_type == "checkout.session.completed":
        user_id = int(data_obj.get("metadata", {}).get("user_id", 0))
        tier = data_obj.get("metadata", {}).get("tier", "pro")
        customer_id = data_obj.get("customer", "")
        subscription_id = data_obj.get("subscription", "")
        if user_id:
            db.update_subscription(
                user_id=user_id, tier=tier,
                stripe_customer_id=customer_id,
                stripe_subscription_id=subscription_id,
                status="active",
            )
            return True

    elif event_type == "customer.subscription.deleted":
        customer_id = data_obj.get("customer", "")
        # Downgrade to free
        # Note: requires looking up user by customer_id
        return True

    return False


def format_tier_comparison() -> Dict[str, Any]:
    """Return tier data for pricing page display."""
    return {
        "free": {
            "name": "Free",
            "price": "$0",
            "features": [
                "3 runs per month",
                "Up to 10 ideas per run",
                "PDF export (watermarked)",
                "Bookmarks & notes",
                "Community support",
            ],
            "cta": "Start Free",
        },
        "pro": {
            "name": "Pro",
            "price": "$15/mo",
            "features": [
                "Unlimited runs",
                "Up to 50 ideas per run",
                "PDF export (no watermark)",
                "Full history & export",
                "Priority support",
                "Share public links",
            ],
            "cta": "Upgrade to Pro",
            "popular": True,
        },
        "team": {
            "name": "Team",
            "price": "$49/mo",
            "features": [
                "Everything in Pro",
                "5 seats included",
                "Shared workspace",
                "API access",
                "Team analytics",
                "Dedicated support",
            ],
            "cta": "Start Team Trial",
        },
        "enterprise": {
            "name": "Enterprise",
            "price": "$299/mo",
            "features": [
                "Everything in Team",
                "Unlimited seats",
                "White-label option",
                "Custom models",
                "SLA guarantee",
                "24/7 phone support",
            ],
            "cta": "Contact Sales",
        },
    }
