"""
billing.py — plan catalog + per-account plan management for IdeaGraph.

What's in here:
  - PLANS catalog (free / pro / team / enterprise) with price, feature
    list, monthly quotas, and which advanced features unlock at each tier
  - get_plan(user_id), set_plan(user_id, tier) wrappers around the
    existing db.subscriptions table
  - has_feature(user_id, feature_name) — feature-gating helper. Default
    is permissive (returns True when no plan info exists or when the
    feature isn't gated) so adding this module doesn't accidentally
    break existing functionality
  - render_plan_card(st, user_id) — a Streamlit panel that shows the
    user's current plan, feature checklist, monthly usage, and upgrade
    CTAs. Drop into any page.
  - render_admin_plan_override(st) — admin-only panel for manually
    setting another user's tier (useful for comps, debugging, freeing
    a tester who hit a quota)

What's deliberately NOT here:
  - Real Stripe checkout. The "Upgrade" buttons currently surface a
    placeholder "Stripe checkout would happen here". Wiring real Stripe
    needs an API key, webhook endpoints, and a billing reconciliation
    job — that's its own multi-day project. The data model + UI shape
    is ready for it: just plug in `stripe.checkout.Session.create()` in
    `start_upgrade_flow()` and add a `/stripe-webhook` route.
  - Hard feature gates on existing functionality. `has_feature()` is
    only called by NEW code that opts in. Existing features keep
    working regardless of tier (we don't want to lock a paying user
    out of a feature they already use just because we shipped this
    file).

Public API:
    PLANS                                         → dict[tier, plan_info]
    DEFAULT_TIER                                  → "free"
    get_plan(user_id)                             → dict
    set_plan(user_id, tier, persist=True)         → None
    has_feature(user_id, feature)                 → bool
    plan_features(tier)                           → list[str]
    monthly_run_limit(tier)                       → int (-1 = unlimited)
    render_plan_card(st, user_id)                 → None
    render_admin_plan_override(st)                → None
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

try:
    import db
    _HAS_DB = True
except Exception:
    _HAS_DB = False


# ── Plan catalog ─────────────────────────────────────────────────────────────

@dataclass
class Plan:
    """One subscription tier."""
    tier: str                          # canonical id (free/pro/team/enterprise)
    label: str                         # display name with emoji
    price_usd_monthly: float           # 0 for free
    tagline: str                       # one-line pitch
    features: List[str] = field(default_factory=list)   # bullet points shown in UI
    monthly_run_limit: int = -1        # -1 = unlimited
    unlocks: List[str] = field(default_factory=list)    # feature ids unlocked at this tier
    cta_label: str = "Upgrade"         # button text


# Feature IDs — referenced by has_feature() across the app. Adding a new
# gated feature: pick an id, add it to a plan's `unlocks` list. All free-
# tier features should be in the "free" plan's unlocks so has_feature()
# returns True for them universally.
FEATURE_IDS = {
    "visual_abstract":      "🎨 Visual abstract (single image)",
    "multi_panel_figures":  "📊 Multi-panel paper figures (4-panel set)",
    "video_generation":     "🎥 Animated explainers (Veo)",
    "n_sample_variants":    "🔀 N-sample variant generation",
    "all_image_styles":     "🎭 All 8 image style presets",
    "all_novelty_modes":    "🧪 All 19 Novelty Lab modes",
    "chat_to_optimize":     "💬 Chat-to-optimize per idea",
    "multi_worker":         "⚡ Multi-worker parallel runs",
    "priority_queue":       "🚀 Priority queue (jump line)",
    "shared_archive":       "👥 Team-shared idea archive",
    "export_paper":         "📄 Export idea as paper draft",
    "custom_provider":      "🔌 Custom LLM provider endpoints",
    "sla_support":          "📞 SLA-backed support",
}


PLANS: Dict[str, Plan] = {
    "free": Plan(
        tier="free",
        label="🆓 Free",
        price_usd_monthly=0.0,
        tagline="Try IdeaGraph — perfect for a single PhD chapter.",
        features=[
            "5 ideation runs per month",
            "All 19 Novelty Lab modes",
            "All 25 sort + group modes",
            "Single-image visual abstracts (FLUX schnell)",
            "Chat-to-optimize per idea",
            "Local SQLite archive (your machine)",
        ],
        monthly_run_limit=5,
        unlocks=[
            "visual_abstract",
            "all_novelty_modes",
            "chat_to_optimize",
        ],
        cta_label="Current plan",
    ),
    "pro": Plan(
        tier="pro",
        label="⭐ Pro",
        price_usd_monthly=15.0,
        tagline="For a working researcher across multiple projects.",
        features=[
            "100 ideation runs per month",
            "Everything in Free",
            "All 8 image style presets (editorial / 3D / sketch / blueprint / …)",
            "📊 Multi-panel paper figure sets (4-panel ZIP export)",
            "🔀 N-sample variants (pick the best of 8)",
            "Imagen 4 + Imagen 4 Ultra image generation",
            "Export ideas as Markdown paper drafts",
            "Email support (48h SLA)",
        ],
        monthly_run_limit=100,
        unlocks=[
            "visual_abstract", "all_novelty_modes", "chat_to_optimize",
            "multi_panel_figures", "n_sample_variants",
            "all_image_styles", "export_paper",
        ],
        cta_label="Upgrade to Pro",
    ),
    "team": Plan(
        tier="team",
        label="👥 Team",
        price_usd_monthly=49.0,
        tagline="For a research group of 3-5 sharing one archive.",
        features=[
            "500 ideation runs per month (shared across team)",
            "Everything in Pro",
            "🎥 Veo 3 animated explainers (4s and 8s clips)",
            "⚡ Multi-worker parallel runs (up to 5 workers)",
            "👥 Team-shared idea archive",
            "Custom LLM provider endpoints (corporate proxies, self-hosted)",
            "Email support (24h SLA)",
        ],
        monthly_run_limit=500,
        unlocks=[
            "visual_abstract", "all_novelty_modes", "chat_to_optimize",
            "multi_panel_figures", "n_sample_variants",
            "all_image_styles", "export_paper",
            "video_generation", "multi_worker", "shared_archive",
            "custom_provider",
        ],
        cta_label="Upgrade to Team",
    ),
    "enterprise": Plan(
        tier="enterprise",
        label="🏢 Enterprise",
        price_usd_monthly=299.0,
        tagline="For a department / lab with SLA + dedicated resources.",
        features=[
            "Unlimited ideation runs",
            "Everything in Team",
            "🚀 Priority queue (jump ahead of free/pro users)",
            "📞 SLA-backed support with named contact",
            "Dedicated worker process on shared VPS",
            "Custom Stripe / PO billing",
            "On-prem deployment guide",
        ],
        monthly_run_limit=-1,  # unlimited
        unlocks=[
            "visual_abstract", "all_novelty_modes", "chat_to_optimize",
            "multi_panel_figures", "n_sample_variants",
            "all_image_styles", "export_paper",
            "video_generation", "multi_worker", "shared_archive",
            "custom_provider",
            "priority_queue", "sla_support",
        ],
        cta_label="Contact sales",
    ),
}


DEFAULT_TIER: str = "free"


# Tier ordering — used to decide if a "downgrade" warning should fire.
_TIER_ORDER = {"free": 0, "pro": 1, "team": 2, "enterprise": 3}


# ── Plan I/O ─────────────────────────────────────────────────────────────────

def get_plan(user_id: Optional[int]) -> Dict[str, Any]:
    """Return the current plan info for a user.

    Returns a dict with the same shape `db.get_user_subscription` returns,
    plus an `_plan` field carrying the matching Plan dataclass. Falls
    back to the free tier (and the Plan object) when no DB row exists
    or no user_id is supplied.
    """
    tier = DEFAULT_TIER
    runs_this_month = 0
    raw: Dict[str, Any] = {}
    if user_id and _HAS_DB:
        try:
            raw = db.get_user_subscription(int(user_id)) or {}
            tier = (raw.get("tier") or DEFAULT_TIER).lower()
            runs_this_month = int(raw.get("runs_this_month") or 0)
        except Exception:
            pass
    plan = PLANS.get(tier) or PLANS[DEFAULT_TIER]
    return {
        **raw,
        "tier": plan.tier,
        "runs_this_month": runs_this_month,
        "_plan": plan,
    }


def set_plan(
    user_id: int,
    tier: str,
    persist: bool = True,
) -> None:
    """Set a user's subscription tier.

    Raises ValueError on an unknown tier. When `persist=False`, only the
    in-memory cache is touched (useful for tests). With `persist=True`
    (the default), writes to db.subscriptions via update_subscription().
    """
    if tier not in PLANS:
        raise ValueError(
            f"unknown tier {tier!r}; must be one of {sorted(PLANS)}"
        )
    if not user_id:
        raise ValueError("user_id is required")
    if persist and _HAS_DB:
        try:
            db.update_subscription(int(user_id), tier=tier)
        except Exception as e:
            raise RuntimeError(f"failed to persist plan: {e}") from e


def plan_features(tier: str) -> List[str]:
    plan = PLANS.get(tier) or PLANS[DEFAULT_TIER]
    return list(plan.features)


def monthly_run_limit(tier: str) -> int:
    plan = PLANS.get(tier) or PLANS[DEFAULT_TIER]
    return plan.monthly_run_limit


def has_feature(user_id: Optional[int], feature: str) -> bool:
    """Check whether the user's plan unlocks `feature`.

    Permissive by default: unknown feature IDs return True (so adding
    feature checks in new code doesn't accidentally lock anyone out of
    something we haven't catalogued yet). For canonical feature IDs in
    FEATURE_IDS, gating is enforced per plan.
    """
    if not feature:
        return True
    if feature not in FEATURE_IDS:
        # Unknown / untracked feature → don't gate.
        return True
    info = get_plan(user_id)
    plan = info.get("_plan") or PLANS[DEFAULT_TIER]
    return feature in plan.unlocks


def runs_remaining(user_id: Optional[int]) -> int:
    """How many runs remain this month for this user. -1 for unlimited."""
    info = get_plan(user_id)
    limit = info["_plan"].monthly_run_limit
    if limit < 0:
        return -1
    used = int(info.get("runs_this_month") or 0)
    return max(0, limit - used)


# ── Upgrade flow (real Stripe Checkout via stripe_integration.py) ──────────

_STRIPE_NOT_CONFIGURED_MSG = (
    "💳 **Stripe is not configured.** Admin needs to set the keys in "
    "Admin Dashboard → 💳 Billing → Stripe configuration. Until then, "
    "admin can grant you the tier manually via the same panel."
)

_STRIPE_MISSING_PRICE_ID_MSG = (
    "💳 Stripe is connected, but no price ID is set for this tier. "
    "Admin needs to create the matching Stripe product (e.g. \"Pro "
    "$15/mo\") in dashboard.stripe.com and paste the resulting "
    "`price_…` ID into .env (STRIPE_PRICE_PRO / _TEAM / _ENTERPRISE)."
)


def start_upgrade_flow(
    user_id: int,
    target_tier: str,
    customer_email: str = "",
) -> Dict[str, Any]:
    """Kick off the upgrade flow for one user.

    Tries the real Stripe Checkout path; falls back to a helpful message
    when Stripe isn't configured or the tier has no price ID yet.

    Returns:
        {"target_tier", "target_label", "target_price",
         "checkout_url": "<url or None>",
         "message": "<status or fallback hint>"}
    """
    if target_tier not in PLANS:
        return {"error": f"unknown tier: {target_tier}"}
    plan = PLANS[target_tier]

    # Enterprise is "contact sales" — no self-serve checkout.
    if target_tier == "enterprise":
        return {
            "target_tier": target_tier,
            "target_label": plan.label,
            "target_price": plan.price_usd_monthly,
            "checkout_url": None,
            "message": (
                "🏢 **Enterprise** is contact-sales only. Email "
                "support@ideagraph.app with your team size and "
                "requirements."
            ),
        }

    try:
        import stripe_integration as _stripe_mod
    except Exception:
        return {
            "target_tier": target_tier,
            "target_label": plan.label,
            "target_price": plan.price_usd_monthly,
            "checkout_url": None,
            "message": _STRIPE_NOT_CONFIGURED_MSG,
        }

    if not _stripe_mod.is_configured():
        return {
            "target_tier": target_tier,
            "target_label": plan.label,
            "target_price": plan.price_usd_monthly,
            "checkout_url": None,
            "message": _STRIPE_NOT_CONFIGURED_MSG,
        }

    if not _stripe_mod.PRICE_IDS.get(target_tier):
        return {
            "target_tier": target_tier,
            "target_label": plan.label,
            "target_price": plan.price_usd_monthly,
            "checkout_url": None,
            "message": _STRIPE_MISSING_PRICE_ID_MSG,
        }

    url = _stripe_mod.create_checkout_session(
        user_id=int(user_id),
        tier=target_tier,
        customer_email=customer_email or "",
    )
    if not url:
        return {
            "target_tier": target_tier,
            "target_label": plan.label,
            "target_price": plan.price_usd_monthly,
            "checkout_url": None,
            "message": (
                "💳 Stripe rejected the checkout request. Check the "
                "server log and Stripe dashboard → Events for the "
                "specific error."
            ),
        }
    return {
        "target_tier": target_tier,
        "target_label": plan.label,
        "target_price": plan.price_usd_monthly,
        "checkout_url": url,
        "message": "Redirecting to Stripe Checkout…",
    }


# ── Streamlit UI helpers ────────────────────────────────────────────────────

def render_plan_card(st_module, user_id: Optional[int]) -> None:
    """Render a user-facing plan card: current tier, feature checklist,
    monthly usage bar, and CTAs to upgrade or downgrade.

    Drop this into a sidebar section, a settings page, or anywhere in
    the main area.
    """
    info = get_plan(user_id)
    plan: Plan = info["_plan"]
    used = int(info.get("runs_this_month") or 0)
    limit = plan.monthly_run_limit
    is_unlimited = limit < 0

    # ── Header card ─────────────────────────────────────────────────────
    st_module.markdown(
        f"<div style='border:1px solid #bae6fd;border-radius:12px;"
        f"padding:16px 18px;margin:8px 0;"
        f"background:linear-gradient(135deg,#f0f9ff 0%,#e0f2fe 100%)'>"
        f"<div style='display:flex;justify-content:space-between;"
        f"align-items:flex-start;gap:12px;flex-wrap:wrap'>"
        f"<div>"
        f"<div style='font-size:11px;color:#0369a1;font-weight:700;"
        f"text-transform:uppercase;letter-spacing:0.06em'>"
        f"Current plan</div>"
        f"<div style='font-size:22px;font-weight:800;color:#0c4a6e;"
        f"margin-top:2px'>{plan.label}</div>"
        f"<div style='font-size:12px;color:#475569;margin-top:4px;"
        f"max-width:340px'>{plan.tagline}</div>"
        f"</div>"
        f"<div style='text-align:right'>"
        f"<div style='font-size:11px;color:#64748b'>Price</div>"
        f"<div style='font-size:24px;font-weight:800;color:#0c4a6e'>"
        f"${plan.price_usd_monthly:.0f}"
        f"<span style='font-size:12px;color:#64748b'>"
        f"/mo</span></div>"
        f"</div></div></div>",
        unsafe_allow_html=True,
    )

    # ── Usage row ───────────────────────────────────────────────────────
    if is_unlimited:
        st_module.caption(
            f"📈 Usage this month: **{used}** runs · unlimited"
        )
    else:
        pct = min(100.0, (used / max(1, limit)) * 100.0)
        bar_color = (
            "#10b981" if pct < 70
            else "#f59e0b" if pct < 100
            else "#ef4444"
        )
        st_module.markdown(
            f"<div style='font-size:13px;color:#475569;margin:4px 0'>"
            f"📈 <b>{used}</b> of <b>{limit}</b> runs this month "
            f"({pct:.0f}%)</div>"
            f"<div style='background:#e2e8f0;border-radius:6px;height:8px;"
            f"overflow:hidden'>"
            f"<div style='background:{bar_color};width:{pct:.0f}%;"
            f"height:100%;border-radius:6px'></div></div>",
            unsafe_allow_html=True,
        )

    # ── Feature checklist ──────────────────────────────────────────────
    st_module.markdown(
        f"<div style='font-size:11px;color:#0369a1;font-weight:700;"
        f"text-transform:uppercase;letter-spacing:0.06em;margin:14px 0 4px 0'>"
        f"What's included</div>",
        unsafe_allow_html=True,
    )
    for line in plan.features:
        st_module.markdown(f"- ✓ {line}")

    # ── Compare with other tiers (collapsed by default) ─────────────────
    with st_module.expander("🔍 Compare with other plans", expanded=False):
        # Render every plan's feature list side-by-side via columns.
        _other_tiers = [t for t in PLANS if t != plan.tier]
        if _other_tiers:
            cols = st_module.columns(len(_other_tiers))
            for col, t in zip(cols, _other_tiers):
                with col:
                    op = PLANS[t]
                    direction = "Upgrade ▲" \
                        if _TIER_ORDER[t] > _TIER_ORDER[plan.tier] \
                        else "Downgrade ▼"
                    st_module.markdown(
                        f"**{op.label}** — `${op.price_usd_monthly:.0f}/mo`  \n"
                        f"_{direction}_"
                    )
                    for line in op.features:
                        st_module.markdown(f"- {line}")

    # ── Upgrade CTAs ────────────────────────────────────────────────────
    # Show buttons only for HIGHER tiers (upgrade direction).
    higher_tiers = [
        t for t in PLANS
        if _TIER_ORDER[t] > _TIER_ORDER[plan.tier]
    ]
    if higher_tiers and user_id:
        st_module.markdown(
            "<div style='font-size:11px;color:#0369a1;font-weight:700;"
            "text-transform:uppercase;letter-spacing:0.06em;"
            "margin:18px 0 4px 0'>Upgrade options</div>",
            unsafe_allow_html=True,
        )
        cols = st_module.columns(len(higher_tiers))
        for col, t in zip(cols, higher_tiers):
            p = PLANS[t]
            if col.button(
                f"{p.cta_label} — ${p.price_usd_monthly:.0f}/mo",
                key=f"plan_upgrade_{t}",
                use_container_width=True,
                type="primary" if t == "pro" else "secondary",
            ):
                _action = start_upgrade_flow(int(user_id), t)
                _url = _action.get("checkout_url")
                if _url:
                    # Streamlit can't programmatically redirect, but a
                    # prominent link + auto-open meta tag does the job.
                    st_module.success(
                        f"✅ Checkout session created. "
                        f"**[Click here to pay →]({_url})**"
                    )
                    st_module.markdown(
                        f'<meta http-equiv="refresh" content="1; url={_url}">',
                        unsafe_allow_html=True,
                    )
                    st_module.caption(
                        "Redirecting automatically in 1 second… If "
                        "nothing happens, click the link above."
                    )
                else:
                    st_module.info(_action.get("message", ""))

    # No CTAs for the highest tier — just acknowledge.
    if plan.tier == "enterprise":
        st_module.success(
            "🏢 You're on the highest tier — enterprise. Contact sales "
            "for custom add-ons (on-prem deployment, dedicated SLAs)."
        )


def render_admin_plan_override(st_module) -> None:
    """Admin-only: manually set any user's tier. Useful for comps,
    testing, or freeing a paying user who hit a soft quota.

    Caller is responsible for the is_admin() check before invoking this.
    """
    import db as _db_mod  # local import — admin tab only renders when db is up
    st_module.markdown("### 💳 Override user plan")
    st_module.caption(
        "Set a specific user's tier directly. Bypasses Stripe — useful "
        "for granting comps to testers or freeing someone who hit a "
        "soft quota."
    )

    # User picker — load all users from db.
    try:
        with _db_mod._lock:
            conn = _db_mod._get_conn()
            try:
                rows = conn.execute(
                    "SELECT u.id, u.username, "
                    "       COALESCE(s.tier, 'free') AS current_tier, "
                    "       COALESCE(s.runs_this_month, 0) AS runs "
                    "  FROM users u "
                    "  LEFT JOIN subscriptions s ON s.user_id = u.id "
                    "  ORDER BY u.id",
                ).fetchall()
            finally:
                conn.close()
    except Exception as e:
        st_module.error(f"Couldn't load users: {e}")
        return

    if not rows:
        st_module.info("No users registered yet.")
        return

    users = [dict(r) for r in rows]
    options = [u["id"] for u in users]
    by_id = {u["id"]: u for u in users}
    sel_id = st_module.selectbox(
        "User",
        options=options,
        format_func=lambda uid: (
            f"#{uid}  {by_id[uid]['username']}  "
            f"[current: {by_id[uid]['current_tier']} · "
            f"{by_id[uid]['runs']} runs/mo]"
        ),
        key="admin_billing_user_pick",
    )
    sel = by_id[sel_id]
    current = sel["current_tier"]

    # Tier picker.
    new_tier = st_module.selectbox(
        "Set tier to",
        options=list(PLANS.keys()),
        index=list(PLANS.keys()).index(current)
            if current in PLANS else 0,
        format_func=lambda t: (
            f"{PLANS[t].label} — ${PLANS[t].price_usd_monthly:.0f}/mo"
        ),
        key="admin_billing_tier_pick",
    )

    # Optional: reset run counter (for monthly comps).
    reset_runs = st_module.checkbox(
        "Reset monthly run counter to 0",
        value=False,
        key="admin_billing_reset_runs",
        help="Useful when granting comp time — gives the user a fresh "
              "monthly quota at the new tier.",
    )

    if st_module.button(
        f"Apply: set user #{sel_id} to {new_tier}",
        type="primary",
        key="admin_billing_apply",
    ):
        try:
            set_plan(int(sel_id), new_tier, persist=True)
            if reset_runs:
                with _db_mod._lock:
                    conn = _db_mod._get_conn()
                    try:
                        conn.execute(
                            "UPDATE subscriptions SET runs_this_month = 0 "
                            "WHERE user_id = ?",
                            (int(sel_id),),
                        )
                        conn.commit()
                    finally:
                        conn.close()
            st_module.success(
                f"✅ User #{sel_id} ({sel['username']}) set to "
                f"**{PLANS[new_tier].label}**."
                + (" Run counter reset." if reset_runs else "")
            )
        except Exception as e:
            st_module.error(f"Failed to update: {e}")
