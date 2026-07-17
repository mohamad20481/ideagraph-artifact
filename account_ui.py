"""
account_ui.py — "Manage my account" page for IdeaGraph.

Single entry point: `render_account_page(st, user_id)`. The function paints
a multi-section page covering profile, password change, billing/plan,
usage stats, data export, and account deletion. Each section is wrapped
in a `try` so a single broken subsystem (missing optional module, DB
hiccup) doesn't take the whole page down.

How to use:
    # somewhere in app.py, after auth check:
    if st.session_state.get("_show_account_page"):
        import account_ui
        account_ui.render_account_page(st, st.session_state["user_id"])
        st.stop()

The "Manage account" button in the sidebar toggles the session-state
flag above. A "← Back to app" button on the account page clears it.
"""
from __future__ import annotations

import json
from typing import Any, Dict, Optional

try:
    import db
    _HAS_DB = True
except Exception:
    _HAS_DB = False

try:
    import billing
    _HAS_BILLING = True
except Exception:
    _HAS_BILLING = False


def render_account_page(st_module, user_id: Optional[int]) -> None:
    """Draw the full Manage Account page in the main area.

    Caller is responsible for the auth check (user_id present + valid).
    A missing user_id renders a "log in first" notice and returns.
    """
    # ── Top bar: Back / title / Logout ─────────────────────────────────────
    top_a, top_b, top_c = st_module.columns([1, 5, 1])
    with top_a:
        if st_module.button("← Back", key="account_back", use_container_width=True):
            st_module.session_state["_show_account_page"] = False
            st_module.rerun()
    with top_b:
        st_module.markdown(
            "<h1 style='margin:0;padding:0;font-size:28px'>⚙️ Manage my account</h1>"
            "<div style='color:#64748b;font-size:13px;margin-top:2px'>"
            "Profile, password, plan, usage, and data controls — all in one place."
            "</div>",
            unsafe_allow_html=True,
        )
    with top_c:
        # Logout — clears the session entirely and bounces to the login
        # page. Mirrors auth_ui.show_user_menu()'s Logout button so users
        # don't have to navigate back to the sidebar to sign out.
        if st_module.button(
            "🚪 Logout",
            key="account_logout",
            use_container_width=True,
            type="secondary",
        ):
            try:
                import auth_ui as _auth_ui
                # Reuse the same cleanup the sidebar Logout uses, so
                # remember/recovery tokens are cleared consistently.
                try:
                    _auth_ui._clear_remember_token()
                except Exception:
                    pass
                try:
                    _auth_ui._clear_session_recovery_token()
                except Exception:
                    pass
            except Exception:
                pass
            for k in (
                "user_id", "username", "logged_in", "running", "done",
                "results", "error", "progress_log", "_show_account_page",
                "_quota_acquired", "_concurrency_acquired",
            ):
                try:
                    st_module.session_state.pop(k, None)
                except Exception:
                    pass
            st_module.session_state["logged_in"] = False
            st_module.success("Logged out 👋")
            st_module.rerun()

    if not user_id:
        st_module.warning("You need to be signed in to manage your account.")
        return

    st_module.divider()

    # Six top-level sections as tabs — keeps the page scannable.
    (tab_profile, tab_security, tab_billing, tab_usage,
     tab_data, tab_danger) = st_module.tabs([
        "👤 Profile",
        "🔒 Security",
        "💳 Plan & billing",
        "📈 Usage",
        "📦 My data",
        "⚠️ Danger zone",
    ])

    with tab_profile:
        _render_profile(st_module, user_id)
    with tab_security:
        _render_security(st_module, user_id)
    with tab_billing:
        _render_billing_tab(st_module, user_id)
    with tab_usage:
        _render_usage(st_module, user_id)
    with tab_data:
        _render_data_export(st_module, user_id)
    with tab_danger:
        _render_danger_zone(st_module, user_id)


# ── Section: Profile ───────────────────────────────────────────────────────

def _render_profile(st_module, user_id: int) -> None:
    st_module.markdown("### 👤 Your profile")
    try:
        profile = db.get_user_profile(user_id) if _HAS_DB else None
    except Exception as e:
        st_module.error(f"Couldn't load profile: {e}")
        return
    if not profile:
        st_module.warning("Profile not found.")
        return

    col_a, col_b = st_module.columns(2)
    with col_a:
        st_module.text_input(
            "Username", value=profile.get("username", ""),
            disabled=True, key="acct_profile_username",
            help="Usernames are immutable on this build. Contact admin to change.",
        )
    with col_b:
        st_module.text_input(
            "Member since", value=profile.get("created_at", ""),
            disabled=True, key="acct_profile_joined",
        )

    # Plan badge (sourced from billing module if available).
    if _HAS_BILLING:
        try:
            info = billing.get_plan(user_id)
            plan = info["_plan"]
            st_module.markdown(
                f"<div style='background:#f0f9ff;border:1px solid #bae6fd;"
                f"border-radius:8px;padding:10px 14px;margin-top:8px;"
                f"display:flex;justify-content:space-between;align-items:center'>"
                f"<div>"
                f"<div style='font-size:11px;color:#0369a1;font-weight:700;"
                f"text-transform:uppercase;letter-spacing:0.05em'>Current plan</div>"
                f"<div style='font-size:18px;font-weight:700;color:#0c4a6e'>"
                f"{plan.label}</div></div>"
                f"<div style='font-size:13px;color:#475569'>"
                f"${plan.price_usd_monthly:.0f}/mo</div></div>",
                unsafe_allow_html=True,
            )
        except Exception:
            pass


# ── Section: Security (password change) ────────────────────────────────────

def _render_security(st_module, user_id: int) -> None:
    st_module.markdown("### 🔒 Change password")
    st_module.caption(
        "Use at least 6 characters. Passwords are hashed with PBKDF2-"
        "HMAC-SHA256 (260k iterations) — we don't store them in plaintext."
    )

    with st_module.form("acct_password_form", clear_on_submit=False):
        old_pw = st_module.text_input(
            "Current password", type="password", key="acct_pw_old",
        )
        col_a, col_b = st_module.columns(2)
        with col_a:
            new_pw = st_module.text_input(
                "New password", type="password", key="acct_pw_new",
            )
        with col_b:
            confirm_pw = st_module.text_input(
                "Confirm new password", type="password", key="acct_pw_confirm",
            )
        submitted = st_module.form_submit_button(
            "Change password", type="primary",
        )

    if submitted:
        if not old_pw or not new_pw:
            st_module.warning("Fill in both the current and new password.")
            return
        if new_pw != confirm_pw:
            st_module.error("The two new-password fields don't match.")
            return
        if not _HAS_DB:
            st_module.error("Database module unavailable — cannot change password.")
            return
        try:
            result = db.change_password(user_id, old_pw, new_pw)
        except Exception as e:
            st_module.error(f"Failed: {e}")
            return
        if result.get("ok"):
            st_module.success("✅ Password updated. Use it on your next login.")
        else:
            st_module.error(result.get("error", "Couldn't change password."))


# ── Section: Plan & billing (delegates to billing module) ──────────────────

def _render_billing_tab(st_module, user_id: int) -> None:
    if not _HAS_BILLING:
        st_module.info(
            "Billing module unavailable. Make sure `billing.py` is "
            "importable from the project root."
        )
        return
    billing.render_plan_card(st_module, user_id)

    # ── Manage payment method / cancel via Stripe Customer Portal ───────
    # Only useful for users who already have a Stripe customer record
    # (i.e. they completed checkout at some point). For free users this
    # silently no-ops.
    try:
        import stripe_integration as _si
    except Exception:
        return
    if not _si.is_configured():
        return
    if not _HAS_DB:
        return
    sub = db.get_user_subscription(user_id) or {}
    if not sub.get("stripe_customer_id"):
        return

    st_module.markdown("---")
    st_module.markdown("### 🔧 Manage subscription on Stripe")
    st_module.caption(
        "Open the Stripe-hosted customer portal to update your payment "
        "method, download invoices, or cancel your subscription. "
        "Cancellations apply at the end of the current billing period."
    )
    if st_module.button(
        "🔗 Open Stripe customer portal",
        key="acct_portal_btn",
        type="secondary",
    ):
        url = _si.create_customer_portal_session(user_id)
        if url:
            st_module.success(
                f"Portal session created. **[Open portal →]({url})**"
            )
            st_module.markdown(
                f'<meta http-equiv="refresh" content="1; url={url}">',
                unsafe_allow_html=True,
            )
        else:
            st_module.error(
                "Couldn't create portal session — check the admin "
                "Stripe configuration panel for details."
            )


# ── Section: Usage (lifetime + current month) ──────────────────────────────

def _render_usage(st_module, user_id: int) -> None:
    st_module.markdown("### 📈 Usage")

    # Current-month runs vs quota.
    if _HAS_BILLING:
        try:
            info = billing.get_plan(user_id)
            plan = info["_plan"]
            used = int(info.get("runs_this_month") or 0)
            limit = plan.monthly_run_limit
            col_a, col_b, col_c = st_module.columns(3)
            col_a.metric("This month's runs", used)
            col_b.metric(
                "Monthly limit",
                "Unlimited" if limit < 0 else str(limit),
            )
            remaining = billing.runs_remaining(user_id)
            col_c.metric(
                "Remaining",
                "∞" if remaining < 0 else str(remaining),
            )
            if limit > 0:
                pct = min(100.0, (used / max(1, limit)) * 100.0)
                st_module.progress(pct / 100.0)
        except Exception as e:
            st_module.caption(f"Usage stats unavailable: {e}")

    st_module.markdown("#### 📚 Lifetime activity")
    if not _HAS_DB:
        st_module.caption("DB unavailable.")
        return
    try:
        results = db.get_user_results(user_id)
        n_runs = len(results)
        n_ideas = sum(int(r.get("ideas_count") or 0) for r in results)
        avg_cov = (
            sum(float(r.get("coverage") or 0) for r in results) / n_runs
            if n_runs else 0.0
        )
        col_a, col_b, col_c = st_module.columns(3)
        col_a.metric("Saved runs", n_runs)
        col_b.metric("Total ideas", n_ideas)
        col_c.metric("Avg coverage", f"{avg_cov:.0%}")
    except Exception as e:
        st_module.caption(f"Couldn't load lifetime stats: {e}")

    # Optional: cost summary if the analytics module wired it up.
    try:
        cost = db.get_user_cost_summary(user_id)
        if cost and any(cost.values()):
            with st_module.expander("💰 Cost summary", expanded=False):
                st_module.json(cost)
    except Exception:
        pass


# ── Section: My data (export takeout) ──────────────────────────────────────

def _render_data_export(st_module, user_id: int) -> None:
    st_module.markdown("### 📦 Export my data")
    st_module.caption(
        "Download everything we store about your account as a single "
        "JSON file: profile, subscription state, and saved results. "
        "Password hashes are excluded."
    )

    # Pre-load lightweight counts so user sees what they're about to export.
    if _HAS_DB:
        try:
            n_runs = len(db.get_user_results(user_id) or [])
            st_module.caption(f"Saved runs in your archive: **{n_runs}**")
        except Exception:
            pass

    if st_module.button(
        "📥 Generate export", key="acct_export_gen", use_container_width=False,
    ):
        if not _HAS_DB:
            st_module.error("DB module unavailable.")
            return
        try:
            payload = db.export_user_data(user_id)
        except Exception as e:
            st_module.error(f"Export failed: {e}")
            return
        blob = json.dumps(payload, indent=2, ensure_ascii=False, default=str)
        st_module.success(
            f"Export ready — {len(blob):,} bytes. Click below to download."
        )
        st_module.download_button(
            label="⬇️ Download ideagraph_export.json",
            data=blob.encode("utf-8"),
            file_name=f"ideagraph_export_user{user_id}.json",
            mime="application/json",
            key="acct_export_dl",
        )


# ── Section: Danger zone (delete account) ──────────────────────────────────

def _render_danger_zone(st_module, user_id: int) -> None:
    st_module.markdown("### ⚠️ Danger zone")
    st_module.markdown(
        "<div style='background:#fef2f2;border:1px solid #fecaca;"
        "border-radius:8px;padding:12px 14px;color:#7f1d1d'>"
        "<b>Delete account</b> — permanently removes your profile, "
        "subscription, saved runs, and every related row. "
        "<u>This cannot be undone.</u>"
        "</div>",
        unsafe_allow_html=True,
    )

    confirm_text = st_module.text_input(
        "Type DELETE to confirm",
        key="acct_delete_confirm_text",
        placeholder="DELETE",
    )
    confirm_pw = st_module.text_input(
        "Re-enter your password",
        type="password",
        key="acct_delete_confirm_pw",
    )
    ok_to_show = (confirm_text == "DELETE")

    if st_module.button(
        "🗑️ Permanently delete my account",
        key="acct_delete_btn",
        type="primary",
        disabled=not ok_to_show,
    ):
        if not confirm_pw:
            st_module.error("Enter your password to confirm.")
            return
        if not _HAS_DB:
            st_module.error("DB module unavailable.")
            return
        try:
            result = db.delete_user(user_id, confirm_pw)
        except Exception as e:
            st_module.error(f"Deletion failed: {e}")
            return
        if not result.get("ok"):
            st_module.error(result.get("error", "Couldn't delete account."))
            return
        # Wipe session and bounce to login.
        for k in (
            "user_id", "username", "logged_in", "running", "done",
            "results", "error", "progress_log", "_show_account_page",
        ):
            try:
                st_module.session_state.pop(k, None)
            except Exception:
                pass
        st_module.success("Account deleted. Goodbye 👋")
        st_module.rerun()
