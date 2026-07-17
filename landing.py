"""
landing.py - Marketing landing page for IdeaGraph.

Shows hero section, features, demo video placeholder, pricing tiers,
email capture, public leaderboard, and social proof.

Usage: called from app.py when user is not logged in or visits ?landing=1
"""

from __future__ import annotations

import streamlit as st

import db
import newsletter
from stripe_integration import format_tier_comparison, is_configured as stripe_configured


def render_landing_page() -> None:
    """Render the full landing page."""

    # ── Custom CSS ────────────────────────────────────────────────────────
    st.markdown(
        """
        <style>
        .hero {
            background: linear-gradient(135deg, #0ea5e9 0%, #0284c7 50%, #0369a1 100%);
            padding: 60px 30px;
            border-radius: 16px;
            text-align: center;
            margin-bottom: 40px;
            color: white;
        }
        .hero h1 {
            font-size: 42px;
            margin: 0 0 16px 0;
            color: white;
            line-height: 1.2;
            font-weight: 800;
        }
        .hero p {
            font-size: 18px;
            opacity: 0.92;
            max-width: 640px;
            margin: 0 auto 24px auto;
            line-height: 1.6;
        }
        .pricing-card {
            background: #f0f9ff;
            border: 1px solid #bae6fd;
            border-radius: 12px;
            padding: 24px;
            height: 100%;
        }
        .pricing-card.popular {
            border: 2px solid #0ea5e9;
            position: relative;
            background: #e0f2fe;
        }
        .pricing-card.popular::before {
            content: "MOST POPULAR";
            position: absolute;
            top: -12px;
            left: 50%;
            transform: translateX(-50%);
            background: #0ea5e9;
            color: white;
            padding: 4px 16px;
            border-radius: 20px;
            font-size: 10px;
            font-weight: bold;
            letter-spacing: 1px;
        }
        .pricing-price {
            font-size: 36px;
            font-weight: 800;
            color: #0284c7;
            margin: 8px 0;
        }
        .feature-icon {
            font-size: 40px;
            margin-bottom: 12px;
        }
        .feature-card {
            background: #f0f9ff;
            border: 1px solid #e0f2fe;
            border-radius: 12px;
            padding: 24px;
            text-align: center;
            height: 100%;
        }
        .feature-card h3 {
            color: #0c4a6e;
            margin: 8px 0;
        }
        .feature-card p {
            color: #475569;
            font-size: 14px;
            line-height: 1.5;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # ── Hero Section ──────────────────────────────────────────────────────
    st.markdown(
        """
        <div class="hero">
            <h1>Generate Breakthrough Research Ideas with AI</h1>
            <p>IdeaGraph is your AI research co-pilot. Enter any topic and get dozens of novel, executable research ideas — grounded in real literature, ranked by quality, and ready to publish.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # CTA buttons
    cta_col1, cta_col2, cta_col3 = st.columns([1, 2, 1])
    with cta_col2:
        c1, c2 = st.columns(2)
        with c1:
            if st.button("🚀 Start Free Now", type="primary", use_container_width=True):
                st.session_state["_show_auth"] = True
                st.rerun()
        with c2:
            if st.button("📺 Watch Demo", type="secondary", use_container_width=True):
                st.session_state["_show_demo"] = True

    if st.session_state.get("_show_demo"):
        st.markdown("### 📺 Demo Video")
        st.info(
            "**Demo video coming soon!** Meanwhile, here's what IdeaGraph does:\n\n"
            "1. Enter a research topic (e.g., 'transformer attention mechanisms')\n"
            "2. AI builds a knowledge graph from 20+ papers in ~30 seconds\n"
            "3. Generates 20-50 novel research ideas using 3 strategies\n"
            "4. Filters through 4 quality probes (code/dataset/compute/novelty)\n"
            "5. Returns ranked ideas with full details + exports\n\n"
            "Each run costs less than $0.10 in API fees."
        )

    st.divider()

    # ── Features Section ──────────────────────────────────────────────────
    st.markdown("## ✨ Why IdeaGraph?")

    f1, f2, f3 = st.columns(3)
    with f1:
        st.markdown(
            '<div class="feature-card">'
            '<div class="feature-icon">🧠</div>'
            '<h3>125+ Optimizations</h3>'
            '<p>Powered by research-grade algorithms: MCTS, Thompson sampling, '
            'Pareto fronts, and more.</p></div>',
            unsafe_allow_html=True,
        )
    with f2:
        st.markdown(
            '<div class="feature-card">'
            '<div class="feature-icon">📚</div>'
            '<h3>Literature-Grounded</h3>'
            '<p>Every idea is built on real papers from Semantic Scholar. '
            'No hallucinations — only grounded research.</p></div>',
            unsafe_allow_html=True,
        )
    with f3:
        st.markdown(
            '<div class="feature-card">'
            '<div class="feature-icon">⚡</div>'
            '<h3>Full Pipeline</h3>'
            '<p>Ideas → Experiment → Code → Execution → Paper → Review. '
            'All automated in one click.</p></div>',
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Public Leaderboard ────────────────────────────────────────────────
    st.markdown("## 🏆 Community Leaderboard")
    st.caption("Top research ideas generated by the community this week")

    try:
        top_ideas = db.get_top_shared_ideas(limit=5)
        if top_ideas:
            for i, item in enumerate(top_ideas, 1):
                idea = item.get("idea", {})
                title = idea.get("title", "Untitled")[:80]
                topic = item.get("topic", "Research")[:40]
                views = item.get("views", 0)
                likes = item.get("likes", 0)
                q = idea.get("quality_score", 0)

                rank_emoji = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"][min(i - 1, 4)]

                st.markdown(
                    f"""
                    <div style="background: rgba(52, 152, 219, 0.05); border-left: 3px solid #3498db;
                                padding: 12px 16px; border-radius: 8px; margin-bottom: 8px;">
                        <div style="font-size: 11px; color: #7f8c8d; text-transform: uppercase;">
                            {rank_emoji} {topic}
                        </div>
                        <div style="font-weight: bold; font-size: 15px; margin: 4px 0;">{title}</div>
                        <div style="font-size: 12px; color: #95a5a6;">
                            Quality: {q:.2f} · 👁️ {views} · ❤️ {likes}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.info("Be the first to share an idea! Sign up and create one to appear here.")
    except Exception:
        st.info("Leaderboard loading...")

    st.divider()

    # ── Pricing Section ───────────────────────────────────────────────────
    st.markdown("## 💰 Simple Pricing")
    st.caption("Start free. Upgrade when you need more.")

    tiers = format_tier_comparison()
    price_cols = st.columns(4)

    for i, (tier_key, tier_data) in enumerate(tiers.items()):
        with price_cols[i]:
            popular_class = "popular" if tier_data.get("popular") else ""
            features_html = "".join(
                f"<li style='color: #7f8c8d; font-size: 13px; line-height: 2;'>✓ {f}</li>"
                for f in tier_data["features"]
            )
            st.markdown(
                f"""
                <div class="pricing-card {popular_class}">
                    <h3 style="margin: 0;">{tier_data['name']}</h3>
                    <div class="pricing-price">{tier_data['price']}</div>
                    <ul style="list-style: none; padding: 0; margin: 16px 0;">
                        {features_html}
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button(tier_data["cta"], key=f"tier_{tier_key}", use_container_width=True):
                if tier_key == "free":
                    st.session_state["_show_auth"] = True
                    st.rerun()
                elif stripe_configured():
                    st.info(f"Redirecting to checkout for {tier_data['name']}...")
                else:
                    st.warning("Payment system not configured yet. Contact support.")

    st.divider()

    # ── Email Capture ─────────────────────────────────────────────────────
    st.markdown("## 📧 Weekly Digest")
    st.caption("Get the best AI-generated research ideas in your inbox every Monday")

    email_col1, email_col2 = st.columns([3, 1])
    with email_col1:
        email_input = st.text_input(
            "Your email",
            placeholder="you@university.edu",
            key="landing_email_input",
            label_visibility="collapsed",
            autocomplete="email",
        )
    with email_col2:
        if st.button("Subscribe", type="primary", use_container_width=True):
            if email_input:
                success, msg = newsletter.subscribe(email_input)
                if success:
                    st.success(msg)
                else:
                    st.error(msg)
            else:
                st.warning("Please enter an email")

    st.divider()

    # ── Footer ────────────────────────────────────────────────────────────
    st.markdown(
        """
        <div style="text-align: center; padding: 30px; color: #95a5a6; font-size: 12px;">
            <p><strong>IdeaGraph</strong> — AI-powered research ideation platform</p>
            <p>Built with 125+ optimization techniques · Powered by state-of-the-art LLMs</p>
            <p>© 2026 IdeaGraph · <a href="#" style="color: #95a5a6;">Privacy</a> · <a href="#" style="color: #95a5a6;">Terms</a> · <a href="#" style="color: #95a5a6;">Contact</a></p>
        </div>
        """,
        unsafe_allow_html=True,
    )
