"""
newsletter.py - Email capture and weekly digest builder.

Features:
  - Email subscription capture
  - Weekly digest HTML generation (top ideas across users)
  - Unsubscribe handling
  - SMTP sending (optional, requires SENDGRID_API_KEY or SMTP env vars)

Setup for sending:
  Option 1 (SendGrid):
    SENDGRID_API_KEY=your_key
    FROM_EMAIL=hello@ideagraph.ai

  Option 2 (SMTP):
    SMTP_HOST=smtp.gmail.com
    SMTP_PORT=587
    SMTP_USER=your@email.com
    SMTP_PASSWORD=app_password
    FROM_EMAIL=hello@ideagraph.ai
"""

from __future__ import annotations

import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

import db


EMAIL_REGEX = re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")


def validate_email(email: str) -> bool:
    """Validate email format."""
    return bool(EMAIL_REGEX.match(email.strip()))


def subscribe(email: str, preferences: str = "weekly") -> tuple:
    """
    Subscribe an email to the newsletter.
    Returns (success: bool, message: str).
    """
    email = email.strip().lower()
    if not validate_email(email):
        return False, "Please enter a valid email address."

    if db.add_email_subscriber(email, preferences):
        return True, f"Subscribed! You'll get the weekly digest at {email}."
    else:
        return False, "This email is already subscribed."


def unsubscribe(email: str) -> tuple:
    """Unsubscribe an email. Returns (success, message)."""
    email = email.strip().lower()
    db.unsubscribe_email(email)
    return True, f"Unsubscribed {email}. We're sorry to see you go."


def build_weekly_digest_html(top_ideas: List[Dict[str, Any]], week_of: str = None) -> str:
    """Build a beautiful HTML email with the top ideas of the week."""
    week = week_of or datetime.now().strftime("%B %d, %Y")

    ideas_html = ""
    for i, item in enumerate(top_ideas[:10], 1):
        idea = item.get("idea", {}) if "idea" in item else item
        title = idea.get("title", "Untitled")[:100]
        topic = item.get("topic", idea.get("topic", "Research"))[:50]
        q = idea.get("quality_score", 0)
        views = item.get("views", 0)
        likes = item.get("likes", 0)
        motivation = (idea.get("motivation", "") or idea.get("method", ""))[:150]

        badge_color = "#2ecc71" if q >= 0.7 else "#f39c12" if q >= 0.5 else "#e74c3c"
        token = item.get("token", "")
        share_url = f"https://ideagraph.ai/?share={token}" if token else "#"

        ideas_html += f"""
        <div style="background: white; border-radius: 8px; padding: 20px; margin-bottom: 16px; border-left: 4px solid {badge_color};">
            <div style="font-size: 11px; color: #7f8c8d; text-transform: uppercase; letter-spacing: 1px;">
                #{i} · {topic}
            </div>
            <h3 style="margin: 6px 0 12px 0; color: #2c3e50;">
                <a href="{share_url}" style="color: #2c3e50; text-decoration: none;">{title}</a>
            </h3>
            <p style="color: #555; line-height: 1.6; margin: 0 0 10px 0;">{motivation}</p>
            <div style="font-size: 12px; color: #95a5a6;">
                Quality: <span style="color: {badge_color}; font-weight: bold;">{q:.2f}</span>
                · 👁️ {views} views · ❤️ {likes} likes
            </div>
        </div>
        """

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>IdeaGraph Weekly Digest</title>
    </head>
    <body style="margin: 0; padding: 0; background: #f5f6fa; font-family: -apple-system, 'Segoe UI', sans-serif;">
        <div style="max-width: 640px; margin: 40px auto; background: white; border-radius: 12px; overflow: hidden; box-shadow: 0 4px 20px rgba(0,0,0,0.08);">

            <!-- Header -->
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 40px; text-align: center;">
                <div style="font-size: 12px; color: rgba(255,255,255,0.8); letter-spacing: 2px; text-transform: uppercase;">Weekly Digest</div>
                <h1 style="color: white; margin: 8px 0 4px 0; font-size: 28px;">IdeaGraph</h1>
                <div style="color: rgba(255,255,255,0.9); font-size: 14px;">{week}</div>
            </div>

            <!-- Intro -->
            <div style="padding: 30px 30px 10px 30px;">
                <h2 style="color: #2c3e50; margin: 0 0 12px 0;">This Week's Best Research Ideas</h2>
                <p style="color: #7f8c8d; line-height: 1.6; margin: 0;">
                    The AI generated over 1,000 research ideas this week across every field. Here are the top 10 that caught the community's attention — ranked by quality, views, and likes.
                </p>
            </div>

            <!-- Ideas -->
            <div style="padding: 20px 30px;">
                {ideas_html}
            </div>

            <!-- CTA -->
            <div style="padding: 30px; text-align: center; background: #f8f9fa;">
                <a href="https://ideagraph.ai" style="display: inline-block; background: #3498db; color: white; padding: 14px 36px; border-radius: 8px; text-decoration: none; font-weight: bold;">
                    Generate Your Own Ideas →
                </a>
                <p style="color: #95a5a6; font-size: 12px; margin: 16px 0 0 0;">
                    Free tier includes 3 runs per month. No credit card required.
                </p>
            </div>

            <!-- Footer -->
            <div style="padding: 20px 30px; text-align: center; font-size: 11px; color: #95a5a6; border-top: 1px solid #ecf0f1;">
                You're receiving this because you subscribed to IdeaGraph.<br>
                <a href="https://ideagraph.ai/unsubscribe" style="color: #95a5a6;">Unsubscribe</a>
                · <a href="https://ideagraph.ai" style="color: #95a5a6;">Visit Site</a>
            </div>

        </div>
    </body>
    </html>
    """


def send_email_sendgrid(to_email: str, subject: str, html_body: str) -> bool:
    """Send email via SendGrid. Returns True on success."""
    api_key = os.getenv("SENDGRID_API_KEY", "")
    from_email = os.getenv("FROM_EMAIL", "hello@ideagraph.ai")
    if not api_key:
        return False

    try:
        import requests
        resp = requests.post(
            "https://api.sendgrid.com/v3/mail/send",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "personalizations": [{"to": [{"email": to_email}]}],
                "from": {"email": from_email, "name": "IdeaGraph"},
                "subject": subject,
                "content": [{"type": "text/html", "value": html_body}],
            },
            timeout=10,
        )
        return resp.status_code < 300
    except Exception:
        return False


def send_email_smtp(to_email: str, subject: str, html_body: str) -> bool:
    """Send email via SMTP. Returns True on success."""
    host = os.getenv("SMTP_HOST", "")
    port = int(os.getenv("SMTP_PORT", "587"))
    user = os.getenv("SMTP_USER", "")
    password = os.getenv("SMTP_PASSWORD", "")
    from_email = os.getenv("FROM_EMAIL", user)

    if not host or not user or not password:
        return False

    try:
        import smtplib
        from email.mime.multipart import MIMEMultipart
        from email.mime.text import MIMEText

        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = from_email
        msg["To"] = to_email
        msg.attach(MIMEText(html_body, "html"))

        with smtplib.SMTP(host, port) as server:
            server.starttls()
            server.login(user, password)
            server.sendmail(from_email, [to_email], msg.as_string())
        return True
    except Exception:
        return False


def send_weekly_digest(max_workers: int = 8) -> Dict[str, int]:
    """
    Send the weekly digest to all active subscribers in parallel.

    `max_workers` bounds concurrent SendGrid/SMTP sessions so we don't blow
    past provider rate limits. Each email is one HTTP/SMTP round-trip; running
    them sequentially blocked for ~N×latency_s, easily 10s+ for 1000 subscribers.

    Returns {'sent': N, 'failed': M, 'total_subscribers': T}.
    """
    subscribers = db.get_active_subscribers()
    top_ideas = db.get_top_shared_ideas(limit=10)

    if not top_ideas:
        return {"sent": 0, "failed": 0, "reason": "no_ideas"}

    html = build_weekly_digest_html(top_ideas)
    subject = f"IdeaGraph Weekly: Top {len(top_ideas)} Research Ideas"

    if not subscribers:
        return {"sent": 0, "failed": 0, "total_subscribers": 0}

    def _send_one(email: str) -> bool:
        # SendGrid first, fall back to SMTP. Both calls are wrapped in
        # try/except inside the helpers so a failure in one subscriber
        # never poisons the whole batch.
        if send_email_sendgrid(email, subject, html):
            return True
        return send_email_smtp(email, subject, html)

    from concurrent.futures import ThreadPoolExecutor

    sent = 0
    failed = 0
    # Cap workers at min(max_workers, len(subscribers)) so we never spin up
    # more threads than work items.
    n_workers = min(max_workers, len(subscribers))
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        for ok in ex.map(_send_one, (s["email"] for s in subscribers)):
            if ok:
                sent += 1
            else:
                failed += 1

    return {"sent": sent, "failed": failed, "total_subscribers": len(subscribers)}
