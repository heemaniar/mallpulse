"""
tools/fivetran_tools.py — Fivetran REST API tools for the Data Unifier agent.

Wraps Fivetran's v1 REST API so the agent can ask about pipeline health,
data freshness, and trigger syncs — all without touching the MCP server directly.

Auth: Basic auth encoded from FIVETRAN_API_KEY + FIVETRAN_API_SECRET in .env.

Trial expiry (June 9 2026): every function returns a friendly message on 401/403
so the app never crashes during the demo if the trial lapses.
"""

import base64
import os
from datetime import datetime

import requests

FIVETRAN_API_BASE = "https://api.fivetran.com/v1"

# Update this if Fivetran grants an extension
_TRIAL_EXPIRY = "2026-06-09"


# ── auth ──────────────────────────────────────────────────────────────────────

def _auth_headers() -> dict | None:
    """Return Basic-auth headers, or None if credentials are missing."""
    api_key = os.getenv("FIVETRAN_API_KEY", "").strip()
    api_secret = os.getenv("FIVETRAN_API_SECRET", "").strip()
    if not api_key or not api_secret:
        return None
    token = base64.b64encode(f"{api_key}:{api_secret}".encode()).decode()
    return {"Authorization": f"Basic {token}", "Content-Type": "application/json"}


def _missing_creds_msg() -> str:
    return (
        "⚠️ Fivetran API key not configured. "
        "Add FIVETRAN_API_KEY and FIVETRAN_API_SECRET to your .env file. "
        "(Get them from Fivetran → Settings → API Configuration.)"
    )


def _handle_response(resp: requests.Response, action: str) -> dict | str:
    """Parse an API response; return a user-friendly string on any error."""
    if resp.status_code == 401:
        return (
            f"⚠️ Fivetran credentials are invalid or expired (trial expires {_TRIAL_EXPIRY}). "
            "Check your API key in Fivetran → Settings → API Configuration, "
            "or email partnerships@fivetran.com to request a trial extension."
        )
    if resp.status_code == 403:
        return (
            f"⚠️ Fivetran access denied. Your trial may have lapsed "
            f"(expected expiry: {_TRIAL_EXPIRY}). "
            "Contact partnerships@fivetran.com for an extension through July 6."
        )
    if resp.status_code == 404:
        return f"⚠️ Fivetran resource not found while {action}. Check the connector ID."
    if resp.status_code == 429:
        return "⚠️ Fivetran rate limit reached. Wait 60 seconds and try again."
    if not resp.ok:
        return f"⚠️ Fivetran API error ({resp.status_code}) while {action}: {resp.text[:300]}"
    return resp.json()


def _fmt_timestamp(ts: str | None) -> str:
    """Format an ISO timestamp to readable UTC, or return 'Never'."""
    if not ts:
        return "Never"
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d %H:%M UTC")
    except Exception:
        return ts


# ── tools ─────────────────────────────────────────────────────────────────────

def list_connectors() -> str:
    """List all Fivetran connectors with their current sync status and last sync time.

    Returns a markdown table showing each connector's name, service type,
    current sync state, and when data last landed successfully in BigQuery.
    Use this to check overall pipeline health before answering data-freshness questions.

    Returns:
        Markdown table of connectors with status, or an error message if
        Fivetran credentials are missing or expired.
    """
    headers = _auth_headers()
    if headers is None:
        return _missing_creds_msg()

    try:
        resp = requests.get(
            f"{FIVETRAN_API_BASE}/connectors",
            headers=headers,
            timeout=15,
        )
    except requests.RequestException as e:
        return (
            f"⚠️ Cannot reach Fivetran API: {e}. "
            "Historical data in BigQuery (up to last successful sync) is still available."
        )

    result = _handle_response(resp, "listing connectors")
    if isinstance(result, str):
        return result

    items = result.get("data", {}).get("items", [])
    if not items:
        return (
            "No Fivetran connectors found in this account. "
            "Set one up at fivetran.com → Connectors → + Add connector."
        )

    lines = ["**Fivetran Pipeline — Connector Status**\n"]
    lines.append("| Connector | Service | State | Last Successful Sync |")
    lines.append("|---|---|---|---|")

    for c in items:
        name = c.get("schema") or c.get("id", "unknown")
        service = c.get("service", "—")
        sync_state = c.get("status", {}).get("sync_state", "unknown")
        last_ok = _fmt_timestamp(c.get("succeeded_at"))

        if sync_state in ("synced", "rescheduled"):
            icon = "✅"
        elif sync_state == "broken":
            icon = "🔴"
        elif sync_state == "paused":
            icon = "⏸️"
        else:
            icon = "🟡"

        lines.append(f"| {name} | {service} | {icon} {sync_state} | {last_ok} |")

    return "\n".join(lines)


def get_connector_status(connector_id: str) -> str:
    """Get detailed health and sync information for a specific Fivetran connector.

    Args:
        connector_id: The Fivetran connector ID. Find it via list_connectors(),
                      or in the Fivetran UI connector URL.

    Returns:
        Connector details: sync state, last success/failure timestamps, active
        tasks, and any warnings. Returns a friendly message if credentials are
        missing or expired.
    """
    headers = _auth_headers()
    if headers is None:
        return _missing_creds_msg()

    try:
        resp = requests.get(
            f"{FIVETRAN_API_BASE}/connectors/{connector_id}",
            headers=headers,
            timeout=15,
        )
    except requests.RequestException as e:
        return f"⚠️ Cannot reach Fivetran API: {e}"

    result = _handle_response(resp, f"fetching connector {connector_id}")
    if isinstance(result, str):
        return result

    data = result.get("data", {})
    status = data.get("status", {})

    sync_state = status.get("sync_state", "—")
    if sync_state in ("synced", "rescheduled"):
        health_icon = "✅ Healthy"
    elif sync_state == "broken":
        health_icon = "🔴 Broken"
    elif sync_state == "paused":
        health_icon = "⏸️ Paused"
    else:
        health_icon = f"🟡 {sync_state}"

    lines = [f"**Connector: {data.get('schema', connector_id)}**\n"]
    lines.append(f"- **Health**: {health_icon}")
    lines.append(f"- **Service**: {data.get('service', '—')}")
    lines.append(f"- **Paused**: {data.get('paused', False)}")
    lines.append(f"- **Sync frequency**: every {data.get('sync_frequency', '?')} minutes")
    lines.append(f"- **Last successful sync**: {_fmt_timestamp(data.get('succeeded_at'))}")
    lines.append(f"- **Last failed sync**: {_fmt_timestamp(data.get('failed_at')) or 'None'}")

    tasks = status.get("tasks", [])
    if tasks:
        lines.append("\n**Active tasks:**")
        for t in tasks[:5]:
            lines.append(f"  - {t.get('message', str(t))}")

    warnings = status.get("warnings", [])
    if warnings:
        lines.append("\n**Warnings:**")
        for w in warnings[:5]:
            lines.append(f"  - ⚠️ {w.get('message', str(w))}")

    return "\n".join(lines)


def trigger_sync(connector_id: str) -> str:
    """Trigger an immediate sync for a Fivetran connector.

    Use this when the GM reports stale data, or after manually adding rows
    to Cloud SQL Postgres and wanting them in BigQuery right away.

    Args:
        connector_id: The Fivetran connector ID (from list_connectors() or the UI URL).

    Returns:
        Confirmation that the sync was triggered, or an error message.
        Note: sync typically takes 5–30 minutes to complete for the MallPulse dataset.
    """
    headers = _auth_headers()
    if headers is None:
        return _missing_creds_msg()

    try:
        resp = requests.post(
            f"{FIVETRAN_API_BASE}/connectors/{connector_id}/sync",
            headers=headers,
            json={"force": True},
            timeout=15,
        )
    except requests.RequestException as e:
        return f"⚠️ Cannot reach Fivetran API: {e}"

    result = _handle_response(resp, f"triggering sync on {connector_id}")
    if isinstance(result, str):
        return result

    return (
        f"✅ Sync triggered for connector `{connector_id}`. "
        "Allow 5–30 minutes for Cloud SQL → BigQuery data to land. "
        "Use get_connector_status() to check progress."
    )


def get_pipeline_health_summary() -> str:
    """Return a plain-English summary of the Fivetran pipeline health.

    Checks all connectors and returns a one-paragraph status the agent can
    relay directly to the GM without further processing.

    Returns:
        A natural-language summary of pipeline health and data freshness.
        If Fivetran is unreachable, explains that historical BigQuery data
        is still available — so the GM knows the agent is still useful.
    """
    detail = list_connectors()

    if detail.startswith("⚠️"):
        return (
            f"The Fivetran pipeline status could not be retrieved. {detail} "
            "All data loaded into BigQuery before the connectivity issue "
            "remains fully available for analysis — revenue figures, tenant "
            "performance, and forecasts are unaffected."
        )

    return detail
