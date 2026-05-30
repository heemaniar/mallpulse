"""
root.py — thin entry point. All logic lives in agents/mallpulse/agent.py.

This file exists only for backward compatibility with any scripts that
import directly from agents/root.py. The canonical root agent is defined
in agents/mallpulse/agent.py.
"""

from agents.mallpulse.agent import root_agent  # noqa: F401 (re-export)
