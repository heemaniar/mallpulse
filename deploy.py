"""
deploy.py — Deploy the MallPulse agent to Vertex AI Agent Engine.

This is the "Google Cloud Agent Builder" deployment step.
Run once when you're ready to go live (Day 8+).

Usage:
    source .venv/bin/activate
    python deploy.py

After deployment, the script prints the resource name:
    projects/.../locations/.../reasoningEngines/...
Save that — it's what Streamlit uses to call the agent.

To test the deployed agent:
    python deploy.py --test "What are the top tenants at Kanyon?"

To delete a deployed agent:
    python deploy.py --delete <resource_name>
"""

import argparse
import vertexai
from vertexai import agent_engines
from vertexai.preview.reasoning_engines import AdkApp

PROJECT  = "mallpulse-hackathon"
LOCATION = "us-central1"
DISPLAY_NAME = "MallPulse"

REQUIREMENTS = [
    "google-adk==2.0.0",
    "google-cloud-bigquery>=3.0.0",
    "google-cloud-aiplatform[adk,agent_engines]>=1.88.0",
]


def deploy():
    """Deploy agent to Vertex AI Agent Engine (Google Cloud Agent Builder)."""
    # Import here so local adk run doesn't need vertexai installed
    from agents.root import root_agent

    print(f"Deploying '{DISPLAY_NAME}' to Vertex AI Agent Engine...")
    print(f"  Project : {PROJECT}")
    print(f"  Location: {LOCATION}")
    print()

    vertexai.init(project=PROJECT, location=LOCATION)

    app = AdkApp(agent=root_agent, enable_tracing=False)

    remote_app = agent_engines.create(
        app,
        requirements=REQUIREMENTS,
        display_name=DISPLAY_NAME,
        description=(
            "MallPulse — AI assistant for shopping mall General Managers. "
            "Powered by Google ADK + Gemini 2.5 Flash on Vertex AI Agent Engine."
        ),
    )

    print(f"\nDeployed successfully!")
    print(f"Resource name: {remote_app.resource_name}")
    print()
    print("Add this to your .env:")
    print(f"  AGENT_ENGINE_RESOURCE={remote_app.resource_name}")

    return remote_app.resource_name


def test_agent(resource_name: str, message: str):
    """Send a test message to the deployed agent."""
    vertexai.init(project=PROJECT, location=LOCATION)
    remote_app = agent_engines.get(resource_name)

    print(f"Sending: {message!r}")
    print()
    session = remote_app.create_session(user_id="test-user")
    for event in remote_app.stream_query(
        user_id="test-user",
        session_id=session["id"],
        message=message,
    ):
        if "content" in event and "parts" in event["content"]:
            for part in event["content"]["parts"]:
                if "text" in part:
                    print(part["text"], end="", flush=True)
    print()


def delete_agent(resource_name: str):
    """Delete a deployed agent engine instance."""
    vertexai.init(project=PROJECT, location=LOCATION)
    remote_app = agent_engines.get(resource_name)
    remote_app.delete(force=True)
    print(f"Deleted: {resource_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MallPulse Agent Engine deployment")
    parser.add_argument("--test",   metavar="MESSAGE",       help="Test a deployed agent")
    parser.add_argument("--delete", metavar="RESOURCE_NAME", help="Delete a deployed agent")
    args = parser.parse_args()

    if args.test:
        resource = input("Resource name: ").strip()
        test_agent(resource, args.test)
    elif args.delete:
        delete_agent(args.delete)
    else:
        deploy()
