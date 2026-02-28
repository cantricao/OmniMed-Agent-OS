"""
main_workflow.py - Placeholder for the OmniMed-Agent-OS LangGraph state machine.
"""

from typing import TypedDict


class AgentState(TypedDict):
    """Shared state passed between nodes in the LangGraph workflow."""

    query: str
    context: str
    response: str


def build_graph():
    """Build and return the LangGraph state machine for OmniMed-Agent-OS."""
    pass


def run_workflow(query: str) -> str:
    """Run the main agent workflow for a given user query."""
    pass


if __name__ == "__main__":
    run_workflow("Bệnh nhân có triệu chứng sốt cao và đau đầu.")
