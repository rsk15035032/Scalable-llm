from langchain.tools import tool, ToolRuntime
from langchain.messages import ToolMessage
from langgraph.types import Command
from typing import Literal

from state import SupportState


@tool
def record_warranty_status(
    status: Literal["in_warranty", "out_of_warranty"],
    runtime: ToolRuntime[None, SupportState],
) -> Command:
    """Record warranty status and move to next step."""
    return Command(
        update={
            "messages": [ToolMessage(
                content=f"Warranty status recorded: {status}",
                tool_call_id=runtime.tool_call_id
            )],
            "warranty_status": status,
            "current_step": "issue_classifier",
        }
    )


@tool
def record_issue_type(
    issue_type: Literal["hardware", "software"],
    runtime: ToolRuntime[None, SupportState],
) -> Command:
    """Record issue type and move to resolution."""
    return Command(
        update={
            "messages": [ToolMessage(
                content=f"Issue type recorded: {issue_type}",
                tool_call_id=runtime.tool_call_id
            )],
            "issue_type": issue_type,
            "current_step": "resolution_specialist",
        }
    )


@tool
def provide_solution(solution: str) -> str:
    """Deliver a solution to the customer."""
    return f"✅ Solution: {solution}"


@tool
def escalate_to_human(reason: str) -> str:
    """Escalate to human agent."""
    return f"🔄 Escalating to human support. Reason: {reason}"


@tool
def go_back_to_warranty() -> Command:
    """Allow customer to correct warranty information."""
    return Command(update={"current_step": "warranty_collector"})


@tool
def go_back_to_classification() -> Command:
    """Allow customer to correct issue type."""
    return Command(update={"current_step": "issue_classifier"})