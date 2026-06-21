from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.middleware import SummarizationMiddleware
from utils import create_chat_model

from config import HUGGINGFACE_MODEL, TEMPERATURE, MAX_TOKENS
from state import SupportState
from tools import (
    record_warranty_status,
    record_issue_type,
    provide_solution,
    escalate_to_human,
    go_back_to_warranty,
    go_back_to_classification,
)
from middleware import apply_step_config


def create_support_agent():
    """Create and return the configured customer support agent."""

    model = create_chat_model()  # Defined in utils or inline

    all_tools = [
        record_warranty_status,
        record_issue_type,
        provide_solution,
        escalate_to_human,
        go_back_to_warranty,
        go_back_to_classification,
    ]

    agent = create_agent(
        model=HUGGINGFACE_MODEL,
        tools=all_tools,
        state_schema=SupportState,
        middleware=[
            apply_step_config,
            SummarizationMiddleware(
                model=model,
                trigger=("tokens", 5000),
                keep=("messages", 10),
            ),
        ],
        checkpointer=InMemorySaver(),
    )
    return agent