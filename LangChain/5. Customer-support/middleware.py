from typing import Callable
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse

from prompts import (
    WARRANTY_COLLECTOR_PROMPT,
    ISSUE_CLASSIFIER_PROMPT,
    RESOLUTION_SPECIALIST_PROMPT,
)

STEP_CONFIG = {
    "warranty_collector": {
        "prompt": WARRANTY_COLLECTOR_PROMPT,
        "tools": ["record_warranty_status"],
        "requires": [],
    },
    "issue_classifier": {
        "prompt": ISSUE_CLASSIFIER_PROMPT,
        "tools": ["record_issue_type"],
        "requires": ["warranty_status"],
    },
    "resolution_specialist": {
        "prompt": RESOLUTION_SPECIALIST_PROMPT,
        "tools": ["provide_solution", "escalate_to_human",
                  "go_back_to_warranty", "go_back_to_classification"],
        "requires": ["warranty_status", "issue_type"],
    },
}


@wrap_model_call
def apply_step_config(
    request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    """Core middleware for the state machine pattern."""
    current_step = request.state.get("current_step", "warranty_collector")

    if current_step not in STEP_CONFIG:
        raise ValueError(f"Unknown step: {current_step}")

    config = STEP_CONFIG[current_step]

    # Validate required state
    for key in config["requires"]:
        if request.state.get(key) is None:
            raise ValueError(f"Missing required state '{key}' for step '{current_step}'")

    # Format prompt with state
    try:
        system_prompt = config["prompt"].format(**request.state)
    except KeyError:
        system_prompt = config["prompt"]

    # Dynamically configure agent
    request = request.override(
        system_prompt=system_prompt,
        tools=config["tools"],   # Tool names (LangChain resolves them)
    )

    return handler(request)