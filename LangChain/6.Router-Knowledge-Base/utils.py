from langchain.chat_models import init_chat_model
from config import HUGGINGFACE_MODEL, TEMPERATURE, MAX_TOKENS, HF_TOKEN


def create_chat_model():
    """Initialize Hugging Face chat model."""
    return init_chat_model(
        HUGGINGFACE_MODEL,
        model_provider="huggingface",
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        token=HF_TOKEN,  # Explicit token passing
    )