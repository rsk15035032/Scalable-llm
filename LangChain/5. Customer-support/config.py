import os
from dotenv import load_dotenv

load_dotenv()

# ========================== LLM CONFIG ==========================
HUGGINGFACE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
#"microsoft/Phi-3-mini-4k-instruct"  # Fast & good for agents
# Alternative strong options:
# "Qwen/Qwen2.5-7B-Instruct"
# "mistralai/Mistral-7B-Instruct-v0.3"

TEMPERATURE = 0.2
MAX_TOKENS = 512

HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if not HF_TOKEN:
    raise ValueError("Please set HUGGINGFACEHUB_API_TOKEN in .env file")