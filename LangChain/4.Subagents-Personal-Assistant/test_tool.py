from dotenv import load_dotenv
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct"
)

model = ChatHuggingFace(llm=llm)

@tool
def hello(name: str) -> str:
    """Say hello"""
    return f"Hello {name}"

agent = create_agent(
    model=model,
    tools=[hello]
)

response = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Say hello to Ravi"
            }
        ]
    }
)

print(response)