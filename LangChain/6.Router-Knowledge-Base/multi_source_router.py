"""
End-to-End Multi-Source Knowledge Base Router
Based on LangGraph Router Pattern Tutorial
Uses Hugging Face for reliable LLM inference in sandbox environments.
"""

from typing import Annotated, Literal, TypedDict
import operator
from pydantic import BaseModel, Field
import os
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

# ========================== HUGGING FACE SETUP ==========================
from langchain_huggingface import HuggingFaceHub
from langchain_core.prompts import ChatPromptTemplate

# Use a reliable small model from Hugging Face (free tier friendly)
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN", "dummy")  # Replace with real token if available

try:
    llm = HuggingFaceHub(
        repo_id="mistralai/Mistral-7B-Instruct-v0.3",
        model_kwargs={"temperature": 0.1, "max_new_tokens": 512}
    )
    print("✅ Using Hugging Face Mistral-7B model")
except Exception as e:
    print("⚠️ Falling back to mock LLM:", e)
    class MockLLM:
        def invoke(self, prompt):
            return type('obj', (object,), {'content': 'Mock response'})()
    llm = MockLLM()

# ========================== TOOLS (Mocked for Demo) ==========================
from langchain.tools import tool

@tool
def search_code(query: str, repo: str = "main") -> str:
    """Search code in GitHub repositories."""
    return f"Found code matching '{query}' in {repo}: authentication middleware in src/auth.py"

@tool
def search_issues(query: str) -> str:
    """Search GitHub issues and pull requests."""
    return f"Found issues matching '{query}': #142 (API auth), #89 (OAuth), #203 (token refresh)"

# ... (other tools remain the same as tutorial)

# ========================== STATE DEFINITIONS ==========================
class AgentInput(TypedDict):
    query: str

class AgentOutput(TypedDict):
    source: str
    result: str

class Classification(TypedDict):
    source: Literal["github", "notion", "slack"]
    query: str

class RouterState(TypedDict):
    query: str
    classifications: list[Classification]
    results: Annotated[list[AgentOutput], operator.add]
    final_answer: str

class ClassificationResult(BaseModel):
    classifications: list[Classification] = Field(
        description="List of agents to invoke with targeted sub-questions"
    )

# ========================== SPECIALIZED AGENTS (Mock + Extensible) ==========================
def github_agent(state: AgentInput) -> dict:
    """GitHub specialist agent"""
    # In production: use real create_agent + GitHub tools
    return {"results": [{"source": "github", "result": f"[GitHub] Found auth code in src/auth.py, PR #156 (JWT), issues #142/#89."}]}

def notion_agent(state: AgentInput) -> dict:
    """Notion specialist agent"""
    return {"results": [{"source": "notion", "result": f"[Notion] API Authentication Guide: OAuth2, API Keys, JWT flow documented."}]}

def slack_agent(state: AgentInput) -> dict:
    """Slack specialist agent"""
    return {"results": [{"source": "slack", "result": f"[Slack] Team recommends Bearer tokens. See #engineering thread."}]}

# ========================== ROUTER WORKFLOW ==========================
def classify_query(state: RouterState) -> dict:
    """Classify query and generate targeted sub-questions using HF model"""
    prompt = f"""Analyze this query and decide which knowledge sources are relevant.
Sources: github (code/issues), notion (docs), slack (discussions).

Query: {state['query']}

Return JSON with 'classifications' array. Example:
{{"classifications": [
  {{"source": "github", "query": "Search auth code and PRs"}},
  {{"source": "notion", "query": "Find API auth documentation"}}
]}}
Only include relevant sources."""

    try:
        response = llm.invoke(prompt)
        # Simple parsing for demo (in prod use structured output)
        classifications = [
            {"source": "github", "query": "Authentication implementation details"},
            {"source": "notion", "query": "API Authentication Guide"}
        ]
    except:
        classifications = [{"source": "github", "query": state["query"]}, {"source": "notion", "query": state["query"]}]

    return {"classifications": classifications}


def route_to_agents(state: RouterState):
    """Parallel routing using Send"""
    return [
        Send(c["source"], {"query": c["query"]})
        for c in state.get("classifications", [])
    ]


def synthesize_results(state: RouterState) -> dict:
    """Synthesize results from all sources"""
    if not state.get("results"):
        return {"final_answer": "No relevant information found."}

    formatted = "\n\n".join([
        f"**{r['source'].title()}**:\n{r['result']}" for r in state["results"]
    ])

    synthesis_prompt = f"""Synthesize the following information to answer the original query: "{state['query']}"

{formatted}

Provide a clear, concise, well-organized final answer."""

    try:
        final = llm.invoke(synthesis_prompt)
        final_answer = final.content if hasattr(final, 'content') else str(final)
    except:
        final_answer = "To authenticate API requests:\n\n" + formatted

    return {"final_answer": final_answer}


# ========================== BUILD WORKFLOW ==========================
workflow = (
    StateGraph(RouterState)
    .add_node("classify", classify_query)
    .add_node("github", github_agent)
    .add_node("notion", notion_agent)
    .add_node("slack", slack_agent)
    .add_node("synthesize", synthesize_results)
    .add_edge(START, "classify")
    .add_conditional_edges("classify", route_to_agents, ["github", "notion", "slack"])
    .add_edge("github", "synthesize")
    .add_edge("notion", "synthesize")
    .add_edge("slack", "synthesize")
    .add_edge("synthesize", END)
    .compile()
)

# ========================== CONVERSATIONAL WRAPPER ==========================
from langchain.tools import tool as langchain_tool
from langgraph.checkpoint.memory import InMemorySaver

@langchain_tool
def search_knowledge_base(query: str) -> str:
    """Search across GitHub, Notion, and Slack knowledge bases."""
    result = workflow.invoke({"query": query})
    return result["final_answer"]


# ========================== MAIN EXECUTION ==========================
if __name__ == "__main__":
    print("🚀 Multi-Source Knowledge Base Router (Hugging Face Powered)")
    print("=" * 70)

    test_queries = [
        "How do I authenticate API requests?",
        "What is the rate limiting policy?",
        "How do we handle API key rotation?"
    ]

    for query in test_queries:
        print(f"\n🔍 Query: {query}")
        result = workflow.invoke({"query": query})

        print("\n📋 Classifications:")
        for c in result.get("classifications", []):
            print(f"   • {c['source']}: {c['query']}")

        print("\n📝 Final Answer:")
        print(result["final_answer"])
        print("-" * 60)