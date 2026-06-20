"""
Personal Assistant using Supervisor Pattern

Features
--------
1. Calendar Agent
   - Schedule meetings
   - Check availability

2. Email Agent
   - Draft and send emails

3. Supervisor Agent
   - Routes requests to correct agent

4. Human Approval
   - Email approval required
   - Calendar approval required

Author: Ravi
"""

import os
from typing import List

from dotenv import load_dotenv

from langchain.tools import tool
from langchain.agents import create_agent

from langchain_huggingface import (
    HuggingFaceEndpoint,
    ChatHuggingFace,
)

# ============================================================
# Load Environment Variables
# ============================================================

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise ValueError(
        "HF_TOKEN not found in .env file"
    )

# ============================================================
# Hugging Face Model
# ============================================================

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    huggingfacehub_api_token=HF_TOKEN,
    temperature=0.2,
    max_new_tokens=512,
)

model = ChatHuggingFace(llm=llm)

# ============================================================
# Calendar Tools
# ============================================================

@tool
def get_available_time_slots(
    attendees: List[str],
    date: str,
    duration_minutes: int,
) -> str:
    """
    Check attendee availability.
    """

    slots = [
        "09:00",
        "11:00",
        "14:00",
        "16:00",
    ]

    return (
        f"Available slots on {date}: "
        f"{', '.join(slots)}"
    )


@tool
def create_calendar_event(
    title: str,
    start_time: str,
    end_time: str,
    attendees: List[str],
    location: str = "",
) -> str:
    """
    Create calendar event.
    """

    return (
        f"Event Created Successfully\n"
        f"Title: {title}\n"
        f"Start: {start_time}\n"
        f"End: {end_time}\n"
        f"Location: {location}\n"
        f"Attendees: {', '.join(attendees)}"
    )

# ============================================================
# Email Tool
# ============================================================

@tool
def send_email(
    to: List[str],
    subject: str,
    body: str,
) -> str:
    """
    Send email.
    Stub implementation.
    """

    return (
        f"Email Sent Successfully\n"
        f"To: {', '.join(to)}\n"
        f"Subject: {subject}"
    )

# ============================================================
# Human Approval
# ============================================================

def approve_action(action_name: str) -> bool:
    """
    Ask for user approval.
    """

    print("\n" + "=" * 60)
    print(f"APPROVAL REQUIRED: {action_name}")
    print("=" * 60)

    choice = input(
        "Approve? (y/n): "
    ).strip().lower()

    return choice == "y"

# ============================================================
# Calendar Agent
# ============================================================

calendar_agent = create_agent(
    model=model,
    tools=[
        get_available_time_slots,
        create_calendar_event,
    ],
    system_prompt="""
You are a Calendar Assistant.

Responsibilities:
- Check availability.
- Schedule meetings.
- Use available tools whenever needed.

Always provide concise responses.
"""
)

# ============================================================
# Email Agent
# ============================================================

email_agent = create_agent(
    model=model,
    tools=[send_email],
    system_prompt="""
You are a Professional Email Assistant.

Responsibilities:
- Draft professional emails.
- Generate good subject lines.
- Send emails using tools.

Always be concise and professional.
"""
)

# ============================================================
# Helper Function
# ============================================================

def extract_text(result) -> str:
    """
    Extract text from LangChain response.
    Compatible with multiple versions.
    """

    try:

        if isinstance(result, dict):

            messages = result.get("messages")

            if messages:

                last = messages[-1]

                if hasattr(last, "content"):
                    return str(last.content)

                return str(last)

        return str(result)

    except Exception:
        return str(result)

# ============================================================
# Calendar Workflow
# ============================================================

def schedule_event(request: str) -> str:

    if not approve_action(
        "Calendar Event Creation"
    ):
        return "Calendar action cancelled."

    result = calendar_agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": request,
                }
            ]
        }
    )

    return extract_text(result)

# ============================================================
# Email Workflow
# ============================================================

def manage_email(request: str) -> str:

    if not approve_action(
        "Send Email"
    ):
        return "Email action cancelled."

    result = email_agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": request,
                }
            ]
        }
    )

    return extract_text(result)

# ============================================================
# Supervisor
# ============================================================

def supervisor(query: str) -> str:
    """
    Route user requests.
    """

    query_lower = query.lower()

    email_keywords = [
        "email",
        "mail",
        "send",
        "message",
    ]

    calendar_keywords = [
        "meeting",
        "schedule",
        "calendar",
        "appointment",
        "availability",
    ]

    has_email = any(
        word in query_lower
        for word in email_keywords
    )

    has_calendar = any(
        word in query_lower
        for word in calendar_keywords
    )

    responses = []

    if has_calendar:
        responses.append(
            schedule_event(query)
        )

    if has_email:
        responses.append(
            manage_email(query)
        )

    if not responses:
        return (
            "I can currently help with:\n"
            "- Scheduling meetings\n"
            "- Checking availability\n"
            "- Drafting emails\n"
            "- Sending emails"
        )

    return "\n\n".join(responses)

# ============================================================
# Interactive CLI
# ============================================================

def run_assistant():

    print("\nPersonal Assistant Started")
    print("Type 'exit' to quit")

    while True:

        query = input("\nYou: ").strip()

        if query.lower() in (
            "exit",
            "quit",
        ):
            print("\nGoodbye!")
            break

        try:

            response = supervisor(query)

            print("\nAssistant:")
            print(response)

        except KeyboardInterrupt:

            print("\nInterrupted.")
            break

        except Exception as e:

            print(
                f"\nError occurred: {e}"
            )

# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    run_assistant()