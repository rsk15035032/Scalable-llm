from langchain.messages import HumanMessage
from langchain_core.utils.uuid import uuid7
from agent import create_support_agent


def run_demo():
    agent = create_support_agent()
    thread_id = str(uuid7())
    config = {"configurable": {"thread_id": thread_id}}

    print("🤖 Customer Support Agent (Hugging Face + State Machine)")
    print("Type 'exit' to end conversation.\n")

    messages = []

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("👋 Goodbye!")
            break

        messages.append(HumanMessage(user_input))

        result = agent.invoke({"messages": messages}, config)

        # Update history
        new_messages = result.get("messages", [])
        messages.extend(new_messages)

        for msg in new_messages:
            if msg.content:
                print(f"Agent: {msg.content}")

        print(f"\n[Step: {result.get('current_step')}]")


if __name__ == "__main__":
    run_demo()