"""
Chat with the ABC Grocery assistant in the terminal.
"""
import uuid

from dotenv import load_dotenv

from src.rag_chain import build_chain


def main():
    load_dotenv()
    chain = build_chain()

    # a new random session id each run, so every chat starts with empty memory
    session = {"configurable": {"session_id": str(uuid.uuid4())}}

    print("Hi, I'm the ABC Grocery virtual assistant. Type 'exit' to leave the chat.\n")
    try:
        while True:
            question = input("You: ").strip()
            if not question or question.lower() in {"exit", "quit"}:
                break

            response = chain.invoke({"input": question}, config=session)
            print("Assistant:", response.content.strip(), "\n")
    except KeyboardInterrupt:
        pass  # Ctrl+C exits cleanly instead of showing an error

    print("\nGoodbye!")


if __name__ == "__main__":
    main()