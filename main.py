from agent import agent
from langchain_core.messages import HumanMessage
from config import settings
import threading
import logging
import time
from pathlib import Path
from uuid import uuid4
from ingest import ingest
from retriever import get_retriever

logger = logging.getLogger(__name__)
SESSION_ID = f"local-{uuid4().hex[:8]}"

def warmup_rag() -> None:
    """Preload retriever and reranker model before first user query."""
    try:
        print("Warming up RAG components (first run may take a few minutes)...")
        retriever = get_retriever() # loads FAISS + BM25 + CrossEncoder
        _ = retriever.search("warmup query")
        print("RAG warm-up completed.")
    except Exception as e:
    # Non-fatal: app can still run, and model may load on first knowledge_search call
        logger.warning("RAG warm-up skipped due to error: %s", e)

def ensure_knowledge_index() -> None:
    index_ok = Path(settings.index_dir).exists()
    chunks_ok = Path(settings.chunks_path).exists()
    if index_ok and chunks_ok:
        return

    print("Knowledge index not found. Running ingestion...")
    ingest()
    print("Knowledge index ready.")

def run_agent(user_input, output_container, thread_id, debug=False):
    try:
        result = agent.invoke(
            {
                "messages": [HumanMessage(content=user_input)],
                "step_count": 0,
            },
            config={"configurable": {"thread_id": thread_id}},
        )

        messages = result.get("messages", [])
        if messages:
            last = messages[-1]
            if hasattr(last, "content") and last.content:
                output_container.append(last.content)
            else:
                output_container.append("⚠️ Агент повернув порожню відповідь.")
        else:
            output_container.append("⚠️ Немає повідомлень у відповіді.")

    except Exception as e:
        error_msg = f"Помилка агента: {str(e)}"
        logger.exception(error_msg)
        output_container.append(error_msg)

def main():
    print("Research Agent (type 'exit' to quit)")
    print("Debug mode:", "ON" if settings.debug else "OFF")
    print(
        f"Limits: max_iterations={settings.max_iterations}, "
        f"tool_timeout={settings.tool_timeout_sec}s, "
        f"wait={settings.agent_first_wait_sec}+{settings.agent_second_wait_sec}s, "
        f"thread_id={SESSION_ID}"
    )
    print("-" * 40)
    ensure_knowledge_index()
    warmup_rag()

    while True:
        try:
            user_input = input("\nYou: ").strip()

            if not user_input:
                print("⚠️ Будь ласка, введіть запит.")
                continue

            if user_input.lower() in ("exit", "quit"):
                print("Goodbye!")
                break

            if user_input.lower() == "debug on":
                settings.debug = True
                print("✅ Debug mode: ON")
                continue

            if user_input.lower() == "debug off":
                settings.debug = False
                print("✅ Debug mode: OFF")
                continue

            if user_input.lower() == "/ingest":
                print("Rebuilding knowledge index...")
                ingest()
                print("Done.")
                continue

            output = []
            # Daemon потік з більшим timeout
            t = threading.Thread(
                target=run_agent, 
                args=(user_input, output, SESSION_ID, settings.debug),
                daemon=True
            )
            t.start()
            t.join(timeout=settings.agent_first_wait_sec)

            if t.is_alive():
                print("⏳ Агент все ще обробляє запит...")
                t.join(timeout=settings.agent_second_wait_sec)
                
                if t.is_alive():
                    print("⏳ Агент завис або не відповідає. Можливо, проблема з інтернетом або інструментами.")
                    print("Спробуй інше формулювання або перевір з'єднання.")
                    continue

            # ✅ ВИПРАВЛЕНО: Перевіряємо результат ПІСЛЯ того, як потік завершився
            if output:
                print(f"Agent: {output[-1]}\n")
            else:
                print("⚠️ Агент не повернув відповідь.\n")

        except KeyboardInterrupt:
            print("\n\nПрограма завершена користувачем.")
            break
        except Exception as e:
            logger.exception("Неочікувана помилка в main loop")
            print(f"❌ Неочікувана помилка: {e}")

if __name__ == "__main__":
    main()
