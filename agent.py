from typing import List, Dict, Any, Literal, Annotated, TypedDict
from langgraph.graph.message import add_messages
import concurrent.futures

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, ToolMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool

from tools import (
    extract_keywords,
    generate_filename_from_query,
    search_tool,
    read_tool,
    save_report_tool,
    list_files_tool,
    read_file_tool,
    calculate_tool,
    knowledge_search,
)
from config import settings, SYSTEM_PROMPT


# ---------------------------------------------------------
# Debug helper
# ---------------------------------------------------------
def debug_print(*args):
    if settings.debug:
        print("[DEBUG]", *args)


# ---------------------------------------------------------
# LangChain tools wrappers
# ---------------------------------------------------------
@tool
def search_tool_lc(query: str) -> str:
    """Виконує пошук у DuckDuckGo."""
    return search_tool(query)

@tool
def read_tool_lc(url: str) -> str:
    """Завантажує та витягує текст зі сторінки."""    
    return read_tool(url)

@tool
def save_report_tool_lc(filename: str, content: str) -> str:
    """Зберігає текст у файл."""
    return save_report_tool(filename, content)

@tool
def list_files_tool_lc(directory: str) -> str:
    """Показує файли в директорії."""
    return list_files_tool(directory)

@tool
def read_file_tool_lc(path: str) -> str:
    """Читає файл і повертає його вміст."""
    return read_file_tool(path)

@tool
def calculate_tool_lc(expression: str) -> str:
    """Обчислює математичний вираз."""
    return calculate_tool(expression)

@tool
def knowledge_search_lc(query: str) -> str:
    """Шукає в локальній базі знань (RAG)."""
    return knowledge_search(query)


TOOLS_LC = [
    search_tool_lc,
    read_tool_lc,
    save_report_tool_lc,
    list_files_tool_lc,
    read_file_tool_lc,
    calculate_tool_lc,
    knowledge_search_lc,
]

TOOLS_MAP = {
    "search_tool_lc": search_tool,
    "read_tool_lc": read_tool,
    "save_report_tool_lc": save_report_tool,
    "list_files_tool_lc": list_files_tool,
    "read_file_tool_lc": read_file_tool,
    "calculate_tool_lc": calculate_tool,
    "knowledge_search_lc": knowledge_search,
}


# ---------------------------------------------------------
# LLMs
# ---------------------------------------------------------
llm_tools = ChatOpenAI(
    model=settings.model_name,
    api_key=settings.openai_api_key.get_secret_value(),
    temperature=0.2,
).bind_tools(TOOLS_LC)

llm_plain = ChatOpenAI(
    model=settings.model_name,
    api_key=settings.openai_api_key.get_secret_value(),
    temperature=0.2,
)


# ---------------------------------------------------------
# Prompt
# ---------------------------------------------------------
prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("user", "{input}"),
])


# ---------------------------------------------------------
# State
# ---------------------------------------------------------
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    step_count: int


# ---------------------------------------------------------
# Agent node
# ---------------------------------------------------------
def agent_node(state: AgentState) -> AgentState:
    debug_print("\n=== AGENT NODE ===")

    if state.get("step_count", 0) >= settings.max_iterations:
        debug_print(f"⚠️ Зупиняю цикл: досягнуто ліміт {settings.max_iterations} кроків.")
        return {
            "messages": state["messages"] + [
                AIMessage(content=f"⚠️ Зупиняю цикл: досягнуто ліміт {settings.max_iterations} кроків.")
            ],
            "step_count": state.get("step_count", 0) + 1,
        }

    last_user = next(
        (msg.content for msg in reversed(state["messages"]) if isinstance(msg, HumanMessage)),
        "Опиши свій запит."
    )
    
    
    state_messages = list(state.get("messages", []))
    debug_print(f"State messages count: {len(state_messages)}")

    last_msg_types = [type(m).__name__ for m in state_messages[-3:]]
    debug_print(f"Last message types (up to 3): {last_msg_types}")

    last_msg_roles = [getattr(m, "type", type(m).__name__) for m in state_messages[-3:]]
    debug_print(f"Last message roles (up to 3): {last_msg_roles}")



    # Pass system prompt + full message history so the agent sees all tool results
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(state["messages"])
    response = llm_tools.invoke(messages)

    debug_print(f"Response type: {type(response)}")
    debug_print(f"Tool calls: {getattr(response, 'tool_calls', None)}")

    return {
        "messages": state["messages"] + [response],
        "step_count": state.get("step_count", 0) + 1,
    }


# ---------------------------------------------------------
# Conditional edge
# ---------------------------------------------------------
def should_continue(state: AgentState) -> Literal["tool", "summarizer"]:
    """Перевіряємо, потрібно ли виконувати інструменти."""
    last = state["messages"][-1]
    
    if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
        debug_print("→ Переходимо до TOOL NODE")
        return "tool"
    
    debug_print("→ Переходимо до SUMMARIZER NODE")
    return "summarizer"


# ---------------------------------------------------------
# Tool node
# ---------------------------------------------------------
def tool_node(state: AgentState) -> AgentState:
    debug_print("\n=== TOOL NODE ===")

    last = state["messages"][-1]
    if not isinstance(last, AIMessage) or not getattr(last, "tool_calls", None):
        return state

    new_messages = list(state["messages"])

    for call in last.tool_calls:
        tool_name = call["name"]
        args = call.get("args", {})
        tool_id = call.get("id")

        debug_print(f"Executing tool: {tool_name} with args: {args}")

        tool_fn = TOOLS_MAP.get(tool_name)
        if not tool_fn:
            new_messages.append(ToolMessage(
                tool_call_id=tool_id,
                content=f"❌ Невідомий інструмент: {tool_name}"
            ))
            continue

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(tool_fn, **args)
                result = future.result(timeout=settings.tool_timeout_sec)
        except concurrent.futures.TimeoutError:
            result = f"⏳ Timeout: інструмент не відповідає за {settings.tool_timeout_sec} секунд."
        except Exception as e:
            result = f"❌ Помилка інструменту: {str(e)}"

        new_messages.append(ToolMessage(
            tool_call_id=tool_id,
            content=str(result)
        ))
        debug_print(f"Tool result: {str(result)[:100]}...")

    return {
        "messages": new_messages,
        "step_count": state.get("step_count", 0),
    }


# ---------------------------------------------------------
# Summarizer node
# ---------------------------------------------------------
def summarizer_node(state: AgentState) -> AgentState:
    debug_print("\n=== SUMMARIZER NODE ===")

# Збираємо tool results лише поточного ходу (після останнього HumanMessage)
    last_human_idx = -1
    for i in range(len(state["messages"]) - 1, -1, -1):
        if isinstance(state["messages"][i], HumanMessage):
            last_human_idx = i
            break

    recent_msgs = (
        state["messages"][last_human_idx + 1:]
        if last_human_idx >= 0
        else state["messages"]
    )

    tool_results = [
        msg.content for msg in recent_msgs
        if isinstance(msg, ToolMessage)
    ]

    if not tool_results:
        debug_print("Немає результатів інструментів.")
        last_ai = next(
            (m for m in reversed(state["messages"]) if isinstance(m, AIMessage)),
            None,
        )
        if last_ai:
            return state
        return state
    
    debug_print(f"tool_results count: {len(tool_results)}")
    for i, tr in enumerate(tool_results, 1):
        preview = tr if len(tr) <= 400 else tr[:400] + "..."
        debug_print(f"\n--- tool_results[{i}] preview ---\n{preview}\n")

    last_user_question = next(
        (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        "",
    )

    summary_instruction = (
        "Сформуй фінальну відповідь мовою запиту користувача.\n"
        "Спочатку дай суть відповіді на запит користувача у 1-3 абзацах.\n"
        "Потім окремо додай розділ 'Джерела' у вигляді маркованого списку.\n"
        "Для knowledge_search: вказуй лише Source, page, Relevance (без Excerpt).\n"
        "Для web/search: вказуй URL або назву джерела, якщо вона є.\n"
        "Не вигадуй джерела або URL, яких немає у tool results.\n"
        "Do not claim lack of internet access."
    )
    
    summary_prompt = [
        HumanMessage(content=summary_instruction),
        HumanMessage(content=f"Запит користувача:\n{last_user_question}"),
        HumanMessage(content="Tool results:\n" + "\n\n".join(tool_results[:5])),
    ]
    
    try:
        final_answer_msg = llm_plain.invoke(summary_prompt)
        debug_print(f"Final answer: {final_answer_msg.content[:100]}...")
        
        # ✅ ВИПРАВЛЕНО: Правильно повертаємо новий стан з фінальною відповіддю
        new_messages = state["messages"] + [final_answer_msg]
        
        return {
            "messages": new_messages,
            "step_count": state.get("step_count", 0)+1,
        }
    except Exception as e:
        debug_print(f"Error in summarizer: {e}")
        error_msg = AIMessage(content=f"❌ Помилка при формуванні відповіді: {str(e)}")
        
        return {
            "messages": state["messages"] + [error_msg],
            "step_count": state.get("step_count", 0)+1,
        }


# ---------------------------------------------------------
# Save node - ВИПРАВЛЕНО: Повертаємо завжди стан
# ---------------------------------------------------------
def save_node(state: AgentState) -> AgentState:
    debug_print("\n=== SAVE NODE ===")

    last = state["messages"][-1]
    if not isinstance(last, AIMessage) or not last.content:
        return state  # ✅ Якщо немає що зберігати, повертаємо стан як є

    try:
        filename = generate_filename_from_query(last.content)
        result = save_report_tool(filename=filename, content=last.content)
        debug_print(f"Saved to: {result}")
    except Exception as e:
        debug_print(f"❌ Error saving file: {e}")

    # ✅ ВИПРАВЛЕНО: Завжди повертаємо стан (без змін)
    return state


# ---------------------------------------------------------
# Build Graph - ВИПРАВЛЕНО
# ---------------------------------------------------------
workflow = StateGraph(AgentState)

workflow.add_node("agent", agent_node)
workflow.add_node("tool", tool_node)
workflow.add_node("summarizer", summarizer_node)
workflow.add_node("save", save_node)

workflow.set_entry_point("agent")

# ✅ ВИПРАВЛЕНО: Правильний потік
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tool": "tool",
        "summarizer": "summarizer"
    }
)

workflow.add_edge("tool", "agent")  # ✅ ВИПРАВЛЕНО: cycle back so agent sees tool results, Changed workflow.add_edge("tool", "summarizer") → workflow.add_edge("tool", "agent"), The graph now properly implements the ReAct pattern: agent → tool → agent → … → summarizer → save → END
workflow.add_edge("summarizer", "save")
workflow.add_edge("save", END)  # ✅ save → END

memory = MemorySaver()
agent = workflow.compile(checkpointer=memory)
