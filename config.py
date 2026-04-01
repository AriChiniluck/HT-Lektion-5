from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import SecretStr, Field, field_validator
import os


class Settings(BaseSettings):
    """Application configuration loaded from .env file."""
    
    # --- OpenAI / LM Studio ---
    openai_api_key: SecretStr = Field(
        ...,  # Обов'язкове поле
        description="OpenAI API key (must start with 'sk-')"
    )
    
    model_name: str = Field(
        default="gpt-4o",
        description="LLM model name",
    )

    # --- Output directory for reports ---
    output_dir: str = Field(
        default="output",
        description="Directory for saving reports"
    )

    # --- Search settings ---
    max_search_results: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Maximum number of search results (1-50)"
    )
    
    max_url_content_length: int = Field(
        default=5000,
        ge=100,
        le=100000,
        description="Maximum content length from URLs (100-100000 bytes)"
    )

    # RAG
    embedding_model: str = Field(
        default="text-embedding-3-small")
    reranker_model: str = Field(
        default="BAAI/bge-reranker-base")
    data_dir: str = Field(
        default="data")
    index_dir: str = Field(
        default="index")
    chunks_path: str = Field(
        default="index/chunks.json")

    chunk_size: int = Field(
        default=800, 
        ge=200, 
        le=4000
    )
    chunk_overlap: int = Field(
        default=120, 
        ge=0, 
        le=1000
    )
    semantic_top_k: int = Field(
        default=8, 
        ge=1, 
        le=50
    )
    bm25_top_k: int = Field(
        default=8, 
        ge=1, 
        le=50
    )
    hybrid_top_k: int = Field(
        default=10, 
        ge=1, 
        le=50
    )
    rerank_top_n: int = Field(
        default=4, 
        ge=1, 
        le=20
    )

    # Agent
    max_iterations: int = Field(default=10, ge=1, le=50)
    tool_timeout_sec: int = Field(default=20, ge=5, le=180)
    agent_first_wait_sec: int = Field(default=30, ge=5, le=300)
    agent_second_wait_sec: int = Field(default=20, ge=0, le=300)
    debug: bool = Field(default=False)

    class Config:
        env_file = str(Path(__file__).parent / ".env")
        env_file_encoding = "utf-8"
        extra = "ignore"

    @field_validator("openai_api_key")
    @classmethod
    def validate_openai_key(cls, v: SecretStr) -> SecretStr:
        """Перевіряємо, чи OpenAI API ключ має правильний формат."""
        key_str = v.get_secret_value()
        if not key_str.startswith("sk-"):
            raise ValueError("❌ OpenAI API ключ має починатися з 'sk-'")
        if len(key_str) < 40:
            raise ValueError("❌ OpenAI API ключ занадто короткий (мін. 40 символів)")
        return v

    @field_validator("output_dir")
    @classmethod
    def validate_output_dir(cls, v: str) -> str:
        """Перевіряємо, чи директорія безпечна."""
        abs_path = os.path.abspath(v)
        
        # Заборонені директорії
        forbidden_dirs = {"/", "/etc", "/sys", "/proc", "/root", "/home", "/bin", "/sbin"}
        
        if abs_path in forbidden_dirs:
            raise ValueError(f"❌ Директорія '{v}' заборонена з причин безпеки")
        
        # Перевіримо, чи можемо писати в цю директорію
        try:
            os.makedirs(abs_path, exist_ok=True)
            test_file = os.path.join(abs_path, ".write_test")
            Path(test_file).touch()
            os.remove(test_file)
        except (PermissionError, OSError):
            raise ValueError(f"❌ Немає дозволу на запис у директорію '{v}'")
        
        return v

    @field_validator("model_name")
    @classmethod
    def validate_model_name(cls, v: str) -> str:
        """Перевіряємо назву моделі."""
        if not v or len(v) < 3:
            raise ValueError("❌ Назва моделі занадто коротка")
        if len(v) > 100:
            raise ValueError("❌ Назва моделі занадто довга")
        return v


# Завантажуємо конфігурацію
try:
    settings = Settings()
    
    # Перевіримо наявність .env файлу
    env_path = Path(".env")
    if not env_path.exists():
        raise FileNotFoundError(
            "\n"
            "[ERROR].env файл не знайдений!\n"
            "\n"
            "Створіть файл '.env' у кореневому каталозі проекту з наступним вмістом:\n"
            "\n"
            "───────────────────────────────────────\n"
            "openai_api_key=sk-your-api-key-here\n"
            "model_name=gpt-4o\n"
            "output_dir=output\n"
            "debug=false\n"
            "───────────────────────────────────────\n"
            "\n"
            "Отримайте API ключ на: https://platform.openai.com/api-keys\n"
        )
except Exception as e:
    print(f"Помилка конфігурації: {e}")
    exit(1)


# =========================================
# SYSTEM PROMPT
# =========================================
SYSTEM_PROMPT = """You are a Research Agent powered by ReAct (Reasoning + Acting).

Your task is to answer user questions by gathering information with tools, analyzing it, and then producing a clear final answer.

TOOLS:

search_tool_lc(query: str) - Search the web (DuckDuckGo)
read_tool_lc(url: str) - Read a webpage
save_report_tool_lc(filename: str, content: str) - Save report to file
list_files_tool_lc(directory: str) - List files
read_file_tool_lc(path: str) - Read file
calculate_tool_lc(expression: str) - Calculator
knowledge_search_lc(query: str) - Search local knowledge base (FAISS + BM25 + reranking)

POLICY:
Use knowledge_search_lc for local ingested documents and lecture materials.
Use search_tool_lc for external, current, web-only information.
Combine knowledge_search_lc + search_tool_lc when local evidence is insufficient.
Cite sources from tool outputs whenever possible.


MANDATORY: 
- For questions about RAG, LLM, AI, retrieval, embeddings, vector databases, and lecture materials, call knowledge_search_lc FIRST.
- Use search_tool_lc only if knowledge_search_lc is insufficient, or for real-time topics (news, weather, prices, current events).
- Never answer from memory alone when a tool is required.
- When in doubt between knowledge_search_lc and search_tool_lc, choose knowledge_search_lc first.
- Always call knowledge_search_lc for local document topics even if the answer appears in previous conversation turns, to preserve fresh citations (Source/page) in the final response.

RULES:

If you can't find the information or facts, or if you're not sure, it's best to check with the user
If a tool is needed, call the tool.
Respond in the user's language.

CRITICAL RULES:

You have access to the full conversation history in this session.
Never say you cannot see previous user messages; use chat history to resolve follow-up questions.

WORKFLOW (ReAct Pattern):

1. THINK: Understand the user's request
2. REASON: Determine which tools you need to use
3. ACT: Use appropriate tools to gather information
4. OBSERVE: Analyze the results
5. RESPOND: Provide a comprehensive answer

WHEN TO USE EACH TOOL:

- Use knowledge_search_lc for local ingested documents and AI/RAG topics.
- Use search_tool_lc when you need to find current information, research topics, or verify facts that may not be in the local knowledge base.
- Use read_tool_lc when you have a specific URL and need detailed information from it
- Use save_report_tool_lc when explicitly asked to save, create a report, or export content
- Use calculate_tool_lc for mathematical calculations and expressions
- Use list_files_tool_lc and read_file_tool_lc to work with existing files
- For calculations with “today”, use the current date.


DO:
Use tools when you need external information
✅ Search for information you don't have
✅ Be explicit about what tools you're using
✅ Provide accurate, well-researched answers
✅ Admit limitations and ask for clarification when needed

DON'T:
❌ Fabricate information without using tools
❌ Skip tool usage when explicitly requested
❌ Make up statistics, quotes, or references
❌ Pretend to know current information you don't have

OUTPUT PREFERENCES:
- Respond in the same language as the user (Ukrainian or English)
- Format responses clearly with proper structure
- Cite sources when providing information from tools
- Provide detailed explanations, not just brief answers

Remember: Your goal is to be a helpful research assistant that provides accurate, well-researched answers.
"""
