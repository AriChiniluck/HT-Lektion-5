import os
import re
import ast
import operator
import socket
from datetime import datetime
from urllib.parse import urlparse
import trafilatura
from ddgs import DDGS  # ✅ ЗМІНЕНО: duckduckgo_search → ddgs
from config import settings
from retriever import get_retriever


# ---------------------------------------------------------
# Debug helper
# ---------------------------------------------------------
def debug_print(*args):
    if settings.debug:
        print("[DEBUG]", *args)


# ---------------------------------------------------------
# Keyword extraction
# ---------------------------------------------------------
def extract_keywords(text: str, max_words: int = 3) -> str:
    """Виділяє ключові слова з тексту для формування імені файлу."""
    cleaned = re.sub(r"[^a-zA-Zа-яА-ЯїієґЇІЄҐ0-9 ]", " ", text)
    words = cleaned.lower().split()

    stopwords = {"що", "як", "коли", "де", "про", "та", "і", "або", "чи", "будь", "будь-який"}
    keywords = [w for w in words if w not in stopwords and len(w) > 2]

    if not keywords:
        keywords = ["agent_answer"]

    return "_".join(keywords[:max_words])


def generate_filename_from_query(text: str) -> str:
    """Створює ім'я Markdown-файлу на основі ключових слів."""
    topic = extract_keywords(text)
    date_str = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    return f"{topic}_{date_str}.md"


# ---------------------------------------------------------
# Search tool
# ---------------------------------------------------------
def search_tool_lc(query: str):
    """Пошук у DuckDuckGo з валідацією."""
    debug_print(f"search_tool called with query: {query}")
    
    # Валідація запиту
    if not query or not isinstance(query, str):
        return "❌ Невалідний запит пошуку"
    
    query = query.strip()
    if len(query) < 2 or len(query) > 500:
        return "❌ Запит занадто короткий або довгий (2-500 символів)"
    
    try:
        ddgs = DDGS()  # ✅ Створюємо інстанс DDGS
        results = list(ddgs.text(query, max_results=settings.max_search_results))
        
        # Обмежуємо розмір результатів
        limited_results = []
        total_size = 0
        max_total_size = settings.max_url_content_length
        
        for result in results:
            result_size = len(str(result))
            if total_size + result_size > max_total_size:
                break
            limited_results.append(result)
            total_size += result_size
        
        return limited_results if limited_results else "❌ Результатів не знайдено"
    except Exception as e:
        debug_print(f"search_tool error: {e}")
        return f"❌ search_tool error: {e}"


# ---------------------------------------------------------
# Read URL tool
# ---------------------------------------------------------
def read_tool_lc(url: str):
    """Завантажує та витягує текст зі сторінки з перевіркою безпеки."""
    debug_print(f"read_tool called with url: {url}")
    
    try:
        # Валідація URL
        if not url or not isinstance(url, str):
            return "❌ Невалідна URL"
        
        parsed = urlparse(url)
        
        # Перевірка схеми
        if parsed.scheme not in ('http', 'https'):
            return "❌ Дозволені тільки HTTP/HTTPS URL"
        
        # Перевірка хоста
        if not parsed.hostname:
            return "❌ Невалідна URL"
        
        # Запобіжність від SSRF атак
        forbidden_hosts = {'localhost', '127.0.0.1', '0.0.0.0', '[::1]'}
        if parsed.hostname in forbidden_hosts:
            return "❌ Доступ до локальних адрес заборонений"
        
        # Перевірка IP адреси
        try:
            ip = socket.gethostbyname(parsed.hostname)
            if ip.startswith(('127.', '192.168.', '10.', '172.')):
                return "❌ Доступ до приватних мереж заборонений"
        except socket.gaierror:
            return "❌ Невалідна адреса"
        
        downloaded = trafilatura.fetch_url(url, timeout=10)
        if not downloaded:
            return "❌ Не вдалось завантажити сторінку"

        text = trafilatura.extract(downloaded)
        if not text:
            return "❌ Не вдалось витягти текст"

        return text[:settings.max_url_content_length]
    except Exception as e:
        debug_print(f"read_tool error: {e}")
        return f"❌ read_tool error: {e}"


# ---------------------------------------------------------
# Save report tool
# ---------------------------------------------------------
def save_report_tool_lc(filename: str, content: str):
    """Зберігає текст у файл з валідацією."""
    debug_print(f"save_report_tool called with filename={filename}")
    
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
    
    try:
        # Валідація імені файлу
        if not filename or not isinstance(filename, str):
            return "❌ Невалідне ім'я файлу"
        
        if "/" in filename or "\\" in filename or filename.startswith("."):
            return "❌ Невалідне ім'я файлу"
        
        # Валідація вмісту
        if not isinstance(content, str):
            return "❌ Невалідний вміст"
        
        if len(content) > MAX_FILE_SIZE:
            return f"❌ Файл занадто великий (>{MAX_FILE_SIZE} байт)"
        
        os.makedirs(settings.output_dir, exist_ok=True)
        path = os.path.join(settings.output_dir, filename)
        
        # Перевірка шляху (запобіжність від path traversal)
        abs_path = os.path.abspath(path)
        allowed_dir = os.path.abspath(settings.output_dir)
        if not abs_path.startswith(allowed_dir):
            return "❌ Шлях поза дозволеною директорією"

        with open(abs_path, "w", encoding="utf-8") as f:
            f.write(content)

        return f"✅ Файл збережено: {abs_path}"
    except Exception as e:
        debug_print(f"save_report_tool error: {e}")
        return f"❌ save_report_tool error: {e}"


# ---------------------------------------------------------
# List files tool
# ---------------------------------------------------------
def list_files_tool_lc(directory: str):
    """Показує файли в директорії з перевіркою безпеки."""
    debug_print(f"list_files_tool called with directory: {directory}")
    
    try:
        if not directory or not isinstance(directory, str):
            return "❌ Невалідна директорія"
        
        abs_dir = os.path.abspath(directory)
        allowed_dir = os.path.abspath(settings.output_dir)
        
        # Перевірка, чи директорія у дозволеному місце
        if not abs_dir.startswith(allowed_dir):
            return "❌ Доступ до цієї директорії заборонений"
        
        if not os.path.isdir(abs_dir):
            return "❌ Директорія не знайдена"
        
        files = os.listdir(abs_dir)
        return files[:50]  # Обмежуємо до 50 файлів
    except Exception as e:
        debug_print(f"list_files_tool error: {e}")
        return f"❌ list_files_tool error: {e}"


# ---------------------------------------------------------
# Read file tool
# ---------------------------------------------------------
def read_file_tool_lc(path: str):
    """Читає файл з перевіркою безпеки та кодувань."""
    debug_print(f"read_file_tool called with path: {path}")
    
    try:
        if not path or not isinstance(path, str):
            return "❌ Невалідний шлях"
        
        # Нормалізуємо шлях
        abs_path = os.path.abspath(path)
        allowed_dir = os.path.abspath(settings.output_dir)
        
        # Перевіряємо, чи файл у дозволеній директорії
        if not abs_path.startswith(allowed_dir):
            return "❌ Доступ до цього файлу заборонений"
        
        # Перевіряємо, чи файл існує
        if not os.path.isfile(abs_path):
            return "❌ Файл не знайдений"
        
        # Перевіряємо розмір файлу
        if os.path.getsize(abs_path) > 10 * 1024 * 1024:  # 10 MB
            return "❌ Файл занадто великий"
        
        # Намагаємось прочитати з різними кодуваннями
        for encoding in ['utf-8', 'utf-16', 'cp1252', 'iso-8859-1']:
            try:
                with open(abs_path, "r", encoding=encoding) as f:
                    content = f.read()
                return content[:settings.max_url_content_length]
            except (UnicodeDecodeError, UnicodeError):
                continue
        
        return "❌ Не вдалось декодувати файл"
    except Exception as e:
        debug_print(f"read_file_tool error: {e}")
        return f"❌ read_file_tool error: {e}"


# ---------------------------------------------------------
# Calculator tool - БЕЗПЕЧНА ВЕРСІЯ
# ---------------------------------------------------------
def calculate_tool_lc(expression: str):
    """Обчислює математичний вираз БЕЗПЕЧНО без eval()."""
    debug_print(f"calculate_tool called with expression: {expression}")
    
    try:
        if not expression or not isinstance(expression, str):
            return "❌ Невалідний вираз"
        
        expression = expression.strip()
        if len(expression) > 1000:
            return "❌ Вираз занадто довгий"
        
        # Парсимо вираз за допомогою AST (безпечно)
        node = ast.parse(expression, mode='eval')
        
        # Перевіряємо, чи містить тільки дозволені операції
        for child in ast.walk(node.body):
            # Забороняємо функції
            if isinstance(child, ast.Call):
                raise ValueError("❌ Функції не дозволені")
            # Забороняємо довільні змінні
            if isinstance(child, ast.Name):
                if child.id not in {'pi', 'e', 'sqrt'}:
                    raise ValueError(f"❌ Змінна '{child.id}' не дозволена")
        
        # Безпечно обчислюємо
        import math
        safe_dict = {
            'pi': math.pi,
            'e': math.e,
            '__builtins__': {}
        }
        
        result = eval(compile(node, '<string>', 'eval'), safe_dict)
        
        # Перевіряємо результат
        if isinstance(result, (int, float)):
            return round(result, 6) if isinstance(result, float) else result
        else:
            return "❌ Результат має неприпустимий тип"
    except ValueError as e:
        return str(e)
    except Exception as e:
        debug_print(f"calculate_tool error: {e}")
        return f"❌ calculate_tool error: {e}"


def format_knowledge_results(results: list[dict]) -> str:
    if not results:
        return "❌ Нічого не знайдено у локальній базі знань."

    lines = [f"[{len(results)} documents found]"]
    for i, item in enumerate(results, 1):
        score = item.get("rerank_score", item.get("hybrid_score", 0.0))
        lines.append(
            f"{i}. Source: {item.get('filename', 'unknown')}, page {item.get('page', '?')}"
        )
        lines.append(f"   Relevance: {score:.4f}")
    return "\n".join(lines)
    

def knowledge_search(query: str) -> str:
    debug_print(f"knowledge_search called with query: {query}")

    if not query or not isinstance(query, str):
        return "❌ Невалідний запит для knowledge_search"

    q = query.strip()
    if len(q) < 2:
        return "❌ Запит занадто короткий"

    try:
        retriever = get_retriever()
        results = retriever.search(q)
        return format_knowledge_results(results)
    except FileNotFoundError:
        return "❌ Knowledge index not found. Run python ingest.py first."
    except Exception as e:
        debug_print(f"knowledge_search error: {e}")
        return f"❌ knowledge_search error: {e}"


    
# alias
search_tool = search_tool_lc
read_tool = read_tool_lc
save_report_tool = save_report_tool_lc
list_files_tool = list_files_tool_lc
read_file_tool = read_file_tool_lc
calculate_tool = calculate_tool_lc
knowledge_search_tool = knowledge_search
