# Research Agent with Local RAG

Локальний multi-tool агент, який поєднує пошук в інтернеті та пошук у локальній базі знань (RAG), а потім формує фінальну відповідь з переліком джерел і зберігає результат у Markdown.

## Що вміє агент

- Виконує web search (DuckDuckGo).
- Читає та очищує вміст веб-сторінок.
- Шукає у локальній базі знань через hybrid retrieval:
  - semantic search (FAISS + OpenAI embeddings)
  - BM25 lexical search
  - reranking через cross-encoder
- Працює у ReAct-циклі: agent -> tool -> agent -> summarizer.
- Зберігає фінальну відповідь у папку output як .md файл.

## Структура проєкту

```text
HT Lektion 5/
|-- main.py
|-- agent.py
|-- tools.py
|-- ingest.py
|-- retriever.py
|-- config.py
|-- requirements.txt
|-- .env.example
|-- data/
|-- index/
`-- output/
```

## Вимоги

- Python 3.11+
- OpenAI API key

## Швидкий старт

1. Встановіть залежності:
```
pip install -r requirements.txt
```
2. Створіть .env на основі шаблону:
```
cp .env.example .env
```
На Windows PowerShell:
```powershell
Copy-Item .env.example .env
```
3. У файлі .env задайте ключ:
```dotenv
openai_api_key=sk-your-real-key-here
```
4. Побудуйте або оновіть індекс знань:
```
python ingest.py
```
5. Запустіть агента:
```
python main.py
```

## Команди в CLI
- exit або quit: завершити програму.
- debug on: увімкнути debug-логи.
- debug off: вимкнути debug-логи.
- /ingest: перебудувати knowledge index без виходу з агента.

## Рекомендований .env профіль (balanced)

```dotenv
openai_api_key=sk-your-real-key-here
model_name=gpt-4o

# Agent control
max_iterations=9
tool_timeout_sec=25
agent_first_wait_sec=35
agent_second_wait_sec=20
debug=false

# Search / content limits
max_search_results=5
max_url_content_length=5000

# RAG
embedding_model=text-embedding-3-small
reranker_model=BAAI/bge-reranker-base
data_dir=data
index_dir=index
chunks_path=index/chunks.json

chunk_size=800
chunk_overlap=120
semantic_top_k=8
bm25_top_k=8
hybrid_top_k=10
rerank_top_n=4

# Output
output_dir=output
```

## Як це працює

1. Користувач вводить запит.
2. Агент вирішує, які інструменти викликати.
3. Tool node виконує інструменти з таймаутом, керованим через .env.
4. Summarizer формує відповідь мовою запиту та додає розділ Джерела.
5. Save node зберігає фінальну відповідь у output.

## Формат відповіді

- Основна відповідь у 1-3 абзацах.
- Окремий блок Джерела.
- Для локального RAG-пошуку в джерелах виводяться Source, page, Relevance.

## Known limitations

- Перевірка наявності .env орієнтується на поточну робочу директорію запуску.
- Захист меж директорій у файлових tools реалізований префіксною перевіркою шляху.
- Для завантаження FAISS-індексу використовується allow_dangerous_deserialization=True.

Ці пункти свідомо залишені для локального single-user режиму та заплановані до hardening у наступній ітерації.

## Troubleshooting

- Якщо агент зависає або довго відповідає:
  - збільшіть tool_timeout_sec на 5-10 секунд
  - збільшіть agent_first_wait_sec та agent_second_wait_sec
- Якщо відповідь обривається занадто рано:
  - збільшіть max_iterations на 1-2
- Якщо не знаходить локальні документи:
  - перевірте вміст data/
  - запустіть python ingest.py ще раз

## Ліцензія
Навчальний проєкт. Використовуйте та адаптуйте під свої задачі.
