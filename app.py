"""
Text-to-SQL Core Engine
========================
Converts natural language questions to SQL queries using a local or cloud LLM.

Supported backends:
  - ollama   (default, fully offline)
  - openai   (requires OPENAI_API_KEY)
  - anthropic (requires ANTHROPIC_API_KEY)

Usage:
  python app.py
"""

import os
import re
import sqlite3
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────────────
LLM_BACKEND   = os.getenv("LLM_BACKEND",   "ollama")
OLLAMA_MODEL  = os.getenv("OLLAMA_MODEL",  "llama3.2")
OLLAMA_URL    = os.getenv("OLLAMA_URL",    "http://localhost:11434/api/generate")
DB_PATH       = os.getenv("DB_PATH",       "data/sample.db")


# ── Database Setup ─────────────────────────────────────────────────────────────
def setup_database():
    """Create sample SQLite database with employees, sales, products."""
    Path("data").mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.executescript("""
        CREATE TABLE IF NOT EXISTS employees (
            id INTEGER PRIMARY KEY,
            name TEXT, department TEXT,
            salary REAL, hire_date TEXT, city TEXT
        );
        CREATE TABLE IF NOT EXISTS sales (
            id INTEGER PRIMARY KEY,
            employee_id INTEGER, product TEXT,
            amount REAL, sale_date TEXT, region TEXT
        );
        CREATE TABLE IF NOT EXISTS products (
            id INTEGER PRIMARY KEY,
            name TEXT, category TEXT,
            price REAL, stock INTEGER
        );
    """)

    if cur.execute("SELECT COUNT(*) FROM employees").fetchone()[0] == 0:
        employees = [
            (1, "Alice Johnson",   "Engineering", 95000,  "2020-03-15", "Riyadh"),
            (2, "Bob Smith",       "Marketing",   72000,  "2019-07-22", "Jeddah"),
            (3, "Carol White",     "Engineering", 88000,  "2021-01-10", "Dubai"),
            (4, "David Brown",     "Engineering", 102000, "2018-11-05", "Riyadh"),
            (5, "Eve Davis",       "Marketing",   65000,  "2022-04-18", "Dammam"),
            (6, "Frank Miller",    "HR",          58000,  "2020-09-30", "Jeddah"),
            (7, "Grace Kim",       "Engineering", 110000, "2017-06-14", "Riyadh"),
            (8, "Henry Wilson",    "HR",          55000,  "2023-02-28", "Dubai"),
        ]
        cur.executemany("INSERT INTO employees VALUES (?,?,?,?,?,?)", employees)

        sales = [
            (1,  1, "Software License", 15000, "2024-01-15", "Gulf"),
            (2,  2, "Marketing Suite",  8500,  "2024-01-20", "Gulf"),
            (3,  3, "Cloud Platform",   22000, "2024-02-05", "Europe"),
            (4,  4, "Enterprise Pack",  35000, "2024-02-14", "Gulf"),
            (5,  1, "Support Plan",     5000,  "2024-02-28", "Asia"),
            (6,  7, "AI Module",        28000, "2024-03-10", "Gulf"),
            (7,  3, "Security Suite",   18000, "2024-03-22", "Europe"),
            (8,  5, "Analytics Tool",   12000, "2024-04-01", "Gulf"),
            (9,  4, "Data Platform",    42000, "2024-04-15", "USA"),
            (10, 7, "ML Package",       31000, "2024-05-02", "Gulf"),
        ]
        cur.executemany("INSERT INTO sales VALUES (?,?,?,?,?,?)", sales)

        products = [
            (1, "Software License", "Software",  1200, 500),
            (2, "Cloud Platform",   "Cloud",     2500, 200),
            (3, "AI Module",        "AI",        3800, 150),
            (4, "Security Suite",   "Security",  1800, 300),
        ]
        cur.executemany("INSERT INTO products VALUES (?,?,?,?,?)", products)

    conn.commit()
    conn.close()
    print("Database ready:", DB_PATH)


def get_schema() -> str:
    """Return the database schema as a formatted string for the LLM."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [r[0] for r in cur.fetchall()]
    lines = []
    for t in tables:
        cur.execute(f"PRAGMA table_info({t})")
        cols = [f"{r[1]} {r[2]}" for r in cur.fetchall()]
        lines.append(f"  {t}({', '.join(cols)})")
    conn.close()
    return "Tables:\n" + "\n".join(lines)


# ── LLM Backends ──────────────────────────────────────────────────────────────
def ask_ollama(question: str, schema: str) -> str:
    """Send question to local Ollama model (fully offline)."""
    try:
        import requests
        prompt = (
            f"You are a SQL expert. Given this SQLite database schema:\n{schema}\n\n"
            f"Write a SQL query to answer: {question}\n\n"
            f"Return ONLY the SQL query, no explanation."
        )
        resp = requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=120,
        )
        return resp.json().get("response", "ERROR: Empty response")
    except Exception as e:
        return f"ERROR: {e}"


def ask_openai(question: str, schema: str) -> str:
    """Send question to OpenAI API. Requires OPENAI_API_KEY env variable."""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": f"You are a SQL expert. Schema:\n{schema}\nReturn ONLY SQL queries."},
                {"role": "user", "content": question}
            ]
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"ERROR: {e}"


def ask_anthropic(question: str, schema: str) -> str:
    """Send question to Anthropic Claude API. Requires ANTHROPIC_API_KEY env variable."""
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=512,
            messages=[{"role": "user", "content": f"Schema:\n{schema}\n\nWrite SQL for: {question}\n\nReturn ONLY SQL."}]
        )
        return resp.content[0].text
    except Exception as e:
        return f"ERROR: {e}"


# ── SQL Utilities ─────────────────────────────────────────────────────────────
def extract_sql(raw: str) -> str:
    """Clean LLM response and extract the SQL query."""
    raw = re.sub(r"```sql\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"```\s*", "", raw)
    raw = raw.strip()
    # Take only the first statement
    for keyword in ["SELECT", "INSERT", "UPDATE", "DELETE", "WITH"]:
        idx = raw.upper().find(keyword)
        if idx != -1:
            raw = raw[idx:]
            break
    # Stop at first semicolon
    if ";" in raw:
        raw = raw[:raw.index(";") + 1]
    return raw.strip()


def run_sql(sql: str):
    """Execute SQL against SQLite and return (columns, rows)."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(sql)
    cols = [d[0] for d in cur.description] if cur.description else []
    rows = [list(r) for r in cur.fetchall()]
    conn.close()
    return cols, rows


def print_table(cols, rows):
    """Print results as a formatted ASCII table."""
    if not rows:
        print("(no results)")
        return
    widths = [max(len(str(c)), max(len(str(r[i])) for r in rows)) for i, c in enumerate(cols)]
    sep = "+-" + "-+-".join("-" * w for w in widths) + "-+"
    header = "| " + " | ".join(str(c).ljust(w) for c, w in zip(cols, widths)) + " |"
    print(sep)
    print(header)
    print(sep)
    for row in rows:
        print("| " + " | ".join(str(v).ljust(w) for v, w in zip(row, widths)) + " |")
    print(sep)
    print(f"  {len(rows)} row(s)")


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    setup_database()
    schema = get_schema()

    print("\n" + "="*50)
    print("  Text-to-SQL CLI")
    print(f"  Backend : {LLM_BACKEND.upper()}")
    print(f"  Model   : {OLLAMA_MODEL}")
    print("="*50)

    examples = [
        "Show all employees in Engineering",
        "What is the average salary by department?",
        "Who are the top 3 salespeople by total amount?",
        "Which products have stock below 200?",
        "Show total sales per region",
        "List employees hired after 2020",
    ]

    print("\nExample questions:")
    for i, q in enumerate(examples, 1):
        print(f"  {i}. {q}")

    print("\nType a number (1-6), your own question, or 'quit' to exit.\n")

    while True:
        user_input = input("Question: ").strip()
        if not user_input or user_input.lower() in ("quit", "exit", "q"):
            break

        question = examples[int(user_input) - 1] if user_input.isdigit() and 1 <= int(user_input) <= 6 else user_input
        print(f"\nAsking: {question}")
        print("Generating SQL...\n")

        if LLM_BACKEND == "openai":
            raw = ask_openai(question, schema)
        elif LLM_BACKEND == "anthropic":
            raw = ask_anthropic(question, schema)
        else:
            raw = ask_ollama(question, schema)

        if raw.startswith("ERROR"):
            print(f"Error: {raw}")
            continue

        sql = extract_sql(raw)
        print(f"SQL: {sql}\n")

        try:
            cols, rows = run_sql(sql)
            print_table(cols, rows)
        except sqlite3.Error as e:
            print(f"SQL Error: {e}")
        print()


if __name__ == "__main__":
    main()
