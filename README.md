# Text-to-SQL + YouTube Analytics Agent

> Ask your database anything in plain English. The AI writes the SQL, runs it, and gives you insights.

![Python](https://img.shields.io/badge/Python-3.x-blue?style=flat-square&logo=python)
![Flask](https://img.shields.io/badge/Flask-Web%20UI-lightgrey?style=flat-square&logo=flask)
![Ollama](https://img.shields.io/badge/Ollama-Offline%20AI-black?style=flat-square)
![SQL Server](https://img.shields.io/badge/SQL%20Server-1M%20Rows-red?style=flat-square&logo=microsoft-sql-server)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## What Is This?

A fully **offline** AI-powered tool that converts natural language into SQL queries and runs them against your real database.

No cloud. No subscriptions. Everything runs on your local machine.

```
You type:   "Which category has the highest average views?"
Agent:      SELECT c.category_name, AVG(m.views) AS avg_views
            FROM dbo.video_metrics m
            JOIN dbo.videos v ON m.video_id = v.video_id
            JOIN dbo.categories c ON v.category_id = c.category_id
            GROUP BY c.category_name
            ORDER BY avg_views DESC

Result:     Table with data + business insight + follow-up suggestions
```

---

## Screenshots

### Natural Language → SQL Query (Agent Demo)
![ Natural Language → SQL Query (Agent Demo)](screenshots/15.jpeg)

### Database Statistics — 1M rows processed
![Database Statistics — 1M rows processed](screenshots/image.png)

---

## Demo

| Feature | Description |
|---|---|
| Natural language to SQL | Type any question, get SQL instantly |
| Agent Insights | AI explains what the results mean |
| Follow-up questions | Suggested next questions after each answer |
| Export to CSV | Download any result with one click |
| Schema viewer | See your full database structure in the sidebar |
| 100% offline | Powered by Ollama — no internet needed |

---

## Project Structure

```
text-to-sql/
├── app.py              # Core engine: LLM backends, SQL runner, schema reader
├── agent.py            # YouTube Analytics Agent (SQL Server, 1M rows)
├── web_app_v2.py       # General web UI (SQLite + PostgreSQL)
├── .env.example        # Environment variables template
├── requirements.txt    # Python dependencies
├── .gitignore
└── data/
    └── sample.db       # Auto-generated SQLite sample database
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.x |
| AI Model | Ollama — llama3.2 (fully offline, runs on CPU) |
| Web Framework | Flask |
| Local Database | SQLite (sample data — employees, sales, products) |
| Production Database | SQL Server via pyodbc |
| Frontend | HTML + CSS + Vanilla JavaScript |

---

## Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/text-to-sql.git
cd text-to-sql
```

### 2. Create virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Ollama (local AI)

Download from [ollama.ai](https://ollama.ai) then pull the model:

```bash
ollama pull llama3.2
```

> This downloads ~2GB. After this, everything runs fully offline.

### 5. Configure environment

```bash
cp .env.example .env
```

Edit `.env` and set your SQL Server name:

```env
SQL_SERVER=YOUR_SERVER_NAME\SQLEXPRESS
SQL_DATABASE=YouTubeAnalytics
```

### 6. Run

```bash
# YouTube Analytics Agent (SQL Server)
python agent.py
# Open: http://localhost:5001

# General web app (SQLite / PostgreSQL)
python web_app_v2.py
# Open: http://localhost:5000

# Command line interface
python app.py
```

---

## YouTube Analytics Database

Dataset from Kaggle: [YouTube 1M Global Creator Analytics](https://www.kaggle.com/datasets/ehsanzx/youtube-1m-global-creator-analytics)

### Schema

```
categories      category_id (PK), category_name
languages       language_id (PK), language_name
regions         region_id (PK), region_name
videos          video_id (PK), category_id (FK), language_id (FK),
                region_id (FK), duration_sec, ads_enabled
video_metrics   metric_id (PK), video_id (FK), timestamp, views,
                likes, comments, shares, sentiment_score
youtube_raw     Raw source data — 1,000,000 rows
```

### Example Questions

```
How many videos are there in each category?
What is the average number of views for each language?
How many videos have ads enabled?
What is the total number of videos in each region?
Top 10 most viewed videos
Which category has the highest average views?
What percentage of videos have comments disabled?
Average likes and dislikes per category
Which language has the most videos?
Show videos with over 1 million views by region
```

---

## How the Agent Works

```
1. You type a question
         ↓
2. Agent reads full schema + row counts
         ↓
3. Ollama AI selects the right tables and JOINs
         ↓
4. SQL generated for SQL Server syntax
         ↓
5. Query runs on real database (1M rows)
         ↓
6. Results + insight + follow-up questions
```

---

## Switching AI Models

```bash
# Pull a different model
ollama pull llama3.3
ollama pull mistral
ollama pull deepseek-coder
```

Then set in `.env`:

```env
OLLAMA_MODEL=llama3.3
```

| Model | Size | Best For |
|---|---|---|
| llama3.2 (default) | ~2GB | Balanced speed and accuracy |
| llama3.3 | ~4GB | Better SQL generation |
| mistral | ~4GB | Fastest responses |
| deepseek-coder | ~4GB | Specialized for SQL and code |

To use cloud AI instead:

```env
LLM_BACKEND=openai
OPENAI_API_KEY=sk-...
```

---

## Requirements

- Python 3.8+
- Ollama installed ([ollama.ai](https://ollama.ai))
- ODBC Driver 17 for SQL Server ([download](https://aka.ms/odbc17))
- SQL Server running locally (Express edition works fine)

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `ollama not found` | Add Ollama to PATH or restart terminal |
| `pyodbc error` | `pip install pyodbc` |
| `ODBC Driver missing` | Download from aka.ms/odbc17 |
| SQL Server not connecting | Check `SQL_SERVER` in `.env` matches your SSMS connection |
| Slow responses | Normal — Ollama runs on CPU, 30–90 seconds per query |
| Port already in use | Change `port=5001` in `agent.py` to another number |

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `LLM_BACKEND` | `ollama` | AI backend: `ollama`, `openai`, `anthropic` |
| `OLLAMA_MODEL` | `llama3.2` | Ollama model name |
| `SQL_SERVER` | `localhost\SQLEXPRESS` | SQL Server instance name |
| `SQL_DATABASE` | `YouTubeAnalytics` | Database name |
| `DB_PATH` | `data/sample.db` | SQLite path (for web_app_v2) |
| `OPENAI_API_KEY` | — | Required only if `LLM_BACKEND=openai` |
| `ANTHROPIC_API_KEY` | — | Required only if `LLM_BACKEND=anthropic` |

---

## License

MIT License — free to use, modify, and share.

---

*Built with Python + Ollama + Flask + SQL Server*
