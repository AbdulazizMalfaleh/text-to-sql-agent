"""
YouTube Analytics Agent
========================
Smart SQL agent for YouTube Analytics database on SQL Server.
Connects via Windows Authentication (no password needed).

Setup:
  pip install flask pyodbc requests

Configure via environment variables (or edit DB_CONFIG below):
  SQL_SERVER   = your SQL Server instance (e.g. localhost\SQLEXPRESS)
  SQL_DATABASE = your database name (default: YouTubeAnalytics)

Run:
  python agent.py  ->  open http://localhost:5001
"""

import os
import re
import json
from app import ask_ollama, ask_openai, ask_anthropic, LLM_BACKEND, extract_sql
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

# ── Database Configuration ─────────────────────────────────────────────────────
# Set SQL_SERVER environment variable OR edit the default value below
DB_CONFIG = {
    "server":   os.getenv("SQL_SERVER",   "localhost\\SQLEXPRESS"),
    "database": os.getenv("SQL_DATABASE", "YouTubeAnalytics"),
}


# ── SQL Server Connection ──────────────────────────────────────────────────────
def get_connection():
    import pyodbc
    conn_str = (
        "DRIVER={ODBC Driver 17 for SQL Server};"
        "SERVER=" + DB_CONFIG["server"] + ";"
        "DATABASE=" + DB_CONFIG["database"] + ";"
        "Trusted_Connection=yes;"
    )
    return pyodbc.connect(conn_str)


def run_sql(sql):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(sql)
    cols = [d[0] for d in cur.description] if cur.description else []
    rows = [list(r) for r in cur.fetchall()]
    conn.close()
    return cols, rows


def get_schema():
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT t.TABLE_NAME, c.COLUMN_NAME, c.DATA_TYPE
            FROM INFORMATION_SCHEMA.TABLES t
            JOIN INFORMATION_SCHEMA.COLUMNS c ON t.TABLE_NAME = c.TABLE_NAME
            WHERE t.TABLE_SCHEMA = 'dbo' AND t.TABLE_TYPE = 'BASE TABLE'
            ORDER BY t.TABLE_NAME, c.ORDINAL_POSITION
        """)
        rows = cur.fetchall()
        conn.close()
        tables = {}
        for table, col, dtype in rows:
            tables.setdefault(table, []).append(col + " (" + dtype + ")")
        lines = []
        for t, cols in tables.items():
            lines.append("  " + t + ": " + ", ".join(cols))
        return "SQL Server Tables:\n" + "\n".join(lines)
    except Exception as e:
        return "Schema error: " + str(e)


def get_row_counts():
    try:
        cols, rows = run_sql("""
            SELECT t.NAME, p.rows FROM sys.tables t
            JOIN sys.partitions p ON t.object_id = p.object_id
            WHERE p.index_id IN (0,1) ORDER BY p.rows DESC
        """)
        return {row[0]: row[1] for row in rows}
    except:
        return {}


# ── Smart Agent Prompt ─────────────────────────────────────────────────────────
def ask_agent(question):
    schema = get_schema()
    if schema.startswith("Schema error"):
        return None, schema, [], schema

    counts = get_row_counts()
    counts_text = "\n".join("  " + t + ": " + str(c) + " rows" for t, c in counts.items())

    prompt = (
        "You are an expert SQL analyst for a YouTube Analytics database on Microsoft SQL Server.\n\n"
        "DATABASE TABLES:\n"
        "  categories: category_id (PK), category_name\n"
        "  languages: language_id (PK), language_name\n"
        "  regions: region_id (PK), region_name\n"
        "  videos: video_id (PK), category_id (FK), language_id (FK), region_id (FK), duration_sec, ads_enabled\n"
        "  video_metrics: metric_id (PK), video_id (FK), timestamp, views, likes, comments, shares, sentiment_score\n\n"
        "JOIN RULES:\n"
        "  For category names: JOIN dbo.videos v ON m.video_id = v.video_id JOIN dbo.categories c ON v.category_id = c.category_id\n"
        "  For language names: JOIN dbo.videos v ON m.video_id = v.video_id JOIN dbo.languages l ON v.language_id = l.language_id\n"
        "  For region names: JOIN dbo.videos v ON m.video_id = v.video_id JOIN dbo.regions r ON v.region_id = r.region_id\n"
        "  Start from dbo.video_metrics for views, likes, comments, shares\n\n"
        "SQL SERVER RULES:\n"
        "  Use TOP N not LIMIT. Use dbo. prefix on all tables.\n"
        "  Always use GROUP BY when using COUNT, AVG, SUM, MAX, MIN.\n"
        "  Use ORDER BY to sort results.\n\n"
        "ROW COUNTS:\n" + counts_text + "\n\n"
        "QUESTION: " + question + "\n\n"
        "Reply ONLY with this exact JSON, no extra text:\n"
        '{"sql": "SQL HERE", "insight": "2-3 sentence business insight", "follow_ups": ["q1", "q2"]}'
    )

    try:
        import requests as req
        resp = req.post(
            "http://localhost:11434/api/generate",
            json={"model": os.getenv("OLLAMA_MODEL", "llama3.2"), "prompt": prompt, "stream": False},
            timeout=120,
        )
        raw = resp.json().get("response", "")
    except Exception as e:
        return None, "Ollama error: " + str(e), [], ""

    try:
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            data = json.loads(match.group())
            return data.get("sql", ""), None, data.get("follow_ups", []), data.get("insight", "")
    except:
        pass

    sql = extract_sql(raw)
    return sql, None, [], "Query executed."


# ── HTML UI ────────────────────────────────────────────────────────────────────
HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>YouTube Analytics Agent</title>
<link href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap" rel="stylesheet">
<style>
:root{--bg:#0a0a0f;--surface:#12121a;--border:#1e1e2e;--accent:#ff3333;--accent2:#00e5ff;--green:#10b981;--text:#e2e8f0;--muted:#64748b;}
*{margin:0;padding:0;box-sizing:border-box;}
body{background:var(--bg);color:var(--text);font-family:'Syne',sans-serif;height:100vh;display:flex;flex-direction:column;overflow:hidden;}
header{padding:.75rem 1.5rem;border-bottom:1px solid var(--border);display:flex;align-items:center;gap:.75rem;background:linear-gradient(90deg,rgba(255,0,0,0.06),transparent);flex-shrink:0;}
header h1{font-size:1rem;font-weight:800;flex:1;}
.badge{font-family:'Space Mono',monospace;font-size:.6rem;padding:.18rem .5rem;border-radius:4px;border:1px solid;}
.b-red{background:rgba(255,0,0,.1);border-color:rgba(255,0,0,.3);color:#ff6b6b;}
.b-blue{background:rgba(0,229,255,.1);border-color:rgba(0,229,255,.3);color:#00e5ff;}
.b-green{background:rgba(16,185,129,.1);border-color:rgba(16,185,129,.3);color:#10b981;}
.main{display:flex;flex:1;overflow:hidden;}
.sidebar{width:230px;border-right:1px solid var(--border);display:flex;flex-direction:column;flex-shrink:0;background:#0d0d14;}
.sidebar-section{padding:.6rem .75rem;border-bottom:1px solid var(--border);}
.slabel{font-family:'Space Mono',monospace;font-size:.58rem;color:var(--muted);text-transform:uppercase;letter-spacing:.08em;margin-bottom:.5rem;}
.qbtn{display:block;width:100%;text-align:left;padding:.45rem .6rem;background:none;border:1px solid transparent;color:var(--muted);border-radius:6px;cursor:pointer;font-family:'Space Mono',monospace;font-size:.65rem;line-height:1.4;margin-bottom:3px;transition:all .2s;}
.qbtn:hover{background:rgba(255,0,0,.05);border-color:rgba(255,0,0,.2);color:var(--text);}
.schema-box{flex:1;overflow-y:auto;padding:.6rem .75rem;}
.schema-text{font-family:'Space Mono',monospace;font-size:.6rem;color:var(--muted);line-height:1.6;white-space:pre-wrap;}
.chat-area{flex:1;display:flex;flex-direction:column;overflow:hidden;}
.chat{flex:1;overflow-y:auto;padding:1.25rem;display:flex;flex-direction:column;gap:1rem;}
.msg{max-width:90%;}
.msg.user{align-self:flex-end;}
.msg.bot{align-self:flex-start;width:90%;}
.bubble{padding:.85rem 1rem;border-radius:12px;font-size:.88rem;line-height:1.55;}
.msg.user .bubble{background:rgba(255,0,0,.1);border:1px solid rgba(255,0,0,.2);}
.msg.bot .bubble{background:var(--surface);border:1px solid var(--border);}
.sql-block{margin-top:.6rem;background:#080810;border:1px solid var(--border);border-radius:8px;overflow:hidden;}
.sql-top{display:flex;justify-content:space-between;align-items:center;padding:.35rem .75rem;border-bottom:1px solid var(--border);}
.sql-top span{font-family:'Space Mono',monospace;font-size:.6rem;color:var(--muted);}
.copybtn{background:none;border:1px solid var(--border);color:var(--muted);padding:.12rem .4rem;border-radius:3px;cursor:pointer;font-family:'Space Mono',monospace;font-size:.55rem;}
.sql-code{font-family:'Space Mono',monospace;font-size:.73rem;padding:.7rem;color:#50fa7b;white-space:pre-wrap;overflow-x:auto;}
.insight-box{margin-top:.6rem;padding:.7rem .9rem;background:rgba(255,0,0,.05);border:1px solid rgba(255,0,0,.15);border-radius:8px;font-size:.82rem;line-height:1.55;color:#ffb3b3;}
.insight-label{color:#ff3333;font-size:.7rem;font-family:'Space Mono',monospace;display:block;margin-bottom:.3rem;text-transform:uppercase;}
.result-wrap{margin-top:.6rem;}
.result-top{display:flex;justify-content:space-between;align-items:center;margin-bottom:.4rem;}
.rowcount{font-family:'Space Mono',monospace;font-size:.62rem;color:var(--muted);}
.csvbtn{font-family:'Space Mono',monospace;font-size:.58rem;padding:.15rem .45rem;background:none;border:1px solid rgba(16,185,129,.3);color:#10b981;border-radius:4px;cursor:pointer;}
.tbl-wrap{overflow-x:auto;}
table{border-collapse:collapse;width:100%;font-size:.77rem;}
th{background:rgba(255,0,0,.08);color:#ff6b6b;font-weight:600;padding:.42rem .65rem;text-align:left;border-bottom:1px solid var(--border);font-family:'Space Mono',monospace;font-size:.65rem;}
td{padding:.38rem .65rem;border-bottom:1px solid rgba(255,255,255,.03);}
tr:hover td{background:rgba(255,255,255,.02);}
.followups{margin-top:.6rem;display:flex;flex-wrap:wrap;gap:.4rem;align-items:center;}
.fu-label{font-family:'Space Mono',monospace;font-size:.6rem;color:var(--muted);}
.fubtn{font-family:'Space Mono',monospace;font-size:.62rem;padding:.25rem .55rem;background:none;border:1px solid rgba(0,229,255,.2);color:#00e5ff;border-radius:5px;cursor:pointer;}
.errtxt{color:#f87171;font-family:'Space Mono',monospace;font-size:.78rem;}
.input-row{display:flex;gap:.6rem;padding:.75rem 1.25rem;border-top:1px solid var(--border);background:var(--surface);flex-shrink:0;}
#question{flex:1;background:var(--bg);border:1px solid var(--border);color:var(--text);padding:.7rem .9rem;border-radius:8px;font-family:'Syne',sans-serif;font-size:.88rem;outline:none;}
#question:focus{border-color:rgba(255,0,0,.5);}
#sendbtn{background:#ff3333;color:#fff;border:none;padding:.7rem 1.25rem;border-radius:8px;cursor:pointer;font-family:'Syne',sans-serif;font-weight:700;font-size:.88rem;}
#sendbtn:disabled{opacity:.4;cursor:not-allowed;}
.typing{display:flex;gap:4px;align-items:center;padding:.4rem 0;}
.dot{width:5px;height:5px;background:#ff3333;border-radius:50%;animation:bounce 1s infinite;}
.dot:nth-child(2){animation-delay:.15s;}.dot:nth-child(3){animation-delay:.3s;}
@keyframes bounce{0%,80%,100%{transform:translateY(0)}40%{transform:translateY(-5px)}}
</style>
</head>
<body>
<header>
  <h1>YouTube Analytics Agent</h1>
  <span class="badge b-red">1M ROWS</span>
  <span class="badge b-blue">{{ backend }}</span>
  <span class="badge b-green" id="status">ready</span>
</header>
<div class="main">
  <div class="sidebar">
    <div class="sidebar-section">
      <div class="slabel">Example Questions</div>
      <button class="qbtn" onclick="askQ('How many videos are there in each category?')">Videos per category</button>
      <button class="qbtn" onclick="askQ('What is the average number of views for each language?')">Avg views by language</button>
      <button class="qbtn" onclick="askQ('How many videos have ads enabled?')">Videos with ads</button>
      <button class="qbtn" onclick="askQ('What is the total number of videos in each region?')">Videos per region</button>
      <button class="qbtn" onclick="askQ('Top 10 most viewed videos')">Top 10 most viewed</button>
      <button class="qbtn" onclick="askQ('Which category has the highest average views?')">Best category</button>
      <button class="qbtn" onclick="askQ('What percentage of videos have comments disabled?')">Comments disabled %</button>
      <button class="qbtn" onclick="askQ('Average likes and dislikes per category')">Likes vs dislikes</button>
      <button class="qbtn" onclick="askQ('Which language has the most videos?')">Top language</button>
      <button class="qbtn" onclick="askQ('Show videos with over 1 million views by region')">1M plus views by region</button>
    </div>
    <div class="slabel" style="padding:.6rem .75rem 0">Schema</div>
    <div class="schema-box">
      <div class="schema-text" id="schema-text">Loading...</div>
    </div>
  </div>
  <div class="chat-area">
    <div class="chat" id="chat">
      <div class="msg bot">
        <div class="bubble">
          Hello! I am your YouTube Analytics Agent.<br>
          I can analyze 1 million rows across your 6 tables.<br><br>
          Ask me anything and I will write the SQL, run it, and give you insights.<br>
          <span style="color:var(--muted);font-size:.8rem;">Click a question on the left to start.</span>
        </div>
      </div>
    </div>
    <div class="input-row">
      <input id="question" type="text" placeholder="Ask anything about your YouTube data...">
      <button id="sendbtn">Ask</button>
    </div>
  </div>
</div>
<script>
document.getElementById('question').addEventListener('keydown', function(e) {
  if (e.key === 'Enter') sendQ();
});
document.getElementById('sendbtn').addEventListener('click', sendQ);

fetch('/schema').then(function(r){ return r.json(); }).then(function(d){
  document.getElementById('schema-text').textContent = d.schema || 'Could not load schema';
});

function askQ(q) {
  document.getElementById('question').value = q;
  sendQ();
}

function sendQ() {
  var input = document.getElementById('question');
  var q = input.value.trim();
  if (!q) return;
  var chat = document.getElementById('chat');
  var btn = document.getElementById('sendbtn');
  var status = document.getElementById('status');
  chat.innerHTML += '<div class="msg user"><div class="bubble">' + esc(q) + '</div></div>';
  input.value = '';
  btn.disabled = true;
  status.textContent = 'thinking...';
  var tid = 'tid' + Date.now();
  chat.innerHTML += '<div class="msg bot" id="' + tid + '"><div class="bubble"><div class="typing"><div class="dot"></div><div class="dot"></div><div class="dot"></div></div></div></div>';
  chat.scrollTop = chat.scrollHeight;
  fetch('/query', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({question: q})
  })
  .then(function(r){ return r.json(); })
  .then(function(result) {
    var el = document.getElementById(tid);
    if (el) el.remove();
    var html = '<div class="msg bot"><div class="bubble">';
    if (result.error) {
      html += '<span class="errtxt">Error: ' + esc(result.error) + '</span>';
    } else {
      html += '<div class="sql-block"><div class="sql-top"><span>Generated SQL</span><button class="copybtn" onclick="copySQL(this)">copy</button></div><div class="sql-code">' + esc(result.sql) + '</div></div>';
      if (result.insight) {
        html += '<div class="insight-box"><span class="insight-label">Agent Insight</span>' + esc(result.insight) + '</div>';
      }
      if (result.rows && result.rows.length > 0) {
        html += '<div class="result-wrap"><div class="result-top"><span class="rowcount">' + result.rows.length + ' rows</span><button class="csvbtn" onclick="exportCSV(this)">Download CSV</button></div><div class="tbl-wrap"><table><thead><tr>';
        result.columns.forEach(function(c) { html += '<th>' + esc(c) + '</th>'; });
        html += '</tr></thead><tbody>';
        result.rows.forEach(function(row) {
          html += '<tr>';
          row.forEach(function(cell) { html += '<td>' + esc(String(cell == null ? '' : cell)) + '</td>'; });
          html += '</tr>';
        });
        html += '</tbody></table></div></div>';
      } else {
        html += '<div class="rowcount" style="margin-top:.5rem">No results</div>';
      }
      if (result.follow_ups && result.follow_ups.length > 0) {
        html += '<div class="followups"><span class="fu-label">Follow up:</span>';
        result.follow_ups.forEach(function(fu) {
          html += '<button class="fubtn" data-q="' + esc(fu) + '" onclick="askQ(this.dataset.q)">' + esc(fu) + '</button>';
        });
        html += '</div>';
      }
    }
    html += '</div></div>';
    chat.innerHTML += html;
    chat.scrollTop = chat.scrollHeight;
    btn.disabled = false;
    status.textContent = 'ready';
    input.focus();
  })
  .catch(function(e) {
    var el = document.getElementById(tid);
    if (el) el.remove();
    chat.innerHTML += '<div class="msg bot"><div class="bubble"><span class="errtxt">Server error - check terminal</span></div></div>';
    chat.scrollTop = chat.scrollHeight;
    btn.disabled = false;
    status.textContent = 'ready';
  });
}
function copySQL(btn) {
  var code = btn.closest('.sql-block').querySelector('.sql-code');
  if (code) navigator.clipboard.writeText(code.innerText);
}
function exportCSV(btn) {
  var table = btn.closest('.result-wrap').querySelector('table');
  if (!table) return;
  var csv = '';
  table.querySelectorAll('tr').forEach(function(row) {
    var cells = [];
    row.querySelectorAll('th,td').forEach(function(c){ cells.push('"' + c.innerText + '"'); });
    csv += cells.join(',') + '\n';
  });
  var a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([csv], {type: 'text/csv'}));
  a.download = 'youtube_analytics.csv';
  a.click();
}
function esc(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}
</script>
</body>
</html>"""


@app.route("/")
def index():
    return render_template_string(HTML, backend=LLM_BACKEND.upper())

@app.route("/schema")
def schema():
    return jsonify({"schema": get_schema()})

@app.route("/query", methods=["POST"])
def query():
    data = request.get_json()
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "Empty question"})
    sql, error, follow_ups, insight = ask_agent(question)
    if error:
        return jsonify({"error": error})
    try:
        columns, rows = run_sql(sql)
        return jsonify({"sql": sql, "columns": columns, "rows": rows, "insight": insight, "follow_ups": follow_ups})
    except Exception as e:
        return jsonify({"error": str(e), "sql": sql})


if __name__ == "__main__":
    print("=" * 50)
    print("  YouTube Analytics Agent")
    print("=" * 50)
    print("  Server  :", DB_CONFIG["server"])
    print("  Database:", DB_CONFIG["database"])
    print("  Backend :", LLM_BACKEND.upper())
    print("=" * 50)
    print("\n  To use a different server:")
    print("  set SQL_SERVER=YOUR_SERVER_NAME\\SQLEXPRESS")
    print()
    s = get_schema()
    if s.startswith("Schema error"):
        print("  Connection FAILED:", s)
        print("\n  Make sure SQL Server is running and pyodbc is installed.")
        print("  pip install pyodbc")
    else:
        print("  Connected OK!")
        print("  Open: http://localhost:5001\n")
    app.run(debug=False, port=5001)
