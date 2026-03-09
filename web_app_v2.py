"""
Text-to-SQL Web UI v2
=====================
Improvements:
- 12 example questions (categorized)
- Connect to SQLite (local file) or PostgreSQL (cloud)
- Export results to CSV
- Query history sidebar
- Better loading states

Run: python web_app_v2.py  ->  open http://localhost:5000
"""

import os
import sqlite3
import re
import csv
import io
from pathlib import Path
from app import setup_database, get_schema, extract_sql
from app import ask_ollama, ask_openai, ask_anthropic, LLM_BACKEND, DB_PATH

try:
    from flask import Flask, request, jsonify, render_template_string, Response
except ImportError:
    print("Run: pip install flask")
    exit(1)

app = Flask(__name__)

#    DB connection state                                                         
current_db = {"type": "sqlite", "path": DB_PATH, "conn_string": ""}


def run_query(sql: str):
    """Run SQL on current DB (sqlite or postgres)."""
    if current_db["type"] == "postgres":
        try:
            import psycopg2
            conn = psycopg2.connect(current_db["conn_string"])
            cur = conn.cursor()
            cur.execute(sql)
            cols = [d[0] for d in cur.description] if cur.description else []
            rows = [list(r) for r in cur.fetchall()]
            conn.close()
            return cols, rows
        except Exception as e:
            raise RuntimeError(str(e))
    else:
        conn = sqlite3.connect(current_db["path"])
        cur = conn.cursor()
        cur.execute(sql)
        cols = [d[0] for d in cur.description] if cur.description else []
        rows = [list(r) for r in cur.fetchall()]
        conn.close()
        return cols, rows


def get_current_schema():
    if current_db["type"] == "postgres":
        try:
            import psycopg2
            conn = psycopg2.connect(current_db["conn_string"])
            cur = conn.cursor()
            cur.execute("""
                SELECT table_name, column_name, data_type
                FROM information_schema.columns
                WHERE table_schema = 'public'
                ORDER BY table_name, ordinal_position
            """)
            rows = cur.fetchall()
            conn.close()
            tables = {}
            for t, c, dt in rows:
                tables.setdefault(t, []).append(f"{c} {dt}")
            return "Tables:\n" + "\n".join(f"  {t}({', '.join(cols)})" for t, cols in tables.items())
        except Exception as e:
            return f"Schema error: {e}"
    else:
        return get_schema()


HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Text-to-SQL v2</title>
<link href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap" rel="stylesheet">
<style>
:root {
  --bg:#0a0a0f; --surface:#12121a; --surface2:#16161f;
  --border:#1e1e2e; --accent:#00e5ff; --accent2:#ff6b35;
  --green:#10b981; --purple:#7c3aed;
  --text:#e2e8f0; --muted:#64748b;
}
*{margin:0;padding:0;box-sizing:border-box;}
body{background:var(--bg);color:var(--text);font-family:'Syne',sans-serif;height:100vh;display:flex;flex-direction:column;overflow:hidden;}

/*    Header    */
header{padding:.75rem 1.5rem;border-bottom:1px solid var(--border);display:flex;align-items:center;gap:.75rem;background:linear-gradient(90deg,rgba(0,229,255,0.05),transparent);flex-shrink:0;}
header h1{font-size:1rem;font-weight:800;flex:1;}
.badge{font-family:'Space Mono',monospace;font-size:.6rem;padding:.18rem .5rem;border-radius:4px;border:1px solid;}
.badge-blue{background:rgba(0,229,255,.1);border-color:rgba(0,229,255,.3);color:var(--accent);}
.badge-orange{background:rgba(255,107,53,.08);border-color:rgba(255,107,53,.3);color:var(--accent2);}
.badge-green{background:rgba(16,185,129,.1);border-color:rgba(16,185,129,.3);color:var(--green);}
.hdr-btn{background:none;border:1px solid var(--border);color:var(--muted);padding:.3rem .7rem;border-radius:6px;cursor:pointer;font-family:'Space Mono',monospace;font-size:.65rem;transition:all .2s;}
.hdr-btn:hover{border-color:var(--accent);color:var(--accent);}

/*    Layout    */
.main{display:flex;flex:1;overflow:hidden;}

/*    Sidebar    */
.sidebar{width:240px;border-right:1px solid var(--border);display:flex;flex-direction:column;flex-shrink:0;background:var(--surface2);}
.sidebar-title{font-family:'Space Mono',monospace;font-size:.65rem;color:var(--muted);padding:.75rem 1rem;border-bottom:1px solid var(--border);text-transform:uppercase;letter-spacing:.08em;}
.history{flex:1;overflow-y:auto;padding:.5rem;}
.hist-item{padding:.5rem .6rem;border-radius:6px;font-size:.75rem;color:var(--muted);cursor:pointer;border:1px solid transparent;transition:all .2s;line-height:1.4;margin-bottom:3px;}
.hist-item:hover{background:rgba(0,229,255,.05);border-color:rgba(0,229,255,.15);color:var(--text);}
.hist-empty{font-family:'Space Mono',monospace;font-size:.65rem;color:var(--border);padding:.75rem;text-align:center;}

/*    DB Panel    */
.db-panel{border-top:1px solid var(--border);padding:.75rem;}
.db-panel-title{font-family:'Space Mono',monospace;font-size:.6rem;color:var(--muted);margin-bottom:.5rem;text-transform:uppercase;}
.db-type-row{display:flex;gap:.4rem;margin-bottom:.5rem;}
.db-type-btn{flex:1;padding:.3rem;border:1px solid var(--border);background:none;color:var(--muted);border-radius:5px;cursor:pointer;font-family:'Space Mono',monospace;font-size:.6rem;transition:all .2s;}
.db-type-btn.active{border-color:var(--accent);color:var(--accent);background:rgba(0,229,255,.08);}
.db-input{width:100%;background:var(--bg);border:1px solid var(--border);color:var(--text);padding:.4rem .6rem;border-radius:5px;font-family:'Space Mono',monospace;font-size:.65rem;margin-bottom:.4rem;outline:none;}
.db-input:focus{border-color:var(--accent);}
.db-connect-btn{width:100%;padding:.4rem;background:var(--accent);color:#000;border:none;border-radius:5px;cursor:pointer;font-family:'Syne',sans-serif;font-weight:700;font-size:.72rem;transition:opacity .2s;}
.db-connect-btn:hover{opacity:.85;}
.db-status{font-family:'Space Mono',monospace;font-size:.6rem;margin-top:.4rem;text-align:center;}

/*    Chat area    */
.chat-area{flex:1;display:flex;flex-direction:column;overflow:hidden;}
.chat{flex:1;overflow-y:auto;padding:1.25rem;display:flex;flex-direction:column;gap:1rem;}
.msg{max-width:88%;}
.msg.user{align-self:flex-end;}
.msg.bot{align-self:flex-start;}
.bubble{padding:.85rem 1rem;border-radius:12px;font-size:.88rem;line-height:1.55;}
.msg.user .bubble{background:rgba(0,229,255,.12);border:1px solid rgba(0,229,255,.2);}
.msg.bot .bubble{background:var(--surface);border:1px solid var(--border);}
.sql-block{margin-top:.6rem;background:#080810;border:1px solid var(--border);border-radius:8px;overflow:hidden;}
.sql-label{display:flex;justify-content:space-between;align-items:center;font-family:'Space Mono',monospace;font-size:.6rem;color:var(--muted);padding:.35rem .7rem;border-bottom:1px solid var(--border);}
.sql-code{font-family:'Space Mono',monospace;font-size:.75rem;padding:.7rem;color:#50fa7b;white-space:pre-wrap;}
.result-wrap{margin-top:.6rem;}
.result-header{display:flex;justify-content:space-between;align-items:center;margin-bottom:.4rem;}
.row-count{font-family:'Space Mono',monospace;font-size:.65rem;color:var(--muted);}
.export-btn{font-family:'Space Mono',monospace;font-size:.6rem;padding:.18rem .5rem;background:none;border:1px solid rgba(16,185,129,.3);color:var(--green);border-radius:4px;cursor:pointer;transition:all .2s;}
.export-btn:hover{background:rgba(16,185,129,.1);}
.result-table{overflow-x:auto;}
table{border-collapse:collapse;width:100%;font-size:.78rem;}
th{background:rgba(0,229,255,.08);color:var(--accent);font-weight:600;padding:.45rem .7rem;text-align:left;border-bottom:1px solid var(--border);font-family:'Space Mono',monospace;font-size:.68rem;}
td{padding:.4rem .7rem;border-bottom:1px solid rgba(255,255,255,.03);}
tr:last-child td{border-bottom:none;}
tr:hover td{background:rgba(255,255,255,.02);}
.error{color:#f87171;font-family:'Space Mono',monospace;font-size:.78rem;margin-top:.5rem;}

/*    Examples    */
.examples{padding:.6rem 1.25rem;border-top:1px solid var(--border);background:rgba(255,255,255,.01);flex-shrink:0;}
.ex-cats{display:flex;gap:.4rem;margin-bottom:.45rem;flex-wrap:wrap;}
.cat-btn{font-family:'Space Mono',monospace;font-size:.6rem;padding:.18rem .5rem;background:none;border:1px solid var(--border);color:var(--muted);border-radius:4px;cursor:pointer;transition:all .2s;}
.cat-btn.active,.cat-btn:hover{border-color:var(--accent2);color:var(--accent2);}
.ex-btns{display:flex;gap:.4rem;flex-wrap:wrap;}
.ex-btn{font-family:'Space Mono',monospace;font-size:.65rem;padding:.25rem .6rem;background:none;border:1px solid var(--border);color:var(--muted);border-radius:5px;cursor:pointer;transition:all .2s;white-space:nowrap;}
.ex-btn:hover{border-color:var(--accent2);color:var(--accent2);}

/*    Input    */
.input-row{display:flex;gap:.6rem;padding:.75rem 1.25rem;border-top:1px solid var(--border);background:var(--surface);flex-shrink:0;}
#question{flex:1;background:var(--bg);border:1px solid var(--border);color:var(--text);padding:.7rem .9rem;border-radius:8px;font-family:'Syne',sans-serif;font-size:.88rem;outline:none;transition:border-color .2s;}
#question:focus{border-color:var(--accent);}
#send-btn{background:var(--accent);color:#000;border:none;padding:.7rem 1.25rem;border-radius:8px;cursor:pointer;font-family:'Syne',sans-serif;font-weight:700;font-size:.88rem;transition:opacity .2s;white-space:nowrap;}
#send-btn:hover{opacity:.85;}
#send-btn:disabled{opacity:.4;cursor:not-allowed;}

/*    Typing    */
.typing{display:flex;gap:4px;align-items:center;padding:.4rem 0;}
.dot{width:5px;height:5px;background:var(--accent);border-radius:50%;animation:bounce 1s infinite;}
.dot:nth-child(2){animation-delay:.15s;}.dot:nth-child(3){animation-delay:.3s;}
@keyframes bounce{0%,80%,100%{transform:translateY(0)}40%{transform:translateY(-5px)}}

/*    Copy btn    */
.copy-sql{background:none;border:1px solid var(--border);color:var(--muted);padding:.12rem .4rem;border-radius:3px;cursor:pointer;font-family:'Space Mono',monospace;font-size:.55rem;transition:all .2s;}
.copy-sql:hover{border-color:var(--accent);color:var(--accent);}
</style>
</head>
<body>

<header>
  <h1>  Text-to-SQL</h1>
  <span class="badge badge-blue" id="backend-badge">{{ backend }}</span>
  <span class="badge badge-orange" id="db-badge">SQLite</span>
  <span class="badge badge-green" id="status-badge">  ready</span>
  <button class="hdr-btn" onclick="clearChat()">clear chat</button>
</header>

<div class="main">

  <!--    Sidebar    -->
  <div class="sidebar">
    <div class="sidebar-title">  History</div>
    <div class="history" id="history">
      <div class="hist-empty" id="hist-empty">No queries yet</div>
    </div>

    <!-- DB Connection Panel -->
    <div class="db-panel">
      <div class="db-panel-title">  Database</div>
      <div class="db-type-row">
        <button class="db-type-btn active" id="btn-sqlite" onclick="setDbType('sqlite')">SQLite</button>
        <button class="db-type-btn" id="btn-postgres" onclick="setDbType('postgres')">PostgreSQL</button>
      </div>

      <div id="sqlite-fields">
        <input class="db-input" id="sqlite-path" placeholder="Path: data/sample.db" value="data/sample.db">
      </div>

      <div id="postgres-fields" style="display:none">
        <input class="db-input" id="pg-host" placeholder="Host: localhost">
        <input class="db-input" id="pg-port" placeholder="Port: 5432">
        <input class="db-input" id="pg-db" placeholder="Database name">
        <input class="db-input" id="pg-user" placeholder="Username">
        <input class="db-input" id="pg-pass" type="password" placeholder="Password">
      </div>

      <button class="db-connect-btn" onclick="connectDb()">Connect -></button>
      <div class="db-status" id="db-status" style="color:var(--green)">OK Connected: SQLite</div>
    </div>
  </div>

  <!--    Chat    -->
  <div class="chat-area">
    <div class="chat" id="chat">
      <div class="msg bot">
        <div class="bubble">
           Hi! Ask anything about your database in plain English.<br>
          I'll generate SQL and show results instantly.<br><br>
          <span style="color:var(--muted);font-size:.82rem;">Try the example buttons below  </span>
        </div>
      </div>
    </div>

    <!-- Example questions by category -->
    <div class="examples">
      <div class="ex-cats">
        <span style="font-family:'Space Mono',monospace;font-size:.6rem;color:var(--muted);align-self:center;">Category:</span>
        <button class="cat-btn active" onclick="showCat('employees',this)">  Employees</button>
        <button class="cat-btn" onclick="showCat('sales',this)">  Sales</button>
        <button class="cat-btn" onclick="showCat('products',this)">  Products</button>
        <button class="cat-btn" onclick="showCat('analytics',this)">  Analytics</button>
      </div>
      <div class="ex-btns" id="ex-btns"></div>
    </div>

    <div class="input-row">
      <input id="question" type="text" placeholder="Ask a question about your data..." onkeydown="if(event.key==='Enter')sendQ()">
      <button id="send-btn" onclick="sendQ()">Ask -></button>
    </div>
  </div>

</div>

<script>
//    Example questions by category                                           
const EXAMPLES = {
  employees: [
    'Show all employees in Engineering',
    'Who has the highest salary?',
    'How many employees per department?',
    'Employees hired after 2021',
    'Show employees in Riyadh',
    'Average salary by city',
  ],
  sales: [
    'Top 3 salespeople by total amount',
    'Total sales per region',
    'Sales in the Gulf region',
    'Which product sold the most?',
    'Monthly sales in 2024',
    'Sales above 20000',
  ],
  products: [
    'Products with stock below 100',
    'Most expensive product',
    'Products in Software category',
    'Total stock value by category',
  ],
  analytics: [
    'Average salary vs total sales per employee',
    'Which department has the highest payroll?',
    'Top region by revenue',
    'Compare product prices',
  ]
};

let currentCat = 'employees';
let history = [];
let lastData = null;

function showCat(cat, btn) {
  currentCat = cat;
  document.querySelectorAll('.cat-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  renderExamples();
}

function renderExamples() {
  const qs = EXAMPLES[currentCat] || [];
  document.getElementById('ex-btns').innerHTML = qs.map(q =>
    `<button class="ex-btn" onclick="ask('${q.replace(/'/g,"\\'")}'">${q}</button>`
  ).join('');
}

renderExamples();

//    DB Type toggle                                                           
let dbType = 'sqlite';
function setDbType(type) {
  dbType = type;
  document.getElementById('btn-sqlite').classList.toggle('active', type === 'sqlite');
  document.getElementById('btn-postgres').classList.toggle('active', type === 'postgres');
  document.getElementById('sqlite-fields').style.display = type === 'sqlite' ? 'block' : 'none';
  document.getElementById('postgres-fields').style.display = type === 'postgres' ? 'block' : 'none';
}

async function connectDb() {
  const btn = document.querySelector('.db-connect-btn');
  const status = document.getElementById('db-status');
  btn.textContent = 'Connecting...';
  btn.disabled = true;

  let payload = { type: dbType };
  if (dbType === 'sqlite') {
    payload.path = document.getElementById('sqlite-path').value || 'data/sample.db';
  } else {
    payload.host = document.getElementById('pg-host').value;
    payload.port = document.getElementById('pg-port').value || '5432';
    payload.database = document.getElementById('pg-db').value;
    payload.user = document.getElementById('pg-user').value;
    payload.password = document.getElementById('pg-pass').value;
  }

  try {
    const res = await fetch('/connect', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload)
    });
    const data = await res.json();
    if (data.success) {
      status.style.color = 'var(--green)';
      status.textContent = 'OK Connected: ' + (dbType === 'sqlite' ? 'SQLite' : 'PostgreSQL');
      document.getElementById('db-badge').textContent = dbType === 'sqlite' ? 'SQLite' : 'PostgreSQL';
      document.getElementById('db-badge').className = 'badge ' + (dbType === 'sqlite' ? 'badge-orange' : 'badge-blue');
      addBotMsg(`  Connected to ${dbType === 'sqlite' ? 'SQLite: ' + payload.path : 'PostgreSQL: ' + payload.database}<br><small style="color:var(--muted)">${data.schema || ''}</small>`);
    } else {
      status.style.color = '#f87171';
      status.textContent = '  Failed';
      addBotMsg(`  Connection failed: ${data.error}`);
    }
  } catch(e) {
    status.style.color = '#f87171';
    status.textContent = '  Error';
  }

  btn.textContent = 'Connect ->';
  btn.disabled = false;
}

//    Chat                                                                     
function ask(q) {
  document.getElementById('question').value = q;
  sendQ();
}

async function sendQ() {
  const input = document.getElementById('question');
  const q = input.value.trim();
  if (!q) return;

  const chat = document.getElementById('chat');
  const btn = document.getElementById('send-btn');

  chat.innerHTML += `<div class="msg user"><div class="bubble">${esc(q)}</div></div>`;
  input.value = '';
  btn.disabled = true;
  document.getElementById('status-badge').textContent = '  thinking...';
  document.getElementById('status-badge').style.color = 'var(--accent2)';

  const tid = 't' + Date.now();
  chat.innerHTML += `<div class="msg bot" id="${tid}"><div class="bubble"><div class="typing"><div class="dot"></div><div class="dot"></div><div class="dot"></div></div></div></div>`;
  chat.scrollTop = chat.scrollHeight;

  try {
    const res = await fetch('/query', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({question: q})
    });
    const data = await res.json();
    document.getElementById(tid).remove();
    lastData = data;

    let html = '<div class="msg bot"><div class="bubble">';
    if (data.error) {
      html += `<span class="error">  ${esc(data.error)}</span>`;
    } else {
      const msgId = 'm' + Date.now();
      html += `<div class="sql-block">
        <div class="sql-label">
          <span>generated SQL</span>
          <button class="copy-sql" onclick="copySQL('${msgId}')">copy</button>
        </div>
        <div class="sql-code" id="${msgId}">${esc(data.sql)}</div>
      </div>`;
      if (data.rows && data.rows.length > 0) {
        const exportId = 'e' + Date.now();
        html += `<div class="result-wrap">
          <div class="result-header">
            <span class="row-count">${data.rows.length} row(s)</span>
            <button class="export-btn" onclick="exportCSV('${exportId}')">  CSV</button>
          </div>
          <div class="result-table" id="${exportId}">
            <table><thead><tr>`;
        data.columns.forEach(c => { html += `<th>${esc(c)}</th>`; });
        html += '</tr></thead><tbody>';
        data.rows.forEach(row => {
          html += '<tr>';
          row.forEach(cell => { html += `<td>${esc(String(cell ?? ''))}</td>`; });
          html += '</tr>';
        });
        html += '</tbody></table></div></div>';
      } else {
        html += '<div class="row-count" style="margin-top:.5rem">(no results)</div>';
      }
      addHistory(q);
    }
    html += '</div></div>';
    chat.innerHTML += html;
  } catch(e) {
    document.getElementById(tid)?.remove();
    addBotMsg('  Request failed   is the server running?');
  }

  chat.scrollTop = chat.scrollHeight;
  btn.disabled = false;
  document.getElementById('status-badge').textContent = '  ready';
  document.getElementById('status-badge').style.color = 'var(--green)';
  input.focus();
}

function addBotMsg(html) {
  const chat = document.getElementById('chat');
  chat.innerHTML += `<div class="msg bot"><div class="bubble">${html}</div></div>`;
  chat.scrollTop = chat.scrollHeight;
}

function addHistory(q) {
  history.unshift(q);
  if (history.length > 20) history.pop();
  const el = document.getElementById('history');
  document.getElementById('hist-empty')?.remove();
  const item = document.createElement('div');
  item.className = 'hist-item';
  item.textContent = q;
  item.onclick = () => ask(q);
  el.insertBefore(item, el.firstChild);
}

function clearChat() {
  document.getElementById('chat').innerHTML = '<div class="msg bot"><div class="bubble">Chat cleared! Ask a new question.</div></div>';
}

function copySQL(id) {
  const text = document.getElementById(id)?.innerText;
  if (text) navigator.clipboard.writeText(text);
}

function exportCSV(tableId) {
  const table = document.querySelector(`#${tableId} table`);
  if (!table) return;
  let csv = '';
  table.querySelectorAll('tr').forEach(row => {
    const cells = [...row.querySelectorAll('th,td')].map(c => `"${c.innerText}"`);
    csv += cells.join(',') + '\n';
  });
  const blob = new Blob([csv], {type: 'text/csv'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'query_results.csv';
  a.click();
}

function esc(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML, backend=LLM_BACKEND.upper())


@app.route("/connect", methods=["POST"])
def connect():
    data = request.get_json()
    db_type = data.get("type", "sqlite")

    if db_type == "sqlite":
        path = data.get("path", DB_PATH)
        if not Path(path).exists() and path != DB_PATH:
            return jsonify({"success": False, "error": f"File not found: {path}"})
        current_db["type"] = "sqlite"
        current_db["path"] = path
        schema = get_current_schema()
        return jsonify({"success": True, "schema": schema[:200] + "..."})

    elif db_type == "postgres":
        try:
            import psycopg2
            conn_str = f"host={data['host']} port={data.get('port','5432')} dbname={data['database']} user={data['user']} password={data['password']}"
            conn = psycopg2.connect(conn_str)
            conn.close()
            current_db["type"] = "postgres"
            current_db["conn_string"] = conn_str
            schema = get_current_schema()
            return jsonify({"success": True, "schema": schema[:200] + "..."})
        except ImportError:
            return jsonify({"success": False, "error": "Run: pip install psycopg2-binary"})
        except Exception as e:
            return jsonify({"success": False, "error": str(e)})

    return jsonify({"success": False, "error": "Unknown db type"})


@app.route("/query", methods=["POST"])
def query():
    data = request.get_json()
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "Empty question"})

    schema = get_current_schema()
    if schema.startswith("Schema error"):
        return jsonify({"error": schema})

    if LLM_BACKEND == "openai":
        raw = ask_openai(question, schema)
    elif LLM_BACKEND == "anthropic":
        raw = ask_anthropic(question, schema)
    else:
        raw = ask_ollama(question, schema)

    if raw.startswith("ERROR"):
        return jsonify({"error": raw})

    sql = extract_sql(raw)
    try:
        columns, rows = run_query(sql)
        return jsonify({"sql": sql, "columns": columns, "rows": rows})
    except Exception as e:
        return jsonify({"error": str(e), "sql": sql})


if __name__ == "__main__":
    setup_database()
    print("\n  Text-to-SQL v2   http://localhost:5000\n")
    print("  Features:")
    print("    16 example questions in 4 categories")
    print("    Connect SQLite (local) or PostgreSQL (cloud)")
    print("    Export results to CSV")
    print("    Query history sidebar")
    print()
    app.run(debug=True, port=5000)
