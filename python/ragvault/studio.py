"""RagVault Studio — optional local inspection UI.

    ragvault studio ./data

Serves a single-page app from the Python standard library (no extra
dependencies, no external requests): run queries, compare retrieval modes,
inspect scores per signal, the query plan, context blocks and citations.
Local-only by default (binds 127.0.0.1).
"""

from __future__ import annotations

import json
import threading
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .kb import KnowledgeBase

PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>RagVault Studio</title>
<style>
  :root { color-scheme: light dark; }
  body { font-family: system-ui, sans-serif; margin: 0; padding: 1.5rem;
         max-width: 1100px; margin-inline: auto; }
  h1 { font-size: 1.3rem; } h1 small { font-weight: normal; opacity: .6; }
  form { display: flex; gap: .5rem; flex-wrap: wrap; margin-bottom: 1rem; }
  input[type=text] { flex: 1; min-width: 280px; padding: .55rem .7rem;
                     font-size: 1rem; border-radius: 8px; border: 1px solid #8884; }
  select, button, input[type=number] { padding: .5rem .7rem; border-radius: 8px;
                     border: 1px solid #8884; font-size: .95rem; }
  button { cursor: pointer; font-weight: 600; }
  .chunk { border: 1px solid #8883; border-radius: 10px; padding: .8rem 1rem;
           margin: .6rem 0; }
  .chunk .meta { font-size: .8rem; opacity: .7; margin-bottom: .4rem; }
  .scores { display: flex; gap: .8rem; font-size: .8rem; margin-top: .4rem;
            opacity: .85; flex-wrap: wrap; }
  .scores span { border: 1px solid #8883; border-radius: 6px; padding: .1rem .45rem; }
  pre { background: #8881; padding: .8rem; border-radius: 8px; overflow-x: auto;
        font-size: .8rem; }
  .cols { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
  @media (max-width: 800px) { .cols { grid-template-columns: 1fr; } }
  .expanded { opacity: .75; border-style: dashed; }
  #stats { font-size: .85rem; opacity: .75; }
</style>
</head>
<body>
<h1>RagVault Studio <small id="stats"></small></h1>
<form id="f">
  <input type="text" id="q" placeholder="Ask your knowledge base..." autofocus>
  <select id="mode">
    <option value="">mode: preset default</option>
    <option value="hybrid">hybrid</option>
    <option value="dense">dense</option>
    <option value="keyword">keyword (BM25)</option>
  </select>
  <input type="number" id="k" value="8" min="1" max="50" title="k">
  <button>Retrieve</button>
</form>
<div class="cols">
  <div id="results"></div>
  <div>
    <h3>Plan</h3><pre id="plan">—</pre>
    <h3>Trace</h3><pre id="trace">—</pre>
    <h3>Citations</h3><pre id="citations">—</pre>
  </div>
</div>
<script>
async function refreshStats() {
  const s = await (await fetch('/api/stats')).json();
  document.getElementById('stats').textContent =
    `${s.documents} docs · ${s.live_chunks} chunks · dim ${s.dim} · ${s.embedding}`;
}
refreshStats();
document.getElementById('f').addEventListener('submit', async (e) => {
  e.preventDefault();
  const body = {
    query: document.getElementById('q').value,
    mode: document.getElementById('mode').value || null,
    k: parseInt(document.getElementById('k').value, 10),
  };
  const res = await fetch('/api/query', {method: 'POST',
    headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)});
  const data = await res.json();
  const out = document.getElementById('results');
  out.innerHTML = '';
  if (data.error) { out.textContent = 'Error: ' + data.error; return; }
  for (const c of data.chunks) {
    const div = document.createElement('div');
    div.className = 'chunk' + (c.expanded ? ' expanded' : '');
    const scores = [`fused ${c.score.toFixed(4)}`];
    if (c.dense_score != null) scores.push(`dense ${c.dense_score.toFixed(4)}`);
    if (c.bm25_score != null) scores.push(`bm25 ${c.bm25_score.toFixed(3)}`);
    if (c.expanded) scores.push('expanded neighbor');
    div.innerHTML = `
      <div class="meta">${c.document_id} · v${c.document_version} · chunk ${c.chunk_index}
        ${c.section_path.length ? '· ' + c.section_path.join(' › ') : ''}</div>
      <div>${c.text.replace(/</g, '&lt;')}</div>
      <div class="scores">${scores.map(s => `<span>${s}</span>`).join('')}</div>`;
    out.appendChild(div);
  }
  document.getElementById('plan').textContent = JSON.stringify(data.plan, null, 2);
  document.getElementById('trace').textContent = JSON.stringify(data.trace, null, 2);
  document.getElementById('citations').textContent =
    JSON.stringify(data.citations, null, 2);
});
</script>
</body>
</html>
"""


def build_handler(kb: "KnowledgeBase"):
    class StudioHandler(BaseHTTPRequestHandler):
        def log_message(self, *args) -> None:  # keep the terminal quiet
            pass

        def _send(self, code: int, body: bytes, content_type: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_json(self, payload: dict, code: int = 200) -> None:
            self._send(code, json.dumps(payload).encode(), "application/json")

        def do_GET(self) -> None:  # noqa: N802 (stdlib API)
            if self.path in ("/", "/index.html"):
                self._send(200, PAGE.encode(), "text/html; charset=utf-8")
            elif self.path == "/api/stats":
                self._send_json(kb.stats())
            else:
                self._send_json({"error": "not found"}, 404)

        def do_POST(self) -> None:  # noqa: N802 (stdlib API)
            if self.path != "/api/query":
                self._send_json({"error": "not found"}, 404)
                return
            length = int(self.headers.get("Content-Length", "0"))
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
                result = kb.retrieve(
                    str(body.get("query", "")),
                    k=int(body.get("k") or 8),
                    mode=body.get("mode") or None,
                    explain=True,
                    trace=True,
                )
                self._send_json({
                    "chunks": [asdict(chunk) for chunk in result.chunks],
                    "citations": [citation.to_dict() for citation in result.citations],
                    "plan": result.plan,
                    "trace": result.trace,
                    "token_count": result.token_count,
                })
            except Exception as exc:
                self._send_json({"error": str(exc)}, 400)

    return StudioHandler


def serve(kb: "KnowledgeBase", host: str = "127.0.0.1", port: int = 7644,
          open_browser: bool = True) -> ThreadingHTTPServer:
    """Start the Studio server (blocking unless you call it via `start`)."""
    server = ThreadingHTTPServer((host, port), build_handler(kb))
    if open_browser:  # pragma: no cover - depends on environment
        import webbrowser

        threading.Timer(
            0.3, lambda: webbrowser.open(f"http://{host}:{server.server_port}/")
        ).start()
    return server
