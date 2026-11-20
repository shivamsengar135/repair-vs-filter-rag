"""Build labelling/label_tool.html: an offline, single-file page for the 150 human labels (EXPERIMENT_PLAN.md §7).

The page embeds ONLY what the labeller may see (row, topic, claim, evidence) — never the judge's label or the source
system (those stay in labelling/.key_DO_NOT_OPEN.json). Labels autosave in the browser (localStorage) and are
exported as a CSV with exactly the columns of labelling/claims_to_label.csv, which scripts/score_labelling.py reads.
Usage: python3 scripts/make_label_tool.py
"""
import csv
import json
from pathlib import Path

rows = list(csv.DictReader(Path("labelling/claims_to_label.csv").open(encoding="utf-8")))
data = [{"row": int(r["row"]), "topic": r["topic"], "claim": r["claim"], "evidence": r["evidence"]} for r in rows]
assert len(data) == 150 and not any("judge" in k for r in data for k in r)
payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")

HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Claim labelling</title>
<style>
:root{--bg:#f6f7f9;--card:#fff;--ink:#1d2330;--muted:#5d6675;--line:#d9dde4;--accent:#2457c5;
 --s:#1f7a45;--ns:#b3261e;--u:#8a6d00;--hl:#fff1a8}
@media (prefers-color-scheme:dark){:root{--bg:#15181e;--card:#1e222a;--ink:#e6e9ef;--muted:#9aa3b2;--line:#323844;
 --accent:#7aa2ff;--s:#5cc98a;--ns:#ff8a80;--u:#e3c35a;--hl:#5a4b00}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.5 system-ui,Segoe UI,sans-serif}
header{position:sticky;top:0;z-index:2;background:var(--card);border-bottom:1px solid var(--line);padding:10px 16px;
 display:flex;flex-wrap:wrap;gap:10px;align-items:center}
header h1{font-size:17px;margin:0 12px 0 0}.bar{flex:1;min-width:160px;height:8px;background:var(--line);border-radius:4px;overflow:hidden}
.bar i{display:block;height:100%;background:var(--accent);width:0}
button{font:inherit;border:1px solid var(--line);background:var(--card);color:var(--ink);border-radius:6px;padding:6px 12px;cursor:pointer}
button:hover{border-color:var(--accent)}button.primary{background:var(--accent);color:#fff;border-color:var(--accent)}
main{max-width:1100px;margin:0 auto;padding:16px;display:grid;gap:16px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px}
.topic{color:var(--muted);font-size:14px}.claim{font-size:21px;font-weight:600;margin:6px 0 0}
.labels{display:flex;gap:10px;flex-wrap:wrap;margin-top:14px}
.labels button{flex:1;min-width:150px;padding:12px;font-weight:600;text-align:left}
.labels small{display:block;font-weight:400;color:var(--muted)}
.labels button.on[data-l=S]{border:2px solid var(--s);color:var(--s)}.labels button.on[data-l=NS]{border:2px solid var(--ns);color:var(--ns)}
.labels button.on[data-l=U]{border:2px solid var(--u);color:var(--u)}
textarea{width:100%;font:inherit;padding:8px;border:1px solid var(--line);border-radius:6px;background:var(--card);color:var(--ink);margin-top:10px}
.nav{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-top:12px}.nav .sp{flex:1}
.ev p{margin:0 0 12px;white-space:pre-wrap}.ev b{color:var(--muted)}mark{background:var(--hl);color:inherit;padding:0 1px}
details{font-size:15px}details li{margin:3px 0}.muted{color:var(--muted);font-size:14px}
.grid{display:flex;flex-wrap:wrap;gap:4px;margin-top:8px}.grid a{width:30px;text-align:center;font-size:12px;border:1px solid var(--line);
 border-radius:4px;padding:2px 0;cursor:pointer;color:var(--ink);text-decoration:none}.grid a.S{background:var(--s);color:#fff}
.grid a.NS{background:var(--ns);color:#fff}.grid a.U{background:var(--u);color:#000}.grid a.cur{outline:2px solid var(--accent)}
kbd{border:1px solid var(--line);border-radius:3px;padding:0 4px;font-size:13px}
</style></head><body>
<header><h1>Claim labelling</h1><div class="bar"><i id="bar"></i></div><span id="count"></span>
<button id="exp" class="primary">Download CSV</button><button id="imp">Load CSV</button>
<input type="file" id="file" accept=".csv" hidden></header>
<main>
<details class="card"><summary><b>Rules</b> (open while you label; same as docs/LABELLING_GUIDE.md)</summary><ul>
<li><b>S — Supported:</b> the evidence states it, or it follows from the evidence without outside knowledge.</li>
<li><b>NS — Not supported:</b> the evidence contradicts it <b>or does not contain it</b>. True in reality but absent from the evidence = NS.</li>
<li><b>U — Unclear:</b> only if the statement itself cannot be judged (not factual, garbled, unclear who/what). Use rarely.</li>
<li>Judge against the evidence only — no own knowledge, no internet.</li>
<li>Every part must be supported (place <i>and</i> year, etc.). One part missing or wrong = NS.</li>
<li>Paraphrase is fine. Numbers and dates must match. Shortened/fuller name of the same person is fine.</li>
<li>Label the statement as written; do not fix it. Torn between S and NS: choose NS and write a note.</li>
</ul><p class="muted">Keys: <kbd>1</kbd> S, <kbd>2</kbd> NS, <kbd>3</kbd> U, <kbd>&rarr;</kbd>/<kbd>&larr;</kbd> next/previous,
<kbd>n</kbd> next unlabelled. Highlighted words are just words shared with the statement, to help you find the right
place; they say nothing about the answer.</p></details>
<section class="card"><div class="topic" id="topic"></div><p class="claim" id="claim"></p>
<div class="labels">
<button data-l="S">S — Supported<small>evidence states it / it follows</small></button>
<button data-l="NS">NS — Not supported<small>contradicted or not in the evidence</small></button>
<button data-l="U">U — Unclear<small>statement cannot be judged</small></button></div>
<textarea id="note" rows="2" placeholder="Note (optional): anything you were unsure about"></textarea>
<div class="nav"><button id="prev">&larr; Previous</button><span id="pos" class="muted"></span><span class="sp"></span>
<label class="muted"><input type="checkbox" id="hl" checked> highlight shared words</label>
<button id="nextu">Next unlabelled</button><button id="next">Next &rarr;</button></div>
<div class="grid" id="grid"></div></section>
<section class="card ev"><b>Evidence</b><div id="ev"></div></section>
<p class="muted">Your labels are saved automatically in this browser on this computer. When all 150 are done, click
<b>Download CSV</b> and give the file to the assistant. Use <b>Download CSV</b> now and then as a backup; <b>Load CSV</b>
restores from such a file (for example on another computer).</p>
</main>
<script id="data" type="application/json">__DATA__</script>
<script>
const D = JSON.parse(document.getElementById('data').textContent);
const KEY = 'claimLabels_v1';
let st = {}; try { st = JSON.parse(localStorage.getItem(KEY) || '{}'); } catch (e) {}
let i = 0; try { i = Math.min(D.length - 1, Number(localStorage.getItem(KEY + '_pos') || 0)); } catch (e) {}
const $ = id => document.getElementById(id);
const save = () => { try { localStorage.setItem(KEY, JSON.stringify(st)); localStorage.setItem(KEY + '_pos', i); } catch (e) {} };
const esc = s => s.replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
const STOP = new Set('the a an and or of in on at to for by with was is are were be been as from that this it its his her their he she they which who'.split(' '));
function highlight(text, claim) {
  if (!$('hl').checked) return esc(text);
  const words = [...new Set((claim.match(/[\p{L}\p{N}][\p{L}\p{N}'.-]*/gu) || []).map(w => w.replace(/[.'-]+$/, ''))
    .filter(w => w.length > 2 && !STOP.has(w.toLowerCase())))];
  if (!words.length) return esc(text);
  const re = new RegExp('(' + words.map(w => w.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('|') + ')', 'giu');
  return esc(text).replace(re, '<mark>$1</mark>');
}
function render() {
  const r = D[i], s = st[r.row] || {};
  $('topic').textContent = 'Topic: ' + r.topic; $('claim').textContent = r.claim;
  document.querySelectorAll('.labels button').forEach(b => b.classList.toggle('on', b.dataset.l === s.label));
  $('note').value = s.note || '';
  $('ev').innerHTML = r.evidence.split(/\n\s*\n/).map(p => '<p>' + highlight(p.trim(), r.claim) + '</p>').join('');
  const done = D.filter(x => st[x.row] && st[x.row].label).length;
  $('count').textContent = done + ' / ' + D.length + ' labelled'; $('bar').style.width = (100 * done / D.length) + '%';
  $('pos').textContent = 'Statement ' + (i + 1) + ' of ' + D.length;
  $('grid').innerHTML = D.map((x, k) => '<a data-k="' + k + '" class="' + ((st[x.row] || {}).label || '') + (k === i ? ' cur' : '') + '">' + x.row + '</a>').join('');
  save();
}
function setLabel(l) { const r = D[i]; st[r.row] = Object.assign(st[r.row] || {}, {label: l}); render(); }
function go(k) { i = Math.max(0, Math.min(D.length - 1, k)); render(); window.scrollTo(0, 0); }
function nextUnlabelled() { for (let d = 1; d <= D.length; d++) { const k = (i + d) % D.length; if (!(st[D[k].row] || {}).label) return go(k); } }
document.querySelectorAll('.labels button').forEach(b => b.onclick = () => setLabel(b.dataset.l));
$('note').oninput = e => { const r = D[i]; st[r.row] = Object.assign(st[r.row] || {}, {note: e.target.value}); save(); };
$('prev').onclick = () => go(i - 1); $('next').onclick = () => go(i + 1); $('nextu').onclick = nextUnlabelled;
$('hl').onchange = render; $('grid').onclick = e => { if (e.target.dataset.k) go(Number(e.target.dataset.k)); };
document.addEventListener('keydown', e => {
  if (e.target.tagName === 'TEXTAREA') return;
  if (e.key === '1') setLabel('S'); else if (e.key === '2') setLabel('NS'); else if (e.key === '3') setLabel('U');
  else if (e.key === 'ArrowRight') go(i + 1); else if (e.key === 'ArrowLeft') go(i - 1); else if (e.key === 'n') nextUnlabelled();
});
const q = s => '"' + String(s).replace(/"/g, '""') + '"';
$('exp').onclick = () => {
  const lines = ['row,topic,claim,evidence,label,note'].concat(D.map(r => {
    const s = st[r.row] || {}; return [r.row, q(r.topic), q(r.claim), q(r.evidence), s.label || '', q(s.note || '')].join(','); }));
  const missing = D.filter(r => !(st[r.row] || {}).label).length;
  if (missing && !confirm(missing + ' statements are not labelled yet. Download anyway (as a backup)?')) return;
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([lines.join('\r\n') + '\r\n'], {type: 'text/csv;charset=utf-8'}));
  a.download = 'claims_to_label.csv'; a.click();
};
function parseCSV(t) {  // RFC 4180: quoted fields may contain commas, quotes ("") and newlines
  const out = []; let row = [], f = '', qd = false;
  for (let k = 0; k < t.length; k++) { const c = t[k];
    if (qd) { if (c === '"') { if (t[k + 1] === '"') { f += '"'; k++; } else qd = false; } else f += c; }
    else if (c === '"') qd = true; else if (c === ',') { row.push(f); f = ''; }
    else if (c === '\n' || c === '\r') { if (c === '\r' && t[k + 1] === '\n') k++; row.push(f); out.push(row); row = []; f = ''; }
    else f += c; }
  if (f || row.length) { row.push(f); out.push(row); } return out;
}
$('imp').onclick = () => $('file').click();
$('file').onchange = async e => {
  const t = (await e.target.files[0].text()).replace(/^﻿/, ''); const rows = parseCSV(t); const h = rows[0];
  const ri = h.indexOf('row'), li = h.indexOf('label'), ni = h.indexOf('note'); let n = 0;
  rows.slice(1).forEach(r => { if (r.length > li && r[li]) { st[r[ri]] = {label: r[li].trim().toUpperCase(), note: r[ni] || ''}; n++; } });
  render(); alert('Loaded ' + n + ' labels.');
};
render();
</script></body></html>
"""
Path("labelling/label_tool.html").write_text(HTML.replace("__DATA__", payload), encoding="utf-8")
print("wrote labelling/label_tool.html with", len(data), "statements")
