// AegisFlow dashboard (React 18 + htm, no build step). Every number shown comes from the backend API;
// risk weights, level bands, speeds and the stage list are read from it, never hard-coded here.
"use strict";
const { useState, useEffect, useCallback, useRef, useMemo } = React;
const html = htm.bind(React.createElement);

// ------------------------------------------------------------------ helpers
const fmt = (x, d = 3) => x === null || x === undefined || Number.isNaN(+x) ? "–" : Number(x).toFixed(d);
const t = iso => iso ? String(iso).replace("T", " ").slice(0, 19) : "–";
const hm = iso => t(iso).slice(11);
async function api(path, opts) {
  const r = await fetch(path, opts);
  if (!r.ok) {
    let msg = r.status;
    try { msg = (await r.json()).detail || msg; } catch (_) { /* not JSON */ }
    const e = new Error(msg); e.status = r.status; throw e;
  }
  return (r.headers.get("content-type") || "").includes("json") ? r.json() : r.text();
}
const post = (path, body) => api(path, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body || {}) });
const storage = {
  get(k, d) { try { return localStorage.getItem(k) ?? d; } catch (_) { return d; } },
  set(k, v) { try { localStorage.setItem(k, v); } catch (_) { /* private mode */ } },
};
function levelOf(risk, cfg) {
  if (!cfg || risk === null || risk === undefined) return "low";
  return risk < cfg.thresholds.low ? "low" : risk < cfg.thresholds.medium ? "medium" : "high";
}
const STATUS_LABEL = { open: "open", acknowledged: "acknowledged", response_approved: "response approved",
                       dismissed: "dismissed", reopened: "reopened" };

// Load once per key; re-run when deps change. Returns {data, error, loading, reload}.
function useLoad(fn, deps) {
  const [state, setState] = useState({ data: null, error: null, loading: true });
  const [n, setN] = useState(0);
  useEffect(() => {
    let live = true;
    setState(s => ({ ...s, loading: true, error: null }));
    fn().then(data => live && setState({ data, error: null, loading: false }))
        .catch(error => live && setState({ data: null, error, loading: false }));
    return () => { live = false; };
  }, [...deps, n]);
  return { ...state, reload: () => setN(x => x + 1) };
}

// ------------------------------------------------------------------ small charts
function Bars({ items, label, value, cls = "acc", max }) {
  const m = max ?? Math.max(1e-9, ...items.map(value));
  return html`<div>${items.map((it, i) => html`<div class="hbar" key=${i}><div class="mono">${label(it)}</div>
    <div class="track"><div class=${"fill " + cls} style=${{ left: 0, width: (value(it) / m * 100) + "%" }}></div></div>
    <div class="small" style=${{ textAlign: "right" }}>${fmt(value(it), 3)}</div></div>`)}</div>`;
}
function StepBars({ values, signed, title }) {
  const max = Math.max(1e-9, ...values.map(v => Math.abs(v)));
  const label = i => i === values.length - 1 ? "last" : `-${values.length - 1 - i}`;
  // Signed: bars grow up (towards attack) or down (towards benign) from a zero line; each side gets height in proportion to its largest bar.
  const up = Math.max(0, ...values), down = Math.max(0, ...values.map(v => -v)), tot = Math.max(1e-9, up + down);
  if (signed) return html`<div class="steps signed" role="img" aria-label=${title}>${values.map((v, i) => html`<div key=${i} title=${`window t-${values.length - 1 - i}: ${fmt(v, 4)}`}>
      <div class="half up" style=${{ flex: Math.max(up / tot, .08) }}>${v > 0 && html`<span class="pos" style=${{ height: (v / Math.max(up, 1e-9) * 100) + "%" }}></span>`}</div>
      <div class="half down" style=${{ flex: Math.max(down / tot, .08) }}>${v < 0 && html`<span class="neg" style=${{ height: (-v / Math.max(down, 1e-9) * 100) + "%" }}></span>`}</div>
      <div class="lab">${label(i)}</div></div>`)}</div>`;
  return html`<div><div class="steps" role="img" aria-label=${title}>${values.map((v, i) => html`<div key=${i} title=${`window t-${values.length - 1 - i}: ${fmt(v, 4)}`}>
      <span class="acc" style=${{ height: (Math.abs(v) / max * 100) + "%" }}></span>
      <div>${label(i)}</div></div>`)}</div></div>`;
}
function Sparkline({ tl, cfg }) {
  if (!tl || !tl.length || !cfg) return null;
  const W = 1200, H = 230, P = 30, n = tl.length, top = cfg.max_reachable_risk || 100;
  const x = i => P + (n === 1 ? 0 : i * (W - 2 * P) / (n - 1)), y = r => H - P - r / top * (H - 2 * P);
  const path = tl.map((p, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(p.risk).toFixed(1)}`).join("");
  return html`<svg viewBox=${`0 0 ${W} ${H}`} width="100%" role="img" aria-label="risk timeline">
    ${[0, cfg.thresholds.low, cfg.thresholds.medium].map(v => html`<g key=${v}><line x1=${P} x2=${W - P} y1=${y(v)} y2=${y(v)} stroke="#2b3642"/><text x="2" y=${y(v) + 3}>${v}</text></g>`)}
    <path d=${path} fill="none" stroke="#4aa3ff" stroke-width="1.5"/>
    ${tl.map((p, i) => p.alert_id ? html`<circle key=${i} cx=${x(i)} cy=${y(p.risk)} r="2.6" fill="#f85149"/>` : null)}
    <text x=${P} y=${H - 6}>${hm(tl[0].t)}</text><text x=${W - P - 40} y=${H - 6}>${hm(tl[n - 1].t)}</text></svg>`;
}

// ------------------------------------------------------------------ alert detail sections
function ExplanationSection({ seqId, feats }) {
  const [model, setModel] = useState("lstm");
  const [method, setMethod] = useState(feats.explanations.shap ? "shap" : "integrated_gradients");
  const ex = useLoad(() => api(`/explain/${encodeURIComponent(seqId)}?model=${model}&method=${method}&top=6`), [seqId, model, method]);
  const d = ex.data, max = d ? Math.max(1e-9, ...d.top_features.map(f => Math.abs(f.attribution))) : 1;
  const steps = d ? d.timestep_attribution : [];
  return html`<div class="section"><h3>Why this prediction (feature attribution)</h3>
    <div class="seg-row">
      <div class="segctl" role="group" aria-label="model"><span class="segctl-k">Model</span>
        <button class=${model === "lstm" ? "on" : ""} onClick=${() => setModel("lstm")}>LSTM</button>
        <button class=${model === "logistic_regression" ? "on" : ""} onClick=${() => setModel("logistic_regression")}>Logistic regression</button></div>
      ${model === "lstm" && html`<div class="segctl" role="group" aria-label="method"><span class="segctl-k">Method</span>
        <button class=${method === "shap" ? "on" : ""} disabled=${!feats.explanations.shap} title=${feats.explanations.hint || ""} onClick=${() => setMethod("shap")}>SHAP</button>
        <button class=${method === "integrated_gradients" ? "on" : ""} onClick=${() => setMethod("integrated_gradients")}>Integrated Gradients</button></div>`}
    </div>
    ${ex.loading ? html`<div class="small muted">Computing attributions…</div>`
      : ex.error ? html`<div class="err">${ex.error.message}</div>`
      : html`<div class="attr">
        <div class="attr-legend small"><span><i class="sw pos"></i>pushes towards attack</span><span><i class="sw neg"></i>pushes towards benign</span></div>
        <div class="attr-list">${d.top_features.map((f, i) => {
          const v = f.attribution, w = Math.abs(v) / max * 50;
          return html`<div class="attr-row" key=${f.feature}
              title=${`most influential at window ${f.most_influential_step}${f.last_window_value !== undefined ? `; last-window value ${f.last_window_value}` : ""}`}>
            <span class="attr-rank">${i + 1}</span>
            <span class="attr-name mono">${f.feature}</span>
            <span class="attr-track"><span class="attr-mid"></span>
              <span class=${"attr-fill " + (v >= 0 ? "pos" : "neg")} style=${{ left: v >= 0 ? "50%" : (50 - w) + "%", width: w + "%" }}></span></span>
            <span class=${"attr-val " + (v >= 0 ? "bad" : "ok")}>${v >= 0 ? "+" : "−"}${fmt(Math.abs(v), 3)}</span>
          </div>`; })}</div>
        <div class="facts" style=${{ marginTop: "12px" }}>
          <div class="fact"><div class="fact-k">Model output (log-odds)</div><div class="fact-v">${fmt(d.output, 3)}</div></div>
          <div class="fact"><div class="fact-k">Reference output</div><div class="fact-v">${fmt(d.reference_output, 3)}</div>
            <div class="small muted">${d.method === "integrated_gradients" || d.model !== "lstm" ? `additivity gap ${fmt(d.additivity_gap, 5)}` : "SHAP expected gradients, sampled"}</div></div>
        </div>
        <div class="small muted" style=${{ margin: "12px 0 4px" }}>Attribution per input window, oldest to last</div>
        <${StepBars} values=${steps} signed=${true} title="attribution per input window"/>
      </div>`}
  </div>`;
}

function AttentionSection({ seqId, models }) {
  const attn = models.filter(m => m.has_attention);
  const [name, setName] = useState(attn.length ? attn[0].name : null);
  const at = useLoad(() => name ? api(`/attention/${encodeURIComponent(seqId)}?model=${encodeURIComponent(name)}`) : Promise.resolve(null), [seqId, name]);
  return html`<div class="section"><h3>Attention over input windows</h3>
    ${!attn.length ? html`<div class="small muted">No attention model is trained on this machine. Train one with${" "}
        <code>python -m aegisflow train --model transformer</code> (or <code>attention_lstm</code>) and it appears here.</div>`
    : html`<div>
      <div class="row tabs" style=${{ marginBottom: "8px" }}>${attn.map(m => html`<button key=${m.name} class=${m.name === name ? "on" : ""} onClick=${() => setName(m.name)}>${m.name}</button>`)}</div>
      ${at.loading ? html`<div class="small muted">Loading…</div>` : at.error ? html`<div class="err">${at.error.message}</div>` : at.data && html`<div>
        <${StepBars} values=${at.data.window_weights} signed=${false} title="attention weight per input window"/>
        <div class="small muted" style=${{ marginTop: "6px" }}>${at.data.model_type} model on the same sequence: p = ${fmt(at.data.attack_probability, 4)}${" "}
          (${at.data.predicted_attack ? "would alert" : "would not alert"} at its threshold ${fmt(at.data.threshold, 4)}). Weights sum to 1.
          This is a separate model from the replayed LSTM; attention shows where it looked, not a causal attribution.</div></div>`}
    </div>`}
  </div>`;
}

function StageSection({ seqId, feats }) {
  const on = feats.stage_model.available;
  const fc = useLoad(() => on ? api(`/forecast/${encodeURIComponent(seqId)}`) : Promise.resolve(null), [seqId, on]);
  if (!on) return html`<div class="section"><h3>Attack stage and future state</h3>
    <div class="small muted">${feats.stage_model.configured ? `Stage model configured at ${feats.stage_model.path} but not trained yet. ` : "No stage model is configured, so the predicted stage is UNCERTAIN. "}
      To enable: <code>${feats.stage_model.hint}</code>.</div></div>`;
  if (fc.loading) return html`<div class="section"><h3>Attack stage and future state</h3><div class="small muted">Loading…</div></div>`;
  if (fc.error) return html`<div class="section"><h3>Attack stage and future state</h3><div class="err">${fc.error.message}</div></div>`;
  const d = fc.data, dist = Object.entries(d.stage_distribution).map(([s, p]) => ({ s, p })).sort((a, b) => b.p - a.p);
  const keys = ["flow_count", "unique_destination_ips", "unique_destination_ports", "syn_count_sum", "bytes_total"]
    .filter(k => d.future_state.length && k in d.future_state[0]);
  return html`<div class="section"><h3>Attack stage and future state (multi-task model)</h3>
    <div class="kv small">
      <div>Predicted stage</div><div><b>${d.predicted_stage}</b> <span class="muted">p = ${fmt(d.stage_probability, 3)}; shown only when ≥ the confidence cutoff</span></div>
      <div>MITRE ATT&CK</div><div>${d.mitre && d.mitre.tactic_id ? `${d.mitre.tactic_id} ${d.mitre.tactic_name} · ${d.mitre.techniques.map(x => `${x.id} ${x.name}`).join(", ")}` : "none"}</div>
      <div>Multi-task attack p</div><div>${fmt(d.attack_probability, 4)} <span class="muted">(threshold ${fmt(d.threshold, 4)})</span></div>
    </div>
    <div style=${{ marginTop: "8px" }}><${Bars} items=${dist} label=${x => x.s} value=${x => x.p} max=${1}/></div>
    ${keys.length > 0 && html`<div class="scroll" style=${{ maxHeight: "200px", marginTop: "8px" }}><table>
      <thead><tr><th>Forecast</th>${keys.map(k => html`<th key=${k} class="num">${k}</th>`)}</tr></thead>
      <tbody>${d.future_state.map(r => html`<tr key=${r.horizon}><td>t + ${r.horizon} window${r.horizon > 1 ? "s" : ""}</td>${keys.map(k => html`<td key=${k} class="num">${fmt(r[k], 1)}</td>`)}</tr>`)}</tbody>
    </table></div>`}
    <div class="small muted" style=${{ marginTop: "6px" }}>Model: ${d.model_dir}. Stage labels are proxies inferred from dataset attack names, not ground truth.</div>
  </div>`;
}

function AnalystSection({ alertId, stages, onChange }) {
  const hist = useLoad(() => api(`/alerts/${alertId}/actions`), [alertId]);
  const [who, setWho] = useState(storage.get("aegisflow.analyst", ""));
  const [note, setNote] = useState(""), [response, setResponse] = useState(""), [stage, setStage] = useState(stages[0] || "");
  const [err, setErr] = useState(null), [busy, setBusy] = useState(false);
  const act = async (action) => {
    setBusy(true); setErr(null);
    try {
      await post(`/alerts/${alertId}/actions`, { action, analyst: who, note: note || null,
        response: action === "approve_response" ? response : null, stage: action === "override_stage" ? stage : null });
      storage.set("aegisflow.analyst", who); setNote(""); hist.reload(); onChange && onChange();
    } catch (e) { setErr(e.message); }
    setBusy(false);
  };
  const h = hist.data, st = h ? h.status : "open";
  const can = { acknowledge: ["open", "reopened"], approve_response: ["open", "acknowledged", "reopened"],
                dismiss: ["open", "acknowledged", "reopened"], reopen: ["response_approved", "dismissed"] };
  const allowed = a => !can[a] || can[a].includes(st);
  const noWho = !who.trim(), whoHint = noWho ? "enter an analyst name first" : undefined;
  return html`<div class="section"><h3>Analyst review</h3>
    ${h && html`<div class="kv small" style=${{ marginBottom: "8px" }}>
      <div>Status</div><div><span class=${"status " + st}>${STATUS_LABEL[st] || st}</span></div>
      <div>Stage</div><div>model: ${h.model_stage || "–"} · effective: <b>${h.effective_stage || "–"}</b></div>
      ${h.approved_response && html`<div>Approved response</div><div>${h.approved_response} <span class="muted">(recorded only; nothing is executed)</span></div>`}
    </div>`}
    <div class="row" style=${{ marginBottom: "6px" }}>
      <input type="text" placeholder="Analyst name" value=${who} onInput=${e => setWho(e.target.value)} style=${{ width: "150px" }}/>
      <input type="text" placeholder="Note (optional; needed to comment)" value=${note} onInput=${e => setNote(e.target.value)} style=${{ flex: 1, minWidth: "160px" }}/>
    </div>
    <div class="row" style=${{ marginBottom: "6px" }}>
      <button disabled=${busy || noWho || !allowed("acknowledge")} title=${whoHint} onClick=${() => act("acknowledge")}>Acknowledge</button>
      <button disabled=${busy || noWho || !allowed("dismiss")} title=${whoHint} onClick=${() => act("dismiss")}>Dismiss</button>
      <button disabled=${busy || noWho || !allowed("reopen")} title=${whoHint} onClick=${() => act("reopen")}>Reopen</button>
      <button disabled=${busy || noWho || !note.trim()} title=${whoHint} onClick=${() => act("comment")}>Comment</button>
    </div>
    <div class="row" style=${{ marginBottom: "6px" }}>
      <input type="text" placeholder="Response to approve, e.g. block 172.16.0.1 at the edge" value=${response} onInput=${e => setResponse(e.target.value)} style=${{ flex: 1, minWidth: "200px" }}/>
      <button class="primary" title=${whoHint} disabled=${busy || noWho || !allowed("approve_response") || !response.trim()} onClick=${() => act("approve_response")}>Approve response</button>
    </div>
    <div class="row">
      <select value=${stage} onChange=${e => setStage(e.target.value)}>${stages.map(s => html`<option key=${s} value=${s}>${s}</option>`)}</select>
      <button disabled=${busy || noWho} title=${whoHint} onClick=${() => act("override_stage")}>Override stage</button>
    </div>
    ${err && html`<div class="err" style=${{ marginTop: "6px" }}>${err}</div>`}
    ${h && h.actions.length > 0 && html`<div class="scroll" style=${{ maxHeight: "180px", marginTop: "8px" }}><table>
      <thead><tr><th>When</th><th>Analyst</th><th>Action</th><th>Detail</th><th>hash</th></tr></thead>
      <tbody>${h.actions.slice().reverse().map(a => html`<tr key=${a.id} style=${{ cursor: "default" }}><td class="mono">${t(a.created_at)}</td><td>${a.analyst}</td>
        <td>${a.action}</td><td class="small">${a.stage || a.response || a.note || ""}</td><td class="mono">${a.hash.slice(0, 10)}…</td></tr>`)}</tbody></table></div>`}
    <div class="small muted" style=${{ marginTop: "6px" }}>Every action is appended to its own SHA-256 hash chain, bound to the alert's ledger hash.</div>
  </div>`;
}

// ------------------------------------------------------------------ alert presentation helpers
const COMPONENT_LABEL = { abnormality_score: "Abnormality", attack_probability: "Attack probability",
  prediction_confidence: "Prediction confidence", recent_attack_history: "Recent attack history", stage_severity: "Stage severity" };
const LEVEL_LABEL = { low: "Low", medium: "Medium", high: "High" };

async function copyText(text) {
  try { await navigator.clipboard.writeText(text); return true; } catch (_) { /* insecure origin: fall back */ }
  const ta = document.createElement("textarea");
  ta.value = text; ta.style.position = "fixed"; ta.style.opacity = "0";
  document.body.appendChild(ta); ta.select();
  let ok = false;
  try { ok = document.execCommand("copy"); } catch (_) { /* unsupported */ }
  ta.remove();
  return ok;
}

// A long hash or id, shortened in place; the full value is in the tooltip and on the copy button.
function Hash({ value, head = 8, tail = 6 }) {
  const [copied, setCopied] = useState(false);
  if (value === null || value === undefined || value === "") return html`<span class="muted">–</span>`;
  const s = String(value), short = s.length > head + tail + 1 ? `${s.slice(0, head)}…${s.slice(-tail)}` : s;
  const copy = async e => {
    e.stopPropagation();
    if (await copyText(s)) { setCopied(true); setTimeout(() => setCopied(false), 1200); }
  };
  return html`<span class="hash" title=${s}><span class="mono">${short}</span>
    <button type="button" class="copy" onClick=${copy} aria-label=${"Copy " + s}>${copied ? "copied" : "copy"}</button></span>`;
}

function Chip({ label, value, tone = "", title }) {
  return html`<span class=${"tag " + tone} title=${title || ""}><span class="tag-k">${label}</span><span class="tag-v">${value}</span></span>`;
}

// 0-100 bar with the level bands from /risk/config, the score marker and the part the current setup cannot reach.
function RiskGauge({ score, cfg }) {
  const lvl = levelOf(score, cfg), lo = cfg.thresholds.low, md = cfg.thresholds.medium;
  const top = cfg.max_reachable_risk, pos = Math.max(0, Math.min(100, score));
  return html`<div class="gauge" role="img" aria-label=${`risk ${fmt(score, 1)} of 100, ${lvl}`}>
    <div class="gauge-track">
      <div class="band low" style=${{ width: lo + "%" }}></div>
      <div class="band medium" style=${{ width: (md - lo) + "%" }}></div>
      <div class="band high" style=${{ width: (100 - md) + "%" }}></div>
      ${top < 100 && html`<div class="unreach" style=${{ left: top + "%", width: (100 - top) + "%" }} title=${`Above ${fmt(top, 0)} is not reachable with the current setup`}></div>`}
      <div class=${"gauge-fill lvl-bg-" + lvl} style=${{ width: pos + "%" }}></div>
      <div class="gauge-mark" style=${{ left: pos + "%" }}></div>
    </div>
    <div class="gauge-scale small muted">
      <span style=${{ left: "0%" }}>0</span><span style=${{ left: lo + "%" }}>${fmt(lo, 0)}</span>
      <span style=${{ left: md + "%" }}>${fmt(md, 0)}</span><span style=${{ left: "100%" }}>100</span>
    </div>
  </div>`;
}

// Each component's share of the score: value × weight × 100 points (the risk formula in replay.py).
function RiskBreakdown({ comp, cfg }) {
  const rows = Object.entries(comp).map(([k, v]) => ({ k, v: +v, w: +cfg.weights[k], pts: 100 * v * cfg.weights[k] }));
  return html`<div>
    <div class="stack" role="img" aria-label="risk score contributions">${rows.filter(r => r.pts > 0).map((r, i) =>
      html`<div key=${r.k} class=${"seg c" + Object.keys(comp).indexOf(r.k)} style=${{ width: r.pts + "%" }} title=${`${COMPONENT_LABEL[r.k] || r.k}: ${fmt(r.pts, 1)} pts`}></div>`)}</div>
    <table class="comp">
      <thead><tr><th>Component</th><th class="num">Value</th><th class="num">Weight</th><th class="num">Points</th></tr></thead>
      <tbody>${rows.map(r => html`<tr key=${r.k} class=${r.pts > 0 ? "" : "zero"}>
        <td title=${r.k}><span class=${"dot c" + Object.keys(comp).indexOf(r.k)}></span>${COMPONENT_LABEL[r.k] || r.k}</td>
        <td class="num">${fmt(r.v, 3)}</td><td class="num">× ${r.w}</td><td class="num"><b>${fmt(r.pts, 1)}</b></td></tr>`)}</tbody>
    </table>
  </div>`;
}

// ------------------------------------------------------------------ cards
function AuditCard({ alerts }) {
  const [v, setV] = useState(null), [va, setVa] = useState(null), [busy, setBusy] = useState(false);
  const check = async () => {
    setBusy(true);
    try { setV(await api("/audit/verify")); setVa(await api("/audit/verify-actions")); } finally { setBusy(false); }
  };
  // unit: "record" for alerts, "action" for analyst actions. The id prefix is skipped when the reason
  // already names it (head-anchor problems point at a record that may no longer exist).
  const detail = (r, what, unit) => {
    if (r.status === "VERIFIED") return `${r[what].toLocaleString()} ${unit}${r[what] === 1 ? "" : "s"}`;
    const id = r.record_id ?? r.action_id;
    return (id != null && !r.reason.includes(`${unit} ${id}`) ? `${unit} #${id}: ` : "") + r.reason;
  };
  const line = (r, what, unit, name) => html`<div class=${"chain " + (!r ? "" : r.status === "VERIFIED" ? "ok" : "bad")}>
    <div class="chain-k">${name}</div>
    ${!r ? html`<div class="chain-v muted">not checked</div>`
      : html`<div class="chain-v"><span class="chain-s">${r.status === "VERIFIED" ? "✓ " : "✗ "}${r.status}</span>
        <span class="small muted">${detail(r, what, unit)}</span></div>`}
  </div>`;
  return html`<section class="card span12 audit"><h2>Verify the chains</h2>
    <div class="row" style=${{ marginBottom: "10px" }}><button class="primary" disabled=${busy} onClick=${check}>${busy ? "Verifying…" : "Verify chains"}</button>
      <span class="small muted">Recomputes every hash from the stored fields.</span></div>
    <div class="chains">${line(v, "records_checked", "record", "Alerts chain")}${line(va, "actions_checked", "action", "Analyst actions")}</div>
    ${v && v.head_hash && html`<div class="small muted" style=${{ margin: "6px 0 0" }}>Head hash <${Hash} value=${v.head_hash}/> · anchor: ${v.anchor}</div>`}
    ${v && html`<div class="small muted" style=${{ margin: "4px 0 0" }}>The chain spans every replay session in the ledger; the table below lists the current session.</div>`}
    <div class="scroll" style=${{ maxHeight: "460px", marginTop: "10px" }}><table class="ledger">
      <thead><tr><th>#</th><th>Host</th><th>Previous hash → hash</th></tr></thead>
      <tbody>${alerts.length ? alerts.map(a => html`<tr key=${a.id} style=${{ cursor: "default" }}><td>${a.id}</td><td class="mono">${a.host_id}</td>
        <td class="links"><${Hash} value=${a.prev_hash} head=${6} tail=${4}/><span class="arrow">→</span><${Hash} value=${a.hash} head=${6} tail=${4}/></td></tr>`)
        : html`<tr><td colspan="3" class="muted">No records yet.</td></tr>`}</tbody></table></div>
    <div class="small muted" style=${{ marginTop: "6px" }}>Each record stores the previous record's hash, so editing any row breaks every hash after it. Hover a hash for the full value.</div>
  </section>`;
}

function CaptureCard({ feats }) {
  const [file, setFile] = useState(null), [kind, setKind] = useState("auto");
  const [busy, setBusy] = useState(false), [res, setRes] = useState(null), [err, setErr] = useState(null);
  const [status, setStatus] = useState(null), [rows, setRows] = useState([]);
  const refresh = useCallback(async () => {
    try { setStatus(await api("/stream/status")); setRows(await api("/stream/results?limit=50")); } catch (e) { setErr(e.message); }
  }, []);
  useEffect(() => { refresh(); const h = setInterval(refresh, 3000); return () => clearInterval(h); }, [refresh]);
  const upload = async () => {
    if (!file) return;
    setBusy(true); setErr(null); setRes(null);
    const fd = new FormData(); fd.append("file", file);
    try { setRes(await api(`/ingest/upload?kind=${kind}&reset=true`, { method: "POST", body: fd })); refresh(); }
    catch (e) { setErr(e.message); }
    setBusy(false);
  };
  return html`<section class="card span7"><h2>Live capture scoring (PCAP / NetFlow → streaming)</h2>
    <div class="row">
      <input type="file" accept=".pcap,.pcapng,.cap,.csv" onChange=${e => setFile(e.target.files[0] || null)}/>
      <select value=${kind} onChange=${e => setKind(e.target.value)}>
        <option value="auto">auto (.csv = nfdump, else packets)</option>
        <option value="pcap">packet capture</option>
        <option value="netflow">NetFlow v5/v9/IPFIX export or nfdump CSV</option>
      </select>
      <button class="primary" disabled=${!file || busy} onClick=${upload}>${busy ? "Scoring…" : "Upload and score"}</button>
      <button disabled=${busy} onClick=${async () => { await post("/stream/reset"); setRes(null); refresh(); }}>Reset stream</button>
    </div>
    <div class="small muted" style=${{ marginTop: "6px" }}>Flows are built from the file (CICFlowMeter conventions, plus TTL and TCP retransmissions for packets),
      windowed per host with the same 60 s / 30 s settings as training, and scored by ${feats.stream.model_dir} as each window closes.
      Results are not written to the alert ledger.${feats.pcap.scapy ? "" : " Scapy is not installed, so packet captures cannot be read."}</div>
    ${err && html`<div class="err" style=${{ marginTop: "6px" }}>${err}</div>`}
    ${res && html`<div class="note info" style=${{ marginTop: "8px" }}>${res.file}: ${res.flows.toLocaleString()} flows from ${res.source_hosts} source hosts →${" "}
      ${res.sequences_scored} host sequences scored, ${res.alerts} above threshold.</div>`}
    ${status && html`<div class="kv small" style=${{ marginTop: "8px", gridTemplateColumns: "160px 1fr" }}>
      <div>Flows seen / late</div><div>${status.flows_seen.toLocaleString()} / ${status.late_flows}</div>
      <div>Windows closed</div><div>${status.windows_closed} · hosts tracked ${status.hosts_tracked}</div>
      <div>Sequences scored</div><div>${status.sequences_scored} · alerts ${status.alerts} (threshold ${fmt(status.threshold, 4)})</div>
      <div>Watermark</div><div class="mono">${t(status.watermark)}</div></div>`}
    <div class="scroll" style=${{ maxHeight: "220px", marginTop: "8px" }}><table>
      <thead><tr><th>Host</th><th>Sequence end</th><th class="num">Attack p</th><th>Alert</th></tr></thead>
      <tbody>${rows.length ? rows.map((r, i) => html`<tr key=${i} style=${{ cursor: "default" }}><td class="mono">${r.host_id}</td><td class="mono">${t(r.seq_end_time)}</td>
        <td class="num">${fmt(r.attack_probability, 4)}</td><td>${r.predicted_attack ? html`<span class="bad">yes</span>` : "no"}</td></tr>`)
        : html`<tr><td colspan="4" class="muted">Nothing scored yet. Upload a capture, or POST flows to /stream/flows.</td></tr>`}</tbody></table></div>
  </section>`;
}

function SiemCard() {
  const [format, setFormat] = useState("cef"), [since, setSince] = useState(0), [preview, setPreview] = useState(null), [err, setErr] = useState(null);
  const url = `/export/alerts?format=${format}&since_id=${since}&limit=10000`;
  const show = async () => { setErr(null); try { const txt = await api(`/export/alerts?format=${format}&since_id=${since}&limit=5`); setPreview(txt || "(no alerts after that id)"); } catch (e) { setErr(e.message); } };
  return html`<section class="card span5"><h2>SIEM export</h2>
    <div class="row">
      <select value=${format} onChange=${e => { setFormat(e.target.value); setPreview(null); }}>
        <option value="cef">CEF (ArcSight, Splunk, QRadar)</option><option value="syslog">RFC 5424 syslog</option><option value="jsonl">JSON lines (ECS-style)</option>
      </select>
      <label class="small muted">after alert #</label><input type="number" min="0" value=${since} onInput=${e => setSince(Math.max(0, +e.target.value || 0))} style=${{ width: "80px" }}/>
      <button onClick=${show}>Preview</button>
      <a href=${url} download=${`aegisflow-alerts.${format === "jsonl" ? "jsonl" : "log"}`}><button class="primary">Download</button></a>
    </div>
    ${err && html`<div class="err">${err}</div>`}
    ${preview && html`<pre class="preview">${preview}</pre>`}
    <div class="small muted" style=${{ marginTop: "6px" }}>One event per alert, carrying risk, probability, predicted stage and the ledger hash. Also available as${" "}
      <code>GET /export/alerts</code> and <code>python -m aegisflow export-alerts --syslog host:514</code>.</div>
  </section>`;
}

function ModelsCard({ models }) {
  const demo = models.find(m => m.demo_model);
  return html`<section class="card span12"><h2>Models on this machine</h2>
    <div class="scroll" data-lenis-prevent><table>
      <thead><tr><th>Model</th><th>Type</th><th>Trained</th><th class="num">Test seqs</th><th class="num">Test positives</th>
        <th class="num">ROC-AUC</th><th class="num">PR-AUC</th><th class="num">F1</th><th class="num">FPR</th><th>Extra</th></tr></thead>
      <tbody>${models.map(m => { const x = m.test_metrics || {}; return html`<tr key=${m.name} style=${{ cursor: "default" }}>
        <td class="mono">${m.name}${m.demo_model && html`<span class="badge demo">replayed</span>`}</td><td>${m.model_type}</td><td class="mono">${t(m.trained_at)}</td>
        <td class="num">${m.test_sequences?.toLocaleString() ?? "–"}</td><td class="num">${m.test_positive_targets ?? "–"}</td>
        <td class="num">${fmt(x.roc_auc)}</td><td class="num">${fmt(x.pr_auc)}</td><td class="num">${fmt(x.f1)}</td><td class="num">${fmt(x.false_positive_rate, 4)}</td>
        <td class="small">${m.model_type === "multitask_lstm" ? `stage acc ${fmt(m.stage_accuracy)} vs majority ${fmt(m.stage_majority_baseline)}` : m.has_attention ? "attention weights" : ""}</td></tr>`; })}</tbody>
    </table></div>
    <div class="small muted" style=${{ marginTop: "8px" }}>Metrics are copied from each model's own metrics.json (held-out test split, chronological).
      ${demo && models.some(m => !m.demo_model && m.test_sequences !== demo.test_sequences) && html`<b class="bad"> Some models were evaluated on a different test split than the replayed model, so their numbers are not comparable.</b>`}${" "}
      Train more with <code>train --model transformer</code>, <code>train-multitask</code>, <code>train-gnn</code>.</div>
  </section>`;
}

function EvaluationCard({ ev, model, feats }) {
  const [variant, setVariant] = useState("any"), [split, setSplit] = useState("test");
  if (!ev) return null;
  const c = ev.consistency, h = ev.host_check_test_lstm, a = h["172.16.0.1"], o = h.other_hosts, d = ev.lodo_dos_cross_day;
  return html`<section class="card span12"><h2>Model evaluation (real held-out numbers)</h2>
    <div class="note" style=${{ marginBottom: "10px" }}><b>Honest finding:</b> ${ev.honest_note}
      ${feats.stage_model.available ? " Stage predictions come from the opt-in multi-task model." : ` Predicted stage is always UNCERTAIN: ${model.stage_note}`}</div>
    <div class=${"note" + (c.consistent ? " info" : "")} style=${{ marginBottom: "10px", borderColor: c.consistent ? undefined : "var(--bad)" }}>
      ${c.consistent ? "✓ These evaluation tables were computed on the same data and model as the live replay (same split sizes and alert threshold)."
        : html`<span><b>⚠ These evaluation tables do NOT match the live replay.</b> They come from a different model or dataset, so do not compare them with the replay's numbers. ${c.mismatches.join("; ")}.</span>`}</div>
    <div class="row tabs" style=${{ marginBottom: "8px" }}>
      <span class="small muted">Target:</span>
      <button class=${variant === "any" ? "on" : ""} onClick=${() => setVariant("any")}>any attack in next window</button>
      <button class=${variant === "onset" ? "on" : ""} onClick=${() => setVariant("onset")}>attack onset only</button>
      <span class="small muted" style=${{ marginLeft: "12px" }}>Split:</span>
      <button class=${split === "test" ? "on" : ""} onClick=${() => setSplit("test")}>test</button>
      <button class=${split === "val" ? "on" : ""} onClick=${() => setSplit("val")}>validation</button>
    </div>
    <div class="scroll" data-lenis-prevent><table>
      <thead><tr><th>Model</th><th class="num">Precision</th><th class="num">Recall</th><th class="num">F1</th><th class="num">ROC-AUC</th><th class="num">PR-AUC</th><th class="num">FPR</th><th>Confusion [[TN,FP],[FN,TP]]</th></tr></thead>
      <tbody>${ev.table[variant].map(r => { const m = r[split]; return html`<tr key=${r.model} style=${{ cursor: "default" }}><td>${r.model}</td>
        <td class="num">${fmt(m.precision)}</td><td class="num">${fmt(m.recall)}</td><td class="num">${fmt(m.f1)}</td><td class="num">${fmt(m.roc_auc)}</td>
        <td class="num">${fmt(m.pr_auc)}</td><td class="num">${fmt(m.false_positive_rate, 4)}</td><td class="mono">${JSON.stringify(m.confusion_matrix)}</td></tr>`; })}</tbody>
    </table></div>
    <div class="small muted" style=${{ marginTop: "10px" }}><b>Host check (LSTM, test):</b> on attacker host 172.16.0.1 it flags ${a.neg_flagged}/${a.neg} benign and ${a.pos_detected}/${a.pos} attack sequences;
      on all other hosts it detects ${o.pos_detected}/${o.pos} attacks. <b>Leave-one-day-out DoS:</b> ${Object.entries(d).map(([k, v]) =>
        `${k}: LSTM ${v.lstm.detected}/${v.lstm.positives}, host-identity rule ${v.ref_host_identity.detected}/${v.ref_host_identity.positives}`).join("; ")}.
      The reference rule reads dataset labels and is not deployable; it is shown for comparison. Sources: ${ev.source.join(", ")}.</div>
  </section>`;
}


// ------------------------------------------------------------------ routing (hash based, so the server needs no new routes)
function parseRoute() {
  const [path, qs = ""] = location.hash.replace(/^#\/?/, "").split("?");
  const parts = path.split("/").filter(Boolean).map(decodeURIComponent);
  return { page: parts[0] || "overview", arg: parts[1] ?? null, query: new URLSearchParams(qs) };
}
function useRoute() {
  const [route, setRoute] = useState(parseRoute);
  useEffect(() => {
    const on = () => { setRoute(parseRoute()); FX.lenis ? FX.lenis.scrollTo(0, { immediate: true }) : window.scrollTo(0, 0); };
    addEventListener("hashchange", on);
    return () => removeEventListener("hashchange", on);
  }, []);
  return route;
}
const href = (...p) => "#/" + p.map(x => encodeURIComponent(String(x))).join("/");
const go = (...p) => { location.hash = href(...p); };
// href with a query string (alert filters travel this way between Alerts, an alert and its host)
const hrefQ = (q, ...p) => href(...p) + (q ? "?" + q : "");
const goQ = (q, ...p) => { location.hash = hrefQ(q, ...p); };

const ICONS = {
  overview: "M4 4h7v7H4zM13 4h7v4h-7zM13 10h7v10h-7zM4 13h7v7H4z",
  alerts: "M12 3a6 6 0 0 0-6 6v4l-2 3h16l-2-3V9a6 6 0 0 0-6-6zM10 19a2 2 0 0 0 4 0",
  hosts: "M4 5h16v5H4zM4 14h16v5H4zM7.5 7.5h.01M7.5 16.5h.01",
  models: "M4 20V10M10 20V4M16 20v-7M22 20H2",
  robustness: "M3 12h4l3-8 4 16 3-8h4",
  audit: "M12 3l8 3v6c0 4.5-3.4 8.2-8 9-4.6-.8-8-4.5-8-9V6zM8.5 12l2.5 2.5 4.5-5",
  capture: "M12 16V4M7 9l5-5 5 5M4 16v4h16v-4",
  about: "M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18zM12 11v6M12 7.5h.01",
  health: "M12 20s-7-4.4-7-10a4 4 0 0 1 7-2.6A4 4 0 0 1 19 10c0 5.6-7 10-7 10zM8 11h2l1-2 2 4 1-2h2",
};
function Icon({ name }) {
  return html`<svg class="ico" viewBox="0 0 24 24" aria-hidden="true"><path d=${ICONS[name]}/></svg>`;
}
const PAGES = [
  { id: "overview", label: "Overview" }, { id: "alerts", label: "Alerts" }, { id: "hosts", label: "Hosts" },
  { id: "health", label: "Model health" }, { id: "models", label: "Models and evaluation" }, { id: "robustness", label: "Robustness" }, { id: "audit", label: "Audit ledger" },
  { id: "capture", label: "Capture and export" }, { id: "about", label: "How it works" },
];

// ------------------------------------------------------------------ shared bits
function PageHead({ title, desc, children }) {
  return html`<div class="page-head"><div><h2>${title}</h2>${desc && html`<p class="muted">${desc}</p>`}</div>
    ${children && html`<div class="row">${children}</div>`}</div>`;
}
function Stat({ label, value, sub, tone }) {
  return html`<div class="stat"><div class="stat-k">${label}</div><div class=${"stat-v " + (tone || "")}>${value}</div>${sub && html`<div class="small muted">${sub}</div>`}</div>`;
}
function Empty({ children }) { return html`<div class="empty muted">${children}</div>`; }
function RiskCell({ v, cfg }) {
  const lvl = levelOf(v, cfg);
  return html`<td class="num"><span class="riskcell"><span class="riskbar"><span class=${"lvl-bg-" + lvl} style=${{ width: Math.max(0, Math.min(100, v)) + "%" }}></span></span>
    <span class=${"lvl-" + lvl}>${fmt(v, 1)}</span></span></td>`;
}

// Throttled fetch of every alert in the current session, refreshed when the live snapshot says something changed.
function useSessionAlerts(key) {
  const [rows, setRows] = useState([]);
  const last = useRef(0), pending = useRef(null);
  useEffect(() => {
    let live = true;
    const run = () => { last.current = Date.now(); pending.current = null;
      api("/alerts?limit=1000").then(r => live && setRows(r)).catch(() => {}); };
    const wait = 2000 - (Date.now() - last.current);
    if (wait <= 0) run(); else if (!pending.current) pending.current = setTimeout(run, wait);
    return () => { live = false; };
  }, [key]);
  useEffect(() => () => clearTimeout(pending.current), []);
  return rows;
}

// ------------------------------------------------------------------ charts (dataviz: thin marks, 2px gaps, legend + hover)
const SERIES = { attack: "#3987e5", benign: "#d95926" };  // categorical slots 1-2, validated for the dark surface

function AlertTimeline({ alerts, window: win }) {
  const [hover, setHover] = useState(null);
  if (!win || !win[0]) return html`<${Empty}>Start the replay to see alerts over time.<//>`;
  const t0 = Date.parse(win[0]), t1 = Date.parse(win[1]), N = 46, span = Math.max(1, t1 - t0) / N;
  const bins = Array.from({ length: N }, (_, i) => ({ i, a: 0, b: 0, from: t0 + i * span }));
  for (const a of alerts) {
    const i = Math.min(N - 1, Math.max(0, Math.floor((Date.parse(a.target_window_start) - t0) / span)));
    a.truth_attack === 1 ? bins[i].a++ : bins[i].b++;
  }
  const max = Math.max(4, ...bins.map(b => b.a + b.b));
  const W = 760, H = 210, L = 34, R = 8, T = 10, B = 26, bw = (W - L - R) / N;
  const y = v => H - B - v / max * (H - T - B);
  const ticks = [0, Math.round(max / 2), max];
  const clock = ms => new Date(ms).toISOString().slice(11, 16);
  return html`<div class="chart">
    <div class="legend small"><span><i style=${{ background: SERIES.attack }}></i>window labelled attack</span>
      <span><i style=${{ background: SERIES.benign }}></i>window labelled benign (false alarm)</span>
      <span class="muted">labels are evaluation-only; the model never sees them</span></div>
    <div class="chart-box">
      <svg viewBox=${`0 0 ${W} ${H}`} width="100%" role="img" aria-label="alerts per time bucket, split by dataset label" onMouseLeave=${() => setHover(null)}>
        ${ticks.map(v => html`<g key=${v}><line x1=${L} x2=${W - R} y1=${y(v)} y2=${y(v)} class="grid"/><text x=${L - 6} y=${y(v) + 3} text-anchor="end">${v}</text></g>`)}
        ${bins.map(b => { const x = L + b.i * bw + 1, w = Math.max(1, bw - 2), ya = y(b.a), yb = y(b.a + b.b);
          return html`<g key=${b.i} onMouseEnter=${() => setHover(b)}>
            <rect x=${L + b.i * bw} y=${T} width=${bw} height=${H - T - B} fill="transparent"/>
            ${b.a > 0 && html`<rect x=${x} y=${ya} width=${w} height=${H - B - ya} rx="2" fill=${SERIES.attack}/>`}
            ${b.b > 0 && html`<rect x=${x} y=${yb} width=${w} height=${Math.max(0, ya - yb - (b.a ? 2 : 0))} rx="2" fill=${SERIES.benign}/>`}
            ${hover === b && html`<rect x=${L + b.i * bw} y=${T} width=${bw} height=${H - T - B} class="hoverband"/>`}</g>`; })}
        ${[0, Math.floor(N / 2), N - 1].map(i => html`<text key=${i} x=${L + i * bw + bw / 2} y=${H - 8} text-anchor=${i === 0 ? "start" : i === N - 1 ? "end" : "middle"}>${clock(bins[i].from)}</text>`)}
      </svg>
      ${hover && html`<div class="tip" style=${{ left: `calc(${(L + (hover.i + .5) * bw) / W * 100}% )`, top: "4px" }}>
        <div class="mono">${clock(hover.from)}–${clock(hover.from + span)}</div>
        <div><i style=${{ background: SERIES.attack }}></i>${hover.a} labelled attack</div>
        <div><i style=${{ background: SERIES.benign }}></i>${hover.b} labelled benign</div></div>`}
    </div></div>`;
}

function LevelSplit({ hosts, cfg }) {
  const n = { low: 0, medium: 0, high: 0 };
  hosts.forEach(h => n[levelOf(h.latest_risk, cfg)]++);
  const tot = Math.max(1, hosts.length);
  return html`<div>
    <div class="levelbar" role="img" aria-label="hosts per risk level">${["low", "medium", "high"].filter(k => n[k]).map(k =>
      html`<div key=${k} class=${"lvl-bg-" + k} style=${{ flex: n[k] }} title=${`${n[k]} hosts at ${k} risk`}></div>`)}</div>
    <div class="levels">${["high", "medium", "low"].map(k => html`<div key=${k} class="lvlrow">
      <span class=${"sev sev-" + k}>${LEVEL_LABEL[k]}</span><b>${n[k]}</b><span class="muted small">${n[k] && n[k] / tot < .01 ? "<1" : fmt(n[k] / tot * 100, 0)}% of hosts</span></div>`)}</div>
    <div class="small muted">Latest risk per host; bands at ${cfg.thresholds.low} and ${cfg.thresholds.medium}. With no stage model the score cannot exceed ${fmt(cfg.max_reachable_risk, 0)}.</div>
  </div>`;
}

// Horizontal bars for one metric across models, with a reference line (chance level).
function MetricBars({ rows, metric, refValue, refLabel, max = 1 }) {
  return html`<div class="mbars">${rows.map(r => html`<div class="mbar" key=${r.model}>
      <div class="mbar-k">${r.model}<span class="muted small">${r.labels_used ? " · supervised" : " · label-free"}</span></div>
      <div class="mbar-t"><span class="mbar-f" style=${{ width: r[metric] / max * 100 + "%" }}></span>
        ${refValue != null && html`<span class="mbar-ref" style=${{ left: refValue / max * 100 + "%" }} title=${refLabel}></span>`}</div>
      <div class="mbar-v">${fmt(r[metric], 3)}</div></div>`)}
    ${refValue != null && html`<div class="small muted mbar-note"><span class="refkey"></span>${refLabel}</div>`}</div>`;
}

// ------------------------------------------------------------------ alert detail page
function AlertDetail({ id, cfg, feats, models, onChange }) {
  const al = useLoad(() => api(`/alerts/${id}`), [id]);
  const a = al.data;
  const mitre = useLoad(() => a ? api(`/mitre/${encodeURIComponent(a.truth_stage)}`).catch(() => null) : Promise.resolve(null), [a && a.truth_stage]);
  if (al.loading) return html`<div class="card span12"><div class="skel"></div><div class="skel" style=${{ height: "120px" }}></div></div>`;
  if (al.error) return html`<div class="card span12"><div class="err">${al.error.message}</div></div>`;
  const comp = JSON.parse(a.risk_components || "{}"), lvl = levelOf(a.risk_score, cfg), m = mitre.data;
  const stageModel = feats.stage_model.available, uncertain = cfg.uncertain_label;
  const lstmMax = Math.max(a.lstm_probability, a.lstm_threshold, 1e-9);
  const hasLr = a.lr_probability !== null && a.lr_probability !== undefined;
  const hasRule = a.reference_rule_flag !== null && a.reference_rule_flag !== undefined;
  const hasAnom = a.anomaly_flag !== null && a.anomaly_flag !== undefined;
  return html`<${React.Fragment}>
    <section class="card span5 alert-detail">
      <div class="hero">
        <div class="hero-score">
          <div class=${"score lvl-" + lvl}>${fmt(a.risk_score, 1)}<span class="of">/ 100</span></div>
          <span class=${"sev sev-" + lvl}>${LEVEL_LABEL[lvl] || lvl} risk</span>
        </div>
        <div class="hero-gauge">
          <${RiskGauge} score=${a.risk_score} cfg=${cfg}/>
          <div class="small muted">Max reachable ${fmt(cfg.max_reachable_risk, 0)}${stageModel ? "" : ": no stage model, so the stage term is 0"}.</div>
        </div>
      </div>
      <div class="tags">
        <${Chip} label="Confidence" value=${a.confidence_label} tone=${a.confidence_label === uncertain ? "warn" : "ok"}
          title=${`${cfg.confidence_calibrated ? "Calibrated probability" : "Probability"} ${a.confidence_label === uncertain ? `below ${cfg.confidence_threshold}: the alert fired on the lower validation-selected threshold` : `≥ ${cfg.confidence_threshold}`}`}/>
        <${Chip} label="Stage" value=${a.predicted_stage || "–"} tone=${a.predicted_stage === uncertain ? "dim" : "acc"}
          title=${stageModel ? "From the multi-task stage head" : "No stage model configured: binary attack detector only"}/>
        ${hasLr && html`<${Chip} label="Logistic regression" value=${a.lr_flag ? "also flags" : "does not flag"} tone=${a.lr_flag ? "bad" : "dim"}
          title=${`p = ${fmt(a.lr_probability, 4)}, threshold ${fmt(a.lr_threshold, 5)}`}/>`}
        ${hasAnom && html`<${Chip} label="Anomaly model" value=${a.anomaly_flag ? "also flags" : "does not flag"} tone=${a.anomaly_flag ? "bad" : "dim"}
          title=${`Label-free isolation forest, second opinion only: score ${fmt(a.anomaly_score, 4)}, threshold ${fmt(a.anomaly_threshold, 4)}, set on benign training traffic only`}/>`}
        ${hasRule && html`<${Chip} label="Reference rule" value=${a.reference_rule_flag ? "under attack" : "not under attack"} tone="dim"
          title="Last input window per dataset labels; comparison only"/>`}
      </div>
      <div class="facts">
        <div class="fact"><div class="fact-k">LSTM attack probability</div>
          <div class="fact-v">${fmt(a.lstm_probability, 4)} <span class="muted small">threshold ${fmt(a.lstm_threshold, 4)}</span></div>
          <div class="pbar"><div style=${{ width: (a.lstm_probability / lstmMax * 100) + "%" }}></div>
            <span class="thr" style=${{ left: (a.lstm_threshold / lstmMax * 100) + "%" }} title=${"threshold " + fmt(a.lstm_threshold, 4)}></span></div></div>
        <div class="fact"><div class="fact-k">Forecast made at</div>
          <div class="fact-v mono">${t(a.predicted_at)}</div><div class="small muted">end of 10 input windows</div></div>
        <div class="fact"><div class="fact-k">Target window</div>
          <div class="fact-v mono">${t(a.target_window_start)} → ${hm(a.target_window_end)}</div><div class="small muted">window being forecast</div></div>
        ${hasLr && html`<div class="fact"><div class="fact-k">Logistic regression p</div>
          <div class="fact-v">${fmt(a.lr_probability, 4)} <span class="muted small">threshold ${fmt(a.lr_threshold, 5)}</span></div></div>`}
      </div>
      <${TemporalContext} seqId=${a.sequence_id}/>
      <div class="section"><h3>How the score is built</h3><${RiskBreakdown} comp=${comp} cfg=${cfg}/></div>
      <div class="evalbox">
        <div class="evalbox-h"><span class="evalbox-tag">Evaluation only</span><span class="small muted">dataset ground truth, never shown to the model</span></div>
        <div class="evalbox-body">
          <div class=${"outcome " + (a.truth_attack === 1 ? "tp" : "fp")}>
            <div class="outcome-v">${a.truth_attack === 1 ? "True positive" : "False positive"}</div>
            <div class="small">${a.truth_attack === 1 ? "alert fired and the window is labelled attack" : "alert fired but the window is labelled benign"}</div>
          </div>
          <div class="kv small">
            <div>Dataset label</div><div>${a.truth_class} <span class="muted">· stage</span> ${a.truth_stage}</div>
            <div>MITRE ATT&CK</div><div>${m && m.tactic_id ? html`${m.tactic_id} ${m.tactic_name}<div class="muted">${m.techniques.map(x => `${x.id} ${x.name}`).join(", ")}</div>` : "none (benign)"}</div>
            <div></div><div class="muted">MITRE is a static lookup from configs/mitre_mapping.yaml for the label, not a model output.</div>
          </div>
        </div>
      </div>
      <div class="section"><h3>Ledger record</h3>
        <div class="kv small prov">
          <div>Ledger hash</div><div><${Hash} value=${a.hash}/></div>
          <div>Previous hash</div><div><${Hash} value=${a.prev_hash}/></div>
          <div>Sequence</div><div><${Hash} value=${a.sequence_id} head=${24} tail=${8}/></div>
          <div>Model</div><div><${Hash} value=${a.model_version} head=${28} tail=${12}/></div>
        </div>
      </div>
    </section>
    <section class="card span7 alert-detail">
      <${ExplanationSection} seqId=${a.sequence_id} feats=${feats}/>
      <${AttentionSection} seqId=${a.sequence_id} models=${models}/>
      <${StageSection} seqId=${a.sequence_id} feats=${feats}/>
      <${AnalystSection} alertId=${a.id} stages=${feats.stages} onChange=${onChange}/>
    </section>
  <//>`;
}

// ------------------------------------------------------------------ pages
function ReplayControls({ ctx }) {
  const { boot: { cfg, model }, live: { st } } = ctx;
  const [speed, setSpeed] = useState(String(cfg.default_speed)), [starting, setStarting] = useState(false);
  const start = async () => {
    setStarting(true);
    try { await post("/replay/start", { speed: +speed, reset: true }); } catch (e) { ctx.setLiveErr(e.message); }
    setStarting(false);
  };
  return html`<section class="card span3"><h2>Replay controls</h2>
    <div class="row"><label class="small muted" for="speed">Speed</label>
      <select id="speed" value=${speed} onChange=${e => setSpeed(e.target.value)}>${cfg.allowed_speeds.map(s => html`<option key=${s} value=${s}>${s}×</option>`)}</select>
      <button class="primary" disabled=${starting || (st && st.state === "running")} onClick=${start}>${starting ? "Starting…" : "Start"}</button>
      <button class="danger" disabled=${!st || st.state !== "running"} onClick=${() => post("/replay/stop")}>Stop</button></div>
    <div class="small muted" style=${{ marginTop: "8px" }}>Model <span class="mono">${model.version}</span><br/>alert when LSTM p ≥ ${fmt(model.lstm_threshold, 4)} (validation-selected)</div>
  </section>`;
}

function OverviewPage({ ctx }) {
  const { boot: { cfg, ds, model }, live: { st, hosts, triage }, liveErr, alerts } = ctx;
  const pct = st ? Math.round(st.progress * 100) : 0, tp = st ? st.alerts_matching_attack_label : 0;
  const triageLine = Object.entries(triage).filter(([, n]) => n).map(([k, n]) => `${n} ${STATUS_LABEL[k] || k}`).join(" · ");
  const lstm = model.test_metrics.lstm || {};
  const topHosts = hosts.slice().sort((a, b) => b.max_risk - a.max_risk).slice(0, 7);
  return html`<main class="page">
    <${PageHead} title="Overview" desc="Live replay of the held-out CIC-IDS2017 test period through the saved model. All scores are real model outputs."/>
    <section class="card span3"><h2>Hosts monitored</h2><div class="big"><${FX.CountUp} value=${st ? st.hosts_seen : 0}/></div>
      <div class="small muted">of ${ds.hosts.toLocaleString()} in dataset · test split ${ds.split.test_sequences.toLocaleString()} sequences</div></section>
    <section class="card span3"><h2>Alerts this replay</h2><div class="big"><${FX.CountUp} value=${st ? st.alerts_emitted : 0}/></div>
      <div class="small muted">${st && st.alerts_emitted ? `${tp} match an attack label, ${st.alerts_emitted - tp} benign` : " "}</div>
      ${st && st.anomaly && st.alerts_emitted > 0 && html`<div class="small muted"><a href="#/alerts">${st.anomaly.alerts_both_models} also flagged by the anomaly model</a></div>`}
      <div class="small muted">${triageLine}</div></section>
    <section class="card span3"><h2>Replay</h2><div class="big"><${FX.CountUp} value=${pct}/>%</div>
      <div class="bar"><div style=${{ width: pct + "%" }}></div></div>
      <div class="small muted">${liveErr ? "backend: " + liveErr : st ? (st.model_ready ? `${st.processed.toLocaleString()} / ${st.total.toLocaleString()} test sequences` : "loading model and scoring test split…") + (st.error ? ` · ERROR ${st.error}` : "") : " "}</div></section>
    <${ReplayControls} ctx=${ctx}/>

    <section class="card span8"><h2>Alerts over the replay</h2><${AlertTimeline} alerts=${alerts} window=${st && st.replay_window}/></section>
    <section class="card span4"><h2>Hosts by current risk</h2>${hosts.length ? html`<${LevelSplit} hosts=${hosts} cfg=${cfg}/>` : html`<${Empty}>No hosts yet.<//>`}</section>

    <section class="card span6"><div class="card-h"><h2>Latest alerts</h2><a class="more" href="#/alerts">All alerts →</a></div>
      ${alerts.length ? html`<div class="scroll"><table><thead><tr><th>#</th><th>Host</th><th>Target</th><th class="num">Risk</th><th>Status</th></tr></thead>
        <tbody>${alerts.slice(0, 7).map(a => { const s = ctx.live.statuses[a.id] || "open"; return html`<tr key=${a.id} onClick=${() => go("alerts", a.id)}>
          <td>${a.id}</td><td class="mono">${a.host_id}</td><td class="mono">${hm(a.target_window_start)}</td><${RiskCell} v=${a.risk_score} cfg=${cfg}/>
          <td><span class=${"status " + s}>${STATUS_LABEL[s] || s}</span></td></tr>`; })}</tbody></table></div>` : html`<${Empty}>No alerts yet. Start the replay.<//>`}
    </section>
    <section class="card span6"><div class="card-h"><h2>Riskiest hosts</h2><a class="more" href="#/hosts">All hosts →</a></div>
      ${topHosts.length ? html`<div class="scroll"><table><thead><tr><th>Host</th><th class="num">Alerts</th><th class="num">Max risk</th><th class="num">Latest risk</th></tr></thead>
        <tbody>${topHosts.map(h => html`<tr key=${h.host_id} onClick=${() => go("hosts", h.host_id)}>
          <td class="mono">${h.host_id}</td><td class="num">${h.alerts}</td><${RiskCell} v=${h.max_risk} cfg=${cfg}/><td class=${"num lvl-" + levelOf(h.latest_risk, cfg)}>${fmt(h.latest_risk, 1)}</td></tr>`)}</tbody></table></div>`
        : html`<${Empty}>No hosts yet.<//>`}
    </section>

    <section class="card span12 honest"><div class="card-h"><h2>How good is the model? (held-out test split)</h2><a class="more" href="#/models">Models and evaluation →</a></div>
      <div class="stats">
        <${Stat} label="F1" value=${fmt(lstm.f1, 2)} sub=${`precision ${fmt(lstm.precision, 2)} · recall ${fmt(lstm.recall, 2)}`}/>
        <${Stat} label="ROC-AUC" value=${fmt(lstm.roc_auc, 2)} sub="0.5 is chance"/>
        <${Stat} label="PR-AUC" value=${fmt(lstm.pr_auc, 2)} sub=${lstm.actual_positives ? `base rate ${fmt(lstm.actual_positives / (lstm.actual_positives + lstm.actual_negatives), 3)}` : ""}/>
        <${Stat} label="False positive rate" value=${fmt(lstm.false_positive_rate * 100, 1) + "%"} sub=${lstm.confusion_matrix ? `${lstm.confusion_matrix[0][1]} benign sequences flagged` : ""}/>
      </div>
      <div class="note" style=${{ marginTop: "12px" }}>${model.limitation}</div>
    </section>
  </main>`;
}

const LEVEL_FILTERS = ["all", "high", "medium", "low"];
const AGREE_FILTERS = { all: "All", both: "Both models", lstm_only: "LSTM only" };
// Second-opinion summary: how many alerts the label-free anomaly model also flags, and how many of each group are real attacks.
function AgreementCard({ alerts, rule }) {
  const an = useLoad(() => api("/evaluation/anomaly").catch(() => null), []);
  const scored = alerts.filter(a => a.anomaly_flag !== null && a.anomaly_flag !== undefined);
  const both = scored.filter(a => a.anomaly_flag), only = scored.filter(a => !a.anomaly_flag);
  const atk = xs => xs.filter(a => a.truth_attack === 1).length;
  const pct = (k, n) => n ? `${Math.round(k / n * 100)}%` : "–";
  const rows = an.data ? an.data.rows : [], lstm = rows[0], iso = rows.find(r => r.model === "Isolation forest");
  return html`<section class="card span12 honest"><h2>Second opinion: label-free anomaly model</h2>
    <div class="stats">
      <${Stat} label="Both models flag" value=${both.length} sub=${`${atk(both)} match an attack label (${pct(atk(both), both.length)})`}/>
      <${Stat} label="LSTM only" value=${only.length} sub=${`${atk(only)} match an attack label (${pct(atk(only), only.length)})`}/>
    </div>
    <div class="note" style=${{ marginTop: "12px" }}>The LSTM alone raises alerts; the anomaly model never adds or removes one, so the alert count is unchanged.
      It is an isolation forest fit on benign training traffic only, flagging a sequence when its score reaches ${rule || "a threshold set on that traffic"}.
      ${iso && lstm ? ` On the test split it ranks attacks better than the LSTM (ROC-AUC ${fmt(iso.roc_auc, 2)} vs ${fmt(lstm.roc_auc, 2)}) and leans less on the attacker host,
      but at the same false-alarm rate it does not catch more attacks (recall ${fmt(iso.at_lstm_fpr.recall * 100, 1)}% vs ${fmt(lstm.at_lstm_fpr.recall * 100, 1)}%). ` :
      " It ranks attacks better than the LSTM but does not catch more attacks at the same false-alarm rate (reports/anomaly_report.md). "}
      Agreement is a triage hint, not proof. The attack-label counts use dataset ground truth and are for evaluation only.</div>
  </section>`;
}

// ------------------------------------------------------------------ investigation filters (kept in the URL, so they survive
// moving to an alert, its host and back)
const FILTER_KEYS = ["risk", "status", "host", "from", "to", "model", "agree"];
let lastAlertQuery = "";  // the Alerts link in the sidebar reopens the last filters
function readFilters(query) {
  const f = {};
  for (const k of FILTER_KEYS) f[k] = query.get(k) || (["risk", "status", "model", "agree"].includes(k) ? "all" : "");
  return f;
}
function filterQuery(f) {
  const q = new URLSearchParams();
  for (const k of FILTER_KEYS) if (f[k] && f[k] !== "all") q.set(k, f[k]);
  return q.toString();
}
const minute = iso => iso ? String(iso).slice(0, 16) : "";
function applyFilters(alerts, f, statuses, cfg) {
  const host = f.host.trim();
  return alerts.filter(a => (f.risk === "all" || levelOf(a.risk_score, cfg) === f.risk)
    && (f.status === "all" || (statuses[a.id] || "open") === f.status)
    && (!host || a.host_id.includes(host))
    && (!f.from || minute(a.target_window_start) >= f.from) && (!f.to || minute(a.target_window_start) <= f.to)
    && (f.model === "all" || a.model_version === f.model)
    && (f.agree === "all" || (f.agree === "both") === (a.anomaly_flag === true)));
}
function describeFilters(f) {
  const parts = [];
  if (f.risk !== "all") parts.push(`${LEVEL_LABEL[f.risk] || f.risk} risk`);
  if (f.status !== "all") parts.push(STATUS_LABEL[f.status] || f.status);
  if (f.host) parts.push(`host contains ${f.host}`);
  if (f.from || f.to) parts.push(`target ${f.from ? t(f.from).slice(5) : "start"} to ${f.to ? t(f.to).slice(5) : "end"}`);
  if (f.model !== "all") parts.push("one model version");
  if (f.agree !== "all") parts.push(AGREE_FILTERS[f.agree]);
  return parts.join(" · ");
}

// Download an investigation bundle (POST /evidence). Dataset labels are left out unless ticked.
function EvidenceButton({ ids, label = "Export evidence" }) {
  const [gt, setGt] = useState(false), [busy, setBusy] = useState(false), [err, setErr] = useState(null);
  const run = async () => {
    setBusy(true); setErr(null);
    try {
      const r = await fetch("/evidence", { method: "POST", headers: { "content-type": "application/json" },
        body: JSON.stringify({ alert_ids: ids, include_ground_truth: gt }) });
      if (!r.ok) { let m = r.status; try { m = (await r.json()).detail || m; } catch (_) { /* not JSON */ } throw new Error(m); }
      const name = (/filename="([^"]+)"/.exec(r.headers.get("content-disposition") || "") || [])[1] || "aegisflow-evidence.json";
      const url = URL.createObjectURL(await r.blob()), el = document.createElement("a");
      el.href = url; el.download = gt ? name.replace(/\.json$/, "-with-labels.json") : name;
      document.body.appendChild(el); el.click(); el.remove(); setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch (e) { setErr(e.message); }
    setBusy(false);
  };
  return html`<span class="export">
    <button class="primary" disabled=${busy || !ids.length} onClick=${run} title="Alert records, temporal context, host timeline, model version, configuration fingerprint, analyst actions and chain verification">${busy ? "Exporting…" : `${label}${ids.length > 1 ? ` (${ids.length})` : ""}`}</button>
    <label class="small muted" title="Dataset ground truth, for evaluation only. Leave off for operational hand-over."><input type="checkbox" checked=${gt} onChange=${e => setGt(e.target.checked)}/> include dataset labels</label>
    ${err && html`<span class="err small">${err}</span>`}</span>`;
}

function AlertsPage({ ctx, query }) {
  const { boot: { cfg, feats }, live: { statuses, st }, alerts } = ctx;
  const [f, setF] = useState(() => readFilters(query));
  const fq = filterQuery(f);
  useEffect(() => {  // mirror filters into the URL without a hashchange (which would scroll to the top)
    lastAlertQuery = fq;
    history.replaceState(null, "", hrefQ(fq, "alerts"));
  }, [fq]);
  const set = (k, v) => setF(x => ({ ...x, [k]: v }));
  const anom = feats.anomaly_model && feats.anomaly_model.available;
  const versions = useMemo(() => [...new Set(alerts.map(a => a.model_version).filter(Boolean))], [alerts]);
  const rows = applyFilters(alerts, f, statuses, cfg);
  const win = st && st.replay_window, lo = win ? minute(win[0]) : undefined, hi = win ? minute(win[1]) : undefined;
  const cols = anom ? 10 : 9, active = fq !== "";
  return html`<main class="page">
    <${PageHead} title="Alerts" desc="Every alert in the current replay session, newest first. Filters stay in the address, so opening an alert or its host and coming back keeps them.">
      <${EvidenceButton} ids=${rows.map(a => a.id)} label=${active ? "Export filtered" : "Export all"}/>
    <//>
    ${anom && alerts.length > 0 && html`<${AgreementCard} alerts=${alerts} rule=${st && st.anomaly && st.anomaly.threshold_rule}/>`}
    <section class="card span12">
      <div class="filters">
        <div class="segctl" role="group" aria-label="risk level"><span class="segctl-k">Risk</span>${LEVEL_FILTERS.map(k =>
          html`<button key=${k} class=${f.risk === k ? "on" : ""} onClick=${() => set("risk", k)}>${k === "all" ? "All" : LEVEL_LABEL[k]}</button>`)}</div>
        ${anom && html`<div class="segctl" role="group" aria-label="model agreement"><span class="segctl-k">Anomaly model</span>${Object.entries(AGREE_FILTERS).map(([k, v]) =>
          html`<button key=${k} class=${f.agree === k ? "on" : ""} onClick=${() => set("agree", k)}>${v}</button>`)}</div>`}
        <select value=${f.status} onChange=${e => set("status", e.target.value)} aria-label="review status">
          <option value="all">Any review status</option>${Object.entries(STATUS_LABEL).map(([k, v]) => html`<option key=${k} value=${k}>${v}</option>`)}</select>
        <input type="text" placeholder="Filter by host IP" value=${f.host} onInput=${e => set("host", e.target.value)} style=${{ width: "160px" }} aria-label="host"/>
      </div>
      <div class="filters">
        <label class="small muted">Target window from</label>
        <input type="datetime-local" value=${f.from} min=${lo} max=${hi} onInput=${e => set("from", e.target.value)} aria-label="target window from"/>
        <label class="small muted">to</label>
        <input type="datetime-local" value=${f.to} min=${lo} max=${hi} onInput=${e => set("to", e.target.value)} aria-label="target window to"/>
        <select value=${f.model} onChange=${e => set("model", e.target.value)} aria-label="model version" disabled=${versions.length < 2}
          title=${versions.length < 2 ? "Every alert in this session comes from the same model" : ""}>
          <option value="all">${versions.length === 1 ? `Model ${versions[0].slice(-12)}` : "Any model version"}</option>
          ${versions.length > 1 && versions.map(v => html`<option key=${v} value=${v}>${v}</option>`)}</select>
        ${active && html`<button onClick=${() => setF(readFilters(new URLSearchParams()))}>Clear filters</button>`}
        <span class="small muted">${rows.length} of ${alerts.length} alerts</span>
      </div>
      <div class="scroll tall" data-lenis-prevent><table>
        <thead><tr><th>#</th><th>Host</th><th>Target window</th><th class="num">Risk</th><th class="num">LSTM p</th><th>Confidence</th><th>Stage</th>${anom && html`<th title="Label-free isolation forest, second opinion only">Anomaly</th>`}<th>Status</th><th>Dataset label <span class="muted">(eval)</span></th></tr></thead>
        <tbody>${rows.length ? rows.map(a => { const s = statuses[a.id] || "open"; return html`<tr key=${a.id} onClick=${() => goQ(fq, "alerts", a.id)}>
          <td>${a.id}</td><td class="mono">${a.host_id}</td><td class="mono">${t(a.target_window_start).slice(5)}</td>
          <${RiskCell} v=${a.risk_score} cfg=${cfg}/><td class="num">${fmt(a.lstm_probability)}</td>
          <td class=${a.confidence_label === cfg.uncertain_label ? "muted" : ""}>${a.confidence_label}</td><td>${a.predicted_stage}</td>
          ${anom && html`<td class=${a.anomaly_flag ? "" : "muted"}>${a.anomaly_flag === true ? "also flags" : a.anomaly_flag === false ? "no" : "–"}</td>`}
          <td><span class=${"status " + s}>${STATUS_LABEL[s] || s}</span></td>
          <td><span class=${"lbl " + (a.truth_attack ? "atk" : "ben")}>${a.truth_class}</span></td></tr>`; })
          : html`<tr><td colspan=${cols} class="muted">${alerts.length ? "No alerts match these filters." : "No alerts yet. Start the replay on the Overview page."}</td></tr>`}</tbody></table></div>
    </section>
  </main>`;
}

// Alerts that belong to the same investigation: this host's other alerts, and other hosts alerting around the same time.
const RELATED_WINDOW_MS = 10 * 60 * 1000;
function RelatedAlerts({ a, alerts, fq, cfg, statuses }) {
  const at = Date.parse(a.target_window_start);
  const near = x => Math.abs(Date.parse(x.target_window_start) - at) <= RELATED_WINDOW_MS;
  const same = alerts.filter(x => x.id !== a.id && x.host_id === a.host_id);
  const others = alerts.filter(x => x.host_id !== a.host_id && near(x));
  const table = (rows, empty, showHost) => rows.length ? html`<div class="scroll" style=${{ maxHeight: "220px" }} data-lenis-prevent><table>
      <thead><tr><th>#</th>${showHost && html`<th>Host</th>`}<th>Target window</th><th class="num">Risk</th><th>Status</th></tr></thead>
      <tbody>${rows.map(x => { const s = statuses[x.id] || "open"; return html`<tr key=${x.id} onClick=${() => goQ(fq, "alerts", x.id)}>
        <td>${x.id}</td>${showHost && html`<td class="mono">${x.host_id}</td>`}<td class="mono">${t(x.target_window_start).slice(5)}</td>
        <${RiskCell} v=${x.risk_score} cfg=${cfg}/><td><span class=${"status " + s}>${STATUS_LABEL[s] || s}</span></td></tr>`; })}</tbody></table></div>`
    : html`<div class="small muted">${empty}</div>`;
  return html`<section class="card span12"><h2>Related alerts</h2>
    <div class="twocol">
      <div><h3 class="sub">Same host <span class="mono">${a.host_id}</span> · ${same.length}</h3>${table(same, "No other alerts for this host in this session.", false)}</div>
      <div><h3 class="sub">Other hosts within 10 min · ${others.length}</h3>${table(others, "No other host alerted within 10 minutes of this target window.", true)}</div>
    </div>
    <div class="small muted" style=${{ marginTop: "6px" }}>From the whole session, not just the current filters. Alerting at the same time is a lead to check, not a link the model found.</div>
  </section>`;
}

const dur = sec => {
  if (sec === null || sec === undefined) return "–";
  const s = Math.round(sec);
  if (s < 60) return `${s} s`;
  if (s < 3600) return `${Math.floor(s / 60)} min${s % 60 ? ` ${s % 60} s` : ""}`;
  if (s < 86400) return `${Math.floor(s / 3600)} h${Math.round(s % 3600 / 60) ? ` ${Math.round(s % 3600 / 60)} min` : ""}`;
  return `${Math.floor(s / 86400)} d ${Math.round(s % 86400 / 3600)} h`;
};
const pctRank = r => r === null || r === undefined ? "" : r >= 0.995 ? "longer than nearly every test sequence" : `longer than ${Math.round(r * 100)}% of test sequences`;

// Temporal context (GET /context/{sequence}): how much real time the input covers and how far ahead the target is.
function TemporalContext({ seqId }) {
  const c = useLoad(() => api(`/context/${encodeURIComponent(seqId)}`), [seqId]);
  if (c.loading) return html`<div class="section"><h3>Temporal context</h3><div class="skel"></div></div>`;
  if (c.error) return html`<div class="section"><h3>Temporal context</h3><div class="small muted">${c.error.message}</div></div>`;
  const d = c.data, nominal = d.window_seconds + d.stride_seconds * (d.observed_windows - 1);
  const leadLong = d.lead_seconds !== null && d.lead_seconds > d.min_lead_seconds;
  const gaps = d.window_starts.length ? d.window_starts.slice(1).map((w, i) => w && d.window_starts[i] ? (Date.parse(w) - Date.parse(d.window_starts[i])) / 1000 : null) : [];
  const gmax = Math.max(d.stride_seconds, ...gaps.filter(g => g !== null));
  return html`<div class="section"><h3>Temporal context</h3>
    <div class="facts">
      <div class="fact"><div class="fact-k">Input history</div><div class="fact-v">${dur(d.input_span_seconds)}</div>
        <div class="small muted">${t(d.input_start).slice(5)} → ${hm(d.input_end)}${d.input_span_seconds > nominal ? ` · ${pctRank(d.span_rank_in_test)}` : " · contiguous"}</div></div>
      <div class="fact"><div class="fact-k">Windows</div><div class="fact-v">${d.observed_windows} with traffic${d.silent_windows ? html`, <span class="warn-t">${d.silent_windows} silent</span>` : ""}</div>
        <div class="small muted">${d.expected_windows} slots of ${d.window_seconds} s every ${d.stride_seconds} s in that span</div></div>
      <div class="fact"><div class="fact-k">Time since previous observation</div><div class="fact-v">${dur(d.previous_gap_seconds)}</div>
        <div class="small muted">between the last two input windows${d.largest_gap_seconds !== null && d.largest_gap_seconds > d.stride_seconds ? `; largest gap ${dur(d.largest_gap_seconds)}` : ""}</div></div>
      <div class="fact"><div class="fact-k">Target window</div><div class="fact-v mono">${t(d.target_window_start).slice(5)} → ${hm(d.target_window_end)}</div>
        <div class=${"small " + (leadLong ? "warn-t" : "muted")}>starts ${dur(d.lead_seconds)} after the input ends${leadLong ? ` (shortest possible ${dur(d.min_lead_seconds)}; ${pctRank(d.lead_rank_in_test)})` : " (the shortest possible)"}</div></div>
    </div>
    ${gaps.length > 0 && html`<div class="gapstrip" role="img" aria-label="gap before each input window">${gaps.map((g, i) => html`<div key=${i} title=${g === null ? "unknown" : `gap ${dur(g)}`}>
      <span class=${g !== null && g > d.stride_seconds ? "long" : ""} style=${{ height: g === null ? "0" : Math.max(8, Math.log1p(g) / Math.log1p(gmax) * 100) + "%" }}></span></div>`)}</div>
      <div class="small muted">Gap before each of the ${gaps.length} later input windows, oldest first (log scale; amber = longer than one ${d.stride_seconds} s step).</div>`}
    <div class="small muted" style=${{ marginTop: "6px" }}>Windows exist only when the host sent traffic, so "the last 10 windows" and "the next window" can span very different amounts of time. Read the forecast against these spans.</div>
  </div>`;
}

function AlertPage({ ctx, id, query }) {
  const { boot: { cfg, feats, models }, alerts, poll, live: { statuses } } = ctx;
  const f = readFilters(query), fq = filterQuery(f), filtered = applyFilters(alerts, f, statuses, cfg);
  const n = +id, i = filtered.findIndex(a => a.id === n);
  const newer = i > 0 ? filtered[i - 1].id : null, older = i >= 0 && i < filtered.length - 1 ? filtered[i + 1].id : null;
  const a = alerts.find(x => x.id === n);
  const hostQ = [fq, `alert=${n}`].filter(Boolean).join("&");
  return html`<main class="page">
    <div class="crumbs span12"><a href=${hrefQ(fq, "alerts")}>← Alerts${fq ? ` (${describeFilters(f)})` : ""}</a></div>
    <${PageHead} title=${`Alert #${id}`} desc=${a ? html`Host <a class="mono" href=${hrefQ(hostQ, "hosts", a.host_id)}>${a.host_id}</a> · target window ${t(a.target_window_start)}` : null}>
      <button disabled=${!newer} onClick=${() => goQ(fq, "alerts", newer)} title=${fq ? "Next newer alert matching the filters" : ""}>← Newer</button>
      <button disabled=${!older} onClick=${() => goQ(fq, "alerts", older)} title=${fq ? "Next older alert matching the filters" : ""}>Older →</button>
      ${i >= 0 && html`<span class="small muted">${i + 1} of ${filtered.length}${fq ? " filtered" : ""}</span>`}
      <${EvidenceButton} ids=${[n]}/>
    <//>
    <${AlertDetail} key=${id} id=${n} cfg=${cfg} feats=${feats} models=${models} onChange=${poll}/>
    ${a && html`<${RelatedAlerts} a=${a} alerts=${alerts} fq=${fq} cfg=${cfg} statuses=${statuses}/>`}
  </main>`;
}

function HostsPage({ ctx }) {
  const { boot: { cfg }, live: { hosts } } = ctx;
  const [q, setQ] = useState(""), [sort, setSort] = useState("max_risk");
  const rows = hosts.filter(h => !q || h.host_id.includes(q.trim())).sort((a, b) => (b[sort] ?? 0) - (a[sort] ?? 0));
  return html`<main class="page">
    <${PageHead} title="Hosts" desc="Every source host seen in the replay so far. Open one for its risk timeline and alerts."/>
    <section class="card span12">
      <div class="filters">
        <input type="text" placeholder="Filter by host IP" value=${q} onInput=${e => setQ(e.target.value)} style=${{ width: "180px" }}/>
        <div class="segctl" role="group" aria-label="sort"><span class="segctl-k">Sort by</span>
          ${[["max_risk", "Max risk"], ["latest_risk", "Latest risk"], ["alerts", "Alerts"], ["sequences", "Sequences"]].map(([k, v]) =>
            html`<button key=${k} class=${sort === k ? "on" : ""} onClick=${() => setSort(k)}>${v}</button>`)}</div>
        <span class="small muted">${rows.length} of ${hosts.length} hosts</span>
      </div>
      <div class="scroll tall" data-lenis-prevent><table>
        <thead><tr><th>Host</th><th class="num">Sequences</th><th class="num">Alerts</th><th class="num">Max risk</th><th class="num">Latest risk</th><th class="num">Latest LSTM p</th><th>Last seen</th></tr></thead>
        <tbody>${rows.length ? rows.map(h => html`<tr key=${h.host_id} onClick=${() => go("hosts", h.host_id)}>
          <td class="mono">${h.host_id}</td><td class="num">${h.sequences}</td><td class="num">${h.alerts}</td><${RiskCell} v=${h.max_risk} cfg=${cfg}/>
          <td class=${"num lvl-" + levelOf(h.latest_risk, cfg)}>${fmt(h.latest_risk, 1)}</td><td class="num">${fmt(h.latest_lstm_p)}</td><td class="mono">${hm(h.last_seen)}</td></tr>`)
          : html`<tr><td colspan="7" class="muted">${hosts.length ? "No hosts match." : "Start the replay on the Overview page to see hosts."}</td></tr>`}</tbody></table></div>
    </section>
  </main>`;
}

function HostPage({ ctx, id, query }) {
  const { boot: { cfg }, tick } = ctx;
  const h = useLoad(() => api(`/hosts/${encodeURIComponent(id)}`), [id, Math.floor(tick / 3)]);
  const d = h.data;
  // Arriving from an alert: keep its filters for the way back and for opening this host's alerts.
  const f = readFilters(query), fq = filterQuery(f), from = query.get("alert");
  return html`<main class="page">
    <div class="crumbs span12">${from ? html`<a href=${hrefQ(fq, "alerts", from)}>← Alert #${from}</a><span class="muted"> · </span><a href=${hrefQ(fq, "alerts")}>Alerts${fq ? ` (${describeFilters(f)})` : ""}</a>`
      : html`<a href="#/hosts">← Hosts</a>`}</div>
    <${PageHead} title=${html`Host <span class="mono">${id}</span>`} desc="Risk for every sequence this host produced in the replay, and the alerts it raised.">
      ${d && d.alerts_list.length > 0 && html`<${EvidenceButton} ids=${d.alerts_list.map(a => a.id)} label="Export host evidence"/>`}
    <//>
    ${!d ? html`<section class="card span12">${h.error ? (h.error.status === 404 ? html`<${Empty}>This host has not appeared in the current replay yet. The page fills in once it does.<//>` : html`<div class="err">${h.error.message}</div>`) : html`<div class="skel"></div>`}</section>` : html`<${React.Fragment}>
      <section class="card span12"><div class="stats">
        <${Stat} label="Sequences replayed" value=${d.sequences}/>
        <${Stat} label="Alerts" value=${d.alerts}/>
        <${Stat} label="Max risk" value=${fmt(d.max_risk, 1)} tone=${"lvl-" + levelOf(d.max_risk, cfg)}/>
        <${Stat} label="Latest risk" value=${fmt(d.latest_risk, 1)} tone=${"lvl-" + levelOf(d.latest_risk, cfg)} sub=${"last seen " + hm(d.last_seen)}/>
      </div></section>
      <section class="card span12"><h2>Risk over replay time</h2><${Sparkline} tl=${d.timeline} cfg=${cfg}/>
        <div class="small muted">Red dots are alerts; grid lines mark the level bands.</div></section>
      <section class="card span12"><h2>Alerts for this host</h2><div class="scroll" data-lenis-prevent><table>
        <thead><tr><th>#</th><th>Target window</th><th class="num">Risk</th><th class="num">LSTM p</th><th>Stage</th><th>Dataset label <span class="muted">(eval)</span></th></tr></thead>
        <tbody>${d.alerts_list.length ? d.alerts_list.map(a => html`<tr key=${a.id} class=${String(a.id) === from ? "hl" : ""} onClick=${() => goQ(fq, "alerts", a.id)}><td>${a.id}</td>
          <td class="mono">${t(a.target_window_start).slice(5)}</td><${RiskCell} v=${a.risk_score} cfg=${cfg}/>
          <td class="num">${fmt(a.lstm_probability)}</td><td>${a.predicted_stage}</td><td><span class=${"lbl " + (a.truth_attack ? "atk" : "ben")}>${a.truth_class}</span></td></tr>`)
          : html`<tr><td colspan="6" class="muted">No alerts for this host.</td></tr>`}</tbody></table></div></section>
    <//>`}
  </main>`;
}

function AnomalyCard() {
  const an = useLoad(() => api("/evaluation/anomaly"), []);
  if (an.error) return html`<section class="card span12"><h2>Label-free anomaly baseline</h2><div class="small muted">${an.error.message}</div></section>`;
  if (!an.data) return html`<section class="card span12"><div class="skel"></div></section>`;
  const d = an.data, base = d.test_counts.positive / (d.test_counts.positive + d.test_counts.negative);
  const host = r => r.host_check || {}, bot = r => (r.per_class || {}).Botnet || {};
  return html`<section class="card span12"><h2>Label-free anomaly baseline vs the LSTM</h2>
    <p class="small muted lead">Can a model that never sees an attack label find attacks? Isolation forest and autoencoder are fit on${" "}
      ${d.fit_sequences.toLocaleString()} benign train sequences only, so they cannot learn "which host" from labels. Test split: ${d.test_counts.positive} attack targets,${" "}
      ${d.test_counts.negative.toLocaleString()} benign.</p>
    <div class="twocol">
      <div><h3 class="sub">Ranking quality · ROC-AUC</h3><${MetricBars} rows=${d.rows} metric="roc_auc" refValue=${0.5} refLabel="0.5 = random ranking"/></div>
      <div><h3 class="sub">Precision–recall · PR-AUC</h3><${MetricBars} rows=${d.rows} metric="pr_auc" refValue=${base} refLabel=${`${fmt(base, 3)} = base rate (random)`} max=${0.5}/></div>
    </div>
    <div class="scroll" data-lenis-prevent style=${{ marginTop: "14px" }}><table>
      <thead><tr><th>Model</th><th>Labels</th><th class="num">Precision</th><th class="num">Recall</th><th class="num">FPR</th>
        <th class="num">ROC-AUC, hosts other than 172.16.0.1</th><th class="num">Benign flagged on 172.16.0.1</th><th class="num">Botnet detected</th></tr></thead>
      <tbody>${d.rows.map(r => html`<tr key=${r.model} style=${{ cursor: "default" }}><td>${r.model}</td><td>${r.labels_used ? "used" : "none"}</td>
        <td class="num">${fmt(r.at_lstm_fpr.precision)}</td><td class="num">${fmt(r.at_lstm_fpr.recall)}</td><td class="num">${fmt(r.at_lstm_fpr.false_positive_rate * 100, 1)}%</td>
        <td class="num">${fmt((host(r).other_hosts || {}).roc_auc)}</td>
        <td class="num">${host(r)["172.16.0.1"] ? `${host(r)["172.16.0.1"].neg_flagged}/${host(r)["172.16.0.1"].neg}` : "–"}</td>
        <td class="num">${bot(r).n ? `${bot(r).detected}/${bot(r).n}` : "–"}</td></tr>`)}</tbody></table></div>
    <div class="note" style=${{ marginTop: "12px" }}><b>Reading:</b> the isolation forest ranks attacks better (ROC-AUC ${fmt(d.rows[1] && d.rows[1].roc_auc, 2)} vs ${fmt(d.rows[0].roc_auc, 2)})
      and depends less on the attacker host, but at the same false-alarm rate it does not detect more, and Botnet is still almost entirely missed. ${d.note}</div>
    <div class="small muted" style=${{ marginTop: "6px" }}>Precision, recall and FPR at the LSTM's test FPR. Source: ${d.source}; full write-up in ${d.report}.</div>
  </section>`;
}

function ModelsPage({ ctx }) {
  const { boot: { model, ev, feats, models, cfg } } = ctx;
  const lstm = model.test_metrics.lstm || {}, lr = model.test_metrics.logistic_regression || {}, cal = model.confidence_calibration;
  return html`<main class="page">
    <${PageHead} title="Models and evaluation" desc="Held-out numbers for the replayed model, a label-free baseline, and every model trained on this machine. Nothing here is recomputed or rounded for effect."/>
    <section class="card span12"><h2>Replayed model: LSTM on the chronological test split</h2>
      <div class="stats">
        <${Stat} label="F1" value=${fmt(lstm.f1, 3)} sub=${`logistic regression ${fmt(lr.f1, 3)}`}/>
        <${Stat} label="ROC-AUC" value=${fmt(lstm.roc_auc, 3)} sub=${`logistic regression ${fmt(lr.roc_auc, 3)}`}/>
        <${Stat} label="PR-AUC" value=${fmt(lstm.pr_auc, 3)} sub=${`logistic regression ${fmt(lr.pr_auc, 3)}`}/>
        <${Stat} label="Precision / recall" value=${`${fmt(lstm.precision, 2)} / ${fmt(lstm.recall, 2)}`} sub=${`FPR ${fmt(lstm.false_positive_rate * 100, 1)}%`}/>
        <${Stat} label="Alert threshold" value=${fmt(model.lstm_threshold, 4)} sub=${model.threshold_selection}/>
      </div>
      <div class="kv small" style=${{ marginTop: "14px" }}>
        <div>Architecture</div><div>LSTM, hidden size ${model.training.hidden_size}, input ${model.input_shape.join(" × ")} (windows × features)</div>
        <div>Training</div><div>${model.training.epochs_run} of ${model.training.epochs_requested} epochs, best checkpoint epoch ${model.training.best_checkpoint_epoch} by ${model.training.checkpoint_selection}</div>
        <div>Confidence</div><div>${cfg.confidence_calibrated ? `calibrated (${cal.status}); confident when ≥ ${cfg.confidence_threshold}` : cal.status}</div>
        <div>Stage prediction</div><div>${feats.stage_model.available ? "opt-in multi-task model is enabled" : model.stage_note}</div>
      </div>
    </section>
    <${AnomalyCard}/>
    <${EvaluationCard} ev=${ev} model=${model} feats=${feats}/>
    <${ModelsCard} models=${models}/>
  </main>`;
}

// ------------------------------------------------------------------ robustness page
const SWEEP_SERIES = [
  { key: "f1", label: "F1", color: "#3987e5" },
  { key: "precision", label: "Precision", color: "#d95926" },
  { key: "recall", label: "Recall", color: "#199e70" },
];
// Precision, recall and F1 against the alert threshold; precision has no value where nothing is flagged.
function SweepChart({ rows, t0 }) {
  const [hover, setHover] = useState(null);
  const W = 960, H = 300, L = 44, R = 96, T = 14, B = 40, iw = W - L - R, ih = H - T - B;
  const xMax = Math.max(...rows.map(r => r.threshold)), x = v => L + v / xMax * iw, y = v => T + (1 - v) * ih;
  const val = (r, k) => k === "precision" && !r.alerts ? null : r[k];
  const path = k => {
    let d = "", pen = false;
    for (const r of rows) { const v = val(r, k); if (v == null) { pen = false; continue; } d += `${pen ? "L" : "M"}${x(r.threshold).toFixed(1)},${y(v).toFixed(1)}`; pen = true; }
    return d;
  };
  const move = e => {
    const box = e.currentTarget.getBoundingClientRect(), vx = (e.clientX - box.left) / box.width * W;
    let best = rows[0];
    for (const r of rows) if (Math.abs(x(r.threshold) - vx) < Math.abs(x(best.threshold) - vx)) best = r;
    setHover(best);
  };
  const lastOf = k => [...rows].reverse().find(r => val(r, k) != null);
  // direct labels at the right edge, nudged apart so they never overlap
  const ends = SWEEP_SERIES.map(s => ({ ...s, x: x(lastOf(s.key).threshold), y: y(val(lastOf(s.key), s.key)) })).sort((a, b) => a.y - b.y);
  for (let i = 1; i < ends.length; i++) ends[i].y = Math.max(ends[i].y, ends[i - 1].y + 15);
  return html`<div class="sweep">
    <div class="legend small">${SWEEP_SERIES.map(s => html`<span key=${s.key}><i style=${{ background: s.color }}></i>${s.label}</span>`)}
      <span><i class="dash"></i>alert threshold ${fmt(t0, 4)}</span></div>
    <div class="sweep-plot">
      <svg viewBox=${`0 0 ${W} ${H}`} role="img" aria-label="Precision, recall and F1 against the alert threshold" onMouseMove=${move} onMouseLeave=${() => setHover(null)}>
        ${[0, .25, .5, .75, 1].map(g => html`<g key=${g}><line x1=${L} x2=${L + iw} y1=${y(g)} y2=${y(g)} class="grid"/>
          <text x=${L - 8} y=${y(g) + 4} text-anchor="end" class="axis">${g.toFixed(2)}</text></g>`)}
        ${[.1, .2, .3, .4, .5, .6, .7, .8, .9].filter(v => v <= xMax).map(v => html`<text key=${v} x=${x(v)} y=${H - 18} text-anchor="middle" class="axis">${v.toFixed(1)}</text>`)}
        <text x=${L + iw / 2} y=${H - 1} text-anchor="middle" class="axis">alert threshold on the LSTM probability</text>
        <line x1=${x(t0)} x2=${x(t0)} y1=${T} y2=${T + ih} class="ref"/>
        ${SWEEP_SERIES.map(s => html`<path key=${s.key} d=${path(s.key)} fill="none" stroke=${s.color} stroke-width="2" stroke-linejoin="round"/>`)}
        ${ends.map(s => html`<text key=${s.key} x=${s.x + 18} y=${s.y + 4} class="lab">${s.label}</text>`)}
        ${hover && html`<g><line x1=${x(hover.threshold)} x2=${x(hover.threshold)} y1=${T} y2=${T + ih} class="cross"/>
          ${SWEEP_SERIES.map(s => val(hover, s.key) != null && html`<circle key=${s.key} cx=${x(hover.threshold)} cy=${y(val(hover, s.key))} r="4" fill=${s.color} stroke="var(--panel)" stroke-width="2"/>`)}</g>`}
        <rect x=${L} y=${T} width=${iw} height=${ih} fill="transparent"/>
      </svg>
      ${hover && html`<div class="tip" style=${{ left: Math.min(x(hover.threshold) / W * 100, 70) + "%" }}>
        <div class="mono"><b>threshold ${fmt(hover.threshold, 4)}</b>${hover.threshold === t0 ? " (current)" : ""}</div>
        ${SWEEP_SERIES.map(s => html`<div key=${s.key}><i style=${{ background: s.color }}></i>${s.label} <b>${val(hover, s.key) == null ? "no alerts" : fmt(val(hover, s.key), 3)}</b></div>`)}
        <div class="muted">${hover.alerts} alerts · ${hover.tp} true · FPR ${fmt(hover.false_positive_rate * 100, 2)}%</div></div>`}
    </div>
  </div>`;
}

// "detected / total" with a thin bar, for attack targets in one host group
function Detected({ n, of }) {
  if (!of) return html`<td class="num muted">–</td>`;
  return html`<td><div class="det"><div class="track"><div class="fill" style=${{ width: n / of * 100 + "%" }}></div></div><span class="num">${n} / ${of}</span></div></td>`;
}

const range = m => m.min === m.max ? fmt(m.mean, 3) : html`${fmt(m.mean, 3)} <span class="muted small">(${fmt(m.min, 3)}–${fmt(m.max, 3)})</span>`;

function PerturbationCard({ p }) {
  if (!p) return html`<section class="card span12"><h2>Noisy and missing inputs</h2>
    <${Empty}>Not computed yet. These tests re-score the test split, so they need the processed sequence table: run <span class="mono">python scripts/evaluation/robustness_eval.py</span> on a machine that has <span class="mono">data/processed/cic_ids2017/sequences.parquet</span>.<//></section>`;
  const c = p.clean;
  const rows = [
    ...p.noise.map(r => ({ kind: "Noise on every value", level: `σ = ${r.sigma} (×${fmt(Math.exp(-r.sigma), 2)} to ×${fmt(Math.exp(r.sigma), 2)} for one σ)`, r })),
    ...p.missing_cells.map(r => ({ kind: "Values missing at random", level: `${Math.round(r.rate * 100)}% of values`, r })),
    ...p.missing_windows.map(r => ({ kind: "Whole windows missing", level: `${r.windows} of 10 windows`, r })),
  ];
  const groups = p.missing_group, feats = p.missing_feature.slice(0, 6);
  const sum = o => Object.values(o).reduce((a, b) => a + b, 0), allAtk = sum(p.test_attacks_by_class);
  const tcp = groups.find(g => g.group === "tcp_flags") || c, div = groups.find(g => g.group === "destination_diversity") || c;
  const tcpDet = sum(tcp.detected_by_class), tcpOf = allAtk, divAlerts = div.alerts_changed_vs_clean, divBot = div.detected_by_class.Botnet ?? 0, divAuc = div.roc_auc;
  const delta = v => { const d = v - c.roc_auc; return html`<span class=${d < -0.02 ? "neg-t" : "muted"}>${d >= 0 ? "+" : ""}${fmt(d, 3)}</span>`; };
  return html`<${React.Fragment}>
    <section class="card span12"><h2>Noisy and missing inputs</h2>
      <p class="small muted lead">The ${c.n.toLocaleString()} test sequences re-scored by the same model with corrupted raw features, alert threshold unchanged. Missing values are filled
        with the model's own train medians, as an unseen gap would be in deployment. Each random test is repeated with ${p.seeds} seeds; the range is shown in brackets.</p>
      <div class="stats" style=${{ marginBottom: "12px" }}>
        <${Stat} label="Clean F1" value=${fmt(c.f1, 3)} sub=${`${c.tp} of ${c.positives} attack targets, ${c.alerts} alerts`}/>
        <${Stat} label="Clean ROC-AUC" value=${fmt(c.roc_auc, 3)} sub=${`hosts other than 172.16.0.1: ${fmt(c.roc_auc_hosts_other_than_172_16_0_1, 3)}`}/>
        <${Stat} label="Scores reproduced" value=${p.max_abs_diff_vs_audit_predictions < 1e-4 ? "yes" : "no"} sub=${`max difference vs audit file ${p.max_abs_diff_vs_audit_predictions.toExponential(1)}`}/>
      </div>
      <div class="scroll" data-lenis-prevent><table>
        <thead><tr><th>Corruption</th><th>Level</th><th class="num">F1</th><th class="num">ROC-AUC</th><th class="num">Recall</th><th class="num">FPR</th><th class="num">Alerts</th><th class="num">Alerts flipped vs clean</th></tr></thead>
        <tbody>
          <tr style=${{ cursor: "default" }}><td>None</td><td class="muted">clean inputs</td><td class="num">${fmt(c.f1)}</td><td class="num">${fmt(c.roc_auc)}</td><td class="num">${fmt(c.recall)}</td>
            <td class="num">${fmt(c.false_positive_rate * 100, 2)}%</td><td class="num">${c.alerts}</td><td class="num">0</td></tr>
          ${rows.map(({ kind, level, r }) => html`<tr key=${kind + level} style=${{ cursor: "default" }}><td>${kind}</td><td class="small">${level}</td>
            <td class="num">${range(r.f1)}</td><td class="num">${range(r.roc_auc)}</td><td class="num">${fmt(r.recall.mean)}</td>
            <td class="num">${fmt(r.false_positive_rate.mean * 100, 2)}%</td><td class="num">${fmt(r.alerts.mean, 0)}</td><td class="num">${fmt(r.alerts_changed_vs_clean.mean, 0)}</td></tr>`)}
          <tr style=${{ cursor: "default" }}><td>Latest window missing</td><td class="small">window 10 of 10</td><td class="num">${fmt(p.missing_last_window.f1)}</td><td class="num">${fmt(p.missing_last_window.roc_auc)}</td>
            <td class="num">${fmt(p.missing_last_window.recall)}</td><td class="num">${fmt(p.missing_last_window.false_positive_rate * 100, 2)}%</td><td class="num">${p.missing_last_window.alerts}</td><td class="num">${p.missing_last_window.alerts_changed_vs_clean}</td></tr>
        </tbody></table></div>
    </section>
    <section class="card span12"><div class="note"><b>Reading:</b> the alerts barely move under noise or random gaps (F1 stays near ${fmt(c.f1, 2)} and ROC-AUC near ${fmt(c.roc_auc, 2)}),
      but that is not strength: ROC-AUC is ${fmt(c.roc_auc, 2)} to begin with, and ${fmt(c.roc_auc_hosts_other_than_172_16_0_1, 2)} on hosts other than 172.16.0.1, so there is little signal to lose.
      The weak spots are structural: with every TCP-flag feature missing the model detects ${tcpDet} of ${tcpOf} test attacks, and with the destination-diversity features missing
      it changes ${fmt(divAlerts, 0)} of ${c.n.toLocaleString()} alert decisions (${divBot} Botnet detected vs ${c.detected_by_class.Botnet ?? 0}). A higher ROC-AUC after removing a group (e.g. ${fmt(divAuc, 2)}) is not an improvement; it shows the ranking was close to chance.
      Missing inputs here are imputed with train medians; a live feed that sends zeros or stale values instead could behave differently.</div></section>
    <section class="card span6"><h2>A whole feature group missing</h2>
      <table><thead><tr><th>Group</th><th class="num">F1</th><th class="num">ROC-AUC</th><th class="num">Δ ROC-AUC</th><th class="num">Alerts flipped</th></tr></thead>
        <tbody>${groups.map(g => html`<tr key=${g.group} style=${{ cursor: "default" }} title=${g.features.join(", ")}><td>${g.group.replace(/_/g, " ")} <span class="muted small">(${g.features.length})</span></td>
          <td class="num">${fmt(g.f1)}</td><td class="num">${fmt(g.roc_auc)}</td><td class="num">${delta(g.roc_auc)}</td><td class="num">${g.alerts_changed_vs_clean}</td></tr>`)}</tbody></table>
    </section>
    <section class="card span6"><h2>Single features the model leans on most</h2>
      <table><thead><tr><th>Feature missing</th><th class="num">F1</th><th class="num">ROC-AUC</th><th class="num">Δ ROC-AUC</th><th class="num">Alerts flipped</th></tr></thead>
        <tbody>${feats.map(f => html`<tr key=${f.feature} style=${{ cursor: "default" }}><td class="mono small">${f.feature}</td>
          <td class="num">${fmt(f.f1)}</td><td class="num">${fmt(f.roc_auc)}</td><td class="num">${delta(f.roc_auc)}</td><td class="num">${f.alerts_changed_vs_clean}</td></tr>`)}</tbody></table>
      <div class="small muted" style=${{ marginTop: "8px" }}>The ${feats.length} of ${p.missing_feature.length} features whose absence lowers test ROC-AUC the most.</div>
    </section>
  <//>`;
}

function RobustnessPage() {
  const rb = useLoad(() => api("/evaluation/robustness"), []);
  const [split, setSplit] = useState("test");
  const head = html`<${PageHead} title="Robustness" desc="How the replayed model's numbers move with the alert threshold, across capture days and hours, and when inputs are noisy or missing. Nothing is retrained and the live threshold is unchanged."/>`;
  if (rb.error) return html`<main class="page">${head}<section class="card span12"><div class="small muted">${rb.error.message}</div></section></main>`;
  if (!rb.data) return html`<main class="page">${head}<section class="card span12"><div class="skel" style=${{ height: "160px" }}></div></section></main>`;
  const d = rb.data, th = d.thresholds, t0 = th.committed_threshold, sw = th.splits[split];
  const at = r => r.threshold === t0, cur = th.splits.test.rows.find(at), ci = th.test_bootstrap_95ci;
  const testHours = d.by_day.test_hours, h172 = testHours.reduce((a, h) => [a[0] + h.tp_on_172_16_0_1, a[1] + h.positives_from_172_16_0_1], [0, 0]);
  const hOther = testHours.reduce((a, h) => [a[0] + h.tp - h.tp_on_172_16_0_1, a[1] + h.positives - h.positives_from_172_16_0_1], [0, 0]);
  return html`<main class="page">${head}
    <section class="card span12"><h2>Alert threshold</h2>
      <div class="stats">
        <${Stat} label="Current threshold" value=${fmt(t0, 4)} sub="chosen on validation (max F1); not changed here"/>
        <${Stat} label="Test F1 at it" value=${fmt(cur.f1, 3)} sub=${`95% bootstrap ${fmt(ci.f1[0], 3)}–${fmt(ci.f1[1], 3)}`}/>
        <${Stat} label="Test precision / recall" value=${`${fmt(cur.precision, 2)} / ${fmt(cur.recall, 2)}`} sub=${`${cur.alerts} alerts, FPR ${fmt(cur.false_positive_rate * 100, 1)}%`}/>
        <${Stat} label="Best test F1, any threshold" value=${fmt(th.splits.test.best_f1, 3)} sub=${`at ${th.splits.test.best_f1_threshold}; picked with hindsight on test`}/>
      </div>
      <div class="row tabs" style=${{ margin: "14px 0 6px" }}><span class="small muted">Split:</span>
        <button class=${split === "test" ? "on" : ""} onClick=${() => setSplit("test")}>test (Fri 7 Jul)</button>
        <button class=${split === "val" ? "on" : ""} onClick=${() => setSplit("val")}>validation (Thu 6 Jul)</button></div>
      <${SweepChart} key=${split} rows=${sw.rows} t0=${t0}/>
      <details class="small" style=${{ marginTop: "8px" }}><summary class="muted">Table view</summary><div class="scroll" data-lenis-prevent><table>
        <thead><tr><th class="num">Threshold</th><th class="num">Alerts</th><th class="num">True</th><th class="num">Precision</th><th class="num">Recall</th><th class="num">F1</th><th class="num">FPR</th></tr></thead>
        <tbody>${sw.rows.map(r => html`<tr key=${r.threshold} class=${at(r) ? "hl" : ""} style=${{ cursor: "default" }}><td class="num mono">${fmt(r.threshold, at(r) ? 4 : 2)}</td>
          <td class="num">${r.alerts}</td><td class="num">${r.tp}</td><td class="num">${r.alerts ? fmt(r.precision) : "–"}</td><td class="num">${fmt(r.recall)}</td>
          <td class="num">${fmt(r.f1)}</td><td class="num">${fmt(r.false_positive_rate * 100, 2)}%</td></tr>`)}</tbody></table></div></details>
      <div class="note" style=${{ marginTop: "12px" }}><b>Reading:</b> raising the threshold trades recall for precision; F1 stays low everywhere on test
        (at most ${fmt(th.splits.test.best_f1, 2)}), so no threshold rescues the model. A higher threshold would have scored a little better on test, but choosing it from test
        results would be hindsight, so the validation-chosen ${fmt(t0, 4)} stays. ${th.bootstrap_note}</div>
    </section>

    <section class="card span12"><h2>Results by capture day</h2>
      <div class="scroll" data-lenis-prevent><table>
        <thead><tr><th>Day</th><th>Split</th><th class="num">Sequences</th><th class="num">Attack targets</th><th class="num">Alerts</th><th class="num">Precision</th><th class="num">Recall</th><th class="num">F1</th><th class="num">FPR</th><th class="num">ROC-AUC</th></tr></thead>
        <tbody>${d.by_day.days.map(r => html`<tr key=${r.day + r.split} style=${{ cursor: "default" }} class=${r.in_sample ? "dim" : ""}>
          <td class="mono">${r.weekday} ${r.day.slice(5)}</td><td>${r.split}${r.in_sample ? html`<span class="badge">in-sample</span>` : ""}</td>
          <td class="num">${r.n.toLocaleString()}</td><td class="num">${r.positives}</td><td class="num">${r.alerts}</td>
          <td class="num">${r.alerts ? fmt(r.precision) : "–"}</td><td class="num">${r.positives ? fmt(r.recall) : "–"}</td><td class="num">${r.positives ? fmt(r.f1) : "–"}</td>
          <td class="num">${fmt(r.false_positive_rate * 100, 2)}%</td><td class="num">${fmt(r.roc_auc)}</td></tr>`)}</tbody></table></div>
      <div class="small muted" style=${{ marginTop: "8px" }}>${d.by_day.note} At threshold ${fmt(t0, 4)}.</div>
    </section>

    <section class="card span12"><h2>Test day hour by hour: who gets detected</h2>
      <div class="scroll" data-lenis-prevent><table class="hours">
        <thead><tr><th>Hour</th><th class="num">Sequences</th><th>Attacks detected, other hosts</th><th>Attacks detected, 172.16.0.1</th><th class="num">Alerts</th><th class="num">False alarms</th></tr></thead>
        <tbody>${testHours.map(h => html`<tr key=${h.hour} style=${{ cursor: "default" }}><td class="mono">${String(h.hour).padStart(2, "0")}:00</td><td class="num">${h.n.toLocaleString()}</td>
          <${Detected} n=${h.tp - h.tp_on_172_16_0_1} of=${h.positives - h.positives_from_172_16_0_1}/><${Detected} n=${h.tp_on_172_16_0_1} of=${h.positives_from_172_16_0_1}/>
          <td class="num">${h.alerts}</td><td class="num">${h.fp}</td></tr>`)}</tbody></table></div>
      <div class="note" style=${{ marginTop: "12px" }}><b>Reading:</b> on the test day the model detects ${h172[0]} of ${h172[1]} attack targets on 172.16.0.1, the attacker host
        behind nearly every attack class in training, but only ${hOther[0]} of ${hOther[1]} on the other hosts (the morning attacks there are Botnet, a class whose traffic does not come from 172.16.0.1).
        That is the attacker-host shortcut seen elsewhere in the evaluation, now visible hour by hour.</div>
    </section>

    <${PerturbationCard} p=${d.perturbation}/>
    <section class="card span12"><div class="small muted">Source: ${d.source}, written by <span class="mono">scripts/evaluation/robustness_eval.py</span> from the committed model
      (${d.model_dir}) on ${t(d.generated_at)}${d.perturbation_generated_at && d.perturbation_generated_at !== d.generated_at ? `; noisy and missing input tests from ${t(d.perturbation_generated_at)}` : ""}.</div></section>
  </main>`;
}

// ------------------------------------------------------------------ model health and drift page (GET /health/model)
const LEVEL_TEXT = { investigate: "investigate", watch: "watch", stable: "stable", "n/a": "n/a" };
function LvlTag({ level }) { return html`<span class=${"lvltag " + level}>${LEVEL_TEXT[level] || level}</span>`; }

function ScoreHist({ s }) {
  const share = xs => { const n = Math.max(1, xs.reduce((a, b) => a + b, 0)); return xs.map(x => x / n); };
  const r = share(s.replay), v = s.validation ? share(s.validation) : null;
  const max = Math.max(1e-9, ...r, ...(v || []));
  return html`<div class="chart">
    <div class="legend small"><span><i style=${{ background: SERIES.attack }}></i>replayed so far</span>
      ${v && html`<span><i style=${{ background: "#8493a3" }}></i>validation split</span>`}<span><i class="dash"></i>alert threshold ${fmt(s.threshold, 4)}</span></div>
    <div class="hist" role="img" aria-label="share of sequences per LSTM probability bin">
      ${r.map((x, i) => html`<div key=${i} title=${`p ${fmt(s.bins[i], 2)}–${fmt(s.bins[i + 1], 2)}: replay ${fmt(x * 100, 1)}%${v ? `, validation ${fmt(v[i] * 100, 1)}%` : ""}`}>
        <span style=${{ height: x / max * 100 + "%", background: SERIES.attack }}></span>${v && html`<span style=${{ height: v[i] / max * 100 + "%", background: "#8493a3" }}></span>`}</div>`)}
      <div class="thr" style=${{ left: s.threshold * 100 + "%" }}></div>
    </div>
    <div class="hist-axis"><span>0</span><span>0.5</span><span>1</span></div>
  </div>`;
}

function VolumeChart({ v }) {
  const rows = v.replay, ref = v.train.flows, max = Math.max(1, ref.p90, ...rows.map(r => r.flows));
  return html`<div>
    <div class="vol" role="img" aria-label="flows per 15 minutes of replay time">
      <div class="band" style=${{ bottom: ref.p10 / max * 100 + "%", height: Math.max(0.5, (ref.p90 - ref.p10) / max * 100) + "%" }} title=${`training: 10th–90th percentile ${fmt(ref.p10, 0)}–${fmt(ref.p90, 0)} flows per 15 min`}></div>
      ${rows.map(r => html`<span key=${r.t} style=${{ height: r.flows / max * 100 + "%" }} title=${`${hm(r.t)}: ${r.flows.toLocaleString()} flows in last input windows, ${r.sequences} sequences`}></span>`)}
    </div>
    <div class="hist-axis"><span>${rows.length ? hm(rows[0].t) : ""}</span><span>${rows.length ? hm(rows[rows.length - 1].t) : ""}</span></div>
    <div class="small muted">Bars: flows in the last input window of each sequence, per 15 minutes of replay time. Shaded band: the middle 80% of the same number in training (median ${fmt(ref.median, 0)}). The last bar may be a partial bucket.</div>
  </div>`;
}

function HealthPage({ ctx }) {
  const { live: { st }, tick } = ctx;
  const hd = useLoad(() => api("/health/model"), [st && st.session_id, Math.floor(tick / 20)]);
  const [all, setAll] = useState(false);
  const head = html`<${PageHead} title="Model health" desc="How the traffic being scored compares with what the model was trained on. Computed without labels; drift is a reason to investigate, not evidence of an attack."/>`;
  if (hd.error) return html`<main class="page">${head}<section class="card span12"><div class="err">${hd.error.message}</div></section></main>`;
  if (!hd.data) return html`<main class="page">${head}<section class="card span12"><div class="skel" style=${{ height: "140px" }}></div></section></main>`;
  const d = hd.data, ing = d.ingestion;
  const ingestion = ing && html`<section class="card span12"><h2>Ingestion checks (at preprocessing time)</h2>
    <div class="stats">
      <${Stat} label="Raw rows" value=${ing.raw_rows.toLocaleString()}/>
      <${Stat} label="Rows dropped" value=${ing.dropped_rows.toLocaleString()} sub=${`${fmt(ing.dropped_rows / ing.raw_rows * 100, 1)}% of raw rows`}/>
      <${Stat} label="Features unavailable" value=${ing.unavailable_features.length} sub=${ing.unavailable_features.join(", ") || "none"}/>
    </div>
    <div class="kv small" style=${{ marginTop: "10px" }}>${Object.entries(ing.dropped_by_reason).map(([k, n]) => html`<${React.Fragment} key=${k}><div>${k.replace(/_/g, " ")}</div><div>${n.toLocaleString()}</div><//>`)}</div>
    <div class="small muted" style=${{ marginTop: "6px" }}>From ${ing.source}. These are schema and cleaning failures when the dataset was preprocessed; the replay reads the already-processed table, so it has no live schema failures to count.</div>
  </section>`;
  if (d.state !== "ok") return html`<main class="page">${head}
    <section class="card span12"><${Empty}>${d.state === "model_not_ready" ? "The model is still loading." : `Drift needs at least ${d.min_sequences} replayed sequences (${d.replayed} so far). Start the replay on the Overview page.`}<//></section>
    ${ingestion}</main>`;
  const feats = all ? d.features : d.features.slice(0, 10), lv = d.features_by_level, sc = d.scores;
  return html`<main class="page">${head}
    <section class="card span12"><div class="stats">
      <${Stat} label="Sequences compared" value=${d.replayed.toLocaleString()} sub=${`replayed ${hm(d.window[0])}–${hm(d.window[1])} vs ${d.train_sequences.toLocaleString()} training`}/>
      <${Stat} label="Features shifted" value=${`${lv.investigate} / ${lv.watch}`} sub=${`large (PSI ≥ ${d.psi_levels.investigate}) / moderate (≥ ${d.psi_levels.watch}) of ${d.features.length}`} tone=${lv.investigate ? "bad" : ""}/>
      <${Stat} label="Score drift (PSI)" value=${sc.psi === null ? "–" : fmt(sc.psi, 2)} sub=${sc.validation ? "LSTM probabilities vs validation" : "no validation scores on this machine"} tone=${sc.level === "investigate" ? "bad" : ""}/>
      <${Stat} label="Alert rate" value=${fmt(sc.replay_alert_rate * 100, 2) + "%"} sub=${sc.validation_alert_rate !== undefined ? `validation ${fmt(sc.validation_alert_rate * 100, 2)}% at the same threshold` : ""}/>
      <${Stat} label="Non-finite inputs" value=${d.non_finite_inputs} sub="NaN or infinite model inputs"/>
    </div>
    <div class="note" style=${{ marginTop: "12px" }}><b>Reading:</b> a shifted feature or score distribution says today's traffic looks different from training. That can be an attack,
      but also a new service, a time-of-day effect or a data-collection change, so it is a reason to look at the affected hosts, not a verdict.
      Recomputed as the replay advances. Processing lag is ${d.processing_lag}.</div>
    </section>
    <section class="card span6"><h2>Score distribution</h2>${sc.validation || sc.replay ? html`<${ScoreHist} s=${sc}/>` : null}
      <div class="small muted" style=${{ marginTop: "6px" }}>Share of sequences per LSTM probability bin.${sc.psi !== null ? html` PSI ${fmt(sc.psi, 3)} <${LvlTag} level=${sc.level}/>.` : ""}</div></section>
    <section class="card span6"><h2>Traffic volume</h2><${VolumeChart} v=${d.volume}/></section>
    <section class="card span12"><h2>Input drift by feature (last input window)</h2>
      <div class="scroll" data-lenis-prevent><table>
        <thead><tr><th>Feature</th><th class="num">PSI</th><th></th><th class="num">Outside training range</th><th class="num">Zeros now</th><th class="num">Zeros in training</th></tr></thead>
        <tbody>${feats.map(f => html`<tr key=${f.feature} style=${{ cursor: "default" }}><td class="mono small" title=${`training range ${fmt(f.train_range[0], 3)} to ${fmt(f.train_range[1], 3)}`}>${f.feature}</td>
          <td class="num">${fmt(f.psi, 3)}</td><td><${LvlTag} level=${f.level}/></td>
          <td class=${"num " + (f.outside_train_range > 0.01 ? "warn-t" : "")}>${fmt(f.outside_train_range * 100, 2)}%</td>
          <td class="num">${fmt(f.zero_share * 100, 1)}%</td><td class="num muted">${fmt(f.train_zero_share * 100, 1)}%</td></tr>`)}</tbody></table></div>
      ${d.features.length > 10 && html`<button style=${{ marginTop: "8px" }} onClick=${() => setAll(x => !x)}>${all ? "Show top 10" : `Show all ${d.features.length}`}</button>`}
      <div class="small muted" style=${{ marginTop: "6px" }}>PSI compares each feature's distribution with training, binned at the training deciles (below ${d.psi_levels.watch} stable, ${d.psi_levels.watch}–${d.psi_levels.investigate} moderate, above large).
        "Outside training range" is the share of values below the training minimum or above its maximum: inputs the model never saw.
        The windowing step stores missing values as 0, so a missing measurement cannot be told apart from a real zero; the share of zeros is shown instead of a missing rate.</div>
    </section>
    ${ingestion}
  </main>`;
}

function AuditPage({ ctx }) {
  const sessions = useLoad(() => api("/audit/sessions"), [ctx.live.st && ctx.live.st.session_id, ctx.alerts.length]);
  return html`<main class="page">
    <${PageHead} title="Audit ledger" desc="Every alert and every analyst action is appended to a SHA-256 hash chain, so any edit after the fact is detectable."/>
    <section class="card span12 chainviz">
      ${["Alert n−1", "Alert n", "Alert n+1"].map((x, i) => html`<${React.Fragment} key=${x}>
        <div class="block"><div class="block-k">${x}</div><div class="mono small">prev_hash</div><div class="mono small">fields…</div><div class="mono small acc-t">hash = SHA-256(fields + prev_hash)</div></div>
        ${i < 2 && html`<div class="link-arrow" aria-hidden="true">→</div>`}<//>`)}
      <div class="small muted chainviz-note">Each hash covers the record's fields and the previous hash. Editing a stored field makes its recomputed hash differ from the stored one, and every later record is linked to the stored value. "Verify chains" recomputes them all.</div>
    </section>
    <${AuditCard} alerts=${ctx.alerts}/>
    <section class="card span12"><h2>Replay sessions in the ledger</h2>
      ${sessions.data && sessions.data.length ? html`<table><thead><tr><th>Session</th><th class="num">Alerts</th><th class="num">First id</th><th class="num">Last id</th><th></th></tr></thead>
        <tbody>${sessions.data.map(s => html`<tr key=${s.session_id} style=${{ cursor: "default" }}><td class="mono">${s.session_id}</td><td class="num">${s.alerts}</td>
          <td class="num">${s.first_id}</td><td class="num">${s.last_id}</td><td>${ctx.live.st && ctx.live.st.session_id === s.session_id ? html`<span class="badge demo">current</span>` : ""}</td></tr>`)}</tbody></table>`
        : html`<${Empty}>No sessions yet.<//>`}
    </section>
  </main>`;
}

function CapturePage({ ctx }) {
  return html`<main class="page">
    <${PageHead} title="Capture and export" desc="Score your own packet capture or NetFlow export, and send alerts to a SIEM."/>
    <${CaptureCard} feats=${ctx.boot.feats}/>
    <${SiemCard}/>
  </main>`;
}

function AboutPage({ ctx }) {
  const { boot: { ds, model, cfg, feats } } = ctx;
  const lstm = model.test_metrics.lstm || {};
  const steps = [
    ["Flows", "CIC-IDS2017 flow records (or your own capture), cleaned and keyed by source host."],
    ["Windows", "Per host, flows are grouped into 60 s windows and summarised as 28 behaviour features."],
    ["Sequences", "The last 10 windows of a host form one input sequence."],
    ["Forecast", "The LSTM gives the probability that the host's next window contains attack traffic."],
    ["Alert", `If p ≥ ${fmt(model.lstm_threshold, 3)}, a 0–100 risk score is computed and the alert is hash-chained into the ledger.`],
  ];
  return html`<main class="page">
    <${PageHead} title="How it works" desc="What the demo does, what the numbers mean, and where it stops."/>
    <section class="card span12"><h2>Pipeline</h2>
      <ol class="pipeline">${steps.map(([k, v], i) => html`<li key=${k}><span class="pn">${i + 1}</span><b>${k}</b><span class="small muted">${v}</span></li>`)}</ol>
    </section>
    <section class="card span6"><h2>What the demo shows</h2><ul class="list">
      <li>The held-out test period (${t(ds.split.test_cutoff_time)} onwards) replayed through the saved model; every score is a real model output.</li>
      <li>Alerts as early warning: the model says a host's next window is likely to carry attack traffic, typically while an attack on that host is already under way.</li>
      <li>Per-alert explanations (SHAP or Integrated Gradients), a risk breakdown, analyst review, and a tamper-evident ledger.</li>
      <li>Your own PCAP or NetFlow scored by the same model through the streaming pipeline.</li>
    </ul></section>
    <section class="card span6"><h2>Known limits</h2><ul class="list">
      <li>Test F1 is ${fmt(lstm.f1, 2)} and ROC-AUC ${fmt(lstm.roc_auc, 2)}: modest, and close to chance at ranking.</li>
      <li>${model.limitation}</li>
      <li>Predicting attack onset (before any attack traffic from that host) did not beat simple baselines; see the "attack onset only" tab under Models.</li>
      <li>${feats.stage_model.available ? "Stage predictions come from an opt-in multi-task model and are proxies, not ground truth." : `No kill-chain stage is predicted: ${model.stage_note}`}</li>
      <li>Confidence is ${cfg.confidence_calibrated ? "a calibrated probability" : "the raw model probability (not calibrated)"}.</li>
    </ul></section>
    <section class="card span12"><h2>Data</h2>
      <div class="kv small">
        <div>Dataset</div><div>${ds.dataset}</div>
        <div>Provenance</div><div>${ds.provenance || "–"}</div>
        <div>Time range</div><div class="mono">${ds.time_range ? `${t(ds.time_range[0])} → ${t(ds.time_range[1])}` : "–"}</div>
        <div>Split</div><div>chronological: train ${ds.split.train_sequences?.toLocaleString() ?? "–"}, validation ${ds.split.val_sequences?.toLocaleString() ?? "–"}, test ${ds.split.test_sequences.toLocaleString()} sequences${ds.split.boundary_excluded_sequences != null ? ` (${ds.split.boundary_excluded_sequences} excluded at split boundaries)` : ""}</div>
      </div>
      <div class="small muted" style=${{ marginTop: "8px" }}>The original single-page dashboard is still available as the <a href="/classic">classic view</a>.</div>
    </section>
  </main>`;
}

// ------------------------------------------------------------------ app shell
function Sidebar({ page, st, counts }) {
  return html`<nav class="side" aria-label="Pages">
    <a class="brand" href="#/overview"><span class="brand-mark" aria-hidden="true"></span><span>AegisFlow</span></a>
    <div class="nav">${PAGES.map(p => html`<a key=${p.id} href=${p.id === "alerts" ? hrefQ(lastAlertQuery, "alerts") : "#/" + p.id} class=${page === p.id ? "on" : ""} aria-current=${page === p.id ? "page" : undefined}>
      <${Icon} name=${p.id}/><span>${p.label}</span>${counts[p.id] ? html`<span class="count">${counts[p.id]}</span>` : null}</a>`)}</div>
    <div class="side-foot">
      <span class=${"chip " + (st ? st.state : "idle")}>${st ? st.state + (st.state === "running" ? ` · ${st.speed}×` : "") : "idle"}</span>
      <div class="small muted">sim clock <span class="mono">${st && st.sim_time ? t(st.sim_time).slice(11) : "–"}</span></div>
      <a class="small muted" href="/classic">classic view</a>
    </div>
  </nav>`;
}

function App() {
  const route = useRoute();
  const [boot, setBoot] = useState(null), [bootErr, setBootErr] = useState(null);
  const [live, setLive] = useState({ st: null, hosts: [], alerts: [], statuses: {}, triage: {} }), [liveErr, setLiveErr] = useState(null);
  const [tick, setTick] = useState(0), [rev, setRev] = useState(0);
  const timer = useRef(null);

  useEffect(() => {
    (async () => {
      const [cfg, ds, model, ev, feats, models] = await Promise.all([api("/risk/config"), api("/dataset/status"), api("/model/status"),
        api("/evaluation"), api("/features"), api("/models")]);
      setBoot({ cfg, ds, model, ev, feats, models });
    })().catch(setBootErr);
  }, []);

  const apply = useCallback(snap => {
    setLive(snap); setLiveErr(null);
    if (snap.st.state === "running") setTick(x => x + 1);
  }, []);
  // one-shot refresh (after an analyst action, or as the fallback loop below)
  const poll = useCallback(async () => {
    try {
      const [st, hosts, alerts, statuses, triage] = await Promise.all([api("/replay/status"), api("/hosts"), api("/alerts?limit=50"),
        api("/triage/statuses"), api("/triage/summary")]);
      apply({ st, hosts, alerts, statuses, triage }); setRev(x => x + 1);
    } catch (e) { setLiveErr(e.message); }
  }, [apply]);
  useEffect(() => {
    if (!boot) return;
    let stop = false;
    const loop = async () => { await poll(); if (!stop) timer.current = setTimeout(loop, 1000); };
    if (!window.EventSource) { loop(); return () => { stop = true; clearTimeout(timer.current); }; }
    // server-sent events: the backend pushes a snapshot only when it changes (GET /live)
    const es = new EventSource("/live");
    es.addEventListener("snapshot", e => apply(JSON.parse(e.data)));
    es.addEventListener("error", e => {
      if (e.data) setLiveErr(JSON.parse(e.data).detail);  // server-reported error event
      else if (es.readyState === EventSource.CONNECTING) setLiveErr("live stream lost, reconnecting…");
      else if (es.readyState === EventSource.CLOSED && !stop) { setLiveErr("live stream closed, polling every second"); loop(); }
    });
    return () => { stop = true; es.close(); clearTimeout(timer.current); };
  }, [boot, poll, apply]);

  const st = live.st;
  const alerts = useSessionAlerts(`${st ? st.session_id : ""}:${st ? st.alerts_emitted : 0}:${rev}`);
  useEffect(() => { if (boot) FX.revealCards(); }, [!!boot, route.page, route.arg]);

  const { page, arg } = route;
  const ctx = { boot, live, liveErr, setLiveErr, tick, poll, alerts };
  let body;
  if (bootErr) body = html`<main class="page"><section class="card span12"><h2>Cannot load</h2><div class="err">${bootErr.message}</div></section></main>`;
  else if (!boot) body = html`<main class="page"><section class="card span12"><div class="skel" style=${{ width: "38%" }}></div><div class="skel" style=${{ height: "160px" }}></div></section></main>`;
  else if (page === "alerts") body = arg ? html`<${AlertPage} key=${arg} ctx=${ctx} id=${arg} query=${route.query}/>` : html`<${AlertsPage} ctx=${ctx} query=${route.query}/>`;
  else if (page === "hosts") body = arg ? html`<${HostPage} key=${arg} ctx=${ctx} id=${arg} query=${route.query}/>` : html`<${HostsPage} ctx=${ctx}/>`;
  else if (page === "health") body = html`<${HealthPage} ctx=${ctx}/>`;
  else if (page === "models") body = html`<${ModelsPage} ctx=${ctx}/>`;
  else if (page === "robustness") body = html`<${RobustnessPage} ctx=${ctx}/>`;
  else if (page === "audit") body = html`<${AuditPage} ctx=${ctx}/>`;
  else if (page === "capture") body = html`<${CapturePage} ctx=${ctx}/>`;
  else if (page === "about") body = html`<${AboutPage} ctx=${ctx}/>`;
  else body = html`<${OverviewPage} ctx=${ctx}/>`;
  const counts = { alerts: st && st.alerts_emitted ? st.alerts_emitted : 0, hosts: st ? st.hosts_seen : 0 };

  return html`<div class="shell">
    <${Sidebar} page=${page} st=${st} counts=${counts}/>
    <div class="content">
      <header>
        <div><h1><${FX.ShinyText}>AegisFlow — network attack forecasting demo<//></h1>
          <div class="sub">Replays the held-out CIC-IDS2017 test period through the saved model. All scores are real model outputs.</div></div>
        <div class="row">${st && st.state === "running" && html`<div class="mini-progress" title="replay progress"><div style=${{ width: Math.round(st.progress * 100) + "%" }}></div></div>`}
          <span class=${"chip " + (st ? st.state : "idle")}>${st ? st.state + (st.state === "running" ? ` · ${st.speed}×` : "") : "idle"}</span>
          <span class="muted small">sim clock</span><span class="mono">${st ? t(st.sim_time) : "–"}</span></div>
      </header>
      ${body}
    </div>
  </div>`;
}

ReactDOM.createRoot(document.getElementById("root")).render(html`<${App}/>`);
