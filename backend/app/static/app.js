// AegisFlow dashboard (React 18 + htm, no build step). Every number shown comes from the backend API;
// risk weights, level bands, speeds and the stage list are read from it, never hard-coded here.
"use strict";
const { useState, useEffect, useCallback, useRef } = React;
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
function SignedBars({ items, label, value, extra }) {
  const max = Math.max(1e-9, ...items.map(i => Math.abs(value(i))));
  return html`<div>${items.map((it, i) => {
    const v = value(it), w = Math.abs(v) / max * 50;
    return html`<div class="hbar" key=${i}><div class="mono" title=${label(it)}>${label(it)}</div>
      <div class="track"><div class="mid"></div>
        <div class=${"fill " + (v >= 0 ? "pos" : "neg")} style=${{ left: v >= 0 ? "50%" : (50 - w) + "%", width: w + "%" }}></div></div>
      <div class="small" style=${{ textAlign: "right" }}>${extra ? extra(it) : fmt(v, 3)}</div></div>`;
  })}</div>`;
}
function Bars({ items, label, value, cls = "acc", max }) {
  const m = max ?? Math.max(1e-9, ...items.map(value));
  return html`<div>${items.map((it, i) => html`<div class="hbar" key=${i}><div class="mono">${label(it)}</div>
    <div class="track"><div class=${"fill " + cls} style=${{ left: 0, width: (value(it) / m * 100) + "%" }}></div></div>
    <div class="small" style=${{ textAlign: "right" }}>${fmt(value(it), 3)}</div></div>`)}</div>`;
}
function StepBars({ values, signed, title }) {
  const max = Math.max(1e-9, ...values.map(v => Math.abs(v)));
  return html`<div><div class="steps" role="img" aria-label=${title}>${values.map((v, i) => html`<div key=${i} title=${`window t-${values.length - 1 - i}: ${fmt(v, 4)}`}>
      <span class=${signed ? (v >= 0 ? "pos" : "neg") : "acc"} style=${{ height: (Math.abs(v) / max * 100) + "%" }}></span>
      <div>${i === values.length - 1 ? "last" : `-${values.length - 1 - i}`}</div></div>`)}</div></div>`;
}
function Sparkline({ tl, cfg }) {
  if (!tl || !tl.length || !cfg) return null;
  const W = 640, H = 150, P = 24, n = tl.length, top = cfg.max_reachable_risk || 100;
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
  return html`<div class="section"><h3>Why this prediction (feature attribution)</h3>
    <div class="row tabs" style=${{ marginBottom: "8px" }}>
      <button class=${model === "lstm" ? "on" : ""} onClick=${() => setModel("lstm")}>LSTM</button>
      <button class=${model === "logistic_regression" ? "on" : ""} onClick=${() => setModel("logistic_regression")}>Logistic regression</button>
      ${model === "lstm" && html`<span class="small muted" style=${{ marginLeft: "8px" }}>Method:</span>
        <button class=${method === "shap" ? "on" : ""} disabled=${!feats.explanations.shap} title=${feats.explanations.hint || ""} onClick=${() => setMethod("shap")}>SHAP</button>
        <button class=${method === "integrated_gradients" ? "on" : ""} onClick=${() => setMethod("integrated_gradients")}>Integrated Gradients</button>`}
    </div>
    ${ex.loading ? html`<div class="small muted">Computing attributions…</div>`
      : ex.error ? html`<div class="err">${ex.error.message}</div>`
      : html`<div>
        <${SignedBars} items=${ex.data.top_features} label=${f => f.feature} value=${f => f.attribution}
          extra=${f => html`<span title=${`most influential at window ${f.most_influential_step}; last-window value ${f.last_window_value}`}>${fmt(f.attribution, 3)}</span>`}/>
        <div class="small muted" style=${{ margin: "6px 0 10px" }}><span class="bad">■</span> pushes towards attack · <span class="ok">■</span> towards benign.
          Attributions explain the log-odds (${fmt(ex.data.output, 3)} vs reference ${fmt(ex.data.reference_output, 3)}; ${ex.data.method === "integrated_gradients" || ex.data.model !== "lstm" ? `additivity gap ${fmt(ex.data.additivity_gap, 5)}` : "SHAP expected gradients, sampled"}).</div>
        <div class="small muted">Attribution per input window (oldest → last):</div>
        <${StepBars} values=${ex.data.timestep_attribution} signed=${true} title="attribution per input window"/>
      </div>`}
  </div>`;
}

function AttentionSection({ seqId, models }) {
  const attn = models.filter(m => m.has_attention);
  const [name, setName] = useState(attn.length ? attn[0].name : null);
  const at = useLoad(() => name ? api(`/attention/${encodeURIComponent(seqId)}?model=${encodeURIComponent(name)}`) : Promise.resolve(null), [seqId, name]);
  return html`<div class="section"><h3>Attention over input windows</h3>
    ${!attn.length ? html`<div class="small muted">No attention model is trained on this machine. Train one with
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
      <button disabled=${busy || !allowed("acknowledge")} onClick=${() => act("acknowledge")}>Acknowledge</button>
      <button disabled=${busy || !allowed("dismiss")} onClick=${() => act("dismiss")}>Dismiss</button>
      <button disabled=${busy || !allowed("reopen")} onClick=${() => act("reopen")}>Reopen</button>
      <button disabled=${busy || !note.trim()} onClick=${() => act("comment")}>Comment</button>
    </div>
    <div class="row" style=${{ marginBottom: "6px" }}>
      <input type="text" placeholder="Response to approve, e.g. block 172.16.0.1 at the edge" value=${response} onInput=${e => setResponse(e.target.value)} style=${{ flex: 1, minWidth: "200px" }}/>
      <button class="primary" disabled=${busy || !allowed("approve_response") || !response.trim()} onClick=${() => act("approve_response")}>Approve response</button>
    </div>
    <div class="row">
      <select value=${stage} onChange=${e => setStage(e.target.value)}>${stages.map(s => html`<option key=${s} value=${s}>${s}</option>`)}</select>
      <button disabled=${busy} onClick=${() => act("override_stage")}>Override stage</button>
    </div>
    ${err && html`<div class="err" style=${{ marginTop: "6px" }}>${err}</div>`}
    ${h && h.actions.length > 0 && html`<div class="scroll" style=${{ maxHeight: "180px", marginTop: "8px" }}><table>
      <thead><tr><th>When</th><th>Analyst</th><th>Action</th><th>Detail</th><th>hash</th></tr></thead>
      <tbody>${h.actions.slice().reverse().map(a => html`<tr key=${a.id} style=${{ cursor: "default" }}><td class="mono">${t(a.created_at)}</td><td>${a.analyst}</td>
        <td>${a.action}</td><td class="small">${a.stage || a.response || a.note || ""}</td><td class="mono">${a.hash.slice(0, 10)}…</td></tr>`)}</tbody></table></div>`}
    <div class="small muted" style=${{ marginTop: "6px" }}>Every action is appended to its own SHA-256 hash chain, bound to the alert's ledger hash.</div>
  </div>`;
}

function AlertDetail({ id, cfg, feats, models, onChange }) {
  const al = useLoad(() => api(`/alerts/${id}`), [id]);
  const a = al.data;
  const mitre = useLoad(() => a ? api(`/mitre/${encodeURIComponent(a.truth_stage)}`).catch(() => null) : Promise.resolve(null), [a && a.truth_stage]);
  if (al.loading) return html`<div class="muted">Loading alert…</div>`;
  if (al.error) return html`<div class="err">${al.error.message}</div>`;
  const comp = JSON.parse(a.risk_components || "{}"), lvl = levelOf(a.risk_score, cfg), m = mitre.data;
  const stageModel = feats.stage_model.available;
  return html`<div>
    <div class="kv">
      <div>Risk score</div><div><b class=${"lvl-" + lvl} style=${{ fontSize: "20px" }}>${fmt(a.risk_score, 1)}</b> / 100${" "}
        <span class="muted small">(${lvl}; max reachable ${fmt(cfg.max_reachable_risk, 0)}${stageModel ? "" : " — no stage model, so the stage term is 0"})</span></div>
      <div>Risk components</div><div class="small">${Object.entries(comp).map(([k, v]) => `${k} ${fmt(v, 3)} × ${cfg.weights[k]}`).join(" · ")}</div>
      <div>Forecast made at</div><div class="mono">${t(a.predicted_at)} <span class="muted">(end of 10 input windows)</span></div>
      <div>Target window</div><div class="mono">${t(a.target_window_start)} → ${hm(a.target_window_end)}</div>
      <div>LSTM attack probability</div><div>${fmt(a.lstm_probability, 4)} <span class="muted small">threshold ${fmt(a.lstm_threshold, 4)}</span></div>
      <div>Confidence</div><div><b>${a.confidence_label}</b></div>
      <div>Predicted stage</div><div><b>${a.predicted_stage || "–"}</b> <span class="muted small">${stageModel ? "from the multi-task stage head" : "no stage model configured"}</span></div>
      <div>Logistic regression</div><div>p = ${fmt(a.lr_probability, 4)} · ${a.lr_flag ? "also flags" : "does not flag"} <span class="muted small">(threshold ${fmt(a.lr_threshold, 5)})</span></div>
    </div>
    <${ExplanationSection} seqId=${a.sequence_id} feats=${feats}/>
    <${AttentionSection} seqId=${a.sequence_id} models=${models}/>
    <${StageSection} seqId=${a.sequence_id} feats=${feats}/>
    <${AnalystSection} alertId=${a.id} stages=${feats.stages} onChange=${onChange}/>
    <div class="note info" style=${{ margin: "12px 0" }}>
      <b>Dataset label (ground truth, revealed for evaluation, not available to the model):</b> ${a.truth_class} / ${a.truth_stage} →
      ${a.truth_attack === 1 ? html` <span class="ok">true positive</span>` : html` <span class="bad">false positive</span>`}<br/>
      <b>MITRE ATT&CK for that label:</b> ${m && m.tactic_id ? `${m.tactic_id} ${m.tactic_name} — ${m.techniques.map(x => `${x.id} ${x.name}`).join(", ")}` : "none (benign)"}
      <span class="muted small"> · static lookup from configs/mitre_mapping.yaml</span>
    </div>
    <div class="kv small">
      <div>Ledger hash</div><div class="mono">${a.hash}</div>
      <div>Previous hash</div><div class="mono">${a.prev_hash}</div>
      <div>Sequence / model</div><div class="mono">${a.sequence_id} · ${a.model_version}</div>
    </div>
  </div>`;
}

function HostDetail({ id, cfg, tick, onAlert }) {
  const h = useLoad(() => api(`/hosts/${encodeURIComponent(id)}`), [id, tick]);
  if (!h.data) return h.error ? html`<div class="err">${h.error.message}</div>` : html`<div class="muted">Loading host…</div>`;
  const d = h.data;
  return html`<div>
    <div class="kv"><div>Sequences replayed</div><div>${d.sequences}</div><div>Alerts</div><div>${d.alerts}</div>
      <div>Latest / max risk</div><div><span class=${"lvl-" + levelOf(d.latest_risk, cfg)}>${fmt(d.latest_risk, 1)}</span> / <span class=${"lvl-" + levelOf(d.max_risk, cfg)}>${fmt(d.max_risk, 1)}</span></div>
      <div>Last seen</div><div class="mono">${t(d.last_seen)}</div></div>
    <div style=${{ marginTop: "10px" }}><${Sparkline} tl=${d.timeline} cfg=${cfg}/></div>
    <div class="small muted">Risk over replay time; red dots are alerts.</div>
    <div class="scroll" style=${{ maxHeight: "230px", marginTop: "8px" }}><table>
      <thead><tr><th>#</th><th>Target window</th><th class="num">Risk</th><th class="num">LSTM p</th><th>Stage</th><th>Dataset label</th></tr></thead>
      <tbody>${d.alerts_list.length ? d.alerts_list.map(a => html`<tr key=${a.id} onClick=${() => onAlert(a.id)}><td>${a.id}</td>
        <td class="mono">${hm(a.target_window_start)}</td><td class=${"num lvl-" + levelOf(a.risk_score, cfg)}>${fmt(a.risk_score, 1)}</td>
        <td class="num">${fmt(a.lstm_probability)}</td><td>${a.predicted_stage}</td><td class=${a.truth_attack ? "ok" : "bad"}>${a.truth_class}</td></tr>`)
        : html`<tr><td colspan="6" class="muted">No alerts for this host.</td></tr>`}</tbody></table></div>
  </div>`;
}

// ------------------------------------------------------------------ cards
function AuditCard({ alerts }) {
  const [v, setV] = useState(null), [va, setVa] = useState(null);
  const check = async () => { setV(await api("/audit/verify")); setVa(await api("/audit/verify-actions")); };
  const line = (r, what) => !r ? html`<span class="verify muted">not checked</span>`
    : html`<span><span class=${"verify " + (r.status === "VERIFIED" ? "ok" : "bad")}>${r.status}</span>
      <span class="small muted"> ${r.status === "VERIFIED" ? `${r[what]} records` : `record #${r.record_id ?? r.action_id}: ${r.reason}`}</span></span>`;
  return html`<section class="card span5"><h2>Audit ledger (SHA-256 hash chains)</h2>
    <div class="row"><button class="primary" onClick=${check}>Verify chains</button></div>
    <div class="kv small" style=${{ margin: "8px 0 10px", gridTemplateColumns: "120px 1fr" }}>
      <div>Alerts chain</div><div>${line(v, "records_checked")}</div>
      <div>Analyst actions</div><div>${line(va, "actions_checked")}</div></div>
    <div class="scroll" style=${{ maxHeight: "220px" }}><table>
      <thead><tr><th>#</th><th>Host</th><th>prev_hash</th><th>hash</th></tr></thead>
      <tbody>${alerts.slice(0, 12).map(a => html`<tr key=${a.id} style=${{ cursor: "default" }}><td>${a.id}</td><td class="mono">${a.host_id}</td>
        <td class="mono">${a.prev_hash.slice(0, 10)}…</td><td class="mono">${a.hash.slice(0, 10)}…</td></tr>`)}</tbody></table></div>
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
    <div class="scroll"><table>
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
    <div class="scroll"><table>
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

// ------------------------------------------------------------------ app
function App() {
  const [boot, setBoot] = useState(null), [bootErr, setBootErr] = useState(null);
  const [live, setLive] = useState({ st: null, hosts: [], alerts: [], statuses: {}, triage: {} }), [liveErr, setLiveErr] = useState(null);
  const [speed, setSpeed] = useState(null), [starting, setStarting] = useState(false);
  const [sel, setSel] = useState(null), [tick, setTick] = useState(0);
  const timer = useRef(null);

  useEffect(() => {
    (async () => {
      const cfg = await api("/risk/config");
      const [ds, model, ev, feats, models] = await Promise.all([api("/dataset/status"), api("/model/status"), api("/evaluation"), api("/features"), api("/models")]);
      setSpeed(cfg.default_speed); setBoot({ cfg, ds, model, ev, feats, models });
    })().catch(setBootErr);
  }, []);

  const poll = useCallback(async () => {
    try {
      const [st, hosts, alerts, statuses, triage] = await Promise.all([api("/replay/status"), api("/hosts"), api("/alerts?limit=50"),
        api("/triage/statuses"), api("/triage/summary")]);
      setLive({ st, hosts, alerts, statuses, triage }); setLiveErr(null);
      if (st.state === "running") setTick(x => x + 1);
    } catch (e) { setLiveErr(e.message); }
  }, []);
  useEffect(() => {
    if (!boot) return;
    let stop = false;
    const loop = async () => { await poll(); if (!stop) timer.current = setTimeout(loop, 1000); };
    loop();
    return () => { stop = true; clearTimeout(timer.current); };
  }, [boot, poll]);

  if (bootErr) return html`<main><section class="card span12"><h2>Cannot load</h2><div class="err">${bootErr.message}</div></section></main>`;
  if (!boot) return html`<p style=${{ padding: "20px" }} class="muted">Loading dashboard…</p>`;
  const { cfg, ds, model, ev, feats, models } = boot, { st, hosts, alerts, statuses, triage } = live;
  const pct = st ? Math.round(st.progress * 100) : 0, tp = st ? st.alerts_matching_attack_label : 0;
  const start = async () => {
    setStarting(true);
    try { await post("/replay/start", { speed: +speed, reset: true }); setSel(null); } catch (e) { setLiveErr(e.message); }
    setStarting(false);
  };
  const triageLine = Object.entries(triage).filter(([, n]) => n).map(([k, n]) => `${n} ${STATUS_LABEL[k] || k}`).join(" · ");

  return html`<div>
  <header>
    <div><h1>AegisFlow — network attack forecasting demo</h1>
      <div class="sub">Replays the held-out CIC-IDS2017 test period through the saved model. All scores are real model outputs. <a href="/classic">classic view</a></div></div>
    <div class="row"><span class=${"chip " + (st ? st.state : "idle")}>${st ? st.state + (st.state === "running" ? ` · ${st.speed}×` : "") : "idle"}</span>
      <span class="muted small">sim clock</span><span class="mono">${st ? t(st.sim_time) : "–"}</span></div>
  </header>
  <main>
    <section class="card span3"><h2>Hosts monitored</h2><div class="big">${st ? st.hosts_seen.toLocaleString() : 0}</div>
      <div class="small muted">of ${ds.hosts.toLocaleString()} in dataset · test split ${ds.split.test_sequences.toLocaleString()} sequences</div></section>
    <section class="card span3"><h2>Alerts this replay</h2><div class="big">${st ? st.alerts_emitted.toLocaleString() : 0}</div>
      <div class="small muted">${st && st.alerts_emitted ? `${tp} match an attack label, ${st.alerts_emitted - tp} benign` : " "}</div>
      <div class="small muted">${triageLine}</div></section>
    <section class="card span3"><h2>Replay</h2><div class="big">${pct}%</div>
      <div class="bar"><div style=${{ width: pct + "%" }}></div></div>
      <div class="small muted">${liveErr ? "backend: " + liveErr : st ? (st.model_ready ? `${st.processed.toLocaleString()} / ${st.total.toLocaleString()} test sequences` : "loading model and scoring test split…") + (st.error ? ` · ERROR ${st.error}` : "") : " "}</div></section>
    <section class="card span3"><h2>Replay controls</h2>
      <div class="row"><label class="small muted" for="speed">Speed</label>
        <select id="speed" value=${speed} onChange=${e => setSpeed(e.target.value)}>${cfg.allowed_speeds.map(s => html`<option key=${s} value=${s}>${s}×</option>`)}</select>
        <button class="primary" disabled=${starting || (st && st.state === "running")} onClick=${start}>${starting ? "Starting…" : "Start"}</button>
        <button class="danger" disabled=${!st || st.state !== "running"} onClick=${() => post("/replay/stop")}>Stop</button></div>
      <div class="small muted" style=${{ marginTop: "8px" }}>Model <span class="mono">${model.version}</span><br/>alert when LSTM p ≥ ${fmt(model.lstm_threshold, 4)} (validation-selected)</div>
    </section>

    <section class="card span6"><h2>Host risk</h2><div class="scroll"><table>
      <thead><tr><th>Host</th><th class="num">Seqs</th><th class="num">Alerts</th><th class="num">Latest risk</th><th class="num">Max risk</th><th class="num">LSTM p</th></tr></thead>
      <tbody>${hosts.length ? hosts.slice(0, 60).map(h => html`<tr key=${h.host_id} class=${sel && sel.host === h.host_id ? "sel" : ""} onClick=${() => setSel({ host: h.host_id })}>
        <td class="mono">${h.host_id}</td><td class="num">${h.sequences}</td><td class="num">${h.alerts}</td>
        <td class=${"num lvl-" + levelOf(h.latest_risk, cfg)}>${fmt(h.latest_risk, 1)}</td><td class=${"num lvl-" + levelOf(h.max_risk, cfg)}>${fmt(h.max_risk, 1)}</td>
        <td class="num">${fmt(h.latest_lstm_p)}</td></tr>`) : html`<tr><td colspan="6" class="muted">Start the replay to see hosts.</td></tr>`}</tbody></table></div></section>
    <section class="card span6"><h2>Alerts (newest first)</h2><div class="scroll"><table>
      <thead><tr><th>#</th><th>Host</th><th>Target</th><th class="num">Risk</th><th class="num">LSTM p</th><th>Stage</th><th>Status</th><th>Dataset label</th></tr></thead>
      <tbody>${alerts.length ? alerts.map(a => { const s = statuses[a.id] || "open"; return html`<tr key=${a.id} class=${sel && sel.alert === a.id ? "sel" : ""} onClick=${() => setSel({ alert: a.id })}>
        <td>${a.id}</td><td class="mono">${a.host_id}</td><td class="mono">${hm(a.target_window_start)}</td>
        <td class=${"num lvl-" + levelOf(a.risk_score, cfg)}>${fmt(a.risk_score, 1)}</td><td class="num">${fmt(a.lstm_probability)}</td>
        <td>${a.predicted_stage}</td><td><span class=${"status " + s}>${STATUS_LABEL[s] || s}</span></td><td>${a.truth_class}</td></tr>`; })
        : html`<tr><td colspan="8" class="muted">No alerts yet.</td></tr>`}</tbody></table></div></section>

    <section class="card span7"><h2>${sel ? (sel.alert ? `Alert #${sel.alert}` : `Host ${sel.host}`) : "Detail"}</h2>
      ${!sel ? html`<div class="muted">Click an alert to see why it fired, its stage forecast and the analyst controls, or click a host for its risk timeline.</div>`
        : sel.alert ? html`<${AlertDetail} key=${sel.alert} id=${sel.alert} cfg=${cfg} feats=${feats} models=${models} onChange=${poll}/>`
        : html`<${HostDetail} key=${sel.host} id=${sel.host} cfg=${cfg} tick=${Math.floor(tick / 3)} onAlert=${id => setSel({ alert: id })}/>`}
    </section>
    <${AuditCard} alerts=${alerts}/>
    <${CaptureCard} feats=${feats}/>
    <${SiemCard}/>
    <${ModelsCard} models=${models}/>
    <${EvaluationCard} ev=${ev} model=${model} feats=${feats}/>
  </main></div>`;
}

ReactDOM.createRoot(document.getElementById("root")).render(html`<${App}/>`);
