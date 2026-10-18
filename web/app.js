const $ = (s) => document.querySelector(s);
const NS = "http://www.w3.org/2000/svg";
const STAR_VAR = ["", "--star-1", "--star-2", "--star-3", "--star-4", "--star-5"];
const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();

let products = [];
let current = null; // product detail
let selected = null;
let shown = { raw: null, bridged: null, true: null };

function svgEl(tag, attrs = {}, parent) {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (parent) parent.appendChild(e);
  return e;
}

function starString(x) {
  const n = Math.round(x);
  return "★".repeat(n) + "☆".repeat(5 - n);
}

function fmtDelta(d) {
  if (Math.abs(d) < 0.005) return "±0.00";
  return (d > 0 ? "+" : "−") + Math.abs(d).toFixed(2);
}

async function api(path, opts) {
  const r = await fetch(path, opts);
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  return r.json();
}

function toast(msg) {
  const t = $("#toast");
  t.textContent = msg;
  t.hidden = false;
  clearTimeout(toast.t);
  toast.t = setTimeout(() => (t.hidden = true), 3500);
}

// ---------- sidebar ----------
async function loadProducts() {
  const data = await api("/api/products");
  products = data.products;
  $("#store-stats").textContent =
    `${data.n_reviews.toLocaleString()} reviews from ${data.n_reviewers.toLocaleString()} accounts. ` +
    `A red dot means the store rating and bridged rating differ by more than 0.3★.`;
  renderList();
}

function renderList() {
  const q = $("#filter").value.toLowerCase();
  const ul = $("#product-list");
  ul.innerHTML = "";
  for (const p of products) {
    if (q && !`${p.name} ${p.brand}`.toLowerCase().includes(q)) continue;
    const li = document.createElement("li");
    li.className = p.id === selected ? "active" : "";
    li.innerHTML = `<span class="nm">${p.name}${p.suspicious ? '<span class="flag" title="Big gap between raw and bridged"></span>' : ""}<small class="muted">${p.n_reviews} reviews</small></span>
      <span class="n raw">${p.raw.toFixed(2)}</span><span class="n br">${p.bridged.toFixed(2)}</span>`;
    li.onclick = () => select(p.id);
    ul.appendChild(li);
  }
}

// ---------- hero numbers ----------
function animateNumber(el, from, to, ms = 900) {
  if (from == null || Math.abs(from - to) < 1e-9) { el.textContent = to.toFixed(2); return; }
  const t0 = performance.now();
  const step = (t) => {
    const k = Math.min(1, (t - t0) / ms);
    const e = 1 - Math.pow(1 - k, 3);
    el.textContent = (from + (to - from) * e).toFixed(2);
    if (k < 1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

function renderHero(d, animate) {
  $("#p-brand").textContent = d.brand;
  $("#p-name").textContent = d.name;
  $("#p-count").textContent = `${d.n_reviews} reviews, ${d.n_established} from accounts with a track record`;
  for (const k of ["raw", "bridged", "true"]) {
    animateNumber($(`#s-${k}`), animate ? shown[k] : null, d[k]);
    $(`#st-${k}`).textContent = starString(d[k]);
    shown[k] = d[k];
  }
  for (const k of ["raw", "bridged"]) {
    const delta = d[k] - d[`baseline_${k}`];
    const el = $(`#d-${k}`);
    el.textContent = Math.abs(delta) >= 0.005 ? `${fmtDelta(delta)} since attacks began` : "";
    el.className = "delta" + (Math.abs(delta) > 0.25 ? " up" : "");
  }
}

// ---------- waterfall ----------
function renderWaterfall(d) {
  const svg = $("#waterfall");
  svg.innerHTML = "";
  const W = svg.clientWidth || 500, H = svg.clientHeight || 280;
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  const w = d.waterfall;
  const steps = [
    { label: "Bridged", value: w.bridged, total: true, color: css("--accent") },
    { label: "Leniency", value: w.leniency },
    { label: "Camp", value: w.viewpoint },
    { label: "New accounts", value: w.new_accounts },
    { label: "Noise", value: w.noise },
    { label: "Store", value: w.raw, total: true, color: css("--raw") },
  ];
  let run = 0;
  const bars = steps.map((s) => {
    if (s.total) { run = s.value; return { ...s, y0: 1, y1: s.value }; }
    const b = { ...s, y0: run, y1: run + s.value };
    run += s.value;
    return b;
  });
  const vals = bars.flatMap((b) => [b.y0, b.y1]);
  const lo = Math.max(1, Math.floor((Math.min(...bars.map((b) => b.total ? b.y1 : Math.min(b.y0, b.y1))) - 0.4) * 2) / 2);
  const hi = Math.min(5, Math.ceil((Math.max(...vals) + 0.3) * 2) / 2);
  const m = { l: 34, r: 8, t: 22, b: 40 };
  const y = (v) => m.t + (H - m.t - m.b) * (1 - (v - lo) / (hi - lo));
  const bw = (W - m.l - m.r) / bars.length;

  for (let v = Math.ceil(lo * 2) / 2; v <= hi + 1e-9; v += 0.5) {
    svgEl("line", { x1: m.l, x2: W - m.r, y1: y(v), y2: y(v), class: "axis" }, svg);
    svgEl("text", { x: m.l - 6, y: y(v) + 4, "text-anchor": "end" }, svg).textContent = v.toFixed(1) + "★";
  }
  bars.forEach((b, i) => {
    const x = m.l + i * bw + bw * 0.18, wd = bw * 0.64;
    const top = b.total ? y(b.y1) : y(Math.max(b.y0, b.y1));
    const bot = b.total ? y(lo) : y(Math.min(b.y0, b.y1));
    const color = b.color || (b.value >= 0 ? css("--raw") : css("--star-4"));
    svgEl("rect", { x, y: top, width: wd, height: Math.max(2, bot - top), rx: 4, fill: color, opacity: b.total ? 1 : 0.75 }, svg);
    const t = svgEl("text", { x: x + wd / 2, y: top - 6, "text-anchor": "middle", class: "lbl" }, svg);
    t.textContent = b.total ? b.value.toFixed(2) : fmtDelta(b.value);
    const lab = svgEl("text", { x: x + wd / 2, y: H - m.b + 16, "text-anchor": "middle" }, svg);
    const words = b.label.split(" ");  // one word per line keeps narrow bars legible
    lab.textContent = words[0];
    if (words.length > 1) svgEl("text", { x: x + wd / 2, y: H - m.b + 30, "text-anchor": "middle" }, svg).textContent = words.slice(1).join(" ");
    if (i < bars.length - 1) {
      const nextY = y(b.total ? b.y1 : b.y1);
      svgEl("line", { x1: x + wd, x2: x + bw, y1: nextY, y2: nextY, stroke: css("--muted"), "stroke-dasharray": "2 3" }, svg);
    }
  });
}

// ---------- reviewer map ----------
function renderMap(d) {
  const svg = $("#map");
  svg.innerHTML = "";
  const W = svg.clientWidth || 500, H = svg.clientHeight || 280;
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  const pts = d.reviewer_map;
  const reveal = document.body.classList.contains("reveal");
  if (!pts.length) {
    svgEl("text", { x: W / 2, y: H / 2, "text-anchor": "middle" }, svg).textContent = "Not enough established reviewers";
    return;
  }
  const xs = pts.map((p) => p.x), ys = pts.map((p) => p.y);
  const pad = (a) => { const lo = Math.min(...a), hi = Math.max(...a); const s = (hi - lo) || 1; return [lo - s * 0.06, hi + s * 0.06]; };
  const [x0, x1] = pad(xs), [y0, y1] = pad(ys);
  const m = { l: 12, r: 12, t: 12, b: 26 };
  const X = (v) => m.l + (W - m.l - m.r) * (v - x0) / (x1 - x0);
  const Y = (v) => m.t + (H - m.t - m.b) * (1 - (v - y0) / (y1 - y0));
  svgEl("line", { x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b, class: "axis" }, svg);
  svgEl("text", { x: W - m.r, y: H - 8, "text-anchor": "end" }, svg).textContent = "viewpoint axis → camp that rates this product higher";
  // Draw low-star dots first so a 5★ cluster sits on top.
  const order = [...pts].sort((a, b) => a.stars - b.stars);
  for (const p of order) {
    const c = svgEl("circle", {
      cx: X(p.x), cy: Y(p.y), r: 3.6, fill: css(STAR_VAR[p.stars]), "fill-opacity": 0.8,
      stroke: reveal && p.fake ? css("--danger") : "none", "stroke-width": 1.6,
    }, svg);
    svgEl("title", {}, c).textContent = `${p.stars}★ · camp effect ${fmtDelta(p.camp)}★${reveal && p.fake ? " · seeded fake" : ""}`;
  }
  const leg = $("#map-legend");
  leg.innerHTML = [1, 2, 3, 4, 5].map((s) => `<span><i style="background:var(${STAR_VAR[s]})"></i>${s}★</span>`).join("") +
    (reveal ? `<span><i style="background:transparent;border:2px solid var(--danger)"></i>seeded fake (ground truth)</span>` : "");
}

// ---------- timeline ----------
function renderTimeline(d) {
  const svg = $("#timeline");
  svg.innerHTML = "";
  const W = svg.clientWidth || 900, H = svg.clientHeight || 200;
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  const weeks = d.timeline;
  const max = Math.max(1, ...weeks.map((w) => w.reduce((a, b) => a + b, 0)));
  const m = { l: 34, r: 6, t: 8, b: 22 };
  const bw = (W - m.l - m.r) / weeks.length;
  const y = (v) => m.t + (H - m.t - m.b) * (1 - v / max);
  for (const v of [0, Math.round(max / 2), max]) {
    svgEl("line", { x1: m.l, x2: W - m.r, y1: y(v), y2: y(v), class: "axis" }, svg);
    svgEl("text", { x: m.l - 6, y: y(v) + 4, "text-anchor": "end" }, svg).textContent = v;
  }
  weeks.forEach((w, i) => {
    let acc = 0;
    w.forEach((n, s) => {
      if (!n) return;
      svgEl("rect", { x: m.l + i * bw + 1, y: y(acc + n), width: Math.max(1, bw - 2), height: y(acc) - y(acc + n), fill: css(STAR_VAR[s + 1]) }, svg);
      acc += n;
    });
    if (i % 8 === 0) svgEl("text", { x: m.l + i * bw + bw / 2, y: H - 6, "text-anchor": "middle" }, svg).textContent = `wk ${i + 1}`;
  });
}

// ---------- reviews ----------
function renderReviews(d) {
  const ul = $("#reviews");
  const reveal = document.body.classList.contains("reveal");
  ul.innerHTML = "";
  for (const r of d.reviews) {
    const li = document.createElement("li");
    const badges = [];
    if (r.excluded) badges.push(`<span class="badge" title="Fewer than 3 reviews, so it isn't counted">new account, not counted</span>`);
    else if (r.camp != null && Math.abs(r.camp) >= 0.5) badges.push(`<span class="badge camp">camp ${fmtDelta(r.camp)}★</span>`);
    if (reveal && r.fake) badges.push(`<span class="badge fake">seeded fake</span>`);
    li.innerHTML = `<span><span style="color:var(${STAR_VAR[r.stars]})">${"★".repeat(r.stars)}</span><br><span class="who">${r.reviewer} · day ${r.day}</span></span>
      <span>${r.text}</span><span class="badges">${badges.join("")}</span>`;
    ul.appendChild(li);
  }
}

function renderAll(d, animate = false) {
  current = d;
  renderHero(d, animate);
  renderWaterfall(d);
  renderMap(d);
  renderTimeline(d);
  renderReviews(d);
}

async function select(id) {
  selected = id;
  renderList();
  $("#result").hidden = true;
  const d = await api(`/api/products/${id}`);
  shown = { raw: null, bridged: null, true: null };
  renderAll(d);
  try { localStorage.setItem("br-selected", String(id)); } catch {}
}

// ---------- attack controls ----------
function segValue(id) { return document.querySelector(`#${id} button.on`).dataset.v; }
for (const id of ["mode", "direction"]) {
  document.querySelectorAll(`#${id} button`).forEach((b) => {
    b.onclick = () => {
      document.querySelectorAll(`#${id} button`).forEach((x) => x.classList.toggle("on", x === b));
    };
  });
}
$("#n-reviews").oninput = (e) => ($("#n-label").textContent = e.target.value);

$("#go").onclick = async () => {
  if (selected == null) return;
  const btn = $("#go");
  btn.disabled = true;
  btn.textContent = "Refitting…";
  try {
    const body = { product: selected, n_reviews: +$("#n-reviews").value, mode: segValue("mode"), direction: segValue("direction") };
    const res = await api("/api/attack", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    const b = res.before, a = res.after;
    renderAll(a, true);
    const dr = a.raw - b.raw, db = a.bridged - b.bridged;
    const ratio = Math.abs(db) > 0.01 ? `, ${(Math.abs(dr) / Math.abs(db)).toFixed(1)}× smaller` : ", effectively unmoved";
    const out = $("#result");
    out.hidden = false;
    out.innerHTML = `${body.n_reviews} ${body.mode === "ring" ? "review-ring" : "one-shot"} ${body.direction === "boost" ? "5★" : "1★"} reviews added.
      Store rating <b class="raw">${b.raw.toFixed(2)} → ${a.raw.toFixed(2)} (${fmtDelta(dr)})</b>.
      Bridged rating <b class="br">${b.bridged.toFixed(2)} → ${a.bridged.toFixed(2)} (${fmtDelta(db)}${ratio})</b>.
      <span class="muted">Whole store refit in ${a.fit_ms} ms.</span>`;
    await loadProducts();
  } catch (e) {
    toast("Attack failed: " + e.message);
  } finally {
    btn.disabled = false;
    btn.textContent = "Inject attack";
  }
};

$("#reset").onclick = async () => {
  await api("/api/reset", { method: "POST" });
  await loadProducts();
  if (selected != null) await select(selected);
  toast("Store reset to its original reviews");
};

$("#reveal").onchange = (e) => {
  document.body.classList.toggle("reveal", e.target.checked);
  if (current) { renderMap(current); renderReviews(current); }
};
$("#filter").oninput = renderList;

let resizeTimer;
window.addEventListener("resize", () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => current && (renderWaterfall(current), renderMap(current), renderTimeline(current)), 120);
});

(async () => {
  await loadProducts();
  let start = products.find((p) => p.name === "Zentrix Z1")?.id ?? 0;
  try { const s = localStorage.getItem("br-selected"); if (s != null && products[+s]) start = +s; } catch {}
  await select(start);
})();
