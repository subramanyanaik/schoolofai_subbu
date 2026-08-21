/* ------------------------------------------------------------------
   app.js — wiring. Renders the timeline from data.js and hooks up
   every interactive lab.
   ------------------------------------------------------------------ */
(function () {
  "use strict";
  const $ = (s, r) => (r || document).querySelector(s);
  const $$ = (s, r) => Array.from((r || document).querySelectorAll(s));
  const fmtBytes = (b) => {
    if (b < 1024) return b.toFixed(0) + " B";
    const u = ["KB", "MB", "GB", "TB", "PB"];
    let i = -1; let v = b;
    do { v /= 1024; i++; } while (v >= 1024 && i < u.length - 1);
    return v.toFixed(v < 10 ? 2 : v < 100 ? 1 : 0) + " " + u[i];
  };
  const famVar = (f) => "var(--" + f + ")";

  /* ================= hero stats ================= */
  $("#stat-count").textContent = MECHANISMS.length;
  $("#stat-notes").textContent = MECHANISMS.filter(m => m.dateNote).length;
  $("#stat-cons").textContent = MECHANISMS.reduce((a, m) => a + m.cons.length, 0);
  $("#src-count").textContent = MECHANISMS.filter(m => m.dateNote).length;

  /* ================= bill 2 static numbers ================= */
  (function bill2() {
    const bytes = (users, ctx) => 2 * 80 * 64 * 128 * 2 * ctx * users;
    $("#b2-1").textContent = fmtBytes(bytes(1, 32768));
    $("#b2-8").textContent = fmtBytes(bytes(8, 32768));
    $("#b2-32").textContent = fmtBytes(bytes(32, 32768));
    $("#b2-1m").textContent = fmtBytes(bytes(32, 1048576));
  })();

  /* ================= legend ================= */
  $("#legend").innerHTML = Object.keys(FAMILIES).map(k =>
    `<span><i class="dot" style="background:${famVar(k)}"></i>${FAMILIES[k].label}</span>`).join("");

  /* ================= timeline ================= */
  const tlRoot = $("#tl-root");
  let filterFam = "all", onlySession = false;

  function detailHTML(m) {
    return `
      <div class="detail" id="d-${m.id}">
        <div class="who">${m.who}</div>
        <h4>The problem it answered</h4>
        <p>${m.problem}</p>
        <h4>What it does</h4>
        <p>${m.idea}</p>
        <div class="math">${m.math}</div>
        ${m.whyScale ? `<h4>Why the &radic;d<sub>k</sub></h4><p>${m.whyScale}</p>` : ""}
        ${m.dateNote ? `<h4>On the date</h4><div class="note"><b>date check</b>${m.dateNote}</div>` : ""}
        <div class="pc" style="margin-top:22px">
          <div><h4 style="margin-top:0">What it buys</h4><ul class="pros">${m.pros.map(p => `<li>${p}</li>`).join("")}</ul></div>
          <div><h4 style="margin-top:0">What it gives up</h4><ul class="cons">${m.cons.map(p => `<li>${p}</li>`).join("")}</ul></div>
        </div>
        <div class="pick" style="margin-top:18px"><b>when you would actually pick it</b>${m.pick}</div>
        <div class="srcs">${m.src.map(s => `<a href="${s.url}" target="_blank" rel="noopener">${s.label} &#8599;</a>`).join("")}</div>
      </div>`;
  }

  function renderTimeline() {
    let html = "";
    ERAS.forEach(era => {
      const items = MECHANISMS.filter(m => m.era === era.id
        && (filterFam === "all" || m.family === filterFam)
        && (!onlySession || m.covered));
      if (!items.length) return;
      html += `<div class="era">
        <div class="era-head">
          <span class="num">ERA ${era.id}</span>
          <h3>${era.title}</h3>
          <span class="yrs">${era.from.slice(0, 4)}&ndash;${era.to.slice(0, 4)}</span>
        </div>
        <p class="era-thesis">&ldquo;${era.thesis}&rdquo;</p>
        <p class="era-body">${era.body}</p>
        <div class="tl">`;
      items.forEach(m => {
        html += `<div class="row ${m.anchor ? "anchor" : ""}" style="--c:${famVar(m.family)}" id="m-${m.id}">
          <button class="card" aria-expanded="false" data-id="${m.id}">
            <div class="card-top">
              <span class="date">${m.dateLabel}</span>
              <span class="nm">${m.name}</span>
              <span class="ab">${m.abbr}</span>
              <span class="badge ${m.covered ? "session" : "extra"}">${m.covered ? "session" : "extra"}</span>
              ${m.dateNote ? '<span class="badge">date note</span>' : ""}
            </div>
            <div class="hl">${m.headline}</div>
          </button>
          ${detailHTML(m)}
        </div>`;
      });
      html += `</div></div>`;
    });
    tlRoot.innerHTML = html || '<p class="mut">Nothing matches that filter.</p>';
  }
  renderTimeline();

  tlRoot.addEventListener("click", e => {
    const btn = e.target.closest(".card");
    if (!btn) return;
    const open = btn.getAttribute("aria-expanded") === "true";
    btn.setAttribute("aria-expanded", String(!open));
    $("#d-" + btn.dataset.id).classList.toggle("open", !open);
  });

  $$(".tl-filters .chip[data-fam]").forEach(b => b.addEventListener("click", () => {
    $$(".tl-filters .chip[data-fam]").forEach(x => x.classList.remove("on"));
    b.classList.add("on"); filterFam = b.dataset.fam; renderTimeline();
  }));
  $("#only-session").addEventListener("click", function () {
    onlySession = !onlySession; this.classList.toggle("on", onlySession); renderTimeline();
  });
  $("#expand-all").addEventListener("click", function () {
    const anyClosed = $$("#tl-root .card").some(c => c.getAttribute("aria-expanded") !== "true");
    $$("#tl-root .card").forEach(c => {
      c.setAttribute("aria-expanded", String(anyClosed));
      $("#d-" + c.dataset.id).classList.toggle("open", anyClosed);
    });
    this.textContent = anyClosed ? "collapse all" : "expand all";
  });

  /* ================= playground ================= */
  const pg = { scale: true, causal: true, softmax: true, amp: 1 };
  function drawPlayground() {
    const tokens = $("#pg-sent").value.split(" ");
    const res = VIZ.attention(tokens, pg);
    VIZ.renderMatrix($("#pg-matrix"), tokens, res, pg);
    const notes = [];
    if (!pg.scale) notes.push("<b>Without &radic;d<sub>k</sub></b> the logits are wide, so softmax saturates &mdash; rows collapse toward one-hot and gradients vanish.");
    if (!pg.causal) notes.push("<b>Without the mask</b> every token reads the future. Training would be trivial and generation impossible.");
    if (!pg.softmax) notes.push("<b>Without softmax</b> the weights no longer sum to 1 and can be negative &mdash; but now (QK&#7488;)V can be reassociated to Q(K&#7488;V), and the past collapses into a fixed d&times;d state. That is <a href='#m-linear'>linear attention</a>, and this is exactly what it costs.");
    if (pg.amp !== 1) notes.push(`<b>Amplitude &times;${pg.amp.toFixed(1)}</b> &mdash; softmax responds to the <em>absolute</em> gap between logits, not the relative one. Scale everything up and the same ranking becomes far sharper.`);
    if (!notes.length) notes.push("Rows sum to 1. Each row is one query&rsquo;s distribution over the keys it is allowed to see. Grey cells are masked future.");
    $("#pg-cap").innerHTML = notes.join(" ");
  }
  $("#pg-sent").addEventListener("change", drawPlayground);
  [["pg-scale", "scale"], ["pg-mask", "causal"], ["pg-softmax", "softmax"]].forEach(([id, key]) => {
    $("#" + id).addEventListener("click", function () {
      pg[key] = !pg[key];
      this.setAttribute("aria-pressed", String(pg[key]));
      drawPlayground();
    });
  });
  $("#pg-amp").addEventListener("input", function () {
    pg.amp = +this.value; $("#pg-amp-v").textContent = pg.amp.toFixed(1) + "×"; drawPlayground();
  });
  drawPlayground();

  /* ================= mask lab ================= */
  const MASK_ORDER = ["dense", "window", "sinks", "strided", "bigbird", "topk", "nsa", "dsa", "csa"];
  const MASK_LABEL = {
    dense: "dense causal", window: "sliding window", sinks: "window + sinks", strided: "strided",
    bigbird: "BigBird", topk: "top-k", nsa: "NSA", dsa: "DSA", csa: "CSA"
  };
  let maskKind = "dense";
  const maskP = { w: 8, sink: 4, k: 6 };
  const MASK_N = 48;
  const maskScore = (i, j) => VIZ.hrand("s" + i, j) + (j === i ? 2 : 0) - (i - j) * 0.004;

  $("#mask-ctl").innerHTML = MASK_ORDER.map(k =>
    `<button class="chip${k === "dense" ? " on" : ""}" data-mask="${k}">${MASK_LABEL[k]}</button>`).join("");

  function drawMaskLab() {
    const M = VIZ.buildMask(maskKind, MASK_N, maskP, maskScore);
    const r = VIZ.drawMask($("#mask-canvas"), M);
    const P = VIZ.PATTERNS[maskKind];
    $("#mask-title").textContent = P.title;
    $("#mask-desc").innerHTML = P.desc;
    $("#mask-note").innerHTML = P.note;
    const pct = (r.count / r.dense) * 100;
    $("#mask-bar").style.width = pct + "%";
    $("#mask-pct").textContent = pct.toFixed(1) + "% of dense";
    // What actually decides the trade-off is keys read per query as N grows,
    // and that is invisible on a 48x48 grid.
    const BIG = 131072;
    const kpq = P.keysPerQuery(BIG, maskP);
    const pct2 = (kpq / (BIG / 2)) * 100;
    $("#mask-bar2").style.width = Math.max(0.4, Math.min(100, pct2)) + "%";
    $("#mask-pct2").textContent = pct2 >= 1 ? pct2.toFixed(1) + "%" : pct2.toFixed(3) + "%";

    $("#mask-scale").innerHTML = maskKind === "dense"
      ? `At 128K a query reads <b>65,536</b> keys on average. That is the number every pattern below
         is trying to bring down &mdash; and the grid above cannot show you the difference, because
         at N=48 dense attention is still cheap.`
      : `A query reads about <b>${Math.round(kpq).toLocaleString()}</b> keys at 128K, against
         <b>65,536</b> for dense. This second bar is the one that matters:
         <em>a mechanism that looks unimpressive at N=48 can be the only survivor at N=1M</em>,
         and vice versa. The grid shows you the shape; this shows you the bill.
         <span class="mut">Projection uses each paper's own constants &mdash; NSA's 32/16 compression
         with 16 selected blocks of 64 and a 512 window, CSA's compress-then-top-k, Sparse
         Transformer's O(&radic;N) &mdash; not the block sizes drawn above, which are chosen to be
         visible at N=48.</span>`;
  }
  $("#mask-ctl").addEventListener("click", e => {
    const b = e.target.closest("[data-mask]"); if (!b) return;
    $$("#mask-ctl .chip").forEach(x => x.classList.remove("on"));
    b.classList.add("on"); maskKind = b.dataset.mask; drawMaskLab();
  });
  [["mask-w", "w"], ["mask-sink", "sink"], ["mask-k", "k"]].forEach(([id, key]) => {
    $("#" + id).addEventListener("input", function () {
      maskP[key] = +this.value; $("#" + id + "-v").textContent = this.value; drawMaskLab();
    });
  });
  drawMaskLab();

  /* ================= RoPE circle ================= */
  function drawRopeLab() {
    const m = +$("#rope-m").value, n = +$("#rope-n").value;
    $("#rope-m-v").textContent = m; $("#rope-n-v").textContent = n;
    VIZ.drawRope($("#rope-canvas"), m, n);
    $("#rope-cap").innerHTML = `Both tokens shifted by the same amount &mdash; say the whole passage moves 1,000 tokens later &mdash;
      and the score is <em>bit-identical</em>, because R<sub>m</sub><sup>T</sup>R<sub>n</sub> = R<sub>n&minus;m</sub>.
      That is why RoPE is applied to Q and K inside the attention block rather than added to the token at the bottom.`;
  }
  $("#rope-m").addEventListener("input", drawRopeLab);
  $("#rope-n").addEventListener("input", drawRopeLab);
  drawRopeLab();

  /* ================= decay curves ================= */
  const decayShow = { rope: true, alibi: true, swa: true, ideal: true };
  $("#decay-ctl").innerHTML = [["rope", "RoPE"], ["alibi", "ALiBi"], ["swa", "sliding window"], ["ideal", "what retrieval needs"]]
    .map(([k, l]) => `<button class="chip on" data-dec="${k}">${l}</button>`).join("");
  function drawDecayLab() {
    VIZ.drawDecay($("#decay-canvas"), decayShow);
    $("#decay-cap").innerHTML = `RoPE degrades sharply just past what it was trained on. ALiBi never degrades &mdash; because it has
      already decayed to nothing, which is a different thing from working. Sliding window is a cliff by construction.
      The dashed line is what a retrieval task actually needs: a token 400 away must still be <em>reachable</em>.
      Nothing here delivers it, which is why the field keeps coming back to memory.`;
  }
  $("#decay-ctl").addEventListener("click", e => {
    const b = e.target.closest("[data-dec]"); if (!b) return;
    decayShow[b.dataset.dec] = !decayShow[b.dataset.dec];
    b.classList.toggle("on", decayShow[b.dataset.dec]);
    drawDecayLab();
  });
  drawDecayLab();

  /* ================= RoPE extension ================= */
  const extShow = { none: true, pi: true, ntk: true, yarn: true };
  function drawExtLab() {
    const s = +$("#ext-s").value;
    $("#ext-s-v").textContent = s + "×";
    VIZ.drawExt($("#ext-canvas"), s, extShow);
    $("#ext-cap").innerHTML = `At ${s}&times; extension: <b>PI</b> pushes every dimension up equally, including the fast ones that only
      ever needed to resolve neighbours &mdash; that is why it damages short-context quality.
      <b>NTK-aware</b> scales the base instead, so the fast dimensions barely move.
      <b>YaRN</b> goes further: it classifies each dimension by how many rotations it completed during training and leaves
      the well-trained ones completely alone, then adds an attention-temperature correction that neither of the others has.`;
  }
  $("#ext-s").addEventListener("input", drawExtLab);
  $$("[data-ext]").forEach(b => b.addEventListener("click", function () {
    extShow[this.dataset.ext] = !extShow[this.dataset.ext];
    this.classList.toggle("on", extShow[this.dataset.ext]);
    drawExtLab();
  }));
  drawExtLab();

  /* ================= KV calculator ================= */
  function drawKV() {
    const L = +$("#kv-L").value, H = +$("#kv-H").value, D = +$("#kv-D").value;
    const N = Math.pow(2, +$("#kv-N").value), U = +$("#kv-U").value, B = +$("#kv-B").value;
    $("#kv-L-v").textContent = L; $("#kv-H-v").textContent = H; $("#kv-D-v").textContent = D;
    $("#kv-N-v").textContent = N >= 1048576 ? (N / 1048576) + "M" : (N / 1024) + "K";
    $("#kv-U-v").textContent = U;

    const base = 2 * L * B * N * U;
    const g = Math.max(1, Math.min(8, H));
    const rows = [
      { k: "MHA", d: "every head its own K,V", bytes: base * H * D, c: "var(--exact)" },
      { k: "GQA (g=8)", d: "8 KV heads shared across all query heads", bytes: base * g * D, c: "var(--kv)" },
      { k: "MQA", d: "one K,V for every head", bytes: base * 1 * D, c: "var(--kv)" },
      { k: "MLA", d: "cache a 512-dim latent + 64-dim decoupled RoPE", bytes: L * B * N * U * (512 + 64), c: "var(--kv)" },
      { k: "SWA (w=4096)", d: "rolling buffer — cache stops growing", bytes: 2 * L * g * D * B * Math.min(N, 4096) * U, c: "var(--sparse)" },
      { k: "hybrid ¼ + MLA", d: "only the full-attention layers cache anything", bytes: Math.ceil(L / 4) * B * N * U * (512 + 64), c: "var(--linear)" }
    ];
    const max = Math.max(...rows.map(r => r.bytes));
    $("#kv-bars").innerHTML = rows.map(r => {
      const rel = rows[0].bytes / r.bytes;
      return `<div class="bar-row" title="${r.d}">
        <span class="lbl">${r.k}</span>
        <div class="bar-track"><div class="bar-fill" style="width:${(r.bytes / max) * 100}%;background:${r.c}"></div></div>
        <span class="val">${fmtBytes(r.bytes)}${rel > 1.05 ? ` <span class="mut">${rel.toFixed(1)}×</span>` : ""}</span>
      </div>`;
    }).join("");
  }
  ["kv-L", "kv-H", "kv-D", "kv-N", "kv-U", "kv-B"].forEach(id =>
    $("#" + id).addEventListener("input", drawKV));
  $("#kv-B").addEventListener("change", drawKV);
  drawKV();

  /* ================= linear state ================= */
  function drawStateLab() {
    const n = +$("#st-n").value;
    $("#st-n-v").textContent = n;
    VIZ.drawState($("#state-canvas"), n);
    $("#state-cap").innerHTML = `At ${n} tokens the softmax cache holds ${n} key/value pairs and will hold ${n + 1} after the next token.
      The linear state is the same d&times;d matrix it was at token 1 &mdash; the new token is folded in by addition.
      That is the entire win, and the bounded capacity is the entire cost.`;
  }
  $("#st-n").addEventListener("input", drawStateLab);
  drawStateLab();

  /* ================= delta rule ================= */
  const dl = { mode: "add", value: 40, done: false };
  function drawDeltaLab() {
    VIZ.drawDelta($("#delta-canvas"), dl);
    $("#delta-cap").innerHTML = dl.done
      ? (dl.mode === "delta"
        ? "This is the delta rule &mdash; Widrow&ndash;Hoff, 1960, applied to a matrix-valued memory. It is one step of online gradient descent on &#8214;Sk &minus; v&#8214;&sup2;. Trivial once you see it; it took until 2021 for someone to put it in a linear transformer, and until 2024 for someone to make it trainable in parallel."
        : "The state has no way to say <em>replace</em>. Keys collide, values pile up, and the memory saturates with stale associations that never leave. This is the single worst failure mode of linear attention.")
      : "Press &lsquo;write the token&rsquo; and compare the two modes.";
  }
  $("#dl-mode-add").addEventListener("click", () => {
    dl.mode = "add"; dl.value = 40; dl.done = false;
    $("#dl-mode-add").classList.add("on"); $("#dl-mode-delta").classList.remove("on"); drawDeltaLab();
  });
  $("#dl-mode-delta").addEventListener("click", () => {
    dl.mode = "delta"; dl.value = 40; dl.done = false;
    $("#dl-mode-delta").classList.add("on"); $("#dl-mode-add").classList.remove("on"); drawDeltaLab();
  });
  $("#dl-step").addEventListener("click", () => {
    if (dl.done) return;
    dl.value = dl.mode === "delta" ? 55 : 95; dl.done = true; drawDeltaLab();
  });
  $("#dl-reset").addEventListener("click", () => { dl.value = 40; dl.done = false; drawDeltaLab(); });
  drawDeltaLab();

  /* ================= schedule ================= */
  function drawSchedLab() {
    const r = +$("#sch-r").value, L = +$("#sch-L").value;
    $("#sch-r-v").textContent = "1 in " + r;
    $("#sch-L-v").textContent = L;
    const full = VIZ.drawSchedule($("#sched-canvas"), r, L);
    const cut = (L / full);
    $("#sched-cap").innerHTML = `${full} of ${L} layers keep a growing KV cache &mdash; a <b>${cut.toFixed(1)}&times;</b> cut on top of
      whatever GQA or MLA already bought you. Push the ratio too far and you run out of exact-recall capacity;
      published models cluster around 1 in 4, but nobody can derive that number. They measured it.
      The instructor's V4 used a <span class="mono">DDDG</span> schedule &mdash; three Gated DeltaNet layers, one Gated Slot Attention layer.`;
  }
  $("#sch-r").addEventListener("input", drawSchedLab);
  $("#sch-L").addEventListener("input", drawSchedLab);
  drawSchedLab();

  /* ================= compare table ================= */
  const BUYS = {
    compute: "cheaper score matrix", memory: "smaller KV cache", both: "both bills",
    length: "usable at longer N", quality: "better use of the same compute"
  };
  function shortest(arr) { return arr.slice().sort((a, b) => a.length - b.length)[0].replace(/<[^>]+>/g, ""); }
  let sortKey = "date", sortDir = 1;
  function renderCmp() {
    const rows = MECHANISMS.slice().sort((a, b) => {
      const A = sortKey === "bill" ? a.bill : (a[sortKey] || ""), B = sortKey === "bill" ? b.bill : (b[sortKey] || "");
      return (A < B ? -1 : A > B ? 1 : 0) * sortDir;
    });
    $("#cmp tbody").innerHTML = rows.map(m => `<tr>
      <td class="dt">${m.dateLabel}</td>
      <td class="nm"><a href="#m-${m.id}">${m.abbr}</a></td>
      <td><span class="fam-pill" style="background:color-mix(in srgb,${famVar(m.family)} 18%,transparent);color:${famVar(m.family)}">${FAMILIES[m.family].label}</span></td>
      <td>${BUYS[m.bill] || m.bill}</td>
      <td>${shortest(m.pros)}</td>
      <td>${shortest(m.cons)}</td>
    </tr>`).join("");
  }
  $$("#cmp th").forEach(th => th.addEventListener("click", () => {
    const k = th.dataset.k;
    if (k === sortKey) sortDir *= -1; else { sortKey = k; sortDir = 1; }
    renderCmp();
  }));
  renderCmp();

  /* ================= findings (what the date order shows) ================= */
  let showGaps = true;
  function drawGapLab() {
    VIZ.drawGaps($("#gap-canvas"), MECHANISMS, { gaps: showGaps });
  }
  $("#gap-toggle").addEventListener("click", function () {
    showGaps = !showGaps;
    this.classList.toggle("on", showGaps);
    this.textContent = showGaps ? "show the intervals" : "just the mechanisms";
    drawGapLab();
  });
  drawGapLab();

  $("#find-root").innerHTML = FINDINGS.map(f => `
    <div class="finding">
      <h4>${f.t}</h4>
      <span class="metric">${f.metric}</span>
      <p>${f.body}</p>
      <div class="hid"><b>what a list hides</b>${f.hidden}</div>
      <p class="lesson"><b>so what</b>${f.lesson}</p>
    </div>`).join("");

  /* ================= predictions ================= */
  $("#pred-root").innerHTML = PREDICTIONS.map(p =>
    `<div class="pred"><h4>${p.t}</h4><p>${p.body}</p></div>`).join("");

  /* ================= sources table ================= */
  $("#srctbl tbody").innerHTML = MECHANISMS.map(m => `<tr>
    <td class="dt">${m.dateLabel}</td>
    <td class="nm"><a href="#m-${m.id}">${m.abbr}</a></td>
    <td>${m.src.map(s => `<a href="${s.url}" target="_blank" rel="noopener">${s.label}</a>`).join("<br>")}</td>
    <td>${m.dateNote ? m.dateNote : '<span class="mut">&mdash;</span>'}</td>
  </tr>`).join("");

  /* ================= deep link ================= */
  if (location.hash.startsWith("#m-")) {
    const id = location.hash.slice(3);
    const btn = $(`.card[data-id="${id}"]`);
    if (btn) { btn.setAttribute("aria-expanded", "true"); $("#d-" + id).classList.add("open"); }
  }
  document.addEventListener("click", e => {
    const a = e.target.closest('a[href^="#m-"]');
    if (!a) return;
    const id = a.getAttribute("href").slice(3);
    const btn = $(`.card[data-id="${id}"]`);
    if (btn && btn.getAttribute("aria-expanded") !== "true") {
      btn.setAttribute("aria-expanded", "true"); $("#d-" + id).classList.add("open");
    }
  });

  /* redraw canvases on theme change */
  if (window.matchMedia) {
    window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
      drawMaskLab(); drawRopeLab(); drawDecayLab(); drawExtLab();
      drawStateLab(); drawDeltaLab(); drawSchedLab(); drawGapLab();
    });
  }

  console.log("[attention-timeline] %d mechanisms, %d with date caveats, %d documented downsides",
    MECHANISMS.length, MECHANISMS.filter(m => m.dateNote).length,
    MECHANISMS.reduce((a, m) => a + m.cons.length, 0));
})();
