/* ------------------------------------------------------------------
   viz.js — all canvas drawing + the actual attention arithmetic.
   Nothing here is a picture of a computation; it is the computation.
   ------------------------------------------------------------------ */
(function (global) {
  "use strict";

  /* ---------- tiny helpers ---------- */
  const css = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  const dpi = (canvas) => {
    const r = window.devicePixelRatio || 1;
    const w = canvas.width, h = canvas.height;
    if (canvas.dataset.scaled !== "1") {
      canvas.style.width = w + "px";
      // height:auto, not a pixel value — otherwise max-width:100% shrinks the
      // width while the height stays put and every circle becomes an ellipse.
      // The backing store keeps the aspect ratio, so the browser follows it.
      canvas.style.height = "auto";
      canvas.width = w * r; canvas.height = h * r;
      canvas.dataset.scaled = "1"; canvas.dataset.lw = w; canvas.dataset.lh = h;
    }
    const ctx = canvas.getContext("2d");
    ctx.setTransform(r, 0, 0, r, 0, 0);
    return { ctx, w: +canvas.dataset.lw, h: +canvas.dataset.lh };
  };
  const clear = (ctx, w, h) => { ctx.clearRect(0, 0, w, h); };

  /* deterministic pseudo-random in [-1,1] from a string + index */
  function hrand(str, i) {
    let h = 2166136261 >>> 0;
    const s = str + "#" + i;
    for (let k = 0; k < s.length; k++) { h ^= s.charCodeAt(k); h = Math.imul(h, 16777619) >>> 0; }
    h ^= h >>> 13; h = Math.imul(h, 0x5bd1e995) >>> 0; h ^= h >>> 15;
    return (h / 4294967295) * 2 - 1;
  }

  /* ================================================================
     1. REAL scaled dot-product attention over a toy sentence
     ================================================================ */
  const DK = 4;

  function embed(tokens) {
    // stable per-token vectors; tokens that repeat get identical vectors,
    // which is exactly the point the session made about "bank" and "bank".
    // Normalised to unit length, because every real transformer applies
    // LayerNorm/RMSNorm before the Q/K/V projections. Skipping that is what
    // makes hand-rolled attention demos produce absurdly peaked rows.
    return tokens.map(t => {
      const v = Array.from({ length: DK }, (_, d) => hrand(t, d));
      const n = Math.hypot(...v) || 1;
      return v.map(x => x / n);
    });
  }

  function attention(tokens, opts) {
    const { scale = true, causal = true, softmax = true, amp = 1 } = opts || {};
    const N = tokens.length;
    const E = embed(tokens);
    // Q, K, V are fixed linear maps of the embedding (deterministic "trained" weights)
    // Scale the projection weights by 1/sqrt(DK) the way a real init would, so
    // the resulting logits are O(1) rather than saturating softmax on their own.
    // Without this the demo shows near-one-hot rows and looks broken, when in
    // fact it is just badly conditioned.
    const proj = (name) => Array.from({ length: DK }, (_, i) =>
      Array.from({ length: DK }, (_, j) => hrand(name, i * DK + j) / Math.sqrt(DK)));
    const WQ = proj("Wq"), WK = proj("Wk"), WV = proj("Wv");
    const mm = (v, W) => W.map(row => row.reduce((s, w, j) => s + w * v[j], 0));
    const Q = E.map(v => mm(v, WQ)), K = E.map(v => mm(v, WK)), V = E.map(v => mm(v, WV));

    const raw = [], scores = [], weights = [];
    for (let i = 0; i < N; i++) {
      raw[i] = []; scores[i] = [];
      for (let j = 0; j < N; j++) {
        let s = Q[i].reduce((a, q, d) => a + q * K[j][d], 0);
        raw[i][j] = s;
        if (scale) s /= Math.sqrt(DK);
        s *= amp;
        scores[i][j] = (causal && j > i) ? -Infinity : s;
      }
      if (softmax) {
        const row = scores[i];
        const mx = Math.max(...row.filter(Number.isFinite));
        const ex = row.map(v => Number.isFinite(v) ? Math.exp(v - mx) : 0);
        const sum = ex.reduce((a, b) => a + b, 0) || 1;
        weights[i] = ex.map(v => v / sum);
      } else {
        // no softmax: raw scores, masked entries zeroed. Note they are
        // unnormalised and can be negative — which is the whole trade.
        weights[i] = scores[i].map(v => Number.isFinite(v) ? v : 0);
      }
    }
    const out = weights.map(w => Array.from({ length: DK }, (_, d) =>
      w.reduce((a, wj, j) => a + wj * V[j][d], 0)));
    return { N, Q, K, V, raw, scores, weights, out };
  }

  /* render the weight matrix as an HTML grid (crisp text at small sizes) */
  function renderMatrix(el, tokens, res, opts) {
    const N = res.N, W = res.weights;
    const finite = W.flat().filter(Number.isFinite);
    const lo = Math.min(...finite), hi = Math.max(...finite);
    const norm = (v) => (hi - lo < 1e-9) ? .5 : (v - lo) / (hi - lo);

    let html = '<div class="matrix" style="grid-template-columns:52px repeat(' + N + ',minmax(0,1fr))">';
    html += '<div></div>';
    tokens.forEach(t => { html += '<div class="axis-t">' + t + '</div>'; });
    for (let i = 0; i < N; i++) {
      html += '<div class="axis-l">' + tokens[i] + '</div>';
      for (let j = 0; j < N; j++) {
        const masked = opts.causal && j > i;
        const v = W[i][j];
        const t = norm(v);
        // Single-hue sequential ramp: light = low weight, dark = high weight.
        // A hue rotation would detour through green, which reads as a
        // categorical scale rather than an ordered one.
        const L = 93 - 60 * t;
        const c = masked ? "transparent" : `hsl(214 ${42 + 26 * t}% ${L}%)`;
        const fg = L > 62 ? "#12203a" : "#f2f6ff";
        // strip the leading zero only for values below 1, so 1.00 stays "1.00"
        const label = masked ? "" : (opts.softmax
          ? (v < 1 ? v.toFixed(2).replace(/^0/, "") : v.toFixed(2))
          : v.toFixed(1));
        html += `<div class="cell${masked ? " masked" : ""}" style="background:${c};color:${fg}" title="q=${tokens[i]} k=${tokens[j]} → ${v.toFixed(4)}">${label}</div>`;
      }
    }
    html += "</div>";
    el.innerHTML = html;
  }

  /* ================================================================
     2. Sparsity patterns
     ================================================================ */

  /* rank of block `blk` among the blocks visible to query `i`, by a stable
     pseudo-score. Stands in for NSA's compression-branch index. */
  function VIZ_rank(i, blk, nblk) {
    const mine = hrand("nsa" + i, blk);
    let r = 0;
    for (let b = 0; b < nblk; b++) if (hrand("nsa" + i, b) > mine) r++;
    return r;
  }

  /* keysPerQuery(N, p) — average keys each query must read at sequence length N.
     This is what actually decides whether a mechanism survives at 1M tokens,
     and it is invisible on a 48x48 grid. */
  const PATTERNS = {
    dense: {
      title: "Dense causal attention (2017)",
      desc: "Every query scores every key it is allowed to see. Nothing is skipped, nothing is approximated — and the number of cells grows with the square of the sequence.",
      note: "This is the baseline every other pattern here is a discount on.",
      keysPerQuery: (N) => N / 2,
      fn: (i, j) => j <= i
    },
    window: {
      title: "Sliding window (2020 · Longformer, 2023 · Mistral)",
      desc: "Attend only to the last w tokens. Linear in N, and the KV cache can be a rolling buffer capped at w — one of very few tricks that bills both meters. Stacked layers grow the reach like a CNN.",
      note: "But receptive field gives you influence at distance, not lookup at distance. Needle-in-a-haystack fails.",
      keysPerQuery: (N, p) => Math.min(N / 2, p.w),
      fn: (i, j, p) => j <= i && (i - j) < p.w
    },
    sinks: {
      title: "Sliding window + attention sinks (2023 · StreamingLLM)",
      desc: "The same window, plus the first few tokens kept forever. Softmax must sum to 1 — it cannot say 'nothing here is relevant' — so heads dump spare probability mass on the earliest visible tokens. Evict those and every distribution distorts.",
      note: "Four extra KV entries buy stable decoding over millions of tokens. It extends fluency, not context.",
      keysPerQuery: (N, p) => Math.min(N / 2, p.w + p.sink),
      fn: (i, j, p) => j <= i && ((i - j) < p.w || j < p.sink)
    },
    strided: {
      title: "Strided / dilated (2019 · Sparse Transformer)",
      desc: "A local band plus a fixed stride, each O(N√N). Alternate them across layers and any pair of positions is reachable in two hops.",
      note: "The pattern is a hand-drawn guess. It suits grid-structured data, where the right stride is obvious. Text has no such stride.",
      keysPerQuery: (N) => Math.min(N / 2, 2 * Math.sqrt(N / 2)),
      fn: (i, j, p) => j <= i && ((i - j) < Math.max(2, p.w >> 1) || (i - j) % Math.max(2, p.w >> 1) === 0)
    },
    bigbird: {
      title: "Window + global + random (2020 · BigBird)",
      desc: "Local window for fluency, a few global tokens that everyone can see, and a handful of random edges. Random-graph theory then gives short paths, and the paper proves universality.",
      note: "The proof needs Θ(N) layers to realise those paths. And random gathers are miserable on GPUs.",
      keysPerQuery: (N, p) => Math.min(N / 2, p.w + 4 + 3),
      fn: (i, j, p) => j <= i && ((i - j) < p.w || j < 2 || i < 2 || hrand("bb" + i, j) > 0.93)
    },
    topk: {
      title: "Content-based top-k (2019 · Explicit Sparse Transformer)",
      desc: "Stop designing the pattern. Score everything, keep the k highest per query, mask the rest to −∞. The surviving mass renormalises, which sharpens the distribution as a bonus.",
      note: "The catch that took five years to fix: you must compute all N² scores to find the top k. As stated, it saves nothing on compute — it is a quality trick wearing an efficiency costume.",
      keysPerQuery: (N, p) => Math.min(N / 2, p.k),
      fn: null // computed by score
    },
    nsa: {
      title: "Compressed + selected + window (2025 · NSA)",
      desc: "Three branches, gated together. Coarse compressed blocks so nothing is invisible; the top-n of those blocks re-read at full resolution; and a local window. The compression branch is the cheap indexer that finally makes selection affordable.",
      note: "Trained natively rather than bolted on, so the model learns to use the sparsity instead of being damaged by it. But you must pretrain with it.",
      keysPerQuery: (N) => Math.min(N / 2, (N / 2) / 16 + 16 * 64 + 512),
      fn: (i, j, p) => {
        if (j > i) return false;
        const B = 6;
        if ((i - j) < Math.max(2, p.w >> 2)) return true;               // local window branch
        const blk = Math.floor(j / B);
        const nblk = Math.floor(i / B) + 1;
        // selection branch: the top-n blocks for this query, chosen by a
        // stable pseudo-score standing in for the compression-branch index
        const sel = VIZ_rank(i, blk, nblk) < 2;
        if (sel) return true;
        return j % B === 0;                                             // compressed summaries
      }
    },
    dsa: {
      title: "Token-level top-k via a cheap indexer (2025 · DSA)",
      desc: "A tiny FP8 'lightning indexer' with few heads scores every past token approximately, and only its top-k go to the real MLA attention. Selection is per token, not per block — nothing relevant is lost inside an irrelevant block.",
      note: "The indexer is a learned approximation and it can miss. When it does, the main attention never sees the token, and you get a confident wrong answer with no signal anything was skipped.",
      keysPerQuery: (N, p) => Math.min(N / 2, p.k + 3),
      fn: null
    },
    csa: {
      title: "Compress along time, then select (2026 · CSA)",
      desc: "Consolidate every m tokens of KV cache into one entry, then run top-k selection over those consolidated entries. Sparse selection cuts compute; compression is what finally cuts the memory too.",
      note: "Compression is lossy and permanent. A skipped token still exists in the cache and could have been chosen; a compressed one is gone.",
      keysPerQuery: (N) => Math.min(N / 2, Math.min(2048, (N / 2) / 8) + 512),
      fn: (i, j, p) => {
        if (j > i) return false;
        const m = 4;
        if ((i - j) < Math.max(2, p.w >> 2)) return true;
        if (j % m !== 0) return false;                 // only compressed entries survive
        return hrand("csa" + i, Math.floor(j / m)) > 0.15 || (i - j) < p.w * 2;
      }
    }
  };

  function buildMask(kind, N, p, scoreFn) {
    const M = Array.from({ length: N }, () => new Array(N).fill(false));
    const P = PATTERNS[kind];
    if (P.fn) {
      for (let i = 0; i < N; i++) for (let j = 0; j <= i; j++) M[i][j] = !!P.fn(i, j, p);
    } else {
      // score-driven selection (topk / dsa)
      for (let i = 0; i < N; i++) {
        const cand = [];
        for (let j = 0; j <= i; j++) cand.push([j, scoreFn(i, j)]);
        cand.sort((a, b) => b[1] - a[1]);
        const k = Math.min(p.k, cand.length);
        for (let t = 0; t < k; t++) M[i][cand[t][0]] = true;
        M[i][i] = true;
        if (kind === "dsa") { for (let j = Math.max(0, i - 2); j <= i; j++) M[i][j] = true; }
      }
    }
    return M;
  }

  function drawMask(canvas, M) {
    const { ctx, w, h } = dpi(canvas);
    clear(ctx, w, h);
    const N = M.length, s = w / N;
    const on = css("--sparse") || "#ec4899";
    const off = css("--bg3") || "#161e30";
    const skip = css("--bg2") || "#111725";
    let count = 0, total = 0;
    for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
      const causal = j <= i;
      if (causal) total++;
      if (M[i][j]) { ctx.fillStyle = on; count++; }
      else if (causal) ctx.fillStyle = off;
      else ctx.fillStyle = skip;
      ctx.fillRect(j * s, i * s, s - 0.6, s - 0.6);
    }
    // frame
    ctx.strokeStyle = css("--line2") || "#2f3f5e"; ctx.lineWidth = 1;
    ctx.strokeRect(.5, .5, w - 1, h - 1);
    return { count, total, dense: total };
  }

  /* ================================================================
     3. RoPE circle
     ================================================================ */
  function drawRope(canvas, m, n) {
    const { ctx, w, h } = dpi(canvas);
    clear(ctx, w, h);
    const cx = w * 0.32, cy = h / 2, R = Math.min(w * 0.28, h * 0.38);
    const theta = 0.42; // one frequency
    const am = m * theta, an = n * theta;
    const line = css("--line") || "#233049";

    ctx.strokeStyle = line; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.arc(cx, cy, R, 0, Math.PI * 2); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(cx - R - 12, cy); ctx.lineTo(cx + R + 12, cy);
    ctx.moveTo(cx, cy - R - 12); ctx.lineTo(cx, cy + R + 12); ctx.stroke();

    // the arc between them
    ctx.strokeStyle = css("--acc") || "#6ea8fe"; ctx.lineWidth = 6;
    ctx.globalAlpha = .28;
    ctx.beginPath(); ctx.arc(cx, cy, R * .55, -Math.max(am, an), -Math.min(am, an)); ctx.stroke();
    ctx.globalAlpha = 1;

    const arm = (a, color, label) => {
      const x = cx + R * Math.cos(a), y = cy - R * Math.sin(a);
      ctx.strokeStyle = color; ctx.lineWidth = 2.5;
      ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(x, y); ctx.stroke();
      ctx.fillStyle = color; ctx.beginPath(); ctx.arc(x, y, 5.5, 0, Math.PI * 2); ctx.fill();
      ctx.font = "600 12px ui-monospace,monospace"; ctx.fillText(label, x + 9, y - 7);
    };
    arm(am, css("--pos") || "#38bdf8", "q at m=" + m);
    arm(an, css("--kv") || "#f59e0b", "k at n=" + n);

    // readouts
    const tx = cx + R + 46;
    ctx.font = "12px ui-monospace,monospace";
    ctx.fillStyle = css("--fg3") || "#71809b";
    ctx.fillText("absolute angles change:", tx, cy - 46);
    ctx.fillStyle = css("--pos") || "#38bdf8"; ctx.fillText("m·θ = " + am.toFixed(2) + " rad", tx, cy - 26);
    ctx.fillStyle = css("--kv") || "#f59e0b"; ctx.fillText("n·θ = " + an.toFixed(2) + " rad", tx, cy - 8);
    ctx.fillStyle = css("--fg3") || "#71809b"; ctx.fillText("the gap does not:", tx, cy + 22);
    ctx.fillStyle = css("--acc") || "#6ea8fe";
    ctx.font = "600 15px ui-monospace,monospace";
    ctx.fillText("(n−m)·θ = " + Math.abs(an - am).toFixed(2), tx, cy + 44);
    ctx.font = "12px ui-monospace,monospace";
    ctx.fillStyle = css("--fg3") || "#71809b";
    // + 0 clears the negative-zero that toFixed happily prints as "-0.000"
    ctx.fillText("qᵀk = cos((n−m)·θ) = " + (Math.cos(an - am) + 0).toFixed(3), tx, cy + 66);
  }

  /* ================================================================
     4. Score-vs-distance decay curves
     ================================================================ */
  function drawDecay(canvas, show) {
    const { ctx, w, h } = dpi(canvas);
    clear(ctx, w, h);
    const pad = { l: 40, r: 12, t: 14, b: 30 };
    const W = w - pad.l - pad.r, H = h - pad.t - pad.b;
    const maxD = 512;
    const line = css("--line") || "#233049";

    ctx.strokeStyle = line; ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(pad.l, pad.t); ctx.lineTo(pad.l, pad.t + H); ctx.lineTo(pad.l + W, pad.t + H);
    ctx.stroke();
    ctx.font = "10px ui-monospace,monospace"; ctx.fillStyle = css("--fg3") || "#71809b";
    ctx.fillText("1.0", 8, pad.t + 5); ctx.fillText("0", 22, pad.t + H + 4);
    ctx.fillText("0", pad.l - 2, pad.t + H + 20);
    ctx.fillText(String(maxD), pad.l + W - 16, pad.t + H + 20);
    ctx.fillText("distance →", pad.l + W / 2 - 28, pad.t + H + 20);

    const curves = {
      rope:  { c: css("--pos") || "#38bdf8", lbl: "RoPE (trained to 128)", f: d => d <= 128 ? Math.max(.05, 1 - .55 * (d / 128) ** .6) : Math.max(0, .45 * Math.exp(-(d - 128) / 55)) },
      alibi: { c: css("--kv") || "#f59e0b", lbl: "ALiBi (head slope 1/16)", f: d => Math.exp(-d / 16 / 4) },
      swa:   { c: css("--sparse") || "#ec4899", lbl: "sliding window w=96", f: d => d < 96 ? .9 : 0 },
      ideal: { c: css("--good") || "#4ade80", lbl: "what retrieval needs", f: () => .62 }
    };

    Object.keys(curves).forEach(k => {
      if (!show[k]) return;
      const cv = curves[k];
      ctx.strokeStyle = cv.c; ctx.lineWidth = 2.2;
      if (k === "ideal") ctx.setLineDash([5, 4]); else ctx.setLineDash([]);
      ctx.beginPath();
      for (let px = 0; px <= W; px++) {
        const d = (px / W) * maxD;
        const y = pad.t + H - Math.min(1, Math.max(0, cv.f(d))) * H;
        px === 0 ? ctx.moveTo(pad.l + px, y) : ctx.lineTo(pad.l + px, y);
      }
      ctx.stroke(); ctx.setLineDash([]);
    });

    // training-length marker
    const tx = pad.l + (128 / maxD) * W;
    ctx.strokeStyle = css("--fg3") || "#71809b"; ctx.setLineDash([3, 3]); ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(tx, pad.t); ctx.lineTo(tx, pad.t + H); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = css("--fg3") || "#71809b"; ctx.font = "10px ui-monospace,monospace";
    ctx.fillText("trained to here", tx + 4, pad.t + 10);

    // legend
    let ly = pad.t + 6;
    ctx.font = "11px ui-monospace,monospace";
    Object.keys(curves).forEach(k => {
      if (!show[k]) return;
      ctx.fillStyle = curves[k].c;
      ctx.fillRect(pad.l + W - 148, ly - 7, 9, 9);
      ctx.fillStyle = css("--fg2") || "#a7b4cc";
      ctx.fillText(curves[k].lbl, pad.l + W - 134, ly + 1);
      ly += 15;
    });
  }

  /* ================================================================
     5. RoPE extension: wavelength per dimension
     ================================================================ */
  function drawExt(canvas, s, show) {
    const { ctx, w, h } = dpi(canvas);
    clear(ctx, w, h);
    const pad = { l: 58, r: 16, t: 18, b: 42 };
    const W = w - pad.l - pad.r, H = h - pad.t - pad.b;
    const D = 64, base = 10000, Ltrain = 4096;

    const wavelength = (i, b) => 2 * Math.PI * Math.pow(b, 2 * i / D);
    const yFor = (wl) => {
      const lo = Math.log10(2 * Math.PI), hi = Math.log10(2 * Math.PI * base * 4);
      return pad.t + H - ((Math.log10(wl) - lo) / (hi - lo)) * H;
    };

    // grid + training-context line
    ctx.strokeStyle = css("--line") || "#233049"; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(pad.l, pad.t); ctx.lineTo(pad.l, pad.t + H); ctx.lineTo(pad.l + W, pad.t + H); ctx.stroke();
    const yL = yFor(Ltrain);
    ctx.setLineDash([4, 4]); ctx.strokeStyle = css("--bad") || "#f87171";
    ctx.beginPath(); ctx.moveTo(pad.l, yL); ctx.lineTo(pad.l + W, yL); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = css("--bad") || "#f87171"; ctx.font = "11px ui-monospace,monospace";
    ctx.fillText("training context (4K) — dimensions above this line never complete a rotation", pad.l + 8, yL - 7);

    ctx.fillStyle = css("--fg3") || "#71809b"; ctx.font = "10px ui-monospace,monospace";
    ctx.fillText("wavelength", 6, pad.t - 4);
    ctx.fillText("fast", 6, pad.t + H - 2); ctx.fillText("slow", 6, pad.t + 10);
    ctx.fillText("RoPE dimension pair  0 → 63", pad.l + W / 2 - 70, pad.t + H + 26);

    const variants = {
      none: { c: css("--fg3") || "#71809b", lbl: "plain RoPE (unscaled)", f: i => wavelength(i, base) },
      pi:   { c: css("--sparse") || "#ec4899", lbl: "PI — everything squashed by s", f: i => wavelength(i, base) * s },
      ntk:  { c: css("--kv") || "#f59e0b", lbl: "NTK-aware — base scaled, fast dims spared", f: i => wavelength(i, base * Math.pow(s, D / (D - 2))) },
      yarn: { c: css("--good") || "#4ade80", lbl: "YaRN — by wavelength; short dims untouched", f: i => {
        const wl = wavelength(i, base);
        const r = Ltrain / wl;                    // rotations completed in training
        const a = 1, b = 32;                       // ramp bounds from the paper
        const g = Math.min(1, Math.max(0, (r - a) / (b - a)));
        return wl * (1 + (s - 1) * (1 - g));       // g→1 (many rotations) ⇒ untouched
      } }
    };

    Object.keys(variants).forEach(k => {
      if (!show[k]) return;
      const v = variants[k];
      ctx.strokeStyle = v.c; ctx.lineWidth = k === "none" ? 1.6 : 2.4;
      if (k === "none") ctx.setLineDash([4, 3]); else ctx.setLineDash([]);
      ctx.beginPath();
      for (let i = 0; i < D; i++) {
        const x = pad.l + (i / (D - 1)) * W, y = yFor(v.f(i));
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      }
      ctx.stroke(); ctx.setLineDash([]);
    });

    let ly = pad.t + 14;
    ctx.font = "11px ui-monospace,monospace";
    Object.keys(variants).forEach(k => {
      if (!show[k]) return;
      ctx.fillStyle = variants[k].c; ctx.fillRect(pad.l + 14, ly - 8, 10, 10);
      ctx.fillStyle = css("--fg2") || "#a7b4cc"; ctx.fillText(variants[k].lbl, pad.l + 30, ly + 1);
      ly += 16;
    });
  }

  /* ================================================================
     6. Linear state vs growing KV cache
     ================================================================ */
  function drawState(canvas, n) {
    const { ctx, w, h } = dpi(canvas);
    clear(ctx, w, h);
    const d = 8, cell = 12;
    const gap = 22;

    ctx.font = "11px ui-monospace,monospace";

    // left: softmax KV cache — grows
    ctx.fillStyle = css("--fg2") || "#a7b4cc";
    ctx.fillText("softmax attention — KV cache", 12, 18);
    const shown = Math.min(n, 40);
    const cw = Math.min(6, (w / 2 - 30) / Math.max(1, shown));
    for (let t = 0; t < shown; t++) {
      const a = 0.35 + 0.65 * (t / Math.max(1, shown));
      ctx.fillStyle = css("--kv") || "#f59e0b"; ctx.globalAlpha = a;
      ctx.fillRect(12 + t * (cw + 1), 30, cw, 90);
      ctx.globalAlpha = 1;
    }
    if (n > 40) {
      ctx.fillStyle = css("--fg3") || "#71809b";
      ctx.fillText("… ×" + n, 12 + shown * (cw + 1) + 6, 80);
    }
    ctx.fillStyle = css("--fg3") || "#71809b";
    ctx.fillText(n + " token" + (n === 1 ? "" : "s") + " cached — grows forever", 12, 140);
    ctx.fillStyle = css("--kv") || "#f59e0b";
    ctx.font = "600 13px ui-monospace,monospace";
    ctx.fillText("O(N) memory", 12, 162);

    // right: linear state — constant
    const rx = w / 2 + gap;
    ctx.font = "11px ui-monospace,monospace";
    ctx.fillStyle = css("--fg2") || "#a7b4cc";
    ctx.fillText("linear attention — state S", rx, 18);
    for (let i = 0; i < d; i++) for (let j = 0; j < d; j++) {
      const v = Math.abs(hrand("S" + n + i, j));
      ctx.fillStyle = `hsl(262 65% ${18 + v * 46}%)`;
      ctx.fillRect(rx + j * (cell + 1), 30 + i * (cell + 1), cell, cell);
    }
    ctx.fillStyle = css("--fg3") || "#71809b";
    ctx.fillText("d × d, identical at 1 or 1,000,000 tokens", rx, 140);
    ctx.fillStyle = css("--linear") || "#a78bfa";
    ctx.font = "600 13px ui-monospace,monospace";
    ctx.fillText("O(1) memory", rx, 162);

    // separator
    ctx.strokeStyle = css("--line") || "#233049"; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(w / 2 + 4, 10); ctx.lineTo(w / 2 + 4, h - 10); ctx.stroke();

    ctx.font = "11px ui-monospace,monospace"; ctx.fillStyle = css("--fg3") || "#71809b";
    ctx.fillText("exact recall of any token", 12, 190);
    ctx.fillText("bounded capacity — cannot hold N ≫ d facts", rx, 190);
  }

  /* ================================================================
     7. Delta rule vs additive write
     ================================================================ */
  function drawDelta(canvas, st) {
    const { ctx, w, h } = dpi(canvas);
    clear(ctx, w, h);
    const cx = w / 2;
    ctx.textAlign = "center";

    ctx.font = "11px ui-monospace,monospace"; ctx.fillStyle = css("--fg3") || "#71809b";
    ctx.fillText("memory cell for this key", cx, 20);

    // the cell
    const bw = 132, bh = 62, bx = cx - bw / 2, by = 34;
    ctx.fillStyle = css("--bg3") || "#161e30";
    ctx.strokeStyle = st.done ? (st.mode === "delta" ? css("--good") : css("--bad")) : css("--line2");
    ctx.lineWidth = 2;
    ctx.beginPath(); ctx.roundRect(bx, by, bw, bh, 10); ctx.fill(); ctx.stroke();
    ctx.fillStyle = css("--fg") || "#e6ecf7";
    ctx.font = "700 30px ui-monospace,monospace";
    ctx.fillText(String(st.value), cx, by + 42);

    // target
    ctx.font = "11px ui-monospace,monospace";
    ctx.fillStyle = css("--acc") || "#6ea8fe";
    ctx.fillText("the model wants this to be 55", cx, by + bh + 24);

    // arithmetic
    ctx.font = "13px ui-monospace,monospace";
    const ay = by + bh + 56;
    if (!st.done) {
      ctx.fillStyle = css("--fg3") || "#71809b";
      ctx.fillText(st.mode === "delta"
        ? "delta rule: read memory first, write only the gap"
        : "plain linear attention: it can only add", cx, ay);
      ctx.fillStyle = css("--fg2") || "#a7b4cc";
      ctx.fillText(st.mode === "delta"
        ? "S ← S + β(v − Sk)kᵀ"
        : "S ← S + v kᵀ", cx, ay + 22);
    } else if (st.mode === "delta") {
      ctx.fillStyle = css("--good") || "#4ade80";
      ctx.fillText("read 40  →  gap = 55 − 40 = 15  →  write 15", cx, ay);
      ctx.font = "600 15px ui-monospace,monospace";
      ctx.fillText("40 + 15 = 55  ✓  exactly what was asked", cx, ay + 24);
    } else {
      ctx.fillStyle = css("--bad") || "#f87171";
      ctx.fillText("write 55 on top of 40", cx, ay);
      ctx.font = "600 15px ui-monospace,monospace";
      ctx.fillText("40 + 55 = 95  ✗  memory is now wrong", cx, ay + 24);
    }
    ctx.textAlign = "left";
  }

  /* ================================================================
     8. Layer schedule
     ================================================================ */
  function drawSchedule(canvas, ratio, L) {
    const { ctx, w, h } = dpi(canvas);
    clear(ctx, w, h);
    const pad = 16;
    const bw = (w - pad * 2) / L;
    let full = 0;
    for (let i = 0; i < L; i++) {
      const isFull = ((i + 1) % ratio === 0);
      if (isFull) full++;
      ctx.fillStyle = isFull ? (css("--acc") || "#6ea8fe") : (css("--linear") || "#a78bfa");
      ctx.globalAlpha = isFull ? 1 : .55;
      ctx.fillRect(pad + i * bw + 1, 44, bw - 2, 52);
      ctx.globalAlpha = 1;
    }
    ctx.font = "11px ui-monospace,monospace";
    ctx.fillStyle = css("--fg3") || "#71809b";
    ctx.fillText("input", pad, 36); ctx.fillText("output", w - pad - 36, 36);
    ctx.fillStyle = css("--linear") || "#a78bfa";
    ctx.fillRect(pad, 116, 10, 10);
    ctx.fillStyle = css("--fg2") || "#a7b4cc";
    ctx.fillText("linear layer (Gated DeltaNet) — fixed state, no KV cache", pad + 16, 125);
    ctx.fillStyle = css("--acc") || "#6ea8fe";
    ctx.fillRect(pad + 360, 116, 10, 10);
    ctx.fillStyle = css("--fg2") || "#a7b4cc";
    ctx.fillText("full attention / MLA — exact recall, grows with N", pad + 376, 125);
    ctx.fillStyle = css("--fg") || "#e6ecf7";
    ctx.font = "600 12px ui-monospace,monospace";
    ctx.fillText(full + " of " + L + " layers keep a KV cache", pad, 20);
    return full;
  }

  /* ================================================================
     9. The timeline on a real time axis, one lane per family.
        A list spaces entries evenly. This spaces them by when they
        actually happened, which is the entire argument.
     ================================================================ */
  function drawGaps(canvas, mechs, opts) {
    const { ctx, w, h } = dpi(canvas);
    clear(ctx, w, h);
    const showGaps = !opts || opts.gaps !== false;
    const pad = { l: 104, r: 22, t: 22, b: 34 };
    const W = w - pad.l - pad.r, H = h - pad.t - pad.b;
    const t0 = Date.UTC(2014, 0, 1), t1 = Date.UTC(2027, 0, 1);
    const X = (iso) => pad.l + ((Date.parse(iso) - t0) / (t1 - t0)) * W;

    const lanes = [
      ["exact", "exact"], ["pos", "position"], ["sparse", "sparse"],
      ["linear", "linear"], ["kv", "KV cache"], ["systems", "systems"]
    ];
    const laneH = H / lanes.length;
    const laneY = (f) => pad.t + lanes.findIndex(l => l[0] === f) * laneH + laneH / 2;

    // year gridlines
    ctx.font = "10px ui-monospace,monospace";
    for (let y = 2014; y <= 2026; y++) {
      const x = X(y + "-01-01");
      ctx.strokeStyle = css("--line") || "#233049";
      ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(x, pad.t - 6); ctx.lineTo(x, pad.t + H); ctx.stroke();
      if (y % 2 === 0) {
        ctx.fillStyle = css("--fg3") || "#71809b";
        ctx.fillText(String(y), x - 11, pad.t + H + 18);
      }
    }

    // lane labels
    lanes.forEach(([f, label]) => {
      ctx.fillStyle = css("--fg3") || "#71809b";
      ctx.font = "11px ui-monospace,monospace";
      ctx.textAlign = "right";
      ctx.fillText(label, pad.l - 12, laneY(f) + 4);
      ctx.textAlign = "left";
      ctx.strokeStyle = css("--line") || "#233049";
      ctx.globalAlpha = .5;
      ctx.beginPath(); ctx.moveTo(pad.l, laneY(f)); ctx.lineTo(pad.l + W, laneY(f)); ctx.stroke();
      ctx.globalAlpha = 1;
    });

    // one tick per mechanism, at its real date
    mechs.forEach(m => {
      const x = X(m.date), y = laneY(m.family);
      ctx.fillStyle = css("--" + m.family) || "#888";
      ctx.beginPath();
      ctx.arc(x, y, m.anchor ? 5 : 3.2, 0, Math.PI * 2);
      ctx.fill();
      if (m.anchor) {
        ctx.globalAlpha = .25;
        ctx.beginPath(); ctx.arc(x, y, 9, 0, Math.PI * 2); ctx.fill();
        ctx.globalAlpha = 1;
      }
    });

    if (!showGaps) return;

    // the two intervals that a list cannot show
    const bracket = (fromISO, toISO, family, label) => {
      const x1 = X(fromISO), x2 = X(toISO), y = laneY(family);
      ctx.strokeStyle = css("--bad") || "#f87171";
      ctx.lineWidth = 1.6;
      ctx.setLineDash([4, 3]);
      ctx.beginPath(); ctx.moveTo(x1, y); ctx.lineTo(x2, y); ctx.stroke();
      ctx.setLineDash([]);
      [x1, x2].forEach(x => {
        ctx.beginPath(); ctx.moveTo(x, y - 7); ctx.lineTo(x, y + 7); ctx.stroke();
      });
      ctx.fillStyle = css("--bad") || "#f87171";
      ctx.font = "600 10.5px ui-monospace,monospace";
      const tw = ctx.measureText(label).width;
      ctx.fillText(label, (x1 + x2) / 2 - tw / 2, y - 12);
    };
    bracket("2019-11-06", "2023-05-22", "kv", "3.5 yr — no one was paying this bill yet");
    bracket("2021-02-22", "2024-06-10", "linear", "3.3 yr — waiting for the hardware");

    // the extinction point of the approximation wave
    const fx = X("2022-05-27");
    ctx.strokeStyle = css("--good") || "#4ade80";
    ctx.setLineDash([3, 3]); ctx.lineWidth = 1.4;
    ctx.beginPath(); ctx.moveTo(fx, pad.t - 6); ctx.lineTo(fx, pad.t + H); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = css("--good") || "#4ade80";
    ctx.font = "600 10.5px ui-monospace,monospace";
    const flabel = "FlashAttention — exact beats the approximations";
    const fw = ctx.measureText(flabel).width;
    // flip the label to the left of the line rather than let it run off the canvas
    const fitsRight = fx + 6 + fw < pad.l + W;
    ctx.fillText(flabel, fitsRight ? fx + 6 : fx - 6 - fw, pad.t + 8);
  }

  global.VIZ = {
    drawGaps,
    attention, renderMatrix, PATTERNS, buildMask, drawMask,
    drawRope, drawDecay, drawExt, drawState, drawDelta, drawSchedule, hrand
  };
})(window);
