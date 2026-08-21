/* ------------------------------------------------------------------
   The attention timeline — data.
   Every `date` is the v1 arXiv submission date or the official release
   date, verified against the primary source listed in `src`.
   `dateNote` records anything that was ambiguous, because a date you
   cannot defend is worse than no date at all.
   ------------------------------------------------------------------ */

const FAMILIES = {
  exact:   { label: "Exact attention",       color: "#cbd5e1", desc: "Changes the arithmetic of softmax attention itself, or its kernels, without approximating the result." },
  pos:     { label: "Position",              color: "#38bdf8", desc: "How a token learns where it is. The whole length story lives here." },
  kv:      { label: "KV-cache shape",        color: "#f59e0b", desc: "Attacks the memory bill: how many bytes each cached token costs." },
  sparse:  { label: "Sparse / selected",     color: "#ec4899", desc: "Keeps softmax, but refuses to score every pair. Attacks the compute bill." },
  linear:  { label: "Linear / stateful",     color: "#a78bfa", desc: "Drops softmax so the past collapses into a fixed-size state. Attacks both bills, pays in fidelity." },
  systems: { label: "Systems / kernels",     color: "#22c55e", desc: "Same math, different memory movement. Free wins, until they run out." }
};

const ERAS = [
  { id: 1, from: "2014-01-01", to: "2018-12-31", title: "Exactness",
    thesis: "Make the model look at everything, and prove it helps.",
    body: "Attention starts life as a fix for a translation bug: an RNN cannot cram a sentence into one vector. Nobody is counting the bill yet, because the sequences are 30 tokens long. The 2017 Transformer throws away recurrence entirely and buys pure parallelism with an N&times;N score matrix. That trade looks free at N=512." },
  { id: 2, from: "2019-01-01", to: "2020-12-31", title: "The bill arrives",
    thesis: "N&times;N is quadratic and nobody can pay it.",
    body: "The moment people want book-length inputs, the score matrix becomes the whole story. Two separate answers appear almost simultaneously: <em>don't compute every cell</em> (sparse patterns) and <em>don't build the matrix at all</em> (linear attention). Quietly, in November 2019, Shazeer notices a different bill nobody was looking at &mdash; the KV cache at decode time &mdash; and answers it with MQA. It is ignored for almost four years." },
  { id: 3, from: "2021-01-01", to: "2022-12-31", title: "Memory back, quietly",
    thesis: "Approximation was too lossy. Get exactness back, cheaper.",
    body: "Almost none of the 2020 efficient transformers survive contact with real language modelling; Long Range Arena and hard-nosed replications show they lose too much. The field's answer is not a better approximation. It is RoPE (a better way to encode position, so the exact model generalises further) and FlashAttention (the exact same softmax, computed without ever writing the N&times;N matrix to HBM). Exactness wins this round on engineering, not on math." },
  { id: 4, from: "2023-01-01", to: "2024-12-31", title: "Length, at any cost",
    thesis: "Ship a bigger number on the context box. Figure out quality later.",
    body: "Llama-2 lands with 4K and the entire ecosystem wants 32K by the weekend. The winning moves are all post-hoc surgery on a model that already exists: interpolate the RoPE frequencies (PI), interpolate them unevenly (NTK), interpolate them unevenly and fix the attention temperature (YaRN). Simultaneously the serving bill gets the attention it deserved in 2019: GQA becomes the default, MLA compresses the cache, and StreamingLLM discovers that the first four tokens are load-bearing." },
  { id: 5, from: "2025-01-01", to: "2026-12-31", title: "Memory back, again &mdash; and hybrids",
    thesis: "Stop choosing. Put different attention in different layers.",
    body: "The delta rule fixes linear attention's worst flaw (it could add to memory but never overwrite), gating fixes the second (it could never forget), and the result is finally good enough to carry most of a stack. But <em>most</em> is not <em>all</em>: every serious 2025&ndash;26 model keeps a minority of full-attention layers for the exact recall that a fixed-size state provably cannot do. Sparse attention gets trained natively rather than bolted on. And DroPE suggests the endgame for position: use it as scaffolding, then take it away." }
];

const MECHANISMS = [
/* ============================ ERA 1 ============================ */
{
  id: "bahdanau", name: "Additive (Bahdanau) attention", abbr: "Additive",
  date: "2014-09-01", dateLabel: "1 Sep 2014",
  who: "Bahdanau, Cho, Bengio",
  src: [{label:"arXiv:1409.0473 v1", url:"https://arxiv.org/abs/1409.0473"}],
  family: "exact", era: 1, covered: false, bill: "quality",
  headline: "The original. Attention is invented to fix a bottleneck, not to save money.",
  problem: "Encoder&ndash;decoder RNN translation crushed a whole source sentence into one fixed vector. Quality fell off a cliff past ~30 words, because the vector had no more room.",
  idea: "Let the decoder, at every output step, compute a learned relevance score against every encoder state and read a weighted average instead of one frozen summary. The score is a small feed-forward net over the pair &mdash; hence <em>additive</em>.",
  math: "e(i,j) = v&#7488; tanh(W&#8339; s&#7522;&#8331;&#8321; + W&#8341; h&#11002;) ;  &alpha; = softmax(e) ;  c&#7522; = &sum;&#11002; &alpha;(i,j) h&#11002;",
  pros: [
    "Fixed the long-sentence collapse outright &mdash; the first result where longer input stopped hurting.",
    "The alignment weights are readable; you can point at what the model looked at.",
    "Works with any encoder/decoder dimensionality, since the scorer is a learned MLP."
  ],
  cons: [
    "A small neural network per query&ndash;key pair. Cannot be folded into one matmul, so it is slow on GPUs.",
    "Still bolted onto a sequential RNN, so training does not parallelise over time.",
    "Introduces the O(N&middot;M) score table that every paper for the next decade will be trying to shrink."
  ],
  pick: "Historical interest, and cross-attention in very small models where the extra parameters are cheap. In 2026 you would use dot-product scoring instead; the quality gap is negligible and the speed gap is not."
},
{
  id: "luong", name: "Multiplicative (Luong) dot-product attention", abbr: "Dot-product",
  date: "2015-08-17", dateLabel: "17 Aug 2015",
  who: "Luong, Pham, Manning",
  src: [{label:"arXiv:1508.04025 v1", url:"https://arxiv.org/abs/1508.04025"}],
  family: "exact", era: 1, covered: false, bill: "compute",
  headline: "Replace the scoring MLP with a dot product. This one line makes the Transformer possible.",
  problem: "Bahdanau scoring is a per-pair MLP. It works, but it cannot be expressed as a single dense matrix multiply, so it wastes the hardware.",
  idea: "Score with s&#7488;h &mdash; a plain inner product. Now the entire score table is one matmul, and the GPU is happy. Luong also compared global vs local (windowed) attention, which is the first sliding window in the literature.",
  math: "score(s, h) = s&#7488;h  (dot)   or  s&#7488;W h  (general)",
  pros: [
    "Turns the whole score table into one GEMM. This is the change that makes attention hardware-friendly.",
    "Fewer parameters than additive scoring, and empirically as good when the dimensions match.",
    "The 'local-p' variant already contains the sliding-window idea that Longformer rediscovers in 2020."
  ],
  cons: [
    "Raw dot products grow with dimension, so logits blow up and softmax saturates &mdash; the paper does not yet fix this. (Vaswani's 1/&radic;d&#8342; is the patch, two years later.)",
    "Requires query and key to live in the same space, unlike additive scoring.",
    "Still inside an RNN, so still sequential."
  ],
  pick: "You already do &mdash; it is the inner loop of every model on this page."
},
{
  id: "learned-abs", name: "Learned absolute position embeddings", abbr: "Learned APE",
  date: "2017-05-08", dateLabel: "8 May 2017",
  who: "Gehring, Auli, Grangier, Yarats, Dauphin (ConvS2S)",
  src: [{label:"arXiv:1705.03122 v1", url:"https://arxiv.org/abs/1705.03122"},
        {label:"also evaluated in arXiv:1706.03762 §3.5", url:"https://arxiv.org/abs/1706.03762"}],
  dateNote: "ConvS2S (8 May 2017) predates 'Attention Is All You Need' (12 Jun 2017) by five weeks and is where learned absolute position vectors are introduced for sequence-to-sequence. Vaswani et al. report trying them and finding them equivalent to sinusoidal. BERT (2018) is what made them ubiquitous, but it is not the origin.",
  family: "pos", era: 1, covered: true, bill: "quality",
  headline: "A lookup table of position vectors, added to the token. Simple, learnable, and hard-capped at the length you trained.",
  problem: "Drop recurrence and convolution and the model is permutation-invariant: 'the bank of the river' and 'river the of bank the' are the same bag. Something must tell Q and K where each token sits.",
  idea: "Keep a trainable matrix P of shape (max_len &times; d). Add P[i] to the token embedding at position i. The network learns whatever positional geometry it wants.",
  math: "x&#7522; = Embed(tok&#7522;) + P[i],  P &isin; &#8477;^(L&#7504;&#7509;&#8339; &times; d), trainable",
  pros: [
    "Zero inductive bias imposed &mdash; the model learns the position geometry that suits the data.",
    "Trivially simple to implement and to reason about.",
    "Works fine, and is still what a large fraction of encoder models (BERT, ViT) use."
  ],
  cons: [
    "<strong>Hard length ceiling.</strong> Train on 1024 and P[20000] simply does not exist. This is the instructor's point in the session: you can never ask about a position the table never had.",
    "Absolute, not relative. The learned difference between positions 2 and 12 is not the same object as between 20 and 30, so relative distance has to be re-learned everywhere.",
    "Costs L&#7504;&#7509;&#8339;&middot;d parameters that do nothing except say 'where'.",
    "Position is injected once at the bottom and must survive every layer's residual stream."
  ],
  pick: "Fixed-length encoders where you know the maximum length at train time and never need to exceed it &mdash; classification, ViT patches, retrieval encoders. Never for a decoder you hope to extend."
},
{
  id: "sinusoidal", name: "Sinusoidal position encoding", abbr: "Sinusoidal",
  date: "2017-06-12", dateLabel: "12 Jun 2017",
  who: "Vaswani et al., §3.5",
  src: [{label:"arXiv:1706.03762 v1", url:"https://arxiv.org/abs/1706.03762"}],
  family: "pos", era: 1, covered: true, bill: "length",
  headline: "Fixed sine/cosine waves at geometrically spaced frequencies. Parameter-free, and the first honest attempt at extrapolation.",
  problem: "A learned table cannot represent a position it never saw. Can position be a <em>function</em> instead of a lookup, so that any index is representable?",
  idea: "Each dimension pair gets its own frequency, spanning wavelengths from 2&pi; up to 10000&middot;2&pi;. Position becomes a fixed multi-scale code. The authors' stated hope: PE(pos+k) is a linear function of PE(pos), so relative offsets should be learnable.",
  math: "PE(pos, 2i)   = sin(pos / 10000^(2i/d))<br>PE(pos, 2i+1) = cos(pos / 10000^(2i/d))",
  pros: [
    "No parameters, and defined for every integer &mdash; you can feed it position 10 million.",
    "Multi-scale by construction: fast dimensions resolve neighbours, slow dimensions carry coarse location.",
    "The frequency ladder here is exactly the one RoPE reuses four years later. This is the direct ancestor."
  ],
  cons: [
    "<strong>Defined at any length is not the same as works at any length.</strong> In practice it extrapolates barely better than a learned table; perplexity degrades quickly past the training length.",
    "Still additive and still absolute &mdash; it is mixed into the residual stream and competes with semantic content for the same dimensions.",
    "The nice 'relative offset is a rotation' property exists in the encoding, but the model has no structural reason to use it. RoPE's whole contribution is <em>forcing</em> it into the dot product."
  ],
  pick: "Almost never, in 2026. It is on this page because it is the idea RoPE is a corrected version of &mdash; same frequency ladder, applied multiplicatively to Q and K instead of added to x."
},
{
  id: "sdpa", name: "Scaled dot-product & multi-head attention", abbr: "Vanilla / MHA",
  date: "2017-06-12", dateLabel: "12 Jun 2017",
  who: "Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser, Polosukhin",
  src: [{label:"arXiv:1706.03762 v1", url:"https://arxiv.org/abs/1706.03762"}],
  dateNote: "The session said '2018 and 17'. The v1 submission is 12 June 2017; NeurIPS publication is December 2017. There is no 2018 version.",
  family: "exact", era: 1, covered: true, bill: "both", anchor: true,
  headline: "The baseline everything else is a discount on. Not wrong &mdash; expensive.",
  problem: "Recurrence forces you to process token t before token t+1. On a GPU with thousands of cores, that is the whole problem: you cannot parallelise over sequence length.",
  idea: "Delete recurrence. Project every token into Query, Key and Value; score all pairs at once with QK&#7488;; divide by &radic;d&#8342; so the logits do not saturate softmax; mask the future so the model cannot cheat; softmax into a distribution; read out a weighted sum of Values. Do it h times in parallel with different projections (multi-head), because one softmax distribution can only point at one thing at a time.",
  math: "Attention(Q,K,V) = softmax( (QK&#7488; + M) / &radic;d&#8342; ) V",
  whyScale: "Without 1/&radic;d&#8342;, the dot product of two random d-dimensional vectors has variance d. At d=128 the logits are ~11&times; wider than at d=1, and softmax &mdash; which amplifies by <em>absolute</em> gap, not relative gap &mdash; collapses to a one-hot. Gradients die. The square root is not decoration.",
  pros: [
    "Perfect recall. Every token can read every other token, exactly, with no approximation anywhere.",
    "Fully parallel over sequence length during training &mdash; the reason this architecture and not an RNN.",
    "Softmax gives you competition (weights sum to 1, so tokens fight for mass) and positivity. Everything downstream that removes softmax spends the rest of its paper trying to buy these two properties back.",
    "Every query re-reads the raw past from scratch, so the model can change its mind about old tokens as context evolves."
  ],
  cons: [
    "<strong>O(N&sup2;) compute.</strong> 1K tokens &rarr; 1M scores. 1M tokens &rarr; 10&sup1;&sup2; scores. Per head, per layer.",
    "<strong>O(N) KV cache that never shrinks.</strong> Every generated token permanently adds a K and a V to memory, per head, per layer. This is the bill that actually kills you in production.",
    "The two bills fight each other: the tricks that shrink the cache usually cost quality, and the tricks that keep quality do not shrink the cache.",
    "It knows nothing about position on its own. Everything in the 'Position' family exists to patch that hole.",
    "Softmax forces the (QK&#7488;)V order of operations. You cannot reassociate to K&#7488;V first, which is exactly the door linear attention kicks open."
  ],
  pick: "Short contexts (&le; 8K), any research baseline, and &mdash; still, in 2026 &mdash; the minority of layers inside every hybrid model that need exact recall. Nobody has replaced it. They have only reduced how many layers use it.",
  viz: "playground"
},
{
  id: "shaw-rel", name: "Relative position representations", abbr: "Relative PE",
  date: "2018-03-06", dateLabel: "6 Mar 2018",
  who: "Shaw, Uszkoreit, Vaswani",
  src: [{label:"arXiv:1803.02155 v1", url:"https://arxiv.org/abs/1803.02155"}],
  family: "pos", era: 1, covered: false, bill: "length",
  headline: "Stop telling the model where tokens are. Tell it how far apart they are.",
  problem: "Absolute position is the wrong variable. Language cares that the adjective is one token before the noun, not that it is at index 4,312.",
  idea: "Add a learned vector (or scalar bias) indexed by the clipped relative distance i&minus;j directly inside the attention score, not in the residual stream. Distances beyond a clip k all share one bucket.",
  math: "e&#7522;&#11002; = q&#7522;&#7488;(k&#11002; + a&#7522;&#8331;&#11002;) / &radic;d&#8342;",
  pros: [
    "Translation-invariant by construction &mdash; the same phrase scores the same anywhere in the sequence.",
    "Clipping gives free, if crude, extrapolation: everything past distance k is 'far'.",
    "Position lives in the attention score where it is actually used, not in the residual stream where it has to survive 40 layers."
  ],
  cons: [
    "Materially slower: the naive form needs a per-pair gather, breaking the clean single-GEMM path.",
    "Beyond the clip, all distances are indistinguishable &mdash; 'far' is one bucket.",
    "Extra learned parameters per layer or per head, and no closed form to extend."
  ],
  pick: "T5's simplified scalar-bias version is still a reasonable default for encoders. For decoders, RoPE or ALiBi get the same invariance with less machinery."
},
/* ============================ ERA 2 ============================ */
{
  id: "txl", name: "Transformer-XL: segment recurrence + relative PE", abbr: "Transformer-XL",
  date: "2019-01-09", dateLabel: "9 Jan 2019",
  who: "Dai, Yang, Yang, Carbonell, Le, Salakhutdinov",
  src: [{label:"arXiv:1901.02860 v1", url:"https://arxiv.org/abs/1901.02860"}],
  family: "pos", era: 2, covered: false, bill: "length",
  headline: "Cache the previous segment's hidden states and attend into them. The first practical long context.",
  problem: "Chunking a document into independent 512-token segments means every chunk boundary is total amnesia. Context fragmentation, in the paper's words.",
  idea: "Keep the previous segment's key/value states, frozen (no gradient), and let the current segment attend into them. Absolute positions break under this scheme &mdash; two segments would both start at 0 &mdash; so the paper derives a sinusoidal relative encoding to make it consistent.",
  math: "h&#7511;&#8319; = Layer( [ stopgrad(h&#7511;&#8331;&#8321;&#8319;&#8331;&#8321;) &#8226; h&#7511;&#8319;&#8331;&#8321; ] )",
  pros: [
    "Effective context grows with depth &mdash; L layers reach back L segments.",
    "Evaluation gets ~1800&times; faster than sliding-window re-computation, because you stop recomputing overlapping windows.",
    "Its relative encoding directly influenced T5's and everything after."
  ],
  cons: [
    "The cached segment gets no gradient, so the model never learns to <em>write</em> a good memory, only to read a frozen one.",
    "Still a hard, finite reach. It is a longer leash, not an unbounded one.",
    "Memory cost is now segment_len &times; layers, permanently resident."
  ],
  pick: "Superseded, but the mechanism &mdash; cache states, attend into them, use relative positions so it stays consistent &mdash; is the direct ancestor of every sliding-window + KV-cache decoder shipping today."
},
{
  id: "sparse-tx", name: "Sparse Transformer (strided / fixed patterns)", abbr: "Sparse Transformer",
  date: "2019-04-23", dateLabel: "23 Apr 2019",
  who: "Child, Gray, Radford, Sutskever (OpenAI)",
  src: [{label:"arXiv:1904.10509 v1", url:"https://arxiv.org/abs/1904.10509"}],
  family: "sparse", era: 2, covered: true, bill: "compute",
  headline: "The first serious attack on N&sup2;: hand-designed masks that still connect everything in two hops.",
  problem: "Modelling images and audio needs tens of thousands of positions. N&sup2; at N=12288 is 150M scores per head per layer. Simply unaffordable.",
  idea: "Factorise the dense mask into two cheap ones &mdash; a local window and a strided/dilated pattern &mdash; each O(N&radic;N). Alternate them across heads or layers so that any pair of positions is reachable within two attention steps.",
  math: "cost: O(N&radic;N) instead of O(N&sup2;);  A = A_local &cup; A_strided",
  pros: [
    "Real asymptotic win with no learned machinery &mdash; the pattern is fixed, so kernels are predictable and fast.",
    "Two-hop connectivity means information can still travel anywhere; it just takes an extra layer.",
    "Trained 12K-token models in 2019, which nothing dense could do."
  ],
  cons: [
    "<strong>The pattern is a guess.</strong> It is tuned to grid-structured data (images, audio) where the right stride is obvious. Text has no such stride.",
    "Two hops is not one hop: an exact single-step lookup at arbitrary distance is now impossible, and induction-head-style copying suffers.",
    "Needs custom kernels; the sparsity is only a win if your hardware can exploit it, which in 2019 mostly meant 'write it yourself'."
  ],
  pick: "When your data has genuine grid structure and you know the stride &mdash; images, spectrograms, tabular sequences. For text, the learned/selected successors (NSA, MoBA, DSA) dominate.",
  viz: "masks"
},
{
  id: "mqa", name: "Multi-Query Attention", abbr: "MQA",
  date: "2019-11-06", dateLabel: "6 Nov 2019",
  who: "Noam Shazeer",
  src: [{label:"arXiv:1911.02150 v1", url:"https://arxiv.org/abs/1911.02150"}],
  dateNote: "Note how early this is. MQA is contemporaneous with the sparse-attention wave but answers a completely different bill, and the field ignores it until 2023 when serving costs start to hurt.",
  family: "kv", era: 2, covered: true, bill: "memory",
  headline: "All heads keep their own Query. They share one Key and one Value. Cache drops by the head count.",
  problem: "Decoding is not compute-bound, it is <em>memory-bandwidth</em>-bound. To generate one token you must stream the entire KV cache from HBM into the cores. With h heads you stream h times more bytes than you need to.",
  idea: "Keep h query heads &mdash; that is where the expressive power is &mdash; but project a single shared K and a single shared V. The KV cache shrinks by exactly h&times;, and so does the bytes-per-token you must read at every decode step.",
  math: "cache/token = 2 &middot; L &middot; 1 &middot; d&#8341; &middot; bytes   (instead of 2 &middot; L &middot; h &middot; d&#8341; &middot; bytes)",
  pros: [
    "Enormous, immediate decode speedup &mdash; the paper reports roughly an order of magnitude faster incremental decoding.",
    "Cuts the exact resource that limits how many users you can serve concurrently.",
    "Training cost and FLOPs are essentially unchanged; this is purely a memory-layout win."
  ],
  cons: [
    "<strong>Measurable quality loss.</strong> All heads now read the same key space, so heads can no longer specialise in what they retrieve &mdash; only in what they ask for.",
    "Training instability was reported at scale, which is part of why GQA exists.",
    "Cannot be added to a trained model for free; you need at least an uptraining pass."
  ],
  pick: "Extreme serving pressure where you are hard-limited by concurrent users and can afford a small quality hit &mdash; on-device, or very high-throughput cheap tiers. Otherwise GQA gets most of the win for almost none of the loss.",
  viz: "kvcalc"
},
{
  id: "topk", name: "Explicit top-k sparse attention", abbr: "Top-k",
  date: "2019-12-25", dateLabel: "25 Dec 2019",
  who: "Zhao, Lin, Zhao, Wang, Sui, Li",
  src: [{label:"arXiv:1912.11637 v1", url:"https://arxiv.org/abs/1912.11637"}],
  family: "sparse", era: 2, covered: true, bill: "compute",
  headline: "Don't design the pattern. Score everything, keep the k best, mask the rest to &minus;&infin;.",
  problem: "Fixed sparse patterns are guesses. The relevant tokens are data-dependent &mdash; sometimes it is the neighbour, sometimes it is a name 3,000 tokens back.",
  idea: "Compute the scores, take the top k per query, set everything else to &minus;&infin; before softmax. The surviving mass renormalises over k entries, which sharpens the distribution as a side effect.",
  math: "M&#7522;&#11002; = 0 if s&#7522;&#11002; &isin; topk(s&#7522;&#8226;), else &minus;&infin;;  A = softmax(s + M)",
  pros: [
    "Content-based, not position-based &mdash; it can find the relevant token wherever it is.",
    "Denoising effect: killing the long tail of near-zero weights measurably improved translation and NLI quality, not just speed.",
    "Conceptually the parent of everything DeepSeek later ships (NSA, DSA) and of MoBA."
  ],
  cons: [
    "<strong>You must compute all N&sup2; scores to find the top k.</strong> This version saves nothing at all on compute; it is a quality trick that looks like an efficiency trick. Getting the actual saving is what took another five years and a cheap <em>indexer</em>.",
    "top-k is not differentiable in the selection, so gradients only flow through survivors &mdash; a token that is never selected is never learned about.",
    "A fixed k is wrong twice: too small for a genuinely diffuse query, too large for a sharply peaked one."
  ],
  pick: "As stated, only as a quality/denoising trick. For real speed you want the 2025 descendants that pair selection with a cheap index so you never materialise the full score matrix.",
  viz: "masks"
},
{
  id: "reformer", name: "Reformer (LSH attention)", abbr: "Reformer",
  date: "2020-01-13", dateLabel: "13 Jan 2020",
  who: "Kitaev, Kaiser, Levskaya",
  src: [{label:"arXiv:2001.04451 v1", url:"https://arxiv.org/abs/2001.04451"}],
  family: "sparse", era: 2, covered: false, bill: "both",
  headline: "Use locality-sensitive hashing to find the high-scoring pairs without computing all pairs.",
  problem: "Top-k needs the full score matrix to find the top k. That is circular. Can you find the large dot products without computing the dot products?",
  idea: "Softmax is dominated by the largest logits, and large dot products mean nearby angles. So hash Q and K with random rotations, sort by bucket, and attend only within a bucket. Plus reversible residual layers so activations need not be stored.",
  math: "O(N log N);  bucket(q) = argmax([qR ; &minus;qR]),  attend within bucket",
  pros: [
    "O(N log N) with a principled argument, not a hand-drawn pattern.",
    "Reversible layers cut activation memory independently of the attention trick &mdash; that idea outlived this paper.",
    "Genuinely handled 64K sequences on one accelerator in 2020."
  ],
  cons: [
    "Requires shared Q/K projections, which constrains the model.",
    "Hashing is stochastic; you need multiple rounds to keep recall acceptable, and the constant factors eat the asymptotic win below ~2K tokens.",
    "The sort/gather is memory-bound and awkward on GPUs. FlashAttention made dense attention faster than this at most practical lengths."
  ],
  pick: "Rarely. Its historical role is proving that <em>selection</em> beats <em>fixed patterns</em>, which is the lesson NSA and DSA cash in five years later."
},
{
  id: "swa", name: "Sliding window attention", abbr: "SWA",
  date: "2020-04-10", dateLabel: "10 Apr 2020",
  who: "Beltagy, Peters, Cohan (Longformer)",
  src: [{label:"arXiv:2004.05150 v1", url:"https://arxiv.org/abs/2004.05150"},
        {label:"decoder-scale deployment: Mistral 7B, arXiv:2310.06825 (10 Oct 2023)", url:"https://arxiv.org/abs/2310.06825"}],
  dateNote: "Local/windowed attention appears earlier &mdash; Luong's 'local-p' (2015) and Sparse Transformer's local pattern (Apr 2019). Longformer is the first to make windowed attention the primary mechanism with dilation and global tokens; Mistral 7B (Oct 2023) is what made it standard in production decoders.",
  family: "sparse", era: 2, covered: true, bill: "both",
  headline: "Each token attends to the last w tokens. Linear in N, and stacking layers grows the receptive field like a CNN.",
  problem: "Attention scores decay with distance for most heads anyway. Why pay N&sup2; to compute a matrix that is mostly near-zero off the diagonal?",
  idea: "Restrict the mask to a band of width w. Cost becomes O(N&middot;w). Crucially, layer &#8467; can see w&middot;&#8467; tokens back, because information hops through intermediate positions &mdash; the same receptive-field argument as a stack of convolutions. Longformer adds dilation and a handful of global tokens; Mistral pairs it with a rolling buffer cache so the cache itself is capped at w.",
  math: "cost O(N&middot;w);  theoretical receptive field = w &middot; n_layers",
  pros: [
    "Linear compute <em>and</em> a KV cache that stops growing &mdash; the rolling buffer is a hard cap, not a trend. One of very few tricks that bills both meters.",
    "Trivial to implement, hardware-friendly (it is a banded GEMM), and needs no learned selection.",
    "Matches dense quality closely on tasks with local structure, which is most of them, most of the time."
  ],
  cons: [
    "<strong>Exact long-range retrieval is gone.</strong> The receptive-field argument gives you <em>influence</em> at distance, not <em>lookup</em> at distance. Needle-in-a-haystack falls over: information must survive being re-encoded at every hop, and it degrades.",
    "The theoretical receptive field is an upper bound that real models do not reach.",
    "Choosing w is a raw guess, and it is a train-time commitment.",
    "Pure SWA breaks catastrophically when the window slides past the first tokens &mdash; which is exactly the discovery that produced attention sinks."
  ],
  pick: "As the cheap majority of a hybrid stack &mdash; Mistral, Gemma 2/3 and Command-R all interleave a few global layers with many sliding-window layers. Almost never as the only mechanism.",
  viz: "masks"
},
{
  id: "linear", name: "Linear attention (kernelised, softmax removed)", abbr: "Linear attention",
  date: "2020-06-29", dateLabel: "29 Jun 2020",
  who: "Katharopoulos, Vyas, Pappas, Fleuret",
  src: [{label:"arXiv:2006.16236 v1", url:"https://arxiv.org/abs/2006.16236"}],
  dateNote: "Efficient Attention (Shen et al., arXiv:1812.01243, Dec 2018) has the same reassociation trick earlier. 'Transformers are RNNs' is the paper that connects it to autoregressive decoding as a recurrent state, which is the property that matters here.",
  family: "linear", era: 2, covered: true, bill: "both", anchor: true,
  headline: "Take softmax out and the parentheses move. The past collapses into one fixed-size matrix that never grows.",
  problem: "softmax(QK&#7488;)V forces you to build the N&times;N matrix, because softmax is a row-wise nonlinearity &mdash; it has to see the whole row before it can normalise. That single nonlinearity is what costs N&sup2;.",
  idea: "Replace exp(q&#7488;k) with &phi;(q)&#7488;&phi;(k) for a feature map &phi; (the paper uses elu(x)+1). Now the product is associative, so compute (&phi;(K)&#7488;V) <em>first</em>. That is a d&times;d matrix &mdash; it does not depend on N at all. Autoregressively it becomes an RNN with a matrix-valued hidden state you update by addition.",
  math: "S&#7511; = S&#7511;&#8331;&#8321; + &phi;(k&#7511;) v&#7511;&#7488;  &nbsp;&nbsp; (d &times; d, constant size)<br>o&#7511; = &phi;(q&#7511;)&#7488; S&#7511; / (&phi;(q&#7511;)&#7488; z&#7511;)",
  pros: [
    "<strong>O(N) compute and O(1) memory per layer.</strong> 1K tokens, 1M tokens &mdash; the state is the same d&times;d matrix. The instructor's whole point.",
    "Decoding cost per token is constant, not growing. This is qualitatively different from every sparse method, which only slows the growth.",
    "The RNN framing means you get streaming inference for free."
  ],
  cons: [
    "<strong>You lose competition.</strong> Softmax weights sum to 1, so tokens fight for mass. Raw &phi; scores do not; a new token can be added to the state without anything being displaced.",
    "<strong>You lose positivity and normalisation.</strong> Outputs can be arbitrary magnitude, and the denominator trick is numerically fragile.",
    "<strong>You lose the fresh re-read.</strong> Under softmax, every query re-examines the raw past and can reinterpret it. Here, every query reads the same compressed state. The model cannot change its mind about old tokens.",
    "<strong>Fixed capacity is a hard information limit.</strong> A d&times;d state cannot store N&gg;d distinct facts. Exact recall of a specific token 100K back is not 'hard' &mdash; it is provably impossible past capacity.",
    "In 2020 it was, bluntly, much worse than softmax at language modelling, and the field wrote it off for four years."
  ],
  pick: "Never alone, in 2026 &mdash; but as the majority of layers in a hybrid (Qwen3-Next, Kimi Linear, MiniMax), with the modern fixes stacked on: delta rule for overwriting, gating for forgetting, and a few full-attention layers for the recall a fixed state cannot do.",
  viz: "state"
},
{
  id: "bigbird", name: "BigBird (window + global + random)", abbr: "BigBird",
  date: "2020-07-28", dateLabel: "28 Jul 2020",
  who: "Zaheer et al. (Google)",
  src: [{label:"arXiv:2007.14062 v1", url:"https://arxiv.org/abs/2007.14062"}],
  family: "sparse", era: 2, covered: false, bill: "compute",
  headline: "Window + a few global tokens + random edges, with a proof that it is still a universal approximator.",
  problem: "Sparse patterns kept being accused of throwing away expressiveness. Is there a sparse pattern you can actually prove things about?",
  idea: "Combine three patterns: local window (fluency), global tokens that attend to and are attended by everything (a routing hub), and a handful of random edges per query. Random-graph theory then gives you short path lengths, and the paper proves the result is Turing-complete and a universal sequence approximator.",
  math: "A = A_window &cup; A_global &cup; A_random,  |A| = O(N)",
  pros: [
    "The theory is real: sparse does not have to mean less expressive in principle.",
    "Global tokens are a genuinely good idea and survive into every modern hybrid, including as attention sinks.",
    "Strong results on long-document QA and summarisation in its era."
  ],
  cons: [
    "<strong>The proof needs &Theta;(N) layers to realise those short paths.</strong> Universality that requires a deeper network than you will ever train is a weak guarantee, and the paper says so.",
    "Random edges are miserable on hardware &mdash; scattered gathers, no coalescing. Real speedups lagged the FLOP counts badly.",
    "Encoder-shaped. The random pattern does not translate cleanly to causal decoding.",
    "Empirically beaten in most later comparisons by simply doing dense attention with FlashAttention."
  ],
  pick: "Long-document encoders where the task is genuinely global (retrieval, classification over 8K+ tokens) and you are not decoding. For generation, look at NSA/MoBA instead.",
  viz: "masks"
},
{
  id: "performer", name: "Performer (FAVOR+ random features)", abbr: "Performer",
  date: "2020-09-30", dateLabel: "30 Sep 2020",
  who: "Choromanski et al. (Google)",
  src: [{label:"arXiv:2009.14794 v1", url:"https://arxiv.org/abs/2009.14794"}],
  family: "linear", era: 2, covered: false, bill: "both",
  headline: "Approximate softmax itself with positive random features, so you keep the softmax kernel and still get linear cost.",
  problem: "Linear attention with &phi;=elu+1 is not approximating softmax at all &mdash; it is a different function that happens to be cheap. Can you get O(N) while provably estimating the real softmax kernel?",
  idea: "exp(q&#7488;k) is a Gaussian kernel in disguise, and Gaussian kernels admit unbiased random-feature estimators. FAVOR+ uses <em>positive</em> orthogonal random features so the estimate stays non-negative and low-variance, then reassociates as usual.",
  math: "exp(q&#7488;k) &asymp; &#120124;[&phi;(q)&#7488;&phi;(k)],  &phi; positive orthogonal random features",
  pros: [
    "Unbiased estimate of true softmax attention with variance bounds &mdash; the most principled member of the linear family.",
    "Drop-in: you can approximate a trained softmax model's attention without retraining from scratch.",
    "The positivity fix (over earlier trigonometric features) was a real and necessary insight."
  ],
  cons: [
    "Unbiased is not accurate. Variance is high exactly where softmax is sharp &mdash; and sharp attention is where the important behaviour lives (induction, copying, retrieval).",
    "Needs many random features to be usable, which eats the constant-factor advantage at moderate N.",
    "Like all 2020 efficient transformers, it underperformed on real autoregressive LM benchmarks and did not survive Long Range Arena scrutiny."
  ],
  pick: "Long non-autoregressive workloads with diffuse attention &mdash; some vision and protein work still uses it. Not for language modelling."
},
/* ============================ ERA 3 ============================ */
{
  id: "deltanet-orig", name: "The delta rule (DeltaNet / fast weight programmers)", abbr: "Delta rule",
  date: "2021-02-22", dateLabel: "22 Feb 2021",
  who: "Schlag, Irie, Schmidhuber",
  src: [{label:"arXiv:2102.11174 v1", url:"https://arxiv.org/abs/2102.11174"},
        {label:"parallel training algorithm: arXiv:2406.06484 (10 Jun 2024)", url:"https://arxiv.org/abs/2406.06484"}],
  dateNote: "The session dated DeltaNet to 2024. The delta rule for linear attention is Schlag, Irie & Schmidhuber, 22 Feb 2021 &mdash; itself building on Schmidhuber's 1992 fast weight controllers. What arrived on 10 Jun 2024 (Yang et al.) is the chunkwise <em>parallel</em> training algorithm that made it trainable at scale, which is arguably the reason anyone uses it now. Both dates are on this timeline; the mechanism is 2021.",
  family: "linear", era: 3, covered: true, bill: "quality",
  headline: "Linear attention could only add to memory. The delta rule lets it overwrite.",
  problem: "S&#7511; = S&#7511;&#8331;&#8321; + k v&#7488; is pure accumulation. If key k already stored value 40 and the model now wants it to be 55, plain linear attention writes 40+55 = 95. The state has no way to say 'replace'. Keys collide, the state saturates, and old junk never leaves.",
  idea: "Before writing, <em>read</em> what the state currently holds for this key, take the difference against what you want it to hold, and write only the correction. This is Widrow&ndash;Hoff, 1960, applied to a matrix-valued memory: it is one step of online gradient descent on &#8214;S k &minus; v&#8214;&sup2;.",
  math: "v&#7511;&#7580;&#7503;&#7496; = S&#7511;&#8331;&#8321; k&#7511;  <span class='mut'>(what memory says now)</span><br>S&#7511; = S&#7511;&#8331;&#8321; + &beta;&#7511; (v&#7511; &minus; v&#7511;&#7580;&#7503;&#7496;) k&#7511;&#7488;  <span class='mut'>(write only the gap)</span>",
  pros: [
    "Fixes the single worst failure mode of linear attention. Key&ndash;value associations become updatable instead of append-only.",
    "Strictly more expressive: it can solve in-context associative recall tasks that additive linear attention provably cannot.",
    "&beta;&#7511; is a learned write strength &mdash; &beta;=0 ignores the token, &beta;=1 fully overwrites, and the model chooses per token.",
    "Still O(1) state and O(N) total. The fix is free in asymptotics."
  ],
  cons: [
    "<strong>Sequentially dependent by construction</strong> &mdash; you must read S&#7511;&#8331;&#8321; before writing S&#7511;. That is why it sat unused for three years until the 2024 chunkwise WY-representation algorithm made it parallel.",
    "It can overwrite but it still cannot <em>forget</em> &mdash; nothing decays, so unreferenced keys occupy capacity forever. Gating fixes that, four years later.",
    "Capacity is still d&times;d. Overwriting uses the space better; it does not create more.",
    "Extra read-then-write per step; wall-clock cost is meaningfully above vanilla linear attention."
  ],
  pick: "As the linear component of any hybrid you are building today &mdash; but use the gated version. Plain DeltaNet is now a stepping stone.",
  viz: "delta"
},
{
  id: "rope", name: "Rotary Position Embedding", abbr: "RoPE",
  date: "2021-04-20", dateLabel: "20 Apr 2021",
  who: "Su, Lu, Pan, Murtadha, Wen, Liu",
  src: [{label:"arXiv:2104.09864 v1", url:"https://arxiv.org/abs/2104.09864"}],
  family: "pos", era: 3, covered: true, bill: "length", anchor: true,
  headline: "Rotate Q and K by an angle proportional to position. The dot product then depends only on the difference.",
  problem: "Additive position encodings put 'where' into the residual stream and hope the dot product extracts relative distance. It mostly does not, and a learned table cannot go past its training length at all.",
  idea: "Treat each consecutive pair of dimensions as a point in a 2-D plane and rotate it by m&theta;&#7522;, where m is the position and &theta;&#7522; is a per-pair frequency from the same 10000^(&minus;2i/d) ladder as sinusoidal. Rotation matrices compose, so R&#7504;&#7488;R&#8345; = R&#8345;&#8331;&#7504;: after rotating, q&#7504;&#7488;k&#8345; depends only on n&minus;m. Relative position is not a hint you added &mdash; it is a mathematical identity of the score.",
  math: "q&#7504; = R&#7504; W&#8339;x&#7504;,  k&#8345; = R&#8345; W&#8342;x&#8345;<br>&rArr; q&#7504;&#7488;k&#8345; = x&#7504;&#7488;W&#8339;&#7488; R&#8345;&#8331;&#7504; W&#8342;x&#8345;  <span class='mut'>(depends on n&minus;m only)</span>",
  pros: [
    "Relative position is <em>enforced</em> by the algebra, not merely available. Shift both tokens by 1000 and the score is bit-identical.",
    "Zero parameters, no length table, defined for any integer position.",
    "Injected directly into Q and K inside the attention block, so it does not consume residual-stream capacity. Modern models (Llama, Qwen, Mistral, DeepSeek) have no position embedding at the bottom at all.",
    "Norm-preserving &mdash; it is a rotation, so it cannot blow up activations.",
    "Its explicit frequency structure is what made all the later context-extension surgery (PI, NTK, YaRN, LongRoPE) possible. You cannot rescale a learned lookup table."
  ],
  cons: [
    "<strong>It does not actually extrapolate.</strong> This is the big one. The high-frequency dimensions wrap many times within the training length, so beyond it the model sees rotation phases it has never encountered. Quality collapses sharply just past L&#7511;&#7523;&#8336;&#7522;&#8345; &mdash; RoPE was validated around 32K and degrades badly past roughly 2&times; whatever you trained on.",
    "Long-range decay is a side effect of the frequency ladder, not a designed property, and it is weakly controllable.",
    "Must be applied at every layer to every Q and K &mdash; a small but real per-layer cost, and a common source of subtle implementation bugs (interleaved vs half-split conventions differ across codebases).",
    "The base &theta;=10000 is a hyperparameter that quietly sets your usable length, and almost everyone inherits it without thinking."
  ],
  pick: "Default for any decoder you are training in 2026. Pick the base &theta; deliberately for your target length, and plan the extension strategy (YaRN or DroPE) before you train, not after.",
  viz: "rope"
},
{
  id: "alibi", name: "ALiBi (Attention with Linear Biases)", abbr: "ALiBi",
  date: "2021-08-27", dateLabel: "27 Aug 2021",
  who: "Press, Smith, Lewis",
  src: [{label:"arXiv:2108.12409 v1", url:"https://arxiv.org/abs/2108.12409"}],
  family: "pos", era: 3, covered: true, bill: "length",
  headline: "Delete position embeddings entirely. Just subtract a distance penalty from the score, with a different slope per head.",
  problem: "Every positional scheme up to here fails past its training length. What if the position signal were so simple that there is nothing to fail?",
  idea: "Add &minus;m&middot;(i&minus;j) to the pre-softmax logit, where m is a fixed per-head slope from a geometric series. Heads with steep slopes become local; heads with shallow slopes stay near-global. There is nothing learned and nothing to extrapolate &mdash; the bias is defined at every distance.",
  math: "score&#7522;&#11002; = q&#7522;&#7488;k&#11002;/&radic;d&#8342; &minus; m&#8341; &middot; (i &minus; j),  m&#8341; = 2^(&minus;8h/H)",
  pros: [
    "<strong>The extrapolation claim is real and was the paper's whole point:</strong> train at 1024, evaluate at 2048+, and perplexity holds. That was not true of anything before it.",
    "Zero parameters and essentially zero cost &mdash; one fused add.",
    "Interpretable: each head has an explicit, inspectable locality scale.",
    "Trains faster and uses less memory than learned position embeddings."
  ],
  cons: [
    "<strong>It extrapolates by becoming more local.</strong> The honest reading of the result is that the linear penalty makes distant tokens exponentially unlikely to be attended, so a longer input mostly does not change what the model reads. Perplexity holds; genuine long-range <em>retrieval</em> does not improve. Later needle-in-a-haystack evaluations exposed this clearly.",
    "The monotone decay is a hard prior. A task where the answer is deliberately far away (retrieval, long code) fights the bias directly.",
    "The slopes are fixed, not learned, so you cannot adapt locality to your data.",
    "Bidirectional/encoder use is awkward, and it composes poorly with the KV-cache tricks that assume position lives in Q/K.",
    "The ecosystem went to RoPE, so kernel and tooling support is comparatively thin."
  ],
  pick: "Streaming or unbounded-length settings where you need graceful degradation and mostly-local dependencies, and you cannot afford context-extension fine-tuning. If long-range lookup matters, RoPE + YaRN beats it.",
  viz: "rope"
},
{
  id: "flash", name: "FlashAttention (IO-aware exact attention)", abbr: "FlashAttention",
  date: "2022-05-27", dateLabel: "27 May 2022",
  who: "Dao, Fu, Ermon, Rudra, Ré",
  src: [{label:"arXiv:2205.14135 v1", url:"https://arxiv.org/abs/2205.14135"},
        {label:"FlashAttention-2: arXiv:2307.08691 (17 Jul 2023)", url:"https://arxiv.org/abs/2307.08691"}],
  family: "systems", era: 3, covered: false, bill: "memory",
  headline: "Same softmax, same numbers, bit-for-bit. It just never writes the N&times;N matrix to memory.",
  problem: "Everyone assumed attention was compute-bound and spent three years approximating the math. It was <em>memory-bandwidth</em>-bound: the expensive part was writing the N&times;N score matrix to HBM and reading it back, not the multiplies.",
  idea: "Tile Q, K, V into blocks that fit in on-chip SRAM. Compute attention block by block, maintaining running max and sum so that softmax stays numerically correct across tiles (online softmax). The full matrix never exists anywhere except in registers. For the backward pass, recompute it instead of storing it.",
  math: "same output as softmax(QK&#7488;/&radic;d)V, exactly<br>HBM traffic: O(N&sup2;d) &rarr; O(N&sup2;d&sup2;/M)",
  pros: [
    "<strong>Exact.</strong> No approximation, no quality trade-off, nothing to argue about. Rare and precious on this page.",
    "Memory goes from O(N&sup2;) to O(N) for activations, which is what actually unblocked long-context training.",
    "2&ndash;4&times; wall-clock speedup, and it made dense attention faster than most of the 2020 'efficient' transformers &mdash; which is why so many of them died.",
    "Composes with everything else here: sparse, GQA, MLA all use Flash-style kernels underneath."
  ],
  cons: [
    "<strong>It does not change the asymptotics.</strong> Still O(N&sup2;) FLOPs and still an O(N) KV cache. It bought roughly one order of magnitude, once. It cannot be bought again.",
    "Deeply hardware-specific &mdash; each GPU generation needs a rewrite (FA-2 for Ampere, FA-3 for Hopper), and support for non-standard masks or biases (ALiBi, custom sparsity) lags by months.",
    "Recomputation in the backward pass trades FLOPs for memory; on compute-bound configurations that is not always a win.",
    "It made everyone complacent about N&sup2; for about two years."
  ],
  pick: "Always. There is no configuration in which you should prefer a naive attention kernel. This is a free win and one of the very few entries on this page with no quality cost."
},
/* ============================ ERA 4 ============================ */
{
  id: "gqa", name: "Grouped-Query Attention", abbr: "GQA",
  date: "2023-05-22", dateLabel: "22 May 2023",
  who: "Ainslie, Lee-Thorp, de Jong, Zemlyanskiy, Lebrón, Sanghai",
  src: [{label:"arXiv:2305.13245 v1", url:"https://arxiv.org/abs/2305.13245"}],
  family: "kv", era: 4, covered: true, bill: "memory", anchor: true,
  headline: "The interpolation between MHA and MQA that everybody actually ships. Groups of query heads share one KV head.",
  problem: "MHA's cache is too big; MQA's quality drop and training instability are too real. The trade-off had been posed as binary for four years.",
  idea: "Put the h query heads into g groups. Each group shares one K and one V head. g = h is MHA, g = 1 is MQA, and g = 8 turns out to sit almost exactly on MHA's quality with MQA's speed. The paper also gives an <em>uptraining</em> recipe: mean-pool the KV heads of an existing MHA checkpoint and fine-tune on ~5% of the original compute, so you do not retrain from scratch.",
  math: "cache/token = 2 &middot; L &middot; g &middot; d&#8341; &middot; bytes,  1 &le; g &le; h",
  pros: [
    "Hits the sweet spot empirically: quality within noise of MHA at 4&ndash;8&times; less cache.",
    "One knob (g) that trades quality for memory smoothly, so you can tune it to your serving constraint rather than picking a corner.",
    "Uptraining means existing MHA checkpoints can be converted cheaply &mdash; a huge practical reason for adoption.",
    "Maps cleanly onto tensor parallelism: one KV head per device when g equals your TP degree.",
    "It is the default in Llama 2/3, Mistral, Qwen and most open models &mdash; the baseline any new KV idea has to beat."
  ],
  cons: [
    "It is a compromise, and it is honest about that. Heads in a group still cannot specialise their retrieval.",
    "<strong>The cache still grows linearly with sequence length.</strong> GQA divides the constant. It does not change the asymptotics, so at 1M tokens you are still in trouble &mdash; just 8&times; later.",
    "Choosing g is empirical, and the right value depends on model size, head count and your target hardware.",
    "Uptraining is cheap but not free, and quality after conversion is slightly below training with GQA from scratch."
  ],
  pick: "The default for anything you are training in 2026 that is not doing something more exotic. Start at g=8 and only move if you have measured a reason to.",
  viz: "kvcalc"
},
{
  id: "pi", name: "Position Interpolation", abbr: "PI",
  date: "2023-06-27", dateLabel: "27 Jun 2023",
  who: "Chen, Wong, Chen, Tian (Meta)",
  src: [{label:"arXiv:2306.15595 v1", url:"https://arxiv.org/abs/2306.15595"}],
  family: "pos", era: 4, covered: false, bill: "length",
  headline: "Don't extrapolate the rotation. Squash the new positions into the range the model already knows.",
  problem: "RoPE past its training length shows the model rotation phases it never saw, and quality falls off a cliff. Meanwhile Llama shipped at 2K and everyone wanted 32K last week.",
  idea: "To go from L to L', divide every position index by L'/L before rotating. Position 8000 in a 2K model is presented as position 2000. The model now only ever sees in-distribution angles &mdash; it just has to learn finer resolution between them, which ~1000 fine-tuning steps is enough for.",
  math: "m &rarr; m &middot; L/L' ,  equivalently &theta;&#7522; &rarr; &theta;&#7522; &middot; L/L'  (uniform)",
  pros: [
    "Extremely cheap: 1000 steps of fine-tuning turns a 2K model into a 32K model. This is what made long context a commodity in mid-2023.",
    "The paper gives an actual bound &mdash; interpolated attention scores are well-behaved, extrapolated ones are not &mdash; so it is not just a hack that happened to work.",
    "Preserves the original architecture completely; it is a change to the position index, nothing else."
  ],
  cons: [
    "<strong>Uniform scaling squashes the high-frequency dimensions too.</strong> Those dimensions encode <em>local</em> distance &mdash; adjacent-token relationships &mdash; and crowding them measurably degrades short-context quality. This exact flaw is what NTK-aware scaling exists to fix, days later.",
    "Requires fine-tuning. Not zero-shot.",
    "Effective resolution drops as the scale factor grows; at 32&times; the model genuinely cannot tell nearby positions apart as well."
  ],
  pick: "Superseded by YaRN, which is strictly better for the same cost. Included because it is the paper that framed the whole interpolate-don't-extrapolate insight."
},
{
  id: "ntk", name: "NTK-aware scaled RoPE", abbr: "NTK-aware",
  date: "2023-06-28", dateLabel: "late Jun 2023",
  who: "bloc97 (Reddit, r/LocalLLaMA)",
  src: [{label:"r/LocalLLaMA post by bloc97", url:"https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/"},
        {label:"earliest independent timestamp: HF text-generation-inference issue #512, opened 30 Jun 2023", url:"https://github.com/huggingface/text-generation-inference/issues/512"},
        {label:"described and credited in YaRN §3.2, arXiv:2309.00071", url:"https://arxiv.org/abs/2309.00071"}],
  dateNote: "This is the weakest date on the page and I am flagging it rather than inventing precision. It is a Reddit post, not a paper, so there is no arXiv timestamp, and Reddit was not fetchable from my environment. What I can verify: the YaRN paper credits 'bloc97, 2023' for NTK-aware interpolation, and a HuggingFace TGI issue quoting the post verbatim was opened on 30 June 2023 &mdash; so the post is on or before 30 Jun 2023, and the community-cited date is 28 Jun 2023. Treat the day as approximate; the month is solid. Worth noting that one of the most widely deployed context-extension methods in history was a forum post.",
  family: "pos", era: 4, covered: true, bill: "length",
  headline: "Interpolate the low frequencies, extrapolate the high ones. Scale the RoPE base instead of the positions.",
  problem: "PI scales every frequency equally, which crushes the high-frequency dimensions that encode local, adjacent-token structure &mdash; so you buy length by damaging short-range quality.",
  idea: "Borrow the neural-tangent-kernel observation that networks struggle to learn high-frequency detail. Rather than scaling positions, scale the RoPE <em>base</em>: &theta; = 10000 &rarr; 10000&middot;s^(d/(d&minus;2)). This spreads the interpolation pressure unevenly &mdash; the fast dimensions are left nearly untouched (they still extrapolate, and they only need to resolve nearby tokens anyway) while the slow dimensions absorb almost all of the stretching.",
  math: "base' = base &middot; s^(d/(d&minus;2)),  s = L'/L<br>&rArr; high freqs &asymp; unchanged, low freqs strongly interpolated",
  pros: [
    "<strong>Works zero-shot.</strong> No fine-tuning at all &mdash; the headline claim was 8K+ from a 2K Llama with minimal perplexity loss, and it held up.",
    "A one-line change to the base constant. Deployable in an afternoon, which is exactly why it spread through the open-weight ecosystem in days.",
    "Preserves local structure, which PI damages &mdash; the specific fix it was designed for.",
    "Directly became the 'rope_theta' scaling knob that half the ecosystem now ships."
  ],
  cons: [
    "The NTK justification is a heuristic analogy, not a derivation. It works; the stated reason for why it works is loose.",
    "It still stretches <em>every</em> dimension by some amount, including ones whose wavelength already exceeds the training context and therefore should not be touched at all. YaRN's 'by-parts' fix addresses precisely this.",
    "Zero-shot quality degrades faster than fine-tuned methods past roughly 4&times; extension.",
    "It does not touch the attention temperature, which drifts as context grows &mdash; another thing YaRN adds.",
    "No paper, no peer review, no ablations at scale. The community adopted it on a Colab notebook and got lucky."
  ],
  pick: "As an emergency zero-shot extension when you cannot fine-tune. If you can fine-tune even briefly, YaRN dominates it.",
  viz: "rope"
},
{
  id: "yarn", name: "YaRN (Yet another RoPE extensioN)", abbr: "YaRN",
  date: "2023-08-31", dateLabel: "31 Aug 2023",
  who: "Peng, Quesnelle, Fan, Shippole",
  src: [{label:"arXiv:2309.00071 v1", url:"https://arxiv.org/abs/2309.00071"}],
  dateNote: "The arXiv identifier is 2309.* (September announcement cycle) but the v1 submission timestamp is 31 August 2023. I am using the submission date, consistently with every other entry on this page.",
  family: "pos", era: 4, covered: true, bill: "length", anchor: true,
  headline: "The one that got RoPE extension right: interpolate by wavelength, and fix the temperature nobody else noticed.",
  problem: "PI over-interpolates; NTK-aware interpolates the wrong dimensions a little; and both ignore that as context grows, the average attention entropy drifts and the softmax gets too flat.",
  idea: "Three fixes, composed. (1) <strong>NTK-by-parts</strong>: classify each dimension by whether its wavelength fits inside the original context. Dimensions with short wavelengths (many rotations already seen) are not interpolated at all; dimensions with wavelengths longer than the context are fully interpolated; a ramp blends the middle. (2) <strong>Attention temperature</strong>: multiply logits by a scalar 1/t &asymp; 0.1&middot;ln(s)+1 to restore the pre-extension entropy &mdash; it folds into the RoPE tables, so it is genuinely free at inference. (3) <strong>Dynamic scaling</strong> at inference so short prompts are not penalised.",
  math: "&theta;&#7522;' = (1&minus;&gamma;&#7522;)&middot;&theta;&#7522;/s + &gamma;&#7522;&middot;&theta;&#7522;,  &gamma;&#7522; from wavelength ratio<br>logits &times;= 1/t,  &radic;(1/t) = 0.1&middot;ln(s) + 1",
  pros: [
    "<strong>10&times; less data and 2.5&times; fewer steps than any prior extension method</strong>, per the paper &mdash; a few hundred steps to get a working 128K model.",
    "The temperature correction is a genuinely original observation and costs nothing to apply.",
    "Transfers: YaRN'd models extrapolate beyond even the length they were fine-tuned on.",
    "Became the de facto standard. Qwen, DeepSeek, Nous and much of the open ecosystem ship YaRN configs, and it is a first-class option in vLLM and HF."
  ],
  cons: [
    "<strong>Still needs fine-tuning.</strong> The zero-shot ('Dynamic-YaRN') variant is decent but not as good, so you pay a training bill.",
    "Several hyperparameters (&alpha;, &beta; ramp bounds, the 0.1 in the temperature) were fitted empirically on Llama. Their transfer to very different architectures is assumed more than demonstrated.",
    "Long-context <em>perplexity</em> is where it shines; retrieval-style benchmarks improve less than the perplexity curves suggest.",
    "It is still surgery on a model that was trained short. DroPE's argument is that the whole category is treating a symptom.",
    "Static scaling degrades short-context performance unless you use the dynamic variant, which complicates serving."
  ],
  pick: "Today's correct answer for extending an already-trained RoPE model, if you can afford a few hundred fine-tuning steps. If you are training from scratch and can plan ahead, look hard at DroPE instead.",
  viz: "rope"
},
{
  id: "paged", name: "PagedAttention (vLLM)", abbr: "PagedAttention",
  date: "2023-09-12", dateLabel: "12 Sep 2023",
  who: "Kwon, Li, Zhuang, Sheng, Zheng, Yu, Gonzalez, Zhang, Stoica",
  src: [{label:"arXiv:2309.06180 v1", url:"https://arxiv.org/abs/2309.06180"}],
  family: "systems", era: 4, covered: false, bill: "memory",
  headline: "The KV cache is not too big. It is too fragmented. Give it virtual memory.",
  problem: "Serving systems pre-allocated a contiguous buffer for each request's maximum possible length. Measured waste: 60&ndash;80% of allocated KV memory was never used. You were paying for tokens that did not exist.",
  idea: "Steal the operating-system idea. Store the KV cache in fixed-size non-contiguous blocks with a page table per sequence. Waste drops to under one block per sequence, and identical prefixes (system prompts, few-shot examples, beam-search branches) can be shared by pointing at the same physical pages, copy-on-write.",
  math: "waste &lt; 4% (from 60&ndash;80%);  prefix sharing via reference-counted pages",
  pros: [
    "2&ndash;4&times; throughput at the same latency, with <em>zero</em> change to model quality &mdash; it is pure allocation strategy.",
    "Prefix sharing is enormous in practice: every user hitting the same system prompt pays for it once.",
    "Became infrastructure. vLLM, TensorRT-LLM and SGLang all do this now; it is part of why the KV cache is manageable at all in production."
  ],
  cons: [
    "Not a model technique &mdash; it does not reduce the bytes a sequence fundamentally needs, only the bytes you waste around them.",
    "Block-table indirection adds kernel complexity and a small per-access overhead; attention kernels must be rewritten to be page-aware.",
    "Block size is a tuning parameter with real throughput consequences.",
    "It bought a large constant factor once. It does not help with the linear growth in N."
  ],
  pick: "You are already using it if you serve with vLLM or SGLang. It is on this timeline because a meaningful share of the '1M context' claims in 2024 were as much about paging and quantised caches as about attention math, and it is honest to say so."
},
{
  id: "sinks", name: "Attention sinks (StreamingLLM)", abbr: "Attention sinks",
  date: "2023-09-29", dateLabel: "29 Sep 2023",
  who: "Xiao, Tian, Chen, Han, Lewis",
  src: [{label:"arXiv:2309.17453 v1", url:"https://arxiv.org/abs/2309.17453"},
        {label:"related earlier observation: E. Miller, 'Attention Is Off By One', 24 Jul 2023", url:"https://www.evanmiller.org/attention-is-off-by-one.html"},
        {label:"trained per-head sink logit: gpt-oss model card, 5 Aug 2025", url:"https://arxiv.org/abs/2508.10925"}],
  family: "sparse", era: 4, covered: true, bill: "memory", anchor: true,
  headline: "Evict the first four tokens from a sliding-window cache and the model collapses. Keep them and it streams forever.",
  problem: "Sliding-window decoding should work indefinitely. In practice, the moment the window slid past the very first tokens, perplexity exploded. Nobody could explain why.",
  idea: "The diagnosis is the contribution. Softmax must sum to 1 &mdash; it cannot say 'nothing here is relevant'. So heads with nothing to attend to dump their probability mass somewhere harmless, and they overwhelmingly choose the first few tokens, purely because those tokens are visible to every query. Those tokens are not semantically important; they are a pressure-release valve. Evict them and the mass is forced onto real tokens, distorting every distribution. The fix is four tokens wide: always keep the first ~4 keys, plus the rolling window.",
  math: "cache = { k&#8320;..k&#8323; } &cup; { k&#8345;&#8331;&#8342;..k&#8345; }<br>root cause: &sum;&#11002; softmax&#11002; = 1, always",
  pros: [
    "<strong>Four tokens.</strong> The fix costs four KV entries and enables stable decoding over 4M+ tokens. Best cost-to-benefit ratio on this entire page.",
    "It is an explanation, not just a patch &mdash; it identified a real structural artefact of softmax that shows up in interpretability, quantisation (sinks are massive activation outliers) and pruning work.",
    "No fine-tuning, no retraining; it is a change to which keys you keep.",
    "Graduated into architecture: gpt-oss (Aug 2025) ships a <em>learned per-head sink logit</em> in the softmax denominator, giving heads an explicit 'attend to nothing' option. The hack became a parameter."
  ],
  cons: [
    "<strong>It does not extend context. It extends fluency.</strong> The paper is explicit: StreamingLLM keeps the model coherent over an infinite stream, but anything evicted from the window is gone. It cannot answer questions about token 3 million.",
    "Trades an obvious failure (gibberish) for a subtle one (confident amnesia), which is arguably worse in production.",
    "Optimal sink count is empirical, and models pretrained without a dedicated sink token behave inconsistently.",
    "It is a workaround for a softmax design flaw. Miller's softmax1 (add 1 to the denominator) and gpt-oss's learned sink both argue the real fix belongs in the normaliser."
  ],
  pick: "Any unbounded streaming workload &mdash; long-running assistants, live transcription, agents that never restart. Combine with sliding window. Do not mistake it for context extension.",
  viz: "masks"
},
{
  id: "longrope", name: "LongRoPE (evolutionary search over frequencies)", abbr: "LongRoPE",
  date: "2024-02-21", dateLabel: "21 Feb 2024",
  who: "Ding, Zhang, Zhang, Xu, Xia, Zhang, Yang (Microsoft)",
  src: [{label:"arXiv:2402.13753 v1", url:"https://arxiv.org/abs/2402.13753"}],
  family: "pos", era: 4, covered: false, bill: "length",
  headline: "Stop hand-designing the rescaling curve. Search for it.",
  problem: "PI, NTK and YaRN all pick a rescaling rule analytically. But the ideal per-dimension scale is model-specific and non-uniform in ways no closed form captures.",
  idea: "Treat the per-dimension rescale factors as a search space and run evolutionary search against perplexity, plus a progressive schedule (256K fine-tune, then extend to 2M) and a short-context readjustment so nearby positions are not damaged.",
  math: "search &lambda;&#7522; per dimension, minimise PPL;  256K FT &rarr; 2048K",
  pros: [
    "2M tokens, which was a genuine order-of-magnitude jump at the time.",
    "Empirically confirms that the non-uniformity YaRN hand-derived is real, and that hand-derived versions leave something on the table.",
    "Explicitly repairs short-context performance, which most extension methods quietly damage."
  ],
  cons: [
    "The search costs real GPU time and must be redone per model &mdash; you cannot publish a constant the way YaRN can.",
    "Optimises perplexity, which correlates only loosely with long-range retrieval quality.",
    "More moving parts than YaRN for a benefit that only shows up at extreme extension ratios."
  ],
  pick: "When you need extension beyond roughly 8&times; and you have compute to burn on the search. Below that, YaRN is simpler and close enough."
},
{
  id: "infini", name: "Infini-attention", abbr: "Infini-attention",
  date: "2024-04-10", dateLabel: "10 Apr 2024",
  who: "Munkhdalai, Faruqui, Gopal (Google)",
  src: [{label:"arXiv:2404.07143 v1", url:"https://arxiv.org/abs/2404.07143"}],
  family: "linear", era: 4, covered: false, bill: "both",
  headline: "Put a compressive linear-attention memory and a local softmax window in the same head, and learn the mix.",
  problem: "Sliding window forgets everything outside the window. Linear attention remembers everything, badly. Both live in different layers in most hybrids &mdash; why not in the same head?",
  idea: "Each head keeps a local causal window (exact) plus a compressive matrix memory over everything evicted from that window, updated with a delta-rule-style write. A learned scalar gate &beta; blends the two readouts per head.",
  math: "A = sigmoid(&beta;)&middot;A_mem + (1&minus;sigmoid(&beta;))&middot;A_local,  bounded memory",
  pros: [
    "Constant memory with unbounded input &mdash; the paper demonstrates 1M-token passkey retrieval from a 1B model fine-tuned at 5K.",
    "The per-head gate is elegant: the model decides head by head whether it is a local head or a memory head.",
    "Anticipates the 2025 hybrid consensus by a year, at finer granularity (within-head rather than across-layer)."
  ],
  cons: [
    "Reproduction was difficult and results outside the paper were mixed; it never got broad independent validation.",
    "The compressive memory has the same fixed-capacity ceiling as any linear state &mdash; passkey retrieval is an easy case because there is exactly one thing to remember.",
    "Nobody shipped it at scale. The layer-level hybrid won instead, largely because it is easier to implement, kernel and reason about."
  ],
  pick: "Read it for the idea. In practice, layer-level hybrids (Qwen3-Next, Kimi Linear) achieved the same goal with less complexity."
},
{
  id: "mla", name: "Multi-head Latent Attention", abbr: "MLA",
  date: "2024-05-07", dateLabel: "7 May 2024",
  who: "DeepSeek-AI (DeepSeek-V2)",
  src: [{label:"arXiv:2405.04434 v1", url:"https://arxiv.org/abs/2405.04434"}],
  family: "kv", era: 4, covered: true, bill: "memory", anchor: true,
  headline: "Don't share KV heads. Compress KV into a low-rank latent and cache only that. Better quality than MHA, far less cache than GQA.",
  problem: "GQA and MQA both shrink the cache by <em>deleting</em> heads, which costs expressiveness. Is the cache actually low-rank? If so you could compress instead of delete, and lose nothing.",
  idea: "Project the hidden state down to a small latent c (e.g. 512 dims for a 7168-dim model) and cache <em>only c</em>. At attention time, project back up to full per-head K and V. The up-projection matrices are static, so they can be absorbed into W&#8339; and W&#8338; &mdash; meaning you never actually materialise the full K/V during decoding. RoPE breaks this absorption (it is position-dependent, so it cannot commute through), so DeepSeek splits each head into a compressed part and a small separate RoPE-carrying part &mdash; 'decoupled RoPE'.",
  math: "c&#7511; = W&#7481;&#7472;&#7620; h&#7511;  <span class='mut'>(cache this only, d&#8342;&#7515; &asymp; 512)</span><br>k&#7511; = W&#7512;&#7472; c&#7511; ; v&#7511; = W&#7512;&#7620; c&#7511;  <span class='mut'>(reconstruct on the fly)</span><br>cache/token = 2 &middot; L &middot; (d&#8342;&#7515; + d&#7523;&#7506;&#7510;&#7497;) &middot; bytes",
  pros: [
    "<strong>93.3% KV-cache reduction vs the MHA baseline</strong>, with quality DeepSeek reports as <em>better</em> than MHA, not merely comparable. It is close to the only entry here that improves both axes at once.",
    "Compression is learned, so the model decides what to keep &mdash; unlike GQA's blunt head-averaging.",
    "Weight absorption means the decode path never expands the latent, so it is bandwidth-efficient in practice, not just on paper.",
    "5.76&times; generation throughput vs DeepSeek 67B. Widely regarded as the leading candidate for 'what replaces GQA', and Kimi Linear uses it as its full-attention layer."
  ],
  cons: [
    "<strong>Genuinely complicated.</strong> Decoupled RoPE exists purely because RoPE and weight absorption are incompatible, and it makes the head layout, the kernels and the KV layout all more intricate. Bugs live here.",
    "Weight absorption increases per-step compute &mdash; it trades FLOPs for bandwidth, which is the right trade on current hardware and might not be on future hardware.",
    "You cannot convert an existing GQA checkpoint cheaply the way GQA converts from MHA; there is meaningful conversion work.",
    "Kernel and framework support took over a year to mature, and non-DeepSeek reproductions of the quality claim are thinner than you would like.",
    "Still O(N) in the cache. A much smaller constant is not a different asymptotic."
  ],
  pick: "Large-scale models where serving cost dominates and you control the whole stack. It is the strongest pure-KV answer available, and if you are choosing a cache strategy in 2026 for a frontier-scale model, this is the one to beat.",
  viz: "kvcalc"
},
{
  id: "pdeltanet", name: "Parallel DeltaNet (chunkwise WY training)", abbr: "Parallel DeltaNet",
  date: "2024-06-10", dateLabel: "10 Jun 2024",
  who: "Yang, Wang, Zhang, Kim, Cui, Kim",
  src: [{label:"arXiv:2406.06484 v1", url:"https://arxiv.org/abs/2406.06484"}],
  family: "linear", era: 5, covered: true, bill: "compute",
  headline: "The delta rule sat unused for three years because it was sequential. This made it parallel.",
  problem: "The 2021 delta rule needs S&#7511;&#8331;&#8321; to compute S&#7511;. That is a strict sequential dependency, so training could not use the GPU &mdash; making it useless at LLM scale regardless of how good the idea was.",
  idea: "Recognise the delta update as a product of generalised Householder transformations, and use the WY representation from numerical linear algebra to express a whole chunk of updates as a small number of matmuls. Sequential over chunks, fully parallel within them.",
  math: "S&#7511; = S&#7511;&#8331;&#8321;(I &minus; &beta;&#7511;k&#7511;k&#7511;&#7488;) + &beta;&#7511;v&#7511;k&#7511;&#7488;<br>&rArr; chunkwise WY form &rArr; matmul-bound",
  pros: [
    "Turned a theoretically-nice-only mechanism into a trainable one. Without this, Gated DeltaNet and Kimi Linear do not exist.",
    "Scaled to 1.3B and beat Mamba and GLA on language modelling and, notably, on associative recall.",
    "The Householder framing is genuinely clarifying &mdash; it shows the delta rule is a <em>rotation-and-rescale</em> of memory, not just a subtraction."
  ],
  cons: [
    "Still chunk-sequential, so it never reaches full-attention training throughput.",
    "The kernel is intricate and hardware-specific; you are dependent on a small number of maintained implementations.",
    "It solves the training bottleneck, not the capacity bottleneck. The state is still d&times;d."
  ],
  pick: "You do not choose this directly &mdash; you get it whenever you use DeltaNet, Gated DeltaNet or KDA. It is on the timeline because it is the moment the delta rule became practical, and dates matter."
},
{
  id: "gsa", name: "Gated Slot Attention", abbr: "GSA",
  date: "2024-09-11", dateLabel: "11 Sep 2024",
  who: "Zhang, Yang, Zhu, Qin, Sun, Zhang et al.",
  src: [{label:"arXiv:2409.07146 v1", url:"https://arxiv.org/abs/2409.07146"}],
  family: "linear", era: 5, covered: true, bill: "both",
  headline: "A bounded-memory linear attention with an explicit, softmax-normalised set of slots. Cheap to convert a trained transformer into.",
  problem: "Linear attention's state is an opaque d&times;d blob with no normalisation and no forgetting. Can you keep O(1) memory but restore the useful softmax properties inside a bounded slot set?",
  idea: "Keep m memory slots. Reading and writing both go through softmax over slots (so you get competition and normalisation back, just over m entries instead of N), and each slot has a learned gate so it can decay &mdash; bounded memory with forgetting. It is equivalent to a two-pass gated linear attention, which makes it efficient to implement.",
  math: "bounded m slots;  softmax over slots (not over N);  per-slot forget gate &alpha;&#7511;",
  pros: [
    "Restores normalisation and competition without paying O(N) &mdash; the competition is over m slots.",
    "<strong>Cheap 'T2R' finetuning</strong>: convert a pretrained transformer to GSA on a modest budget, rather than pretraining from scratch. Practically important.",
    "Strong in-context recall for a linear model, which is exactly where linear models usually lose.",
    "The instructor's V4 used it as the periodic global layer in a DDDG schedule &mdash; a real deployment data point, not just a paper."
  ],
  cons: [
    "m slots is still a fixed capacity ceiling. It is a better-organised bound, not the absence of one.",
    "Softmax over slots reintroduces a nonlinearity in the recurrence, so the kernels are more involved than plain gated linear attention.",
    "Less adopted than Gated DeltaNet; thinner tooling and fewer independent replications.",
    "Another hyperparameter (m) with no principled way to set it."
  ],
  pick: "When you want to convert an existing transformer to a mostly-linear model on a budget, or as the recall-heavy layer type in a hybrid schedule."
},
{
  id: "diff", name: "Differential Transformer", abbr: "DIFF Transformer",
  date: "2024-10-07", dateLabel: "7 Oct 2024",
  who: "Ye, Dong, Xia, Sun, Zhu, Huang, Wei (Microsoft)",
  src: [{label:"arXiv:2410.05258 v1", url:"https://arxiv.org/abs/2410.05258"}],
  family: "exact", era: 5, covered: false, bill: "quality",
  headline: "Compute two attention maps and subtract them, like a differential amplifier cancelling common-mode noise.",
  problem: "Softmax attention allocates non-trivial mass to irrelevant context &mdash; the same structural pressure that produces attention sinks. That noise floor is what causes 'lost in the middle' and hallucination from distractors.",
  idea: "Split Q and K in two, compute two separate softmax maps, and take A&#8321; &minus; &lambda;A&#8322; with a learned &lambda;. Shared noise cancels; the signal, which differs between the two maps, survives. Directly analogous to differential signalling in electronics.",
  math: "DiffAttn = ( softmax(Q&#8321;K&#8321;&#7488;/&radic;d) &minus; &lambda;&middot;softmax(Q&#8322;K&#8322;&#7488;/&radic;d) ) V",
  pros: [
    "Measurably sparser, cleaner attention maps and better long-context retrieval, with markedly less sensitivity to prompt-order and distractors.",
    "Reaches comparable quality with roughly 65% of the parameters or training tokens, per the paper.",
    "Reduces activation outliers, which makes quantisation easier &mdash; a real deployment benefit.",
    "Composes with FlashAttention and with everything else on this page; it is a change to the head, not to the pattern."
  ],
  cons: [
    "<strong>Roughly 2&times; the attention compute</strong> for the same context length. It attacks the quality bill, not the cost bill &mdash; and this page is mostly about the cost bill.",
    "The subtraction can produce negative effective weights, which breaks the 'attention is a convex combination' intuition and some interpretability tooling.",
    "&lambda; needs a careful initialisation schedule; naive training is unstable.",
    "Not widely adopted in shipped frontier models, so the scaling evidence is largely from one group."
  ],
  pick: "When retrieval accuracy in long, distractor-heavy context matters more than throughput &mdash; RAG-heavy and agentic pipelines. Not if you are already compute-bound."
},
{
  id: "gdn", name: "Gated DeltaNet", abbr: "Gated DeltaNet",
  date: "2024-12-09", dateLabel: "9 Dec 2024",
  who: "Yang, Kautz, Hatamizadeh (NVIDIA)",
  src: [{label:"arXiv:2412.06464 v1", url:"https://arxiv.org/abs/2412.06464"}],
  family: "linear", era: 5, covered: true, bill: "both", anchor: true,
  headline: "Delta rule gives you precise overwriting. Gating gives you forgetting. Together they finally make linear attention shippable.",
  problem: "Two failure modes remained, and each mechanism only fixed one. Mamba2's gating can decay the whole state but writes bluntly &mdash; it cannot target a specific key. DeltaNet writes precisely but never decays &mdash; unreferenced keys occupy capacity forever, so after a topic change the state is full of stale associations.",
  idea: "Do both in one update. Multiply the state by a scalar forget gate &alpha;&#7511; (global decay: 'the conversation moved on') and apply the delta rule's targeted correction (precise edit: 'this key's value is now 55, not 95'). One gate for erasing broadly, one rule for writing exactly.",
  math: "S&#7511; = &alpha;&#7511; S&#7511;&#8331;&#8321; (I &minus; &beta;&#7511; k&#7511;k&#7511;&#7488;) + &beta;&#7511; v&#7511;k&#7511;&#7488;<br><span class='mut'>&alpha; = forget everything a little &nbsp;|&nbsp; &beta; = rewrite one thing exactly</span>",
  pros: [
    "Beats Mamba2 and DeltaNet on language modelling, common-sense reasoning, in-context retrieval and length extrapolation &mdash; it dominates both parents rather than trading against them.",
    "Still O(1) state and O(N) total. The capability gain is free asymptotically.",
    "Trains efficiently via a delta-rule-aware extension of the chunkwise parallel algorithm.",
    "<strong>This is the one that actually shipped.</strong> Qwen3-Next (Sep 2025) and Qwen3.5 use it as the majority layer type; Kimi Delta Attention is a refinement of it. Linear attention stopped being a research curiosity here.",
    "The paper's own hybrids (interleaving with sliding-window and full attention) established the layer-schedule pattern that is now standard."
  ],
  cons: [
    "<strong>The fixed-capacity ceiling is untouched and unbeatable.</strong> A d&times;d state cannot hold N&gg;d exact facts. No amount of gating or overwriting changes an information-theoretic bound &mdash; which is precisely why every model that ships it also keeps full-attention layers.",
    "A scalar forget gate decays <em>everything</em> uniformly. It cannot forget one topic while keeping another, which is why KDA moved to per-channel decay ten months later.",
    "Two extra learned scalars per token per head, and training is sensitive to their parameterisation.",
    "Needs specialised chunkwise kernels; you are dependent on a small number of maintained implementations.",
    "Hybrid ratio is empirical. Nobody has a theory for how many full-attention layers you need &mdash; reported answers cluster around 1 in 4, but that is measurement, not derivation."
  ],
  pick: "The default linear component for anything you build today. Use it as roughly three quarters of the layers with full attention or MLA for the rest. Do not use it alone unless you genuinely never need exact long-range recall.",
  viz: "delta"
},
/* ============================ ERA 5 ============================ */
{
  id: "nsa", name: "Native Sparse Attention", abbr: "NSA",
  date: "2025-02-16", dateLabel: "16 Feb 2025",
  who: "Yuan, Gao, Dai, Luo, Zhao, Zhang, Wu et al. (DeepSeek)",
  src: [{label:"arXiv:2502.11089 v1", url:"https://arxiv.org/abs/2502.11089"}],
  family: "sparse", era: 5, covered: true, bill: "both", anchor: true,
  headline: "Sparse attention that is trained from scratch and hardware-aligned, instead of bolted onto a dense model at inference.",
  problem: "Every sparse method before this had two flaws. (1) It was applied post-hoc to a model trained dense, so the model never learned to use the sparsity &mdash; you were degrading it and hoping. (2) The theoretical FLOP savings did not translate into wall-clock savings, because irregular gathers destroy GPU efficiency.",
  idea: "Three branches per query, combined by a learned gate. <strong>Compression</strong>: pool blocks of past tokens into coarse summaries so nothing is invisible. <strong>Selection</strong>: use those compression scores to pick the top-n blocks and attend to them at full resolution &mdash; note the compression branch <em>is</em> the cheap indexer that makes top-k affordable, closing the circularity that killed 2019-era top-k. <strong>Window</strong>: a local window for fluency. All three are blockwise, so memory access is coalesced, and the whole thing is differentiable end-to-end so you pretrain with it.",
  math: "o = g&#7580;&middot;Attn(q, K&#7580;&#7580;&#7504;&#7477;) + g&#8347;&middot;Attn(q, K&#8347;&#7529;&#8343;) + g&#7595;&middot;Attn(q, K&#7615;&#8339;&#7504;)<br>blockwise &rArr; coalesced memory access",
  pros: [
    "<strong>Trained natively</strong>, so the model learns to exploit the sparsity rather than being damaged by it. It matched or beat full attention on general benchmarks &mdash; sparse stopped being a pure trade.",
    "Real measured speedups: up to 11.6&times; decode, 9.0&times; forward, 6.0&times; backward at 64K. It speeds up <em>training</em>, which almost nothing else in the sparse family does.",
    "The three-branch design is principled: coarse global coverage, fine selected detail, guaranteed local. Nothing is structurally unreachable.",
    "Hardware-alignment is a first-class design constraint, not an afterthought. This is the paper that made sparse attention practical rather than theoretical."
  ],
  cons: [
    "<strong>You must pretrain with it.</strong> You cannot take an existing dense checkpoint and switch it on, which is a very large commitment.",
    "Block granularity means selection is coarse &mdash; you get a block of 64 tokens, not the one token you wanted.",
    "Meaningfully more complex than dense attention: three branches, a gate, custom kernels, and a block-selection path that must stay consistent between train and inference.",
    "Several hyperparameters (block size, compression stride, number of selected blocks) with limited public ablation outside the paper.",
    "Still not O(1) in memory &mdash; the compressed representation grows with N, just much more slowly."
  ],
  pick: "If you are pretraining a long-context model from scratch and can commit to the architecture. Its lineage (DSA in V3.2, CSA in V4) is now DeepSeek's production answer.",
  viz: "masks"
},
{
  id: "moba", name: "Mixture of Block Attention", abbr: "MoBA",
  date: "2025-02-18", dateLabel: "18 Feb 2025",
  who: "Lu, Jiang, Chen, Sun et al. (Moonshot AI)",
  src: [{label:"arXiv:2502.13189 v1", url:"https://arxiv.org/abs/2502.13189"}],
  dateNote: "Two days after NSA. Two labs converged on 'learned block selection, trained natively' independently and simultaneously &mdash; a good sign the idea was correct rather than lucky.",
  family: "sparse", era: 5, covered: false, bill: "both",
  headline: "Apply the MoE routing idea to attention: each query routes to its top-k blocks of keys.",
  problem: "Same as NSA. The interesting difference is the framing &mdash; if experts can be routed, so can context blocks.",
  idea: "Partition the KV sequence into blocks, compute a cheap gate score per (query, block) from block means, and let each query attend to its top-k blocks plus its own. Critically, MoBA is designed to switch seamlessly between sparse and full attention, so you can pretrain full and fine-tune sparse.",
  math: "gate(q, block b) = q&#7488; mean(K&#7495;);  attend to top-k blocks",
  pros: [
    "<strong>Switchable</strong> &mdash; unlike NSA, you can transition an existing full-attention model into MoBA, which drastically lowers the adoption cost.",
    "Conceptually simple: it is MoE routing, which everyone already understands and has tooling for.",
    "6.5&times; speedup at 1M tokens, deployed in Kimi's production stack.",
    "Parameter-free gating (block means), so no extra learned indexer to train."
  ],
  cons: [
    "Mean-pooled block keys are a crude summary; a block containing one highly relevant token among 511 irrelevant ones may not score highly.",
    "Fixed k per query is the same rigidity that has dogged top-k since 2019.",
    "Less architecturally thorough than NSA &mdash; no compression branch, so there is no guaranteed coarse view of unselected context.",
    "Block boundary effects: a relevant span straddling two blocks can be split and under-scored."
  ],
  pick: "When you want most of NSA's benefit but cannot commit to pretraining from scratch. The switchability is the real selling point."
},
{
  id: "qwen3next", name: "Layer-schedule hybrids (linear + full attention interleaved)", abbr: "Hybrid schedules",
  date: "2025-09-11", dateLabel: "11 Sep 2025",
  who: "Qwen team (Qwen3-Next); pattern also in Jamba, Samba, MiniMax-01, Nemotron",
  src: [{label:"vLLM day-0 support post, 11 Sep 2025", url:"https://vllm.ai/blog/2025-09-11-qwen3-next"},
        {label:"hybrid design established in Gated DeltaNet, arXiv:2412.06464", url:"https://arxiv.org/abs/2412.06464"}],
  dateNote: "Layer-level hybrids are older (Jamba, Mar 2024; Samba, Jun 2024). Qwen3-Next is dated here as the point at which a mostly-linear hybrid became a mainstream, widely-served open-weight model. The exact linear:full layer ratio Qwen uses is widely quoted as 3:1, but I could not confirm it from a primary source, so I am not asserting it. Kimi Linear's 3:1 ratio <em>is</em> stated in its abstract and is cited on that entry instead.",
  family: "linear", era: 5, covered: true, bill: "both", anchor: true,
  headline: "Stop asking which attention is best. Use cheap linear layers for most of the stack and a few exact layers where recall must be perfect.",
  problem: "Linear attention is cheap and forgets. Full attention is exact and expensive. Every paper before this treated it as a choice about the whole model, when it is actually a choice per layer.",
  idea: "Interleave. Most layers use Gated DeltaNet (O(1) state, cheap, good at local and compositional structure); a minority use full or gated attention (exact, expensive, good at long-range lookup). The instructor's V4 did exactly this with a DDDG schedule &mdash; three DeltaNet layers, one GSA layer &mdash; and reports roughly 40% of the FLOPs of a comparable dense stack.",
  math: "[D D D G] &times; n   <span class='mut'>(3 linear : 1 global, repeated)</span><br>KV cache scales with the <em>global</em> layers only",
  pros: [
    "<strong>The KV cache is now proportional to the number of full-attention layers, not the number of layers.</strong> A 1-in-4 schedule is a 4&times; cut on top of whatever GQA/MLA already bought you.",
    "You keep exact long-range retrieval, because some layers really can do it. This is the specific failure that pure-linear models never solved.",
    "There is a defensible reason for the ordering: early layers do local, fine-grained composition (cheap linear is fine); long-range relations are only meaningful once fine detail is resolved, so the global layers earn their cost higher up.",
    "It is empirically the 2026 consensus &mdash; Qwen3-Next/3.5, Kimi Linear, MiniMax, Nemotron and DeepSeek all ship some version of it.",
    "Composable with everything: your global layers can be MLA, your linear layers can be KDA, your sparse layers can be DSA."
  ],
  cons: [
    "<strong>The ratio is pure empiricism.</strong> Nobody can derive how many global layers you need; everyone measures. That means it may not transfer across scales or data mixtures, and you will re-run the ablation.",
    "Two kernel paths, two memory layouts, two sets of numerical quirks. Inference engines had to be substantially rewritten to support it.",
    "Cache management is heterogeneous &mdash; some layers have a growing KV cache, some have a fixed state. Paging, eviction and prefix-sharing all get harder.",
    "Debugging is worse: a quality regression could be the schedule, the linear layers, the global layers, or the interaction.",
    "It is an architecture, not a mechanism &mdash; you cannot bolt it onto a trained dense model."
  ],
  pick: "This is the default shape of a new long-context model in 2026. If you are designing an architecture today and you are not interleaving, you should have a reason.",
  viz: "schedule"
},
{
  id: "dsa", name: "DeepSeek Sparse Attention (lightning indexer)", abbr: "DSA",
  date: "2025-09-29", dateLabel: "29 Sep 2025",
  who: "DeepSeek-AI (DeepSeek-V3.2-Exp)",
  src: [{label:"DeepSeek-V3.2-Exp release, 29 Sep 2025", url:"https://github.com/deepseek-ai/DeepSeek-V3.2-Exp"},
        {label:"vLLM day-0 post, 29 Sep 2025", url:"https://vllm.ai/blog/2025-09-29-deepseek-v3-2"},
        {label:"DeepSeek-V3.2 paper, arXiv:2512.02556 (2 Dec 2025)", url:"https://arxiv.org/abs/2512.02556"}],
  family: "sparse", era: 5, covered: true, bill: "both", anchor: true,
  headline: "The 2019 top-k dream, finally affordable: a tiny cheap indexer picks the tokens, so you never build the full score matrix.",
  problem: "Top-k selection has always been circular &mdash; to find the top k scores you compute all N scores, which is the thing you were trying to avoid. NSA solved it with block compression. Can you do it at <em>token</em> granularity instead of block granularity?",
  idea: "Run a very small, low-precision 'lightning indexer' &mdash; few heads, FP8 &mdash; whose only job is to produce approximate relevance scores cheaply. Take the top-k tokens by that index (k on the order of 2048), then run the real MLA attention on only those. Selection is fine-grained (per token, not per block) because the indexer is cheap enough to run over everything.",
  math: "I(q, k&#11002;) = &sum;&#7530; w&#7530; &middot; ReLU(q&#7530;&#7488;k&#11002;&#7530;)  <span class='mut'>(FP8, few heads)</span><br>attend over top-k by I",
  pros: [
    "Fine-grained token-level selection, which is strictly more precise than block selection &mdash; no relevant token is lost inside an irrelevant block.",
    "O(N&middot;k) instead of O(N&sup2;) with a small constant, and it showed up as a genuine ~50% API price cut for long context. Rare to see an architecture change appear directly on a price list.",
    "Built on top of MLA, so the cache-size win and the compute win compose.",
    "Introduced via continued training from V3.1 rather than a from-scratch pretrain, which is a much cheaper adoption path than NSA."
  ],
  cons: [
    "<strong>The indexer is a learned approximation and it can be wrong.</strong> If it fails to surface the one token that mattered, the main attention never sees it &mdash; and you get a confident wrong answer with no signal that anything was skipped.",
    "A fixed k is a guess that is too small for genuinely diffuse tasks (summarising a whole book) and wasteful for sharp ones.",
    "You are now training and maintaining two attention-ish modules, and the indexer needs its own distillation phase against the dense model's attention.",
    "Released as 'Exp' &mdash; explicitly experimental, with DeepSeek noting mild regressions on some reasoning tasks.",
    "The instructor's caution stands: too sparse and you lose the context. k is where that risk lives."
  ],
  pick: "Long-context serving where cost per token is the binding constraint and you can tolerate occasional retrieval misses. Its successor (CSA in V4) is where this line is going.",
  viz: "masks"
},
{
  id: "kda", name: "Kimi Delta Attention (per-channel gating)", abbr: "KDA",
  date: "2025-10-30", dateLabel: "30 Oct 2025",
  who: "Moonshot AI (Kimi Linear)",
  src: [{label:"arXiv:2510.26692 v1", url:"https://arxiv.org/abs/2510.26692"}],
  family: "linear", era: 5, covered: false, bill: "both",
  headline: "Gated DeltaNet's forget gate is one scalar for the whole state. Give every channel its own.",
  problem: "A scalar decay &alpha;&#7511; forgets everything uniformly. But 'the user changed topic' and 'this specific fact is stale' are different events, and a single number cannot express both.",
  idea: "Replace the scalar gate with a diagonal per-channel decay, so the state can forget along some feature directions while preserving others. Implemented as a constrained Diagonal-Plus-Low-Rank transition that stays cheap. Then hybridise 3 KDA layers to 1 MLA layer.",
  math: "S&#7511; = Diag(&alpha;&#7511;) S&#7511;&#8331;&#8321; (I &minus; &beta;&#7511;k&#7511;k&#7511;&#7488;) + &beta;&#7511;v&#7511;k&#7511;&#7488;<br>3 KDA : 1 MLA layers",
  pros: [
    "First hybrid linear architecture reported to beat full attention under fair comparison across short context, long context <em>and</em> RL scaling &mdash; not just at equal FLOPs.",
    "Up to 75% KV cache reduction and roughly 6&times; decode throughput at 1M tokens.",
    "Per-channel decay is a clean, well-motivated generalisation, with a real efficiency argument behind the DPLR restriction.",
    "Kernels and vLLM integration released with the paper, which is not the norm."
  ],
  cons: [
    "More gate parameters and a more delicate parameterisation; the DPLR constraint exists to keep it tractable, which means it is not the fully general form.",
    "The 3:1 ratio is again empirical.",
    "One lab, one model family so far. The 'beats full attention' claim needs independent replication before you build on it.",
    "Still a fixed-capacity state in the linear layers &mdash; the MLA layers are still doing the exact recall."
  ],
  pick: "If you are building a hybrid today and want the strongest published linear block. Watch for independent replication before betting a pretraining run on it."
},
{
  id: "drope", name: "DroPE (drop the positional embeddings)", abbr: "DroPE",
  date: "2025-12-13", dateLabel: "13 Dec 2025",
  who: "Gelberg, Eguchi, Akiba, Cetin (Sakana AI)",
  src: [{label:"arXiv:2512.12167 v1", url:"https://arxiv.org/abs/2512.12167"},
        {label:"Sakana AI project page", url:"https://sakana.ai/drope/"},
        {label:"precursor: NoPE, arXiv:2305.19466 (31 May 2023)", url:"https://arxiv.org/abs/2305.19466"}],
  dateNote: "The instructor called this 'drop rope' in the session. The published name is DroPE and it is Sakana AI, v1 submitted 13 Dec 2025. Related but distinct earlier work: NoPE (arXiv:2305.19466, 31 May 2023) showed decoder-only models trained <em>without</em> any positional encoding can still learn position implicitly from the causal mask. DroPE's contribution is that you can get there from a <em>pretrained</em> RoPE model with a short recalibration, which is the practically useful version.",
  family: "pos", era: 5, covered: true, bill: "length", anchor: true,
  headline: "Use RoPE as training-time scaffolding, then take it away. The model already knows where it is.",
  problem: "Every context-extension method since 2023 &mdash; PI, NTK, YaRN, LongRoPE &mdash; is surgery on RoPE's frequencies. They all treat the symptom. The disease is that the model has become <em>dependent</em> on a positional signal that is only in-distribution up to the training length.",
  idea: "Invert the assumption. Positional embeddings help convergence during pretraining, but a causal decoder can infer order from the mask alone (this is NoPE's finding). So: pretrain normally with RoPE, then remove the positional embeddings entirely and recalibrate briefly at the <em>original</em> context length &mdash; under 1% of the pretraining budget. Perplexity spikes on removal and then recovers under annealing. What you get back is a model with no length-dependent signal to go out of distribution, so it generalises to lengths it never saw, zero-shot.",
  math: "1. pretrain with RoPE at L<br>2. remove PE, recalibrate at L  <span class='mut'>(&lt;1% of budget)</span><br>3. evaluate at &gg;L, zero-shot",
  pros: [
    "<strong>No long-context fine-tuning at all.</strong> Every other method in the position family requires training on long sequences, which is the expensive part. This one does not.",
    "Recalibration is under 1% of the pretraining budget, at the short context length &mdash; so it is cheap in both senses.",
    "Reported to outperform RoPE-scaling methods on LongBench and RULER, which are retrieval-style benchmarks rather than perplexity &mdash; the harder test, and the one ALiBi quietly failed.",
    "It removes a component rather than adding one. The result is a <em>simpler</em> model, which is unusual and worth something.",
    "Independently corroborates the instructor's V4 result: he reported training with RoPE, dropping it, taking a loss spike, and recovering with annealing to reach a large extension factor. Same phenomenon, arrived at independently."
  ],
  cons: [
    "<strong>Very new.</strong> December 2025, one lab. There is little independent replication and no frontier model has shipped it. Treat the numbers as promising, not settled.",
    "The recalibration is only cheap relative to <em>pretraining</em>. You still need the pretrained model and a training pipeline &mdash; this is not something you apply to a downloaded checkpoint on a laptop.",
    "The loss spike on removal is real and needs careful annealing. That is a training-stability risk on a large run.",
    "NoPE-style implicit position is known to be weaker at fine-grained short-range order than explicit RoPE in some settings; the recalibration is supposed to fix that, and mostly does, but 'mostly' is doing work.",
    "It presumes a causal decoder. There is no obvious bidirectional-encoder version, because the mask is what carries the position signal.",
    "Claimed extension factors depend on the base model and the evaluation. Do not assume the headline number transfers to your setup without measuring."
  ],
  pick: "If you control pretraining and long context is a requirement, this is the most interesting bet on the board right now &mdash; cheaper than YaRN's fine-tuning, and it attacks the cause rather than the symptom. If you only have a downloaded checkpoint and a small budget, YaRN is still the safe answer.",
  viz: "rope"
},
{
  id: "csa", name: "Compressed Sparse Attention + Heavily Compressed Attention", abbr: "CSA + HCA",
  date: "2026-04-26", dateLabel: "26 Apr 2026",
  who: "DeepSeek-AI (DeepSeek-V4)",
  src: [{label:"arXiv:2606.19348 v1, submitted 26 Apr 2026", url:"https://arxiv.org/abs/2606.19348"},
        {label:"DeepSeek-V4 preview release, 24 Apr 2026", url:"https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro"}],
  dateNote: "Careful here. The arXiv identifier is 2606.* (a June 2026 announcement cycle) but the v1 submission line reads 26 Apr 2026, and the model preview was released 24 Apr 2026. At least one aggregator reports 19 Jun 2026, which is the announcement date, not the submission date. I am using the v1 submission date, consistently with the rest of this page, and flagging the discrepancy rather than hiding it.",
  family: "sparse", era: 5, covered: true, bill: "both", anchor: true,
  headline: "The newest answer, and it is two answers: compress the cache along time, then be sparse over what remains &mdash; in different layers, at different rates.",
  problem: "At 1M tokens, even DSA is not enough. Sparse selection cuts the compute bill but every selected token still needs its KV bytes resident, so the memory bill is barely touched.",
  idea: "Attack the sequence dimension itself, in two flavours, and put them in different layers. <strong>CSA</strong> consolidates every m tokens of KV cache into one entry, then runs DSA-style top-k selection over those consolidated entries &mdash; sparse over compressed. <strong>HCA</strong> compresses far more aggressively but keeps attention <em>dense</em> over the survivors &mdash; on the reasoning that once you have compressed heavily, there is little left worth skipping. The two cut different bills, so the stack uses both.",
  math: "CSA: compress m&rarr;1, then top-k  <span class='mut'>(compress, then select)</span><br>HCA: compress hard, keep all  <span class='mut'>(dense over survivors)</span><br>V4-Pro @1M: 27% of V3.2 FLOPs, 10% of V3.2 KV cache",
  pros: [
    "<strong>Hits both bills at once.</strong> 27% of the per-token FLOPs and 10% of the KV cache of V3.2 at 1M context. Sparse-only methods could not do the second number.",
    "The CSA/HCA split is the layer-schedule idea applied to the sparsity axis &mdash; different compression regimes at different depths, rather than one global setting. That is a genuinely new degree of freedom.",
    "Compressing along the sequence dimension is orthogonal to compressing along the feature dimension (MLA), so they multiply rather than compete.",
    "1M context shipped in open weights under MIT, which keeps the pressure on everyone else."
  ],
  cons: [
    "<strong>Compression is lossy and permanent.</strong> Once m tokens become one entry, the individual tokens are unrecoverable &mdash; unlike sparse selection, where a skipped token still exists in the cache and could have been chosen. This is a strictly harder failure mode.",
    "The compression rate m is a train-time commitment. Get it wrong and you cannot fix it at inference.",
    "Very new, one lab, and the paper is explicitly a preview. Independent evaluation on hard long-range retrieval is thin.",
    "Compounding approximation: MLA compresses features, CSA compresses time, top-k skips entries. Each is defensible alone; the interaction of all three at 1M tokens is not well characterised.",
    "Substantial architectural complexity. This is not something a small team reimplements over a weekend."
  ],
  pick: "The frontier answer for million-token serving if you are DeepSeek-scale. For everyone else it is the direction to watch: compression along the sequence axis is the axis that had not been seriously attacked until now.",
  viz: "masks"
}
];

/* ------------------------------------------------------------------
   What the date order shows that a list does not.
   Every interval quoted here is measured from the dates above by
   tools/gaps.py, not estimated.
   ------------------------------------------------------------------ */
const FINDINGS = [
  {
    t: "A mechanism can be four years early and simply be ignored",
    metric: "MQA &rarr; GQA: 1,293 days",
    body: "MQA lands 6 Nov 2019, in the middle of the sparse-attention gold rush &mdash; Sparse Transformer, Reformer, Longformer, linear attention, BigBird and Performer all appear within fourteen months either side of it. Every one of those attacks the <em>compute</em> bill. MQA attacks the <em>memory</em> bill, and nothing happens for three and a half years until GQA in May 2023.",
    hidden: "Grouped by family, MQA and GQA sit next to each other and read as a tidy progression. In date order there is a four-year hole between them, and the hole is the finding: nobody optimises a bill they are not yet paying. In 2019 these models were not being served to millions of people, so the KV cache was not anybody's problem yet.",
    lesson: "The bill you can see is the bill you are currently paying. Ideas that answer the <em>next</em> bottleneck get published and then sit."
  },
  {
    t: "The 2020 approximation wave dies at a single point, and an engineer killed it",
    metric: "five approximations in nine months, then a two-year silence",
    body: "Reformer (13 Jan 2020), Longformer (10 Apr), linear attention (29 Jun), BigBird (28 Jul), Performer (30 Sep). Then the family goes quiet. FlashAttention arrives 27 May 2022 and makes <em>exact</em> dense attention faster than most of the approximations that were trying to avoid it.",
    hidden: "In a family-grouped list FlashAttention is filed under 'systems' and looks like it belongs to a different conversation. In date order it sits exactly at the extinction boundary of the approximation family. They were not beaten by a better approximation. They were beaten by someone who noticed attention was memory-bandwidth-bound, not compute-bound &mdash; which means the whole wave had been optimising the wrong resource.",
    lesson: "Before you approximate something, check which resource is actually scarce. Three years of clever math lost to one profiling insight."
  },
  {
    t: "The scramble for context length is measured in days, and it happened outside academia",
    metric: "PI &rarr; NTK-aware: 1 day. NTK &rarr; YaRN: 64 days.",
    body: "Position Interpolation is submitted 27 Jun 2023. NTK-aware scaled RoPE appears the next day &mdash; as a Reddit post, with no paper, no peer review and no ablations. YaRN follows nine weeks later and fixes both.",
    hidden: "A list files all three under 'RoPE scaling' and the ordering inside is arbitrary. The dates show a community scramble triggered by Llama-2 shipping a 4K window, not a research programme. It also shows that for several months the most widely deployed context-extension method in the world was a forum post &mdash; which is exactly why it is the one date on this page I could not pin to a day.",
    lesson: "Deployment pressure sets the pace, and the artefact that wins is not always the one with a paper."
  },
  {
    t: "Good ideas wait about three years for the hardware, and the lag is consistent",
    metric: "delta rule &rarr; parallel training: 1,204 days",
    body: "The delta rule for linear attention is published 22 Feb 2021 and then goes essentially unused until the chunkwise parallel algorithm arrives 10 Jun 2024. The mechanism was never the problem &mdash; it was sequentially dependent, so training could not use a GPU, which made it worthless at scale no matter how good it was.",
    hidden: "As a list entry, 'DeltaNet' is one bullet with one date, and which date you pick is a coin flip. Splitting it into the idea (2021) and the thing that made it runnable (2024) exposes a gap that is almost identical to MQA's, and for a related reason: not conceptual, but engineering.",
    lesson: "The distance between 'this works' and 'this runs on the hardware we have' is roughly three years, twice, independently. Budget for it."
  },
  {
    t: "The best ideas arrive twice, within days, from different labs",
    metric: "NSA &rarr; MoBA: 2 days. Learned positions &rarr; sinusoidal: 35 days.",
    body: "NSA (DeepSeek, 16 Feb 2025) and MoBA (Moonshot, 18 Feb 2025) independently propose the same thing: learned block selection, trained natively rather than bolted on at inference. Forty-eight hours apart. The same pattern appears at the very start of the timeline, with learned absolute positions and sinusoidal five weeks apart.",
    hidden: "A list shows two similar entries and invites you to ask which one was first, or which is better. The dates show that the question is wrong. When two labs with no contact ship the same idea in the same week, the idea was <em>due</em> &mdash; determined by the hardware and the context lengths people wanted, not by anyone's insight.",
    lesson: "Simultaneity is a signal that the constraint, not the researcher, is driving. It is also the best evidence that an idea is correct rather than lucky."
  },
  {
    t: "Position encoding is being deleted, monotonically, and you can see where it ends",
    metric: "2017 &rarr; 2025: parameters go L&times;d &rarr; 0 &rarr; 0 &rarr; none at all",
    body: "A learned table (2017, L<sub>max</sub>&middot;d parameters and a hard length ceiling) &rarr; sinusoidal (2017, no parameters, still added to the residual stream) &rarr; RoPE (2021, no parameters, moved out of the residual stream and into the score) &rarr; ALiBi (2021, just a bias on the logit) &rarr; DroPE (2025, remove it entirely after training and recalibrate).",
    hidden: "As a list this is a taxonomy of positional encoding methods and every entry looks like an alternative to the others. In date order every single step <em>removes</em> something, and none ever adds it back. That is not a taxonomy, it is a trend line with a visible endpoint.",
    lesson: "This is the one place on the timeline where you can extrapolate with real confidence, because the direction has never once reversed in eight years. The endpoint is zero."
  },
  {
    t: "The oscillation is a control loop, and the trigger is always a benchmark",
    metric: "exactness &rarr; length &rarr; memory &rarr; length &rarr; memory",
    body: "Every swing back toward memory is triggered by an evaluation exposing what the previous swing forgot. Long Range Arena and failed replications killed the 2020 approximations. Needle-in-a-haystack and RULER exposed ALiBi's extrapolation as increasing locality rather than increasing reach, and exposed pure-linear models' fixed-capacity ceiling.",
    hidden: "A list of mechanisms contains no benchmarks at all, so the cause of each turn is invisible. On a timeline the benchmark lands between the cheap mechanism and the correction, every time, and the pattern stops looking like fashion and starts looking like feedback.",
    lesson: "The swings are predictable from which benchmark lands next &mdash; which is what makes the question answerable at all. As of April 2026, aggressive sequence-axis compression has not yet met its adversarial benchmark."
  },
  {
    t: "Attacking both bills at once is a recent capability, not an old one",
    metric: "first headline claim on both meters: Apr 2026",
    body: "Era 2 attacks compute. Era 4 attacks memory. Almost nothing before 2025 bills both meters at once &mdash; sliding window is the rare early exception, and it pays for it in retrieval. DeepSeek-V4's CSA + HCA (26 Apr 2026) is the first to report both as headline results: 27% of the FLOPs and 10% of the KV cache of its predecessor.",
    hidden: "In a family-grouped list, compute methods and memory methods are separate sections, so you never notice that for six years essentially nobody managed both. The date order makes the alternation obvious, and makes the recent convergence look like the genuinely new thing it is.",
    lesson: "If you are choosing a mechanism today, the pre-2025 ones make you pick a meter. Only the newest generation lets you refuse the choice."
  }
];

const PREDICTIONS = [
  { t: "Compression along the sequence axis", body: "MLA compressed the <em>feature</em> axis and won. CSA is the first serious attack on the <em>time</em> axis, and that axis has far more redundancy in it. Expect learned, content-dependent compression rates rather than a fixed m &mdash; a boring paragraph should compress harder than a table of numbers, and right now nothing knows the difference." },
  { t: "The hybrid ratio becomes a theory instead of an ablation", body: "Every hybrid on this page picks its linear:global ratio by measurement. Somebody will derive how much exact-recall capacity a stack actually needs as a function of task and depth, and that paper will be cited for a decade." },
  { t: "Position keeps disappearing", body: "Learned table &rarr; fixed sinusoid &rarr; rotation inside the score &rarr; nothing at all. DroPE and NoPE both point the same way: the causal mask already carries order, and explicit position is scaffolding. The trend line runs to zero." },
  { t: "Softmax itself gets edited", body: "Attention sinks, softmax1 and gpt-oss's learned sink logit are all the same admission &mdash; forcing weights to sum to 1 was a mistake, because sometimes the right answer is 'nothing here is relevant'. Expect the fix to move into the normaliser rather than staying a cache hack." },
  { t: "Memory comes back a third time", body: "The pattern on this timeline is exactness &rarr; length &rarr; memory &rarr; length &rarr; memory. Each swing back to memory is triggered by a retrieval benchmark exposing the previous swing's amnesia. RULER did it to ALiBi and to pure linear models. The next hard benchmark will do it to aggressive compression." }
];

if (typeof module !== "undefined") module.exports = { MECHANISMS, FAMILIES, ERAS, FINDINGS, PREDICTIONS };
