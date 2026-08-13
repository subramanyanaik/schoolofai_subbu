# Part III — Rebuild from scratch

Everything needed to recreate this project on a different machine, from an
empty directory. This file is the build specification; `REPRODUCE.md` is the
command list for re-running code that already exists.

## III.1 Environment

```bash
pip install torch numpy tiktoken pytest matplotlib pyarrow
```

Optional, only for the corpus paths:

```bash
pip install datasets truststore
```

`truststore` matters only behind a TLS-intercepting corporate proxy. It makes
`huggingface_hub` validate against the OS certificate store instead of
certifi's bundle, and it does **not** disable verification. Without it you get
`CERTIFICATE_VERIFY_FAILED`. `data.py` calls it automatically when present.

Create `pytest.ini` so tests can import the package:

```ini
[pytest]
pythonpath = src
testpaths = tests
```

`tiktoken` downloads the GPT-2 encoding on first use, so the very first run
needs network access regardless of which corpus you pick.

## III.2 Corpus

```bash
python experiments/prepare_data.py --source fineweb --tokens 26000000
```

`--source fineweb` streams `HuggingFaceFW/fineweb-edu` sample-10BT — the corpus
the base paper used — and needs network access. `--source local_parquet`
expects `./.hfhome/ccnews/plain_text/train-00000-of-00005.parquet`; a bare path
is read as a UTF-8 text file split on blank lines.

A different corpus changes absolute bits-per-byte but not the relative
comparisons: every arm sees the identical token stream either way, which is
what the experiment tests. Say which corpus you used when reporting numbers.

Tokens are cached as uint32, stored **in length-sorted vocabulary order**
(`PackedVocab.inv_order` maps GPT-2 ids in, `.order` maps back). Validation
takes the first 4,000 documents; training skips them, so the two are disjoint.

## III.3 What to build, and in what order

Build test-first. `codec.py` and `kas_head.py` are where correctness *is* the
contribution.

### 1. `src/kronecker_v2/codebooks.py` — deterministic, zero-parameter, float64

- `sylvester_hadamard(n=256)` → `H[i,j] = (-1)**popcount(i & j)`.
- `byte_codebook(d_c, seed, method)` → `(256, d_c)`, unit-norm rows.
  `method="hadamard_bits"` selects Hadamard columns forced to include the 8
  Walsh bit-columns `{1,2,4,8,16,32,64,128}` **first**, remainder seeded.
  Those 8 guarantee all 256 rows stay distinct and enable an O(8) sign decode.
  Exclude column 0 (all-ones) from the candidate pool: it carries no
  information about the byte and raises coherence for free. Also provide
  `onehot` (= V1's byte code, d_c=256), `hadamard_random`, `rademacher`.
- `position_basis(L_max, kind="dft")` → orthogonal `(L_max, L_max)`. Real
  orthonormal Fourier with period **P = L_max**: column 0 is `1/sqrt(N)`; for
  `k = 1..(N-1)//2` a `sqrt(2/N)*cos(2*pi*k*p/N)` and a matching sine; if N is
  even append the Nyquist term `(-1)**p / sqrt(N)`. `kind="identity"` = V1.
  P = L_max is load-bearing — a longer period clusters positions `0..L-1` into
  a small arc of the circle and the system becomes ill-conditioned.
- `coherence_stats(G)` → max/mean off-diagonal `|cos|`, the Welch bound
  `sqrt((n-d)/(d(n-1)))`, and `rows_distinct`. This is the quantity E5 finds
  predicts language-model quality.

### 2. `src/kronecker_v2/codec.py`

`KroneckerCodec(d_c=64, L_max=64, seed=0, byte_method, pos_kind, variant="v2",
overflow="truncate")`. `variant="v1"` forces `d_c=256, onehot, identity,
truncate` — the published codec.

- `encode_positions(seqs)` → `(N, d_c, L_max)` float64,
  `M[c,p] = G[b_p,c] / sqrt(L_eff)` for `p < L_eff`, where
  `L_eff = min(L, L_max)` under truncation. Using `L_eff` rather than the true
  length is what keeps **every codec vector exactly unit norm**, which E4's
  whole diagnosis rests on.
  Vectorise it or it takes ~105 s on the GPT-2 vocab: concatenate all bytes,
  build `row = repeat(arange(N), lens)` and `pos = (arange(total) - starts) %
  L_max`, then scatter `out[row, pos] = G[all_bytes]` into an
  `(N, L_max, d_c)` buffer and transpose. Use `np.add.at` **only** when some
  length exceeds `L_max` (aliased slots must accumulate); plain assignment is
  far faster otherwise.
- `encode(seqs)` → `(N, D) = (M_pos @ Phi).reshape(N, D)`, `D = d_c * L_max`.
- `decode(K, method=None)` → `M_pos = M_freq @ Phi.T`. Phi is orthogonal, so
  this is exact and needs no pseudo-inverse. A column is "present" if its norm
  exceeds `1e-3`; length falls out of that. Bytes come from either a sign read
  (bit k is 1 iff coordinate k < 0, valid because the bit-columns come first)
  or a 256-way nearest neighbour. **Test that the two agree.**
  Vectorise the decode too — the per-sequence Python loop is the difference
  between E1 taking seconds and taking minutes.

### 3. `src/kronecker_v2/vocab.py`

- `gpt2_token_bytes()` — `tiktoken.get_encoding("gpt2")`,
  `decode_single_token_bytes(i)` per id, with a `b"\x00"` fallback for reserved
  ids so no occupancy row is ever empty.
- `PackedVocab(token_bytes, L_max)` — sort ascending by **true** byte length,
  stable. This single choice is what makes `{v : L_v > p}` a contiguous suffix
  for every p. Sorting by *true* rather than clipped length additionally makes
  `order`/`inv_order` independent of `L_max`, so one tokenized data file is
  valid across the whole E5 sweep. Expose `order`, `inv_order`, `lengths`
  (clipped), `true_lengths`, `max_len`, `total_bytes` (= B_V),
  `hi[p] = searchsorted(lengths, p, "right")`, `flat_bytes`/`offsets`,
  `inv_sqrt_len` (float64), `pos_mod`, `_sorted_bytes`.
- `build_large_vocab(target, base, texts, L_max, seed)` — harvest frequent
  whitespace word forms, disclose any synthetic filler; used by E6.

### 4. `src/kronecker_v2/kas_head.py`

`Occupancy(packed, dtype)` — a **parameter-free** `nn.Module`. Buffers only:
`cols = byte*L_max + (pos % L_max)`, `vals = 1/sqrt(L_v)` repeated per byte,
`offsets = concat([0], cumsum(lengths))[:-1]`.

```python
def matmul(self, dense):                      # (256*L_max, k) -> (V, k)
    w = dense.contiguous()
    return F.embedding_bag(self.cols, w, self.offsets, mode="sum",
                           per_sample_weights=self.vals.to(w.dtype))
```

`KASHead(d_model, codec, packed, tied_weight=None, dtype, impl="bag",
unigram_bias=False, occ=None)` → `W_out (d_model, D)` init `N(0, D**-0.5)`,
`beta (L_max + 1)`, optional `unigram (V)`.

```python
S   = (h @ W_out).view(-1, d_c, L_max)
A_T = einsum("vc,ncp->vpn", G, S).reshape(256*L_max, -1)   # TRANSPOSED: no copy
logits = occ.matmul(A_T).t() * beta[lengths]                # (+ unigram)
```

Size `beta` from **`L_max`, never from the observed max token length** — the
parameter audit catches that one, and it is the single number this project is
about.

Also ship `dense_codec_table` and `dense_logits` — the naive `S @ K.T` head
that KAS must reproduce bit-for-bit. Keep the sparse (CSR) and loop
implementations too, and test all three agree; they are the evidence that the
fast path is not cheating.

### 5. `src/kronecker_v2/model.py`

`KroneckerEmbedding` is the **dual** of the head. Never materialise a
`|V| x D` table (412 MB in fp16 at the GPT-2 vocab and D=4096):

```python
def table(self):                                            # -> (V, d_model)
    Wpos = einsum("pk,ckm->cpm", Phi, W_proj.view(d_c, L_max, -1))
    B    = einsum("vc,cpm->vpm", G, Wpos)
    return self.occ.matmul(B.reshape(256*L_max, -1))
```

Test it against `codec.encode(token_bytes) @ W_proj` directly — that identity
is what licenses the whole construction, and it is the input-side twin of E2.

The V1 arm additionally z-normalises the codec vector (base paper §3). Do it
in closed form rather than by encoding the vocabulary: `sum(K_v^2) = 1` because
every codec vector is unit norm, and `sum(K_v)` is itself an occupancy product
over `g_sum[b] * phi_row_sum[p]`. So mu and sigma are exact, not estimated.

`GPT(arm, vocab_size, packed, codec, d_model=384, n_layer=6, n_head=6,
block_size=256)`. Create **one** `Occupancy` and share it between both
pathways. Arms:

| arm | input | head |
|---|---|---|
| `a0_bpe_tied` | `nn.Embedding` | dense, tied to the embedding |
| `a1_v1_dense` | V1 codec (d_c=256, L_max=16) + z-norm | dense `nn.Linear` |
| `a2_kronf_kas` | Kronecker-F | KAS |
| `a3_kronf_kas_tied` | Kronecker-F | KAS, `W_out = W_proj` |
| `a4_kas_unigram` | Kronecker-F | KAS + one scalar per token |

### 6. `data.py`, `train.py`, `eval.py`, `experiments/e1…e8`, `make_figures.py`

See `REPRODUCE.md` for the run order.

## III.4 Ten traps — read before writing code

Each of these cost real debugging time.

1. **Build codebooks in float64.** Constructing them in float32 and *casting*
   to float64 silently caps end-to-end exactness at ~5e-8, so the fp64 gate
   fails for a reason that looks like a maths bug and isn't. Downcast at
   exactly one boundary (`KASHead.__init__`).
2. **cuSPARSE cannot mix an fp32 sparse operand with a bf16 dense one.** Under
   `autocast(bfloat16)`, `torch.sparse.mm` dies with
   `cusparseSpMM_bufferSize … not supported`. Index dtype is irrelevant. Use
   `F.embedding_bag`, which works under autocast and is a fused kernel.
3. **The naive gather OOMs.** `A[:, all_vocab_bytes, all_pos]` materialises
   `N x B_V` (~2e9 floats at N=8192). Never form it.
4. **The obvious loop is an order of magnitude slower than dense.** A
   length-sorted slice-add over `L_max` steps launches `L_max` tiny CUDA
   kernels and is latency-bound. Recognise the operation as a sparse matmul
   instead.
5. **Do not assign an `nn.Module` you only want for its buffers.**
   `self._occ = <KASHead>` makes `nn.Module` register that head's unused
   `W_out` as a parameter of the host arm. It gets no gradient and AdamW skips
   it, so training is unaffected and *only the reported parameter count is
   wrong* — which is the number this whole project is about. Hence the
   parameter-free `Occupancy`, and `test_no_phantom_parameters`.
6. **Size `beta` by `L_max`, not by the vocabulary's longest token.** Same
   failure mode as trap 5 and much easier to miss: the head's parameter count
   creeps with `|V|` and the zero-vocabulary-parameter claim quietly dies.
   `params_audit.py` exists to catch it.
7. **Overflow past `L_max` must truncate, not alias.** Aliasing was the
   original design intent and measurement killed it. Aliasing corrupts every
   position; truncation keeps a correct prefix.
8. **Compare token-matched, not time-matched.** Arms differ in throughput by
   up to ~2.8x, so a per-arm wall-clock cap silently compares them at
   different token counts. Run a calibration probe, pick one budget every arm
   can afford, hold it fixed. A wall-clock cap is an abort, not a stopping
   rule.
9. **Do not call `loss.item()` every step.** It forces a device sync, which
   serialises numpy batch preparation against GPU compute. It cost ~2.5x
   wall-clock here and *understates the reported tok/s*, which is a number the
   write-up quotes. Accumulate the loss on-device and read it only when
   logging.
10. **Windows stdout is cp1252.** Printing Devanagari/Telugu/emoji raises
    `UnicodeEncodeError`, and `pathlib.read_text()` on a file containing them
    raises `UnicodeDecodeError`. Use `sys.stdout.reconfigure(encoding="utf-8")`
    and always pass `encoding="utf-8"` to file I/O.

Plus three experiment-design traps:

- **Make the sweep script merge, not overwrite.** A timed-out sweep can rewrite
  its own results file with a single partial run, and the next invocation then
  merges that partial forward — completed runs gone. Key by `(d_c, L_max)`,
  skip configs already present, write after *every* run.
- **E6 must swap the input embedding *and* the head.** Swapping only the head
  leaves the input table at `|V|=50257` while remapped ids run to 1M — an
  out-of-bounds gather. Sharing one `Occupancy` between both pathways makes
  this structurally impossible to get wrong.
- **`D = d_c * L_max` cannot be varied in isolation**, so E5 must sweep both
  axes and name the confound rather than claim a controlled experiment.
