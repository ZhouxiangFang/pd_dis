# Method 2 — Attention-based Prompt Pruning

This module implements **Method 2: Attention-based pruning/eviction** of the
PD Disaggregation project.

The current implementation is an **approximate (input-side proxy) version**:
before the prefill stage, we score each prompt token with a lightweight
"attention-proxy" heuristic and keep only the top-k most important ones, which
directly reduces both the compute and the KV cache footprint that needs to be
transferred to the decode node. A strict version — evicting low-importance
tokens from the real KV cache using actual attention weights — requires
modifying the vLLM kernel and is left as future work. The public interface
already exposes a `method` field so a strict backend can be plugged in later
without touching the call sites.

---

## 1. Code Layout

```
methods/
├── __init__.py
└── attn_pruning/
    ├── __init__.py            # exports PruneConfig / PruneResult / prune_prompt
    ├── pruner.py              # core logic: tokenize, score, top-k selection
    ├── README.md              # this file
    └── tests/
        ├── __init__.py
        └── test_pruner.py     # 12 unit tests, no GPU required
```

Integration with the main entry point (`pd_dis.py`):

- Top of `pd_dis.py`: `from methods.attn_pruning import PruneConfig, prune_prompt`
- In `run_decode()`, each prompt is passed through `prune_prompt()` before it
  is sent to the prefill service. The pruned text replaces the original one
  for the two-phase disaggregated completion call.
- The result JSON records `prompt_after_pruning`,
  `prompt_tokens_before_pruning`, and `prompt_tokens_after_pruning` so the
  downstream analysis can inspect the effect per-prompt.

---

## 2. Python API

```python
from methods.attn_pruning import PruneConfig, prune_prompt

cfg = PruneConfig(
    method="attn_proxy",   # "none" | "attn_proxy" | "random"
    keep_ratio=0.30,       # fraction of tokens to keep, 0.0-1.0
    min_tokens=0,          # skip pruning for prompts shorter than this
    seed=42,               # only used by the "random" method
)
result = prune_prompt("Explain prefill-decode disaggregation in LLM inference.", cfg)

print(result.text)              # pruned prompt
print(result.original_tokens)   # token count before pruning
print(result.kept_tokens)       # token count after pruning
print(result.applied)           # whether pruning actually changed the prompt
print(result.method)            # method that was used
```

Scoring features (`_score_token`): position prior (tokens near the beginning
and end are weighted higher) + salient-term whitelist (question / answer /
must / error / ...) + token length as a rough rarity proxy + digit/uppercase
flags. The top-k tokens by score are kept **in their original order** so the
pruned prompt remains syntactically reasonable.

---

## 3. CLI Flags (`pd_dis.py`)

| Flag | Default | Description |
|---|---|---|
| `--pruning-method` | `none` | `none` / `attn_proxy` / `random` |
| `--pruning-keep-ratio` | `1.0` | Fraction of tokens kept; `1.0` is a no-op |
| `--pruning-min-tokens` | `0` | `0` = always prune; `>0` skips prompts shorter than this |
| `--pruning-seed` | `42` | Only used when method is `random` |

> `--pruning-method=none` or `--pruning-keep-ratio=1.0` is functionally
> identical to the baseline and will not affect any existing result.

---

## 4. Testing

### 4.1 Unit tests (no GPU, <1s)

```bash
cd /home/fl38/comp529/pd_dis
python3 -m unittest methods.attn_pruning.tests.test_pruner -v
```

Covers: `none` passthrough, empty prompts, `keep_ratio` clamping, `min_tokens`
threshold, `random` reproducibility, `attn_proxy` determinism and order
preservation, and idempotency across repeated calls.

### 4.2 Syntax check

```bash
python3 -m py_compile pd_dis.py methods/attn_pruning/pruner.py
```

### 4.3 End-to-end GPU test (2-node Lovelace, ~5-10 min per run)

**Baseline** (no pruning):

```bash
sbatch -J pd_ib_baseline pd_dis.sh \
  --prompts-file eval/results/baseline_20260421_220526/prompts_long_ctx.txt
```

**Aggressive pruning** (attn_proxy, keep=0.30):

```bash
sbatch -J pd_ib_attn_k30 pd_dis.sh \
  --prompts-file eval/results/baseline_20260421_220526/prompts_long_ctx.txt \
  --pruning-method attn_proxy \
  --pruning-keep-ratio 0.30
```

**Random pruning control** (to verify that attn_proxy does better than random):

```bash
sbatch -J pd_ib_rand_k30 pd_dis.sh \
  --prompts-file eval/results/baseline_20260421_220526/prompts_long_ctx.txt \
  --pruning-method random \
  --pruning-keep-ratio 0.30 \
  --pruning-seed 42
```

Job output is written to `~/comp529/log/pd_disagg_<jobid>.txt`.

### 4.4 Single-host smoke test (pruning logic only, no vLLM)

```bash
python3 - <<'PY'
from methods.attn_pruning import PruneConfig, prune_prompt
p = ("Explain what prefill-decode disaggregation means in LLM inference, "
     "why RDMA helps KV cache transfer, and how continuous batching improves "
     "throughput. " * 10)
r = prune_prompt(p, PruneConfig(method="attn_proxy", keep_ratio=0.3))
print("orig:", r.original_tokens, "kept:", r.kept_tokens, "applied:", r.applied)
print("text:", r.text[:300])
PY
```

---

## 5. How to Verify Pruning Actually Ran

For every prompt `pd_dis.py` emits lines like:

```
[Prune] method=attn_proxy keep_ratio=0.30 tokens 2543 -> 763 (reduction=70.00%)
[Prune] original: Explain prefill-decode disaggregation ...
[Prune] pruned  : Explain prefill-decode disaggregation important ...
```

Checklist:
1. `reduction` is non-zero → pruning actually fired. If you see `0.00%`,
   either `method=none`, `keep_ratio=1.0`, or the prompt is shorter than
   `--pruning-min-tokens`.
2. The result JSON contains `prompt_tokens_after_pruning <
   prompt_tokens_before_pruning`.
3. When comparing against the baseline, look at `Avg TTFT` and
   `Avg ms/output-token`. The longer the prompts and the smaller the
   `keep_ratio`, the more TTFT drops because the prefill compute cost is
   roughly linear in prompt length.

---

## 6. Known Limitations and Strict vs. Proxy

| Dimension | Current proxy version | Strict attn-based eviction (TODO) |
|---|---|---|
| Target of pruning | Input prompt text | KV cache tensors after prefill |
| Requires vLLM changes | No | Yes (hook attention or patch KV manager) |
| Pruning signal | Heuristic score | Real attention / importance score |
| Semantic risk | May drop mid-prompt keywords | Lower, model decides itself |
| Where end-to-end savings show up | Prefill compute → TTFT | Transfer + decode, TTFT unchanged |

**Why ship the proxy first?**
- No vLLM kernel changes, so it is fully independent of the teammates'
  Method 1 / Method 3 work.
- Reuses the same eval infrastructure (`eval/run_sweep.sh`).
- The `method` / `keep_ratio` CLI surface is designed so the strict backend
  can be added later as a new method value without breaking existing scripts.

---

## 7. Validation Status

- [x] All 12 unit tests pass
- [x] CPU smoke test: different `keep_ratio` values produce prompts of
      different lengths, with the original order preserved
- [x] End-to-end GPU run works (Qwen2.5-3B, 2-node Lovelace, IB +
      NixlConnector)
- [x] `--pruning-min-tokens` defaults to `0`, so pruning fires on any prompt
      length; bump it up manually if you want to protect very short prompts
- [ ] On this cluster IB bandwidth is high (2-3 GB/s) so KV transfer is not
      the bottleneck; expect the savings to come mainly from reduced prefill
      compute (lower TTFT), not from faster KV transfer
