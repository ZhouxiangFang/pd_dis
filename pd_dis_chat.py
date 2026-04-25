"""
pd_dis_chat.py — Chat-templated wrapper around pd_dis.py.

Goal:  apply Qwen's native chat template so the instruct model sees proper
       user/assistant boundaries, WITHOUT interfering with prompt pruning.

Pipeline order (important):
       load raw prompt  →  prune_prompt(raw)  →  apply_chat_template(pruned)  →  vLLM
       ───────────────     ──────────────────     ────────────────────────     ────

       The chat-template wrap happens as late as possible (inside
       two_phase_disagg_completion) so that:
         (a) the pruner scores and keeps the user-content tokens only, and
             never strips <|im_start|>/<|im_end|>/<think> boundary tokens;
         (b) vLLM still receives a well-formed Qwen chat turn.

The original pd_dis.py / pd_dis.sh remain untouched and are still usable
for A/B comparisons.
"""

from __future__ import annotations

import re
import string
import sys

import pd_dis


# ---------------------------------------------------------------------------
# CLI peek
# ---------------------------------------------------------------------------

def _peek_cli(argv: list[str]) -> tuple[str, bool]:
    model = "Qwen/Qwen3-4B"
    enable_thinking = True
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--model" and i + 1 < len(argv):
            model = argv[i + 1]; i += 2; continue
        if a.startswith("--model="):
            model = a.split("=", 1)[1]; i += 1; continue
        if a == "--thinking":       enable_thinking = True
        elif a == "--no-thinking":  enable_thinking = False
        i += 1
    return model, enable_thinking


# ---------------------------------------------------------------------------
# Lazy tokenizer holder — shared across threads via a dict (tokenizer.encode
# and tokenizer.apply_chat_template are thread-safe for read-only use).
# ---------------------------------------------------------------------------

_cache: dict = {"tokenizer": None, "warned": False,
                "model_name": None, "enable_thinking": True}


def _get_tokenizer():
    if _cache["tokenizer"] is None and not _cache["warned"]:
        try:
            from transformers import AutoTokenizer
            print(f"[chat] loading tokenizer for {_cache['model_name']} ...",
                  flush=True)
            _cache["tokenizer"] = AutoTokenizer.from_pretrained(
                _cache["model_name"], trust_remote_code=True
            )
        except Exception as exc:
            print(f"[chat] WARNING: tokenizer load failed ({exc}); "
                  f"chat template DISABLED", file=sys.stderr, flush=True)
            _cache["warned"] = True
    return _cache["tokenizer"]


def _apply_template(raw_text: str) -> str:
    """Wrap raw prompt text in a Qwen chat template (single user turn)."""
    tok = _get_tokenizer()
    if tok is None:
        return raw_text  # fallback: no-op if tokenizer unavailable
    messages = [{"role": "user", "content": raw_text}]
    try:
        return tok.apply_chat_template(
            messages, tokenize=False,
            add_generation_prompt=True,
            enable_thinking=_cache["enable_thinking"],
        )
    except TypeError:
        return tok.apply_chat_template(
            messages, tokenize=False,
            add_generation_prompt=True,
        )


# ---------------------------------------------------------------------------
# The actual monkey-patch: templatize the prompt EXACTLY before the two-phase
# HTTP call. Everything upstream (dataset loading, pruning) sees raw text.
# ---------------------------------------------------------------------------

def _make_templated_two_phase(original_fn):
    def wrapped(*, model: str, prompt: str, max_tokens: int,
                temperature: float, prefill_host: str,
                timeout: float = 600):
        templated = _apply_template(prompt)
        return original_fn(
            model=model,
            prompt=templated,
            max_tokens=max_tokens,
            temperature=temperature,
            prefill_host=prefill_host,
            timeout=timeout,
        )
    return wrapped


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Scoring fixes — monkey-patch pd_dis.score_result so that:
#   (a) LVEval answer extraction strips Qwen3's <think>...</think> block
#       before looking for the final answer, and
#   (b) f1_score_tokens uses a SQuAD-style normalization (lowercase, strip
#       punctuation, collapse whitespace) so that "Timisoara," matches
#       "Timisoara".
# Both are standard practice for QA benchmarks; we patch them in the wrapper
# instead of editing pd_dis.py so the raw path remains a faithful mirror
# of upstream.
# ---------------------------------------------------------------------------

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def _normalize_for_f1(text: str) -> list[str]:
    """SQuAD-style normalization: lower, strip punctuation, split on ws."""
    return text.lower().translate(_PUNCT_TABLE).split()


def _patched_f1_score_tokens(pred: str, ref: str) -> float:
    p_toks = _normalize_for_f1(pred)
    r_toks = _normalize_for_f1(ref)
    if not p_toks or not r_toks:
        return 0.0
    from collections import Counter
    common = Counter(p_toks) & Counter(r_toks)
    num = sum(common.values())
    if num == 0:
        return 0.0
    prec = num / len(p_toks)
    rec  = num / len(r_toks)
    return 2 * prec * rec / (prec + rec)


def _patched_score_result(output: str, ground_truth, dataset: str):
    """Drop-in replacement for pd_dis.score_result.

    Changes vs. the upstream version:
      * Removes any <think>...</think> block from the model output before
        extracting — otherwise the LVEval regex pulls the literal '<think>'
        token as the answer.
      * Calls our punctuation-robust f1_score_tokens for LVEval.
    """
    # Strip Qwen3 thinking blocks first (affects both datasets harmlessly).
    clean = _THINK_RE.sub(" ", output).strip()

    if dataset == "aime25":
        extracted = pd_dis.extract_answer_aime(clean)
        if extracted is None:
            return 0.0, None
        ref = ground_truth[0] if isinstance(ground_truth, (list, tuple)) else str(ground_truth)
        try:
            score = 1.0 if int(extracted) == int(ref) else 0.0
        except (ValueError, TypeError):
            score = 1.0 if extracted.strip() == ref.strip() else 0.0
        return score, extracted

    if dataset == "lveval":
        answer_marker = re.search(r'[Aa]nswer\s*:\s*', clean)
        if answer_marker:
            extracted = clean[answer_marker.end():].strip()
        else:
            extracted = clean.strip()
        # Keep only the first non-empty sentence / line.
        parts = re.split(r'[\n。]', extracted)
        extracted = ""
        for p in parts:
            p = p.strip()
            if p:
                extracted = p[:200]
                break
        refs = ground_truth if isinstance(ground_truth, (list, tuple)) else [str(ground_truth)]
        score = max((_patched_f1_score_tokens(extracted, r) for r in refs), default=0.0)
        return score, extracted

    return 0.0, None


def _patched_hf_load(original_fn):
    """Inject trust_remote_code=True into load_dataset calls.

    The older `datasets` (2.x) we pin via PYTHONPATH supports LVEval's
    script-based loader but requires explicit trust_remote_code. Upstream
    `pd_dis._hf_load_dataset` doesn't forward that kwarg, so we inject it
    here without modifying pd_dis.py.
    """
    def wrapped(name, *args, **kwargs):
        kwargs.setdefault("trust_remote_code", True)
        return original_fn(name, *args, **kwargs)
    return wrapped


def main() -> None:
    import os
    model, enable_thinking = _peek_cli(sys.argv[1:])
    _cache["model_name"] = model
    _cache["enable_thinking"] = enable_thinking

    # Patch 0: give every sbatch job a unique port triplet so two jobs can
    # safely share a node. SLURM commons queue can be saturated for days
    # when we pass --exclusive; this lets us drop --exclusive entirely.
    # Offset = SLURM_JOB_ID mod 100 (gives 100 unique port sets).
    job_id = int(os.environ.get("SLURM_JOB_ID", "0"))
    offset = job_id % 100
    pd_dis.PREFILL_PORT           = 8100 + offset
    pd_dis.DECODE_PORT            = 8200 + offset
    pd_dis.NIXL_SIDE_CHANNEL_PORT = 5559 + offset

    print(f"[chat] wrapper v2 active  model={model}  "
          f"enable_thinking={enable_thinking}  "
          f"ports=(prefill={pd_dis.PREFILL_PORT}, decode={pd_dis.DECODE_PORT}, "
          f"nixl={pd_dis.NIXL_SIDE_CHANNEL_PORT}; offset={offset} from JOB_ID={job_id})",
          flush=True)

    # Patch 1: templatize prompts at the HTTP boundary (after pruning).
    pd_dis.two_phase_disagg_completion = _make_templated_two_phase(
        pd_dis.two_phase_disagg_completion
    )

    # Patch 2: force trust_remote_code=True so LVEval's script-based loader
    # runs under the older `datasets` library pinned via PYTHONPATH.
    pd_dis._hf_load_dataset = _patched_hf_load(pd_dis._hf_load_dataset)

    # Patch 3: robust scoring — strip Qwen3 <think>…</think>, SQuAD-style
    # punctuation-insensitive F1.
    pd_dis.score_result = _patched_score_result
    pd_dis.f1_score_tokens = _patched_f1_score_tokens

    pd_dis.main()


if __name__ == "__main__":
    main()
