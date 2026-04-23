"""Minimal attention-guided prompt pruning for PD experiments.

This module provides a lightweight proxy for attention-based token pruning:
it scores tokens with simple salience signals and keeps the top-k tokens
while preserving their original order.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
import re


TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)

# A tiny hand-written set of likely-important words in QA/instruction prompts.
SALIENT_TERMS = {
    "question",
    "answer",
    "instruction",
    "requirement",
    "important",
    "must",
    "not",
    "explain",
    "summary",
    "reason",
    "because",
    "therefore",
    "code",
    "function",
    "error",
    "fix",
    "result",
}


@dataclass(frozen=True)
class PruneConfig:
    method: str = "none"
    keep_ratio: float = 1.0
    min_tokens: int = 0
    seed: int = 42


@dataclass(frozen=True)
class PruneResult:
    text: str
    original_tokens: int
    kept_tokens: int
    applied: bool
    method: str


def _tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text)


def _detokenize(tokens: list[str]) -> str:
    # Simple reconstruction: keeps punctuation tight, words space-separated.
    out: list[str] = []
    for tok in tokens:
        if not out:
            out.append(tok)
            continue
        if re.fullmatch(r"[^\w\s]", tok):
            out[-1] = out[-1] + tok
        else:
            out.append(" " + tok)
    return "".join(out)


def _score_token(token: str, pos: int, total: int) -> float:
    # Position prior: beginning/end tokens often carry higher relevance.
    if total <= 1:
        edge_prior = 1.0
    else:
        center = (total - 1) / 2.0
        dist = abs(pos - center) / max(center, 1.0)
        edge_prior = 0.5 + 0.5 * dist

    lower = token.lower()
    salient = 1.0 if lower in SALIENT_TERMS else 0.0
    is_number = 1.0 if any(ch.isdigit() for ch in token) else 0.0
    is_upper = 1.0 if token.isupper() and len(token) > 1 else 0.0
    # Rough rarity proxy: longer words often hold more content.
    rarity = min(math.log2(max(len(token), 1)), 4.0) / 4.0

    return 0.55 * edge_prior + 0.25 * salient + 0.10 * rarity + 0.06 * is_number + 0.04 * is_upper


def _keep_indices_attention_proxy(tokens: list[str], keep_k: int) -> set[int]:
    scored = [
        (_score_token(tok, i, len(tokens)), i)
        for i, tok in enumerate(tokens)
    ]
    scored.sort(reverse=True)
    return {idx for _, idx in scored[:keep_k]}


def _keep_indices_random(tokens: list[str], keep_k: int, seed: int) -> set[int]:
    rng = random.Random(seed)
    indices = list(range(len(tokens)))
    rng.shuffle(indices)
    return set(indices[:keep_k])


def prune_prompt(prompt: str, cfg: PruneConfig) -> PruneResult:
    tokens = _tokenize(prompt)
    n = len(tokens)
    method = cfg.method.lower()

    if method == "none" or n == 0:
        return PruneResult(prompt, n, n, False, method)

    keep_ratio = max(0.0, min(1.0, cfg.keep_ratio))
    keep_k = max(1, int(round(n * keep_ratio)))
    keep_k = min(keep_k, n)

    if keep_k >= n or n < cfg.min_tokens:
        return PruneResult(prompt, n, n, False, method)

    if method == "random":
        keep = _keep_indices_random(tokens, keep_k, cfg.seed)
    elif method == "attn_proxy":
        keep = _keep_indices_attention_proxy(tokens, keep_k)
    else:
        raise ValueError(f"Unsupported pruning method: {cfg.method}")

    kept_tokens = [tok for i, tok in enumerate(tokens) if i in keep]
    pruned = _detokenize(kept_tokens).strip()
    if not pruned:
        pruned = prompt
        keep_k = n
        applied = False
    else:
        applied = keep_k < n

    return PruneResult(pruned, n, keep_k, applied, method)
