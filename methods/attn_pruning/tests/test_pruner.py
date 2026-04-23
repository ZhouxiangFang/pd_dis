"""Unit tests for methods.attn_pruning.pruner.

These tests exercise the public API only and require no GPU / vLLM.
Run from the repo root with:

    python3 -m unittest methods.attn_pruning.tests.test_pruner
"""

from __future__ import annotations

import unittest

from methods.attn_pruning import PruneConfig, prune_prompt
from methods.attn_pruning.pruner import _tokenize


LONG_PROMPT = (
    "Please summarize the following instruction carefully. "
    "The user wants a concise answer that explains why the 42 errors "
    "happened in the CODE and provides two clear fixes with reasons. "
    "Do not skip important details, and keep the result under 200 words."
)


class PruneNoneTests(unittest.TestCase):
    def test_none_method_is_identity(self):
        cfg = PruneConfig(method="none", keep_ratio=0.5, min_tokens=0)
        res = prune_prompt(LONG_PROMPT, cfg)

        self.assertEqual(res.text, LONG_PROMPT)
        self.assertFalse(res.applied)
        self.assertEqual(res.method, "none")
        self.assertEqual(res.original_tokens, res.kept_tokens)

    def test_empty_prompt_is_safe(self):
        cfg = PruneConfig(method="attn_proxy", keep_ratio=0.5, min_tokens=0)
        res = prune_prompt("", cfg)

        self.assertEqual(res.text, "")
        self.assertEqual(res.original_tokens, 0)
        self.assertEqual(res.kept_tokens, 0)
        self.assertFalse(res.applied)


class KeepRatioTests(unittest.TestCase):
    def test_keep_ratio_one_keeps_everything(self):
        cfg = PruneConfig(method="attn_proxy", keep_ratio=1.0, min_tokens=0)
        res = prune_prompt(LONG_PROMPT, cfg)

        self.assertFalse(res.applied)
        self.assertEqual(res.kept_tokens, res.original_tokens)

    def test_keep_ratio_reduces_tokens(self):
        cfg = PruneConfig(method="attn_proxy", keep_ratio=0.5, min_tokens=0)
        res = prune_prompt(LONG_PROMPT, cfg)

        self.assertTrue(res.applied)
        self.assertLess(res.kept_tokens, res.original_tokens)
        self.assertGreaterEqual(res.kept_tokens, 1)

    def test_extreme_keep_ratio_keeps_at_least_one_token(self):
        cfg = PruneConfig(method="attn_proxy", keep_ratio=0.0, min_tokens=0)
        res = prune_prompt(LONG_PROMPT, cfg)

        self.assertGreaterEqual(res.kept_tokens, 1)

    def test_min_tokens_skips_short_prompts(self):
        short_prompt = "Short prompt."
        cfg = PruneConfig(method="attn_proxy", keep_ratio=0.5, min_tokens=1000)
        res = prune_prompt(short_prompt, cfg)

        self.assertFalse(res.applied)
        self.assertEqual(res.kept_tokens, res.original_tokens)
        self.assertEqual(res.text, short_prompt)


class RandomMethodTests(unittest.TestCase):
    def test_random_is_deterministic_for_same_seed(self):
        cfg_a = PruneConfig(method="random", keep_ratio=0.4, min_tokens=0, seed=123)
        cfg_b = PruneConfig(method="random", keep_ratio=0.4, min_tokens=0, seed=123)

        res_a = prune_prompt(LONG_PROMPT, cfg_a)
        res_b = prune_prompt(LONG_PROMPT, cfg_b)

        self.assertEqual(res_a.text, res_b.text)
        self.assertEqual(res_a.kept_tokens, res_b.kept_tokens)

    def test_random_different_seeds_likely_differ(self):
        cfg_a = PruneConfig(method="random", keep_ratio=0.4, min_tokens=0, seed=1)
        cfg_b = PruneConfig(method="random", keep_ratio=0.4, min_tokens=0, seed=2)

        self.assertNotEqual(
            prune_prompt(LONG_PROMPT, cfg_a).text,
            prune_prompt(LONG_PROMPT, cfg_b).text,
        )


class AttnProxyTests(unittest.TestCase):
    def test_kept_tokens_preserve_input_order(self):
        cfg = PruneConfig(method="attn_proxy", keep_ratio=0.5, min_tokens=0)
        res = prune_prompt(LONG_PROMPT, cfg)

        kept = _tokenize(res.text)
        original = _tokenize(LONG_PROMPT)

        for token in kept:
            self.assertIn(token, original)

        last_idx = -1
        for token in kept:
            idx = original.index(token, last_idx + 1) if (last_idx + 1) < len(original) else -1
            self.assertGreater(idx, last_idx, f"token order broken near {token!r}")
            last_idx = idx

    def test_attn_proxy_is_deterministic(self):
        cfg = PruneConfig(method="attn_proxy", keep_ratio=0.5, min_tokens=0)
        a = prune_prompt(LONG_PROMPT, cfg)
        b = prune_prompt(LONG_PROMPT, cfg)

        self.assertEqual(a.text, b.text)
        self.assertEqual(a.kept_tokens, b.kept_tokens)


class ConfigValidationTests(unittest.TestCase):
    def test_unknown_method_raises(self):
        cfg = PruneConfig(method="does_not_exist", keep_ratio=0.5, min_tokens=0)
        with self.assertRaises(ValueError):
            prune_prompt(LONG_PROMPT, cfg)

    def test_out_of_range_ratio_is_clamped(self):
        cfg_high = PruneConfig(method="attn_proxy", keep_ratio=1.5, min_tokens=0)
        cfg_low = PruneConfig(method="attn_proxy", keep_ratio=-0.2, min_tokens=0)

        res_high = prune_prompt(LONG_PROMPT, cfg_high)
        res_low = prune_prompt(LONG_PROMPT, cfg_low)

        self.assertLessEqual(res_high.kept_tokens, res_high.original_tokens)
        self.assertGreaterEqual(res_low.kept_tokens, 1)


if __name__ == "__main__":
    unittest.main()
