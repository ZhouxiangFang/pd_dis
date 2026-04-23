"""Generate synthetic workload prompts targeting a specific token count.

We approximate 1 token ≈ 0.75 English words. Exact token counts will vary by
tokenizer, but for KV-size sweeps this is close enough — the pd_dis.py log
records the real prompt_tokens from vLLM, which the parser picks up.
"""

import argparse
from pathlib import Path

BASE = (
    "In recent years, large language models have been deployed across a wide "
    "range of applications including code generation, summarization, "
    "translation, and dialogue systems. These models typically consist of "
    "many transformer layers and use attention mechanisms to capture "
    "dependencies between tokens. Efficient inference is a central concern "
    "because memory bandwidth and compute requirements scale with both model "
    "size and sequence length. Techniques such as paged attention, "
    "quantization, and prefix caching have been developed to improve "
    "throughput and reduce latency. When models are served in a disaggregated "
    "fashion, the prefill stage and the decode stage can run on separate "
    "hardware, which increases flexibility but introduces new communication "
    "challenges between nodes. The KV cache produced during prefill must be "
    "transported to the decode worker, and its size grows linearly with the "
    "number of prompt tokens as well as with the number of transformer "
    "layers and attention heads in the model."
)

TAIL_QUESTIONS = [
    "Summarize the passage above in three sentences and highlight the main tradeoffs involved.",
    "Based on the context above, explain why KV cache transfer becomes a bottleneck in long-context serving.",
    "Given the background above, propose one concrete optimization for reducing inter-node transfer overhead.",
    "Using the material above, describe the difference between the prefill and decode stages in detail.",
    "From the text above, list three metrics that should be tracked when benchmarking disaggregated serving.",
    "Given the passage, explain when disaggregation is preferable to colocated serving.",
    "Based on the text, describe how attention sinks affect token importance estimation.",
    "Using the content above, analyze the tradeoff between quantization fidelity and transfer time.",
]


def build_prompt(target_words: int, idx: int) -> str:
    base_words = BASE.split()
    out = []
    while len(out) < target_words:
        out.extend(base_words)
    out = out[:target_words]
    q = TAIL_QUESTIONS[idx % len(TAIL_QUESTIONS)]
    return " ".join(out) + " " + q


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-tokens", type=int, required=True,
                    help="Approximate prompt length in tokens (1 tok ~ 0.75 words)")
    ap.add_argument("--n-prompts", type=int, default=16)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    target_words = max(1, int(args.target_tokens * 0.75))
    lines = [
        f"# auto-generated: target_tokens={args.target_tokens} n_prompts={args.n_prompts}"
    ]
    for i in range(args.n_prompts):
        p = build_prompt(target_words, i).replace("\n", " ")
        lines.append(p)
    args.out.write_text("\n".join(lines) + "\n")
    print(f"[gen_prompts] wrote {args.n_prompts} prompts "
          f"(~{args.target_tokens} tok each) → {args.out}")


if __name__ == "__main__":
    main()
