"""Analytical KV-cache payload size calculator.

KV bytes per request =
    2                       (K and V)
  × num_hidden_layers
  × num_key_value_heads     (GQA/MQA aware)
  × head_dim
  × seq_len
  × bytes_per_element

Reads the model config via transformers if installed; otherwise falls back to
hard-coded Qwen2.5-3B-Instruct dimensions.
"""

import argparse

try:
    from transformers import AutoConfig
    HAVE_HF = True
except Exception:
    HAVE_HF = False


def kv_bytes(L: int, KV: int, D: int, S: int, b: float) -> float:
    return 2 * L * KV * D * S * b


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:6.2f} {unit}"
        n /= 1024
    return f"{n:6.2f} TB"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--seq-lens", type=int, nargs="+",
                    default=[128, 256, 512, 1024, 2048, 4096, 8192])
    args = ap.parse_args()

    if HAVE_HF:
        cfg = AutoConfig.from_pretrained(args.model)
        L = cfg.num_hidden_layers
        KV = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
        D = cfg.hidden_size // cfg.num_attention_heads
        src = "transformers"
    else:
        # Fallback for Qwen2.5-3B-Instruct.
        L, KV, D = 36, 2, 128
        src = "hardcoded (Qwen2.5-3B)"

    print(f"Model: {args.model}   [{src}]")
    print(f"  num_hidden_layers    = {L}")
    print(f"  num_key_value_heads  = {KV}")
    print(f"  head_dim             = {D}")
    print(f"  per-token KV (fp16)  = {human(2 * L * KV * D * 2)}")
    print()
    print(f"{'seq_len':>8} | {'fp16':>10} | {'int8':>10} | {'int4':>10}"
          f" | fp16 MB | 10 GiB/s s | 100 GiB/s ms")
    print("-" * 92)
    for S in args.seq_lens:
        b16 = kv_bytes(L, KV, D, S, 2)
        b8 = kv_bytes(L, KV, D, S, 1)
        b4 = kv_bytes(L, KV, D, S, 0.5)
        mb = b16 / (1024 * 1024)
        t_10g = b16 / (10 * 1024 ** 3)    # seconds at 10 GiB/s
        t_100g = b16 / (100 * 1024 ** 3)  # seconds at 100 GiB/s
        print(f"{S:>8} | {human(b16)} | {human(b8)} | {human(b4)}"
              f" | {mb:7.2f} | {t_10g*1000:8.1f}ms | {t_100g*1000:8.2f}ms")


if __name__ == "__main__":
    main()
