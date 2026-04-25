# Sweep Aggregate

Results root: `eval/results/comp_pipe_20260424_122441`
Jobs aggregated: 6

| tag | dataset | subset | conc | n_ok | acc_pct | avg_score | TTFT mean | TTFT p95 | TTFT p99 | E2E mean | prefill mean | tpot ms/tok | tok/s | KV MB/req | err |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bs128_aime25_r1 | aime25 |  | n/a | 30 | 43.3% | 0.433 | 0.074s | 0.147s | 0.185s | 176.08s | 0.029s | 13.4 | 74.1 | n/a | 0 |
| fp8_bs128_aime25_r1 | aime25 |  | n/a | 30 | 36.7% | 0.367 | 0.044s | 0.052s | 0.056s | 192.72s | 0.032s | 13.7 | 72.7 | n/a | 0 |
| fp8_e5m2_aime25_r1 | aime25 |  | n/a | 30 | 36.7% | 0.367 | 0.053s | 0.064s | 0.065s | 194.43s | 0.032s | 13.8 | 72.1 | n/a | 0 |
| bs128_lv16k_r1 | lveval | hotpotwikiqa_mixup_16k | n/a | 35 | 14.3% | 0.135 | 0.523s | 0.605s | 0.625s | 7.79s | 2.524s | 18.8 | 39.4 | n/a | 15 |
| fp8_bs128_lv16k_r1 | lveval | hotpotwikiqa_mixup_16k | n/a | 35 | 14.3% | 0.115 | 0.301s | 0.347s | 0.363s | 5.26s | 2.490s | 17.7 | 41.2 | n/a | 15 |
| fp8_e5m2_lv16k_r1 | lveval | hotpotwikiqa_mixup_16k | n/a | 35 | 14.3% | 0.115 | 3.003s | 3.479s | 3.592s | 7.98s | 2.487s | 43.6 | 30.7 | n/a | 15 |
