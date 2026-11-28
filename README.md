# Silicon Tuner (Open-Core)

**Zero-dependency GQA KV-cache & Mixture-of-Experts (`--n-cpu-moe`) GPU VRAM + System RAM calculator for local LLMs.**

Stop guessing `-ngl`, `--n-cpu-moe`, and `-ctk/-ctv` flags by trial-and-error OOM crashes on 12 GiB – 24 GiB GPUs.

## Quickstart

```bash
pip install git+https://github.com/argoness-prime/silicon-tuner.git
silicon-tuner-core plan --preset qwen3-35b-a3b --vram 16 --ram 64 --quant iq3_m --ctx 32768
```

## Open-Core vs. Veristrux Pro Kits

| Capability | `silicon-tuner-core` (MIT Free) | **Hybrid Silicon Tuner Pro ($9)** | **All-Access Agent & Silicon Bundle ($12)** |
| :--- | :---: | :---: | :---: |
| GQA KV-Cache + Weight + OS Headroom Math | Yes | Yes | Yes |
| MoE Cold-Expert Offload Split (`--n-cpu-moe`) | Yes | Yes | Yes |
| **`llama-server` / `Ollama Modelfile` / `vllm serve` Config Generator** | — | **Yes** | **Yes** |
| **Streaming `p50/p95 TTFT` & Decode `TPS` OpenAI Benchmark CLI** | — | **Yes** | **Yes** |
| **Iron Veil `<5 ms` `PreToolUse` Agent Firewall + Token Miser AST Guard** | — | — | **Yes** |

Get the commercial Pro packages & support Veristrux engineering:
- **Storefront:** https://veristrux.com/
- **Hybrid Silicon Local-AI Tuner ($9):** https://www.paypal.com/ncp/payment/X9AGQCZUJF7R6
- **All-Access Agent & Silicon Bundle ($12):** https://www.paypal.com/ncp/payment/FTR3BGX8V6LEJ
- **Contact:** `argoness@veristrux.com`
