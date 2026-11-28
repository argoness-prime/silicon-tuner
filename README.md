# Silicon Tuner (Open-Core `v1.1.0`)

**Zero-dependency GQA KV-cache & Mixture-of-Experts (`--n-cpu-moe`) GPU VRAM + System DDR5 RAM calculator and local `stdio` MCP server for local LLMs.**

Stop guessing `-ngl`, `--n-cpu-moe`, and `-ctk/-ctv` flags by trial-and-error OOM crashes on `8 GiB`–`32 GiB` GPUs. `silicon-tuner-core` computes exact Grouped-Query Attention (GQA) element counts (`2 * layers * kv_heads * head_dim * tokens` — e.g., `1,610,612,736` elements for 48-layer 4-KV-head 32K context) alongside explicitly separated **GPU App Budget** vs **Physical VRAM (including OS/DWM reserve)** ledgers.

## Quickstart

```bash
# Run instantly via uvx (zero clone required):
uvx --from https://github.com/argoness-prime/silicon-tuner/releases/download/v1.1.0/silicon_tuner_core-1.1.0-py3-none-any.whl silicon-tuner-core selftest
uvx --from https://github.com/argoness-prime/silicon-tuner/releases/download/v1.1.0/silicon_tuner_core-1.1.0-py3-none-any.whl silicon-tuner-core plan --preset qwen3-35b-a3b --vram 16 --ram 64 --quant q4_k_m --ctx 32768

# Register as a local stdio MCP server in Claude Code:
claude mcp add silicon-tuner -- uvx --from https://github.com/argoness-prime/silicon-tuner/releases/download/v1.1.0/silicon_tuner_core-1.1.0-py3-none-any.whl silicon-tuner-core mcp

# Install as an Agent Skill:
npx skills add argoness-prime/silicon-tuner
```

### Exposed MCP Tools (`stdio` JSON-RPC 2.0)
1. **`plan_hybrid_moe_split`** — Exact Dense & MoE layer/expert VRAM + DDR5 RAM planner returning `gpuAppLedger` (`14.64 / 14.75 GiB`), `physicalVramLedger` (`1.25 GiB` DWM + `14.64 GiB` = `15.89 / 16.00 GiB` vs `24.56 GiB` naive fp16 OOM), and `llamaCppFlags`.
2. **`compute_gqa_kv_cache`** — Deterministic GQA KV-cache element & byte math across `f16`, `bf16`, `q8_0`, and `q4_0` vs counterfactual Multi-Head Attention (MHA).

## Companion Veristrux Engineering Resources
- **MCP Registry Identifier:** `mcp-name: io.github.argoness-prime/silicon-tuner`
- **1080p & 9:16 GQA KV-Cache & MoE Split Walkthrough:** [1080p MP4](https://veristrux.com/media/wave5/silicon_moe_35b_16gb_16x9.mp4) · [9:16 Short MP4](https://veristrux.com/media/wave5/silicon_moe_35b_16gb_short.mp4)
- **Interactive Token Czar AST & Cache Lab:** https://veristrux.com/tokenczar-demo.html
- **Full Veristrux Systems Catalog & Pro Toolkits:** https://veristrux.com/
- **License:** MIT License — Copyright (c) 2026 Veristrux Inc. (`argoness@veristrux.com`)
- **Interactive Client-Side GQA KV-Cache & Sparse MoE Split Planner (0-signup):** https://veristrux.com/silicon-tuner-demo.html
