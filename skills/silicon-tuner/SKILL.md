---
name: silicon-tuner
description: "Calculate exact GQA KV-cache element/GiB footprints (e.g. 1,610,612,736 elements), separated GPU App vs Physical DWM VRAM ledgers, and Mixture-of-Experts (--n-cpu-moe) splits via CLI or stdio MCP server."
---

# Hybrid Silicon Local-AI Tuner (`silicon-tuner-core` `v1.1.0`)

Use this skill or the `silicon-tuner-core mcp` (`stdio` JSON-RPC 2.0) server when sizing context windows, KV-cache quantization (`q8_0`, `q4_0`, `f16`), or Mixture-of-Experts (`--n-cpu-moe`) GPU/DDR5 splits for `llama.cpp`, `Ollama`, or `vLLM`.

## Quickstart

```bash
uvx --from https://github.com/argoness-prime/silicon-tuner/releases/download/v1.1.0/silicon_tuner_core-1.1.0-py3-none-any.whl silicon-tuner-core selftest
claude mcp add silicon-tuner -- uvx --from https://github.com/argoness-prime/silicon-tuner/releases/download/v1.1.0/silicon_tuner_core-1.1.0-py3-none-any.whl silicon-tuner-core mcp
```
