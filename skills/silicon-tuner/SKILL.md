---
name: silicon-tuner
description: "Calculate exact GQA KV-cache memory, OS compositor headroom, and Mixture-of-Experts (--n-cpu-moe) cold-expert RAM splits for running 14B-70B local LLMs on 8-24 GiB GPUs without OOM crashes."
---

# Hybrid Silicon Local-AI Tuner (`silicon-tuner-core`)

Use this skill when sizing context windows, KV-cache quantization (`q8_0`, `q4_1`, `f16`), or Mixture-of-Experts (`--n-cpu-moe`) GPU/DDR5 splits for `llama.cpp`, `Ollama`, or `vLLM` on consumer GPUs.

## Quickstart

```bash
uvx --from https://github.com/argoness-prime/silicon-tuner/releases/download/v1.0.0/silicon_tuner_core-1.0.0-py3-none-any.whl silicon-tuner-core --help
```
