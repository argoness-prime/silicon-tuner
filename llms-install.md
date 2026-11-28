# LLM / Autonomous Agent Installation Guide (`silicon-tuner` v1.1.0)

Zero-dependency Python 3.10+ stdlib `stdio` MCP server for Grouped-Query Attention (GQA) KV-cache sizing and Mixture-of-Experts (`--n-cpu-moe`) GPU VRAM + DDR5 RAM planning. Zero API keys or external packages required.

## Option A — Zero-Clone `uvx` Configuration

```json
{
  "mcpServers": {
    "silicon-tuner": {
      "command": "uvx",
      "args": [
        "--from",
        "https://github.com/argoness-prime/silicon-tuner/releases/download/v1.1.0/silicon_tuner_core-1.1.0-py3-none-any.whl",
        "silicon-tuner-core",
        "mcp"
      ]
    }
  }
}
```

## Option B — Pure-Python Wheel Installation via `pip`

```bash
python -m pip install --no-deps https://github.com/argoness-prime/silicon-tuner/releases/download/v1.1.0/silicon_tuner_core-1.1.0-py3-none-any.whl
python -m silicon_tuner_core selftest
```

```json
{
  "mcpServers": {
    "silicon-tuner": {
      "command": "python",
      "args": ["-u", "-m", "silicon_tuner_core", "mcp"]
    }
  }
}
```
