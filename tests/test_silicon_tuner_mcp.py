"""Pytest suite for Silicon Tuner v1.1.0 zero-dependency stdio MCP server & exact GQA/MoE math."""

from __future__ import annotations

import io
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from silicon_tuner_core import mcp_server  # noqa: E402


def _rpc(msg: dict) -> dict | None:
    return mcp_server.handle_rpc_message(msg)


def test_initialize_and_tools_list() -> None:
    init = _rpc({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-06-18"}})
    assert init is not None
    assert init["result"]["serverInfo"] == {"name": "silicon-tuner", "version": "1.1.0"}
    assert init["result"]["protocolVersion"] == "2025-06-18"

    tl = _rpc({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    assert tl is not None
    names = [t["name"] for t in tl["result"]["tools"]]
    assert names == ["plan_hybrid_moe_split", "compute_gqa_kv_cache"]


def test_compute_gqa_kv_cache_exact_1610612736() -> None:
    resp = _rpc(
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "compute_gqa_kv_cache",
                "arguments": {"preset": "qwen3-35b-a3b", "context_tokens": 32768, "kv_quant": "q8_0"},
            },
        }
    )
    assert resp is not None
    sc = resp["result"]["structuredContent"]
    assert sc["totalKvElements"] == 1_610_612_736
    assert sc["kvCacheGib"] == 1.59
    assert sc["gqaFp16Gib"] == 3.0
    assert sc["counterfactualMhaFp16Gib"] == 24.0


def test_plan_hybrid_moe_split_separated_ledgers() -> None:
    resp = _rpc(
        {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {
                "name": "plan_hybrid_moe_split",
                "arguments": {
                    "preset": "qwen3-35b-a3b",
                    "vram_gib": 16.0,
                    "ram_gib": 64.0,
                    "os_headroom_gib": 1.25,
                    "quant": "q4_k_m",
                    "context_tokens": 32768,
                    "kv_quant": "q8_0",
                },
            },
        }
    )
    assert resp is not None
    sc = resp["result"]["structuredContent"]
    app_l = sc["gpuAppLedger"]
    phys_l = sc["physicalVramLedger"]

    assert app_l["sharedAttentionRouterGib"] == 3.16
    assert app_l["gpuExpertLayersGib"] == 9.34
    assert app_l["kvCacheGib"] == 1.59
    assert app_l["computeBufferGib"] == 0.55
    assert app_l["gpuAppUsedGib"] == 14.64
    assert app_l["gpuAppBudgetGib"] == 14.75

    assert phys_l["osDwmReserveGib"] == 1.25
    assert phys_l["physicalUsedGib"] == 15.89
    assert phys_l["physicalTotalGib"] == 16.0
    assert phys_l["naiveFullGpuFp16PhysicalGib"] == 24.56
    assert "--n-cpu-moe 21" in sc["llamaCppFlags"]


def test_stdio_framing_roundtrip() -> None:
    body = json.dumps({"jsonrpc": "2.0", "id": 9, "method": "ping"}).encode("utf-8")
    framed = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii") + body
    inp = io.BytesIO(framed)
    out = io.BytesIO()
    assert mcp_server.serve_stdio(inp, out) == 0
    assert b'"id":9' in out.getvalue()
