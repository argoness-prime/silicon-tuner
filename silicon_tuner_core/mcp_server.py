"""Zero-dependency Model Context Protocol (MCP) stdio server for Silicon Tuner Open-Core (v1.1.0).

Exposes:
  1. `plan_hybrid_moe_split` — Exact Dense & MoE layer/expert VRAM + DDR5 RAM planner with separated
     GPU App Budget vs Physical VRAM (with OS/DWM reserve) ledgers and llama.cpp CLI flags.
  2. `compute_gqa_kv_cache` — Deterministic GQA KV-cache element & byte math (e.g. 1,610,612,736 elements)
     across f16/bf16/q8_0/q4_0 vs counterfactual Multi-Head Attention (MHA).
"""

from __future__ import annotations

import json
import sys
from typing import Any, BinaryIO

from . import __version__
from .planner import (
    HardwareSpec,
    KV_BYTES_PER_ELEMENT,
    ModelSpec,
    PRESET_MODELS,
    QUANT_BITS_PER_WEIGHT,
    compute_detailed_ledger,
    compute_gqa_kv_breakdown,
)

SUPPORTED_PROTOCOL_VERSIONS: tuple[str, ...] = (
    "2025-06-18",
    "2025-03-26",
    "2024-11-05",
)
DEFAULT_PROTOCOL_VERSION = "2024-11-05"

TOOLS_SPEC: tuple[dict[str, Any], ...] = (
    {
        "name": "plan_hybrid_moe_split",
        "description": (
            "Compute an exact Dense or Mixture-of-Experts (--n-cpu-moe) GPU VRAM + System DDR5 RAM execution plan "
            "with separated GPU App Budget vs Physical VRAM (including OS/DWM reserve) ledgers, GQA KV-cache math, "
            "and ready-to-run llama.cpp flags."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "preset": {
                    "type": "string",
                    "enum": sorted(PRESET_MODELS.keys()),
                    "description": "Built-in model preset (ignored only when custom `total_params_b` is supplied).",
                    "default": "qwen3-35b-a3b",
                },
                "vram_gib": {
                    "type": "number",
                    "description": "Total physical GPU VRAM in GiB (e.g. 16.0).",
                    "default": 16.0,
                    "minimum": 2.0,
                    "maximum": 1024.0,
                },
                "ram_gib": {
                    "type": "number",
                    "description": "Total host System RAM in GiB (e.g. 64.0).",
                    "default": 64.0,
                    "minimum": 4.0,
                    "maximum": 4096.0,
                },
                "os_headroom_gib": {
                    "type": "number",
                    "description": "OS / Desktop Window Manager (DWM) VRAM reserve in GiB (default 1.25 GiB).",
                    "default": 1.25,
                    "minimum": 0.0,
                    "maximum": 32.0,
                },
                "quant": {
                    "type": "string",
                    "enum": sorted(QUANT_BITS_PER_WEIGHT.keys()),
                    "description": "Weight quantization format (e.g. 'q4_k_m', 'iq3_m', 'q8_0').",
                    "default": "q4_k_m",
                },
                "context_tokens": {
                    "type": "integer",
                    "description": "Context window length in tokens (e.g. 32768).",
                    "default": 32768,
                    "minimum": 512,
                    "maximum": 2097152,
                },
                "kv_quant": {
                    "type": "string",
                    "enum": sorted(KV_BYTES_PER_ELEMENT.keys()),
                    "description": "Optional explicit KV-cache quantization override ('f16', 'bf16', 'q8_0', 'q4_0').",
                },
                "total_params_b": {"type": "number", "minimum": 0.1, "maximum": 2000.0},
                "active_params_b": {"type": "number", "minimum": 0.1, "maximum": 2000.0},
                "num_layers": {"type": "integer", "minimum": 1, "maximum": 512},
                "num_kv_heads": {"type": "integer", "minimum": 1, "maximum": 256},
                "head_dim": {"type": "integer", "minimum": 16, "maximum": 1024},
                "is_moe": {"type": "boolean"},
                "routed_expert_fraction": {"type": "number", "minimum": 0.0, "maximum": 0.99},
            },
            "additionalProperties": False,
        },
        "annotations": {
            "title": "Plan Hybrid MoE / Dense GPU+RAM Split",
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
    },
    {
        "name": "compute_gqa_kv_cache",
        "description": (
            "Compute exact Grouped-Query Attention (GQA) KV-cache element counts (2 * layers * kv_heads * head_dim * tokens) "
            "and byte/GiB footprints across f16, bf16, q8_0, and q4_0 vs counterfactual Multi-Head Attention (MHA)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "preset": {
                    "type": "string",
                    "enum": sorted(PRESET_MODELS.keys()),
                    "description": "Optional built-in model preset (used unless `num_layers` is explicitly supplied).",
                    "default": "qwen3-35b-a3b",
                },
                "num_layers": {"type": "integer", "minimum": 1, "maximum": 512},
                "num_kv_heads": {"type": "integer", "minimum": 1, "maximum": 256},
                "head_dim": {"type": "integer", "minimum": 16, "maximum": 1024},
                "context_tokens": {
                    "type": "integer",
                    "default": 32768,
                    "minimum": 512,
                    "maximum": 2097152,
                },
                "kv_quant": {
                    "type": "string",
                    "enum": sorted(KV_BYTES_PER_ELEMENT.keys()),
                    "default": "q8_0",
                },
                "counterfactual_mha_heads": {
                    "type": "integer",
                    "default": 32,
                    "minimum": 1,
                    "maximum": 512,
                },
            },
            "additionalProperties": False,
        },
        "annotations": {
            "title": "Compute Exact GQA KV-Cache Footprint",
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
    },
)


class _ToolParamError(ValueError):
    """Raised when MCP tool arguments fail schema validation."""


def _num(args: dict[str, Any], key: str, default: float, lo: float, hi: float) -> float:
    val = args.get(key, default)
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        raise _ToolParamError(f"`{key}` must be a number.")
    fval = float(val)
    if not (lo <= fval <= hi):
        raise _ToolParamError(f"`{key}` must be between {lo} and {hi}.")
    return fval


def _int(args: dict[str, Any], key: str, default: int, lo: int, hi: int) -> int:
    val = args.get(key, default)
    if isinstance(val, bool) or not isinstance(val, int):
        raise _ToolParamError(f"`{key}` must be an integer.")
    if not (lo <= val <= hi):
        raise _ToolParamError(f"`{key}` must be between {lo} and {hi}.")
    return val


def _resolve_model(args: dict[str, Any]) -> ModelSpec:
    if "total_params_b" in args or "num_layers" in args:
        preset_base = PRESET_MODELS.get(str(args.get("preset", "qwen3-35b-a3b")), PRESET_MODELS["qwen3-35b-a3b"])
        tot = _num(args, "total_params_b", preset_base.total_params_b, 0.1, 2000.0)
        act = _num(args, "active_params_b", min(tot, preset_base.active_params_b), 0.1, 2000.0)
        layers = _int(args, "num_layers", preset_base.num_layers, 1, 512)
        kv_heads = _int(args, "num_kv_heads", preset_base.num_kv_heads, 1, 256)
        hdim = _int(args, "head_dim", preset_base.head_dim, 16, 1024)
        is_moe = bool(args.get("is_moe", preset_base.is_moe))
        ref = _num(args, "routed_expert_fraction", preset_base.routed_expert_fraction if is_moe else 0.0, 0.0, 0.99)
        return ModelSpec(
            name=f"Custom ({tot:g}B)",
            total_params_b=tot,
            active_params_b=act,
            num_layers=layers,
            num_kv_heads=kv_heads,
            head_dim=hdim,
            is_moe=is_moe,
            routed_expert_fraction=ref,
        )

    preset_name = str(args.get("preset", "qwen3-35b-a3b"))
    if preset_name not in PRESET_MODELS:
        raise _ToolParamError(f"Unknown preset '{preset_name}'. Valid presets: {sorted(PRESET_MODELS)}")
    return PRESET_MODELS[preset_name]


def _call_plan_split(args: dict[str, Any]) -> dict[str, Any]:
    model = _resolve_model(args)
    vram_gib = _num(args, "vram_gib", 16.0, 2.0, 1024.0)
    ram_gib = _num(args, "ram_gib", 64.0, 4.0, 4096.0)
    os_headroom = _num(args, "os_headroom_gib", 1.25, 0.0, 32.0)
    if os_headroom >= vram_gib:
        raise _ToolParamError("`os_headroom_gib` must be smaller than `vram_gib`.")

    quant = str(args.get("quant", "q4_k_m")).lower()
    if quant not in QUANT_BITS_PER_WEIGHT:
        raise _ToolParamError(f"Unsupported `quant` '{quant}'.")
    ctx = _int(args, "context_tokens", 32768, 512, 2097152)
    kv_q = args.get("kv_quant")
    if kv_q is not None:
        kv_q = str(kv_q).lower()
        if kv_q not in KV_BYTES_PER_ELEMENT:
            raise _ToolParamError(f"Unsupported `kv_quant` '{kv_q}'.")

    hw = HardwareSpec(vram_gib=vram_gib, ram_gib=ram_gib, os_headroom_gib=os_headroom)
    payload = compute_detailed_ledger(hw, model, quant=quant, context_tokens=ctx, kv_quant=kv_q)
    return {
        "content": [{"type": "text", "text": json.dumps(payload, indent=2)}],
        "structuredContent": payload,
        "isError": False,
    }


def _call_gqa_kv(args: dict[str, Any]) -> dict[str, Any]:
    model = _resolve_model(args)
    ctx = _int(args, "context_tokens", 32768, 512, 2097152)
    kv_q = str(args.get("kv_quant", "q8_0")).lower()
    if kv_q not in KV_BYTES_PER_ELEMENT:
        raise _ToolParamError(f"Unsupported `kv_quant` '{kv_q}'.")
    mha_heads = _int(args, "counterfactual_mha_heads", 32, 1, 512)
    payload = {"ok": True, **compute_gqa_kv_breakdown(model, context_tokens=ctx, kv_quant=kv_q, counterfactual_mha_heads=mha_heads)}
    return {
        "content": [{"type": "text", "text": json.dumps(payload, indent=2)}],
        "structuredContent": payload,
        "isError": False,
    }


def _jsonrpc_result(req_id: Any, result: Any) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def _jsonrpc_error(req_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}}


def handle_rpc_message(msg: Any) -> dict[str, Any] | None:
    if not isinstance(msg, dict) or msg.get("jsonrpc") != "2.0":
        req_id = msg.get("id") if isinstance(msg, dict) and "id" in msg else None
        return _jsonrpc_error(req_id, -32600, "Invalid JSON-RPC 2.0 request.")

    method = msg.get("method")
    if not isinstance(method, str) or not method:
        return _jsonrpc_error(msg.get("id"), -32600, "Missing string `method` in request.")

    is_notification = "id" not in msg
    req_id = msg.get("id")
    params = msg.get("params")
    if params is None:
        params = {}
    elif not isinstance(params, dict):
        return None if is_notification else _jsonrpc_error(req_id, -32602, "`params` must be a JSON object.")

    if is_notification:
        return None

    if method == "initialize":
        req_ver = str(params.get("protocolVersion") or "")
        negotiated = req_ver if req_ver in SUPPORTED_PROTOCOL_VERSIONS else DEFAULT_PROTOCOL_VERSION
        return _jsonrpc_result(
            req_id,
            {
                "protocolVersion": negotiated,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": "silicon-tuner", "version": __version__},
            },
        )
    if method == "ping":
        return _jsonrpc_result(req_id, {})
    if method == "tools/list":
        return _jsonrpc_result(req_id, {"tools": list(TOOLS_SPEC)})
    if method == "resources/list":
        return _jsonrpc_result(req_id, {"resources": []})
    if method == "prompts/list":
        return _jsonrpc_result(req_id, {"prompts": []})

    if method == "tools/call":
        tool_name = params.get("name")
        tool_args = params.get("arguments") or {}
        if not isinstance(tool_name, str) or not tool_name:
            return _jsonrpc_error(req_id, -32602, "Missing string tool `name`.")
        if not isinstance(tool_args, dict):
            return _jsonrpc_error(req_id, -32602, "Tool `arguments` must be an object.")

        dispatch = {
            "plan_hybrid_moe_split": _call_plan_split,
            "compute_gqa_kv_cache": _call_gqa_kv,
        }
        handler = dispatch.get(tool_name)
        if handler is None:
            return _jsonrpc_error(req_id, -32602, f"Unknown tool: {tool_name}")

        try:
            return _jsonrpc_result(req_id, handler(tool_args))
        except _ToolParamError as exc:
            return _jsonrpc_error(req_id, -32602, str(exc))
        except Exception as exc:
            err_payload = {"ok": False, "error": str(exc)}
            return _jsonrpc_result(
                req_id,
                {
                    "content": [{"type": "text", "text": json.dumps(err_payload)}],
                    "structuredContent": err_payload,
                    "isError": True,
                },
            )

    return _jsonrpc_error(req_id, -32601, f"Method not found: {method}")


def process_payload(raw_bytes: bytes) -> bytes | None:
    try:
        parsed = json.loads(raw_bytes.decode("utf-8"))
    except Exception:
        return json.dumps(_jsonrpc_error(None, -32700, "Parse error: invalid JSON."), separators=(",", ":")).encode("utf-8")

    if isinstance(parsed, list):
        if not parsed:
            return json.dumps(_jsonrpc_error(None, -32600, "Empty JSON-RPC batch."), separators=(",", ":")).encode("utf-8")
        responses = [r for item in parsed if (r := handle_rpc_message(item)) is not None]
        return json.dumps(responses, separators=(",", ":")).encode("utf-8") if responses else None

    resp = handle_rpc_message(parsed)
    return json.dumps(resp, separators=(",", ":")).encode("utf-8") if resp is not None else None


def serve_stdio(stdin: BinaryIO | None = None, stdout: BinaryIO | None = None) -> int:
    in_stream = stdin if stdin is not None else sys.stdin.buffer
    out_stream = stdout if stdout is not None else sys.stdout.buffer

    while True:
        line = in_stream.readline()
        if not line:
            break
        stripped = line.strip()
        if not stripped:
            continue

        if stripped.lower().startswith(b"content-length:"):
            try:
                length = int(stripped.split(b":", 1)[1].strip())
            except ValueError:
                err_b = json.dumps(_jsonrpc_error(None, -32700, "Invalid Content-Length header.")).encode("utf-8")
                out_stream.write(f"Content-Length: {len(err_b)}\r\n\r\n".encode("ascii") + err_b)
                out_stream.flush()
                continue
            while True:
                hdr = in_stream.readline()
                if not hdr or hdr in (b"\r\n", b"\n"):
                    break
            if length < 0 or length > 16 * 1024 * 1024:
                err_b = json.dumps(_jsonrpc_error(None, -32600, "Content-Length out of bounds.")).encode("utf-8")
                out_stream.write(f"Content-Length: {len(err_b)}\r\n\r\n".encode("ascii") + err_b)
                out_stream.flush()
                continue
            body = in_stream.read(length)
            if not body:
                break
            resp_b = process_payload(body)
            if resp_b is not None:
                out_stream.write(f"Content-Length: {len(resp_b)}\r\n\r\n".encode("ascii") + resp_b)
                out_stream.flush()
        else:
            resp_b = process_payload(stripped)
            if resp_b is not None:
                out_stream.write(resp_b + b"\n")
                out_stream.flush()

    return 0


def main() -> int:
    return serve_stdio()


if __name__ == "__main__":
    raise SystemExit(main())
