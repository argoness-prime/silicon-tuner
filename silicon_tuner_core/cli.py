"""CLI for Silicon Tuner Open-Core v1.1.0 (Presets, Planner, Selftest & Stdio MCP Server)."""

from __future__ import annotations

import argparse
import json

from .mcp_server import serve_stdio
from .planner import (
    HardwareSpec,
    PRESET_MODELS,
    compute_detailed_ledger,
    compute_gqa_kv_breakdown,
)


def run_selftest() -> dict:
    checks: list[tuple[str, bool]] = []
    hw = HardwareSpec(vram_gib=16.0, ram_gib=64.0, os_headroom_gib=1.25)
    m = PRESET_MODELS["qwen3-35b-a3b"]
    kv = compute_gqa_kv_breakdown(m, context_tokens=32768, kv_quant="q8_0")
    checks.append(("gqa_32k_exact_elements_1610612736", kv["totalKvElements"] == 1_610_612_736))
    checks.append(("gqa_32k_q8_0_1_59_gib", kv["kvCacheGib"] == 1.59))
    checks.append(("counterfactual_mha_fp16_24_gib", kv["counterfactualMhaFp16Gib"] == 24.0))

    ledger = compute_detailed_ledger(hw, m, quant="q4_k_m", context_tokens=32768, kv_quant="q8_0")
    checks.append(("moe_35b_q4km_gpu_app_14_64_gib", ledger["gpuAppLedger"]["gpuAppUsedGib"] == 14.64))
    checks.append(("moe_35b_q4km_gpu_budget_14_75_gib", ledger["gpuAppLedger"]["gpuAppBudgetGib"] == 14.75))
    checks.append(("moe_35b_q4km_physical_15_89_gib", ledger["physicalVramLedger"]["physicalUsedGib"] == 15.89))
    checks.append(("moe_35b_q4km_naive_24_56_gib", ledger["physicalVramLedger"]["naiveFullGpuFp16PhysicalGib"] == 24.56))
    checks.append(("moe_35b_q4km_cpu_moe_21", ledger["plan"]["cpu_moe_layers"] == 21))

    passed = sum(1 for _, ok in checks if ok)
    return {
        "ok": passed == len(checks),
        "score": f"{passed}/{len(checks)}",
        "passed": passed,
        "total": len(checks),
        "checks": [{"name": n, "passed": ok} for n, ok in checks],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="silicon-tuner-core", description="Silicon Tuner Open-Core v1.1.0 VRAM & MoE Calculator")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("presets", help="List built-in Dense & MoE model presets")

    p_plan = sub.add_parser("plan", help="Compute GQA KV-cache + MoE cold-expert split with separated GPU App & Physical DWM ledgers")
    p_plan.add_argument("--preset", choices=sorted(PRESET_MODELS), default="qwen3-35b-a3b")
    p_plan.add_argument("--vram", type=float, default=16.0)
    p_plan.add_argument("--ram", type=float, default=64.0)
    p_plan.add_argument("--quant", default="q4_k_m")
    p_plan.add_argument("--ctx", type=int, default=32768)
    p_plan.add_argument("--kv-quant", default=None)

    sub.add_parser("mcp", help="Run zero-dependency JSON-RPC 2.0 stdio MCP server")
    sub.add_parser("selftest", help="Run offline 8-check deterministic math verification suite")

    args = parser.parse_args(argv)
    if args.cmd == "mcp":
        return serve_stdio()
    if args.cmd == "selftest":
        res = run_selftest()
        print(json.dumps(res, indent=2))
        return 0 if res["ok"] else 1
    if args.cmd == "presets":
        print(json.dumps({k: v.__dict__ for k, v in PRESET_MODELS.items()}, indent=2))
        return 0
    if args.cmd == "plan":
        hw = HardwareSpec(vram_gib=args.vram, ram_gib=args.ram)
        out = compute_detailed_ledger(hw, PRESET_MODELS[args.preset], quant=args.quant, context_tokens=args.ctx, kv_quant=args.kv_quant)
        print(json.dumps(out, indent=2))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
