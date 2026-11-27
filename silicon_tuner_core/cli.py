"""CLI for Silicon Tuner Open-Core."""

from __future__ import annotations

import argparse
import json

from .planner import HardwareSpec, PRESET_MODELS, compute_tune_plan


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="silicon-tuner-core", description="Silicon Tuner Open-Core VRAM & MoE Calculator")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("presets", help="List built-in Dense & MoE model presets")
    p_plan = sub.add_parser("plan", help="Compute GQA KV-cache + MoE cold-expert split")
    p_plan.add_argument("--preset", choices=sorted(PRESET_MODELS), default="qwen3-35b-a3b")
    p_plan.add_argument("--vram", type=float, default=16.0)
    p_plan.add_argument("--ram", type=float, default=64.0)
    p_plan.add_argument("--quant", default="iq3_m")
    p_plan.add_argument("--ctx", type=int, default=32768)

    args = parser.parse_args(argv)
    if args.cmd == "presets":
        print(json.dumps({k: v.__dict__ for k, v in PRESET_MODELS.items()}, indent=2))
        return 0
    if args.cmd == "plan":
        hw = HardwareSpec(vram_gib=args.vram, ram_gib=args.ram)
        plan = compute_tune_plan(hw, PRESET_MODELS[args.preset], quant=args.quant, context_tokens=args.ctx)
        print(json.dumps({"ok": True, "plan": plan.to_dict()}, indent=2))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
