"""Silicon Tuner Open-Core — GQA KV-Cache & MoE GPU/RAM Planner + Stdio MCP Server."""

from .planner import (
    HardwareSpec,
    ModelSpec,
    PRESET_MODELS,
    TunePlan,
    compute_detailed_ledger,
    compute_gqa_kv_breakdown,
    compute_kv_cache_gib,
    compute_tune_plan,
)

__all__ = [
    "HardwareSpec",
    "ModelSpec",
    "PRESET_MODELS",
    "TunePlan",
    "compute_detailed_ledger",
    "compute_gqa_kv_breakdown",
    "compute_kv_cache_gib",
    "compute_tune_plan",
]
__version__ = "1.1.0"
