"""Silicon Tuner Open-Core — GQA KV-Cache & MoE GPU/RAM Planner."""

from .planner import HardwareSpec, ModelSpec, PRESET_MODELS, TunePlan, compute_tune_plan

__all__ = ["HardwareSpec", "ModelSpec", "PRESET_MODELS", "TunePlan", "compute_tune_plan"]
__version__ = "1.0.0"
