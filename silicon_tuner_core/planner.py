"""Exact Dense & MoE VRAM + System RAM Layer-Split & GQA KV-Cache Planner."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

QUANT_BITS_PER_WEIGHT: dict[str, float] = {
    "iq3_m": 3.45,
    "q3_k_m": 3.90,
    "iq4_xs": 4.25,
    "q4_k_m": 4.85,
    "q5_k_m": 5.70,
    "q6_k": 6.55,
    "q8_0": 8.50,
    "fp8": 8.25,
    "fp16": 16.0,
    "bf16": 16.0,
}

KV_BYTES_PER_ELEMENT: dict[str, float] = {
    "f16": 2.0,
    "bf16": 2.0,
    "q8_0": 1.06,
    "q4_0": 0.56,
}


@dataclass(frozen=True)
class HardwareSpec:
    vram_gib: float = 16.0
    ram_gib: float = 64.0
    vram_bw_gbps: float = 672.0
    ram_bw_gbps: float = 102.4  # Dual-channel 64-bit DDR5-6400 theoretical peak (2 * 8 B * 6.4 GT/s)
    os_headroom_gib: float = 1.25
    has_npu_sidecar: bool = False


@dataclass(frozen=True)
class ModelSpec:
    name: str
    total_params_b: float
    active_params_b: float
    num_layers: int
    num_kv_heads: int
    head_dim: int
    is_moe: bool = False
    routed_expert_fraction: float = 0.0


PRESET_MODELS: dict[str, ModelSpec] = {
    "qwen3-35b-a3b": ModelSpec(
        name="Qwen3.6-35B-A3B (MoE)",
        total_params_b=35.0,
        active_params_b=3.2,
        num_layers=48,
        num_kv_heads=4,
        head_dim=128,
        is_moe=True,
        routed_expert_fraction=0.84,
    ),
    "qwen3-30b-a3b": ModelSpec(
        name="Qwen3-30B-A3B (MoE)",
        total_params_b=30.5,
        active_params_b=3.3,
        num_layers=48,
        num_kv_heads=4,
        head_dim=128,
        is_moe=True,
        routed_expert_fraction=0.83,
    ),
    "llama3-70b-dense": ModelSpec(
        name="Llama-3.3-70B (Dense)",
        total_params_b=70.6,
        active_params_b=70.6,
        num_layers=80,
        num_kv_heads=8,
        head_dim=128,
        is_moe=False,
        routed_expert_fraction=0.0,
    ),
    "qwen3-14b-dense": ModelSpec(
        name="Qwen3-14B (Dense)",
        total_params_b=14.8,
        active_params_b=14.8,
        num_layers=40,
        num_kv_heads=8,
        head_dim=128,
        is_moe=False,
        routed_expert_fraction=0.0,
    ),
}


@dataclass
class TunePlan:
    model_name: str
    quant: str
    kv_quant: str
    context_tokens: int
    fits_memory: bool
    fit_mode: str
    gpu_layers: int
    total_layers: int
    cpu_moe_layers: int
    vram_used_gib: float
    vram_budget_gib: float
    ram_offload_gib: float
    kv_cache_gib: float
    weights_total_gib: float
    estimated_decode_tps: float
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _gib_from_bytes(num_bytes: float) -> float:
    return num_bytes / (1024.0 ** 3)


def compute_kv_cache_gib(model: ModelSpec, context_tokens: int, kv_quant: str) -> float:
    bpe = KV_BYTES_PER_ELEMENT.get(kv_quant.lower(), 2.0)
    # 2 tensors (K + V) * layers * kv_heads * head_dim * tokens * bytes_per_elem
    total_bytes = 2.0 * model.num_layers * model.num_kv_heads * model.head_dim * context_tokens * bpe
    return _gib_from_bytes(total_bytes)


def compute_gqa_kv_breakdown(
    model: ModelSpec,
    context_tokens: int = 32768,
    kv_quant: str = "q8_0",
    counterfactual_mha_heads: int = 32,
) -> dict[str, Any]:
    """Compute exact GQA KV-cache element counts (e.g. 1,610,612,736 elements) and byte/GiB totals."""
    kv_q = kv_quant.lower()
    if kv_q not in KV_BYTES_PER_ELEMENT:
        raise ValueError(f"Unsupported kv_quant '{kv_quant}'. Supported: {sorted(KV_BYTES_PER_ELEMENT)}")

    total_elements = int(2 * model.num_layers * model.num_kv_heads * model.head_dim * context_tokens)
    bpe = KV_BYTES_PER_ELEMENT[kv_q]
    total_bytes = total_elements * bpe
    kv_gib = _gib_from_bytes(total_bytes)

    fp16_gib = _gib_from_bytes(total_elements * 2.0)
    mha_elements = int(2 * model.num_layers * counterfactual_mha_heads * model.head_dim * context_tokens)
    mha_fp16_gib = _gib_from_bytes(mha_elements * 2.0)
    savings_pct = (1.0 - (total_bytes / max(1.0, mha_elements * 2.0))) * 100.0

    quant_table = {
        q_name: round(_gib_from_bytes(total_elements * q_bpe), 2)
        for q_name, q_bpe in KV_BYTES_PER_ELEMENT.items()
    }
    return {
        "modelName": model.name,
        "numLayers": model.num_layers,
        "numKvHeads": model.num_kv_heads,
        "headDim": model.head_dim,
        "contextTokens": context_tokens,
        "kvQuant": kv_q,
        "bytesPerElement": bpe,
        "formula": "2 (K+V) * num_layers * num_kv_heads * head_dim * context_tokens",
        "totalKvElements": total_elements,
        "kvCacheBytes": int(round(total_bytes)),
        "kvCacheGib": round(kv_gib, 2),
        "gqaFp16Gib": round(fp16_gib, 2),
        "counterfactualMhaHeads": counterfactual_mha_heads,
        "counterfactualMhaElements": mha_elements,
        "counterfactualMhaFp16Gib": round(mha_fp16_gib, 2),
        "savingsVsMhaFp16Pct": round(savings_pct, 2),
        "allKvQuantsGib": quant_table,
    }


def compute_tune_plan(
    hw: HardwareSpec,
    model: ModelSpec,
    *,
    quant: str = "q4_k_m",
    context_tokens: int = 32768,
    kv_quant: str | None = None,
) -> TunePlan:
    q_key = quant.lower()
    if q_key not in QUANT_BITS_PER_WEIGHT:
        raise ValueError(f"Unsupported quant '{quant}'. Supported: {sorted(QUANT_BITS_PER_WEIGHT)}")

    bits = QUANT_BITS_PER_WEIGHT[q_key]
    weights_bytes = model.total_params_b * 1e9 * (bits / 8.0)
    weights_gib = _gib_from_bytes(weights_bytes)

    usable_vram = max(0.5, hw.vram_gib - hw.os_headroom_gib)
    usable_ram = max(1.0, hw.ram_gib - 4.0)
    compute_buffer_gib = 0.55

    chosen_kv = (kv_quant or ("q8_0" if weights_gib + 1.5 > usable_vram else "f16")).lower()
    if chosen_kv not in KV_BYTES_PER_ELEMENT:
        raise ValueError(f"Unsupported kv_quant '{chosen_kv}'. Supported: {sorted(KV_BYTES_PER_ELEMENT)}")
    kv_gib = compute_kv_cache_gib(model, context_tokens, chosen_kv)
    if weights_gib + kv_gib + compute_buffer_gib > usable_vram and kv_quant is None and chosen_kv != "q8_0":
        chosen_kv = "q8_0"
        kv_gib = compute_kv_cache_gib(model, context_tokens, chosen_kv)

    notes: list[str] = []
    if hw.has_npu_sidecar:
        notes.append("Route RAG embeddings/rerankers to NPU sidecar to preserve 100% of GPU VRAM for KV + attention.")

    # Case 1: Entire model + KV + compute fits inside GPU VRAM
    full_gpu_vram = weights_gib + kv_gib + compute_buffer_gib
    if full_gpu_vram <= usable_vram:
        active_weight_bytes = model.active_params_b * 1e9 * (bits / 8.0)
        est_tps = (hw.vram_bw_gbps * 1e9 * 0.62) / max(1.0, active_weight_bytes)
        notes.append("100% VRAM resident: zero PCIe/DDR5 expert traffic during token decode.")
        return TunePlan(
            model_name=model.name,
            quant=q_key,
            kv_quant=chosen_kv,
            context_tokens=context_tokens,
            fits_memory=True,
            fit_mode="full_gpu_vram",
            gpu_layers=model.num_layers,
            total_layers=model.num_layers,
            cpu_moe_layers=0,
            vram_used_gib=round(full_gpu_vram, 2),
            vram_budget_gib=round(usable_vram, 2),
            ram_offload_gib=0.0,
            kv_cache_gib=round(kv_gib, 2),
            weights_total_gib=round(weights_gib, 2),
            estimated_decode_tps=round(est_tps, 1),
            notes=notes,
        )

    # Case 2: MoE Selective Expert Offload (keep attention + router + KV in VRAM, spill cold experts to DDR5)
    if model.is_moe and model.routed_expert_fraction > 0.0:
        shared_gib = weights_gib * (1.0 - model.routed_expert_fraction)
        experts_gib = weights_gib * model.routed_expert_fraction
        expert_per_layer_gib = experts_gib / model.num_layers
        vram_for_experts = usable_vram - (shared_gib + kv_gib + compute_buffer_gib)

        if vram_for_experts >= 0.0:
            gpu_expert_layers = min(model.num_layers, int(math.floor(vram_for_experts / max(1e-6, expert_per_layer_gib))))
            cpu_moe_layers = model.num_layers - gpu_expert_layers
            ram_needed = cpu_moe_layers * expert_per_layer_gib
            vram_used = shared_gib + (gpu_expert_layers * expert_per_layer_gib) + kv_gib + compute_buffer_gib
            fits = ram_needed <= usable_ram

            gpu_frac = gpu_expert_layers / model.num_layers
            active_bytes = model.active_params_b * 1e9 * (bits / 8.0)
            shared_active_bytes = active_bytes * 0.30
            routed_active_bytes = active_bytes * 0.70
            t_gpu = (shared_active_bytes + routed_active_bytes * gpu_frac) / (hw.vram_bw_gbps * 1e9 * 0.60)
            t_cpu = (routed_active_bytes * (1.0 - gpu_frac)) / (hw.ram_bw_gbps * 1e9 * 0.72)
            est_tps = 1.0 / max(1e-6, t_gpu + t_cpu)
            notes.append(
                f"Hybrid MoE split: attention + KV cache + {gpu_expert_layers}/{model.num_layers} expert layers in VRAM; "
                f"{cpu_moe_layers} cold expert FFN layers in DDR5."
            )
            return TunePlan(
                model_name=model.name,
                quant=q_key,
                kv_quant=chosen_kv,
                context_tokens=context_tokens,
                fits_memory=fits,
                fit_mode="hybrid_moe_expert_split",
                gpu_layers=model.num_layers,
                total_layers=model.num_layers,
                cpu_moe_layers=cpu_moe_layers,
                vram_used_gib=round(vram_used, 2),
                vram_budget_gib=round(usable_vram, 2),
                ram_offload_gib=round(ram_needed, 2),
                kv_cache_gib=round(kv_gib, 2),
                weights_total_gib=round(weights_gib, 2),
                estimated_decode_tps=round(est_tps, 1),
                notes=notes,
            )

    # Case 3: Standard Dense / Full-Layer GPU+RAM Split
    per_layer_gib = weights_gib / model.num_layers
    vram_for_layers = max(0.0, usable_vram - (kv_gib + compute_buffer_gib))
    gpu_layers = min(model.num_layers, int(math.floor(vram_for_layers / max(1e-6, per_layer_gib))))
    cpu_layers = model.num_layers - gpu_layers
    ram_needed = cpu_layers * per_layer_gib + (0.0 if gpu_layers > 0 else kv_gib)
    vram_used = (gpu_layers * per_layer_gib) + (kv_gib + compute_buffer_gib if gpu_layers > 0 else 0.0)
    fits = ram_needed <= usable_ram

    gpu_frac = gpu_layers / model.num_layers
    active_bytes = model.active_params_b * 1e9 * (bits / 8.0)
    t_gpu = (active_bytes * gpu_frac) / (hw.vram_bw_gbps * 1e9 * 0.58)
    t_cpu = (active_bytes * (1.0 - gpu_frac)) / (hw.ram_bw_gbps * 1e9 * 0.68)
    est_tps = 1.0 / max(1e-6, t_gpu + t_cpu)
    notes.append(f"Dense layer split: {gpu_layers}/{model.num_layers} layers on GPU, {cpu_layers} layers in System RAM.")
    if not fits:
        notes.append("WARNING: Model + KV exceeds combined usable VRAM + System RAM; choose a tighter quant (e.g. iq3_m / iq4_xs).")

    return TunePlan(
        model_name=model.name,
        quant=q_key,
        kv_quant=chosen_kv,
        context_tokens=context_tokens,
        fits_memory=fits,
        fit_mode="hybrid_layer_split",
        gpu_layers=gpu_layers,
        total_layers=model.num_layers,
        cpu_moe_layers=0,
        vram_used_gib=round(vram_used, 2),
        vram_budget_gib=round(usable_vram, 2),
        ram_offload_gib=round(ram_needed, 2),
        kv_cache_gib=round(kv_gib, 2),
        weights_total_gib=round(weights_gib, 2),
        estimated_decode_tps=round(est_tps, 1),
        notes=notes,
    )


def compute_detailed_ledger(
    hw: HardwareSpec,
    model: ModelSpec,
    *,
    quant: str = "q4_k_m",
    context_tokens: int = 32768,
    kv_quant: str | None = None,
) -> dict[str, Any]:
    """Return TunePlan alongside explicitly separated GPU App Budget vs Physical VRAM (with OS/DWM reserve) ledgers."""
    plan = compute_tune_plan(hw, model, quant=quant, context_tokens=context_tokens, kv_quant=kv_quant)
    compute_buffer_gib = 0.55 if plan.gpu_layers > 0 else 0.0
    kv_breakdown = compute_gqa_kv_breakdown(model, context_tokens=context_tokens, kv_quant=plan.kv_quant)

    bits = QUANT_BITS_PER_WEIGHT[plan.quant]
    weights_gib = _gib_from_bytes(model.total_params_b * 1e9 * (bits / 8.0))
    fp16_kv_gib = compute_kv_cache_gib(model, context_tokens, "f16")
    naive_physical_gib = weights_gib + fp16_kv_gib + 0.55 + hw.os_headroom_gib

    if plan.fit_mode == "hybrid_moe_expert_split":
        shared_gib = weights_gib * (1.0 - model.routed_expert_fraction)
        gpu_expert_layers = model.num_layers - plan.cpu_moe_layers
        expert_per_layer_gib = (weights_gib * model.routed_expert_fraction) / model.num_layers
        gpu_experts_gib = gpu_expert_layers * expert_per_layer_gib
    elif plan.fit_mode == "full_gpu_vram":
        shared_gib = weights_gib * (1.0 - model.routed_expert_fraction) if model.is_moe else weights_gib
        gpu_experts_gib = weights_gib * model.routed_expert_fraction if model.is_moe else 0.0
    else:
        shared_gib = (plan.gpu_layers / model.num_layers) * weights_gib
        gpu_experts_gib = 0.0

    gpu_app_used = round(plan.vram_used_gib, 2)
    gpu_app_budget = round(plan.vram_budget_gib, 2)
    os_dwm = round(hw.os_headroom_gib, 2)
    phys_used = round(os_dwm + gpu_app_used, 2)

    llama_flags = [f"-ngl {plan.gpu_layers}", f"-c {context_tokens}", f"-ctk {plan.kv_quant}", f"-ctv {plan.kv_quant}"]
    if plan.cpu_moe_layers > 0:
        llama_flags.insert(1, f"--n-cpu-moe {plan.cpu_moe_layers}")

    return {
        "ok": True,
        "plan": plan.to_dict(),
        "gpuAppLedger": {
            "sharedAttentionRouterGib": round(shared_gib, 2),
            "gpuExpertLayersGib": round(gpu_experts_gib, 2),
            "kvCacheGib": round(plan.kv_cache_gib, 2),
            "computeBufferGib": round(compute_buffer_gib, 2),
            "gpuAppUsedGib": gpu_app_used,
            "gpuAppBudgetGib": gpu_app_budget,
        },
        "physicalVramLedger": {
            "osDwmReserveGib": os_dwm,
            "gpuAppUsedGib": gpu_app_used,
            "physicalUsedGib": phys_used,
            "physicalTotalGib": round(hw.vram_gib, 2),
            "naiveFullGpuFp16PhysicalGib": round(naive_physical_gib, 2),
        },
        "kvCacheBreakdown": kv_breakdown,
        "llamaCppFlags": " ".join(llama_flags),
    }
