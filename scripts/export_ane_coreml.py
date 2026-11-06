#!/usr/bin/env python3
"""
Export EmbeddingGemma 2 (`embeddinggemma-2-MLX-BF16`) sub-models to Apple Neural Engine (ANE)
CoreML `.mlpackage` / `.mlmodelc` bundles for Argmax Playground-style per-tower placement in `mbed`.

Exports three independent CoreML bundles matching `EmbeddingComponent.coreMLBundleName`:
1. `EmbeddingGemma2VisionANE.mlpackage` (`vision_tower` + `embed_vision`, 167.7M params)
2. `EmbeddingGemma2AudioANE.mlpackage` (`audio_tower` + `embed_audio`, 305.4M params)
3. `EmbeddingGemma2TextANE.mlpackage` (`language_model` 24L + `PLE` + `embedding_projection`, 271.0M params)
   plus `embed_tokens_fp16.bin` (`[262144, 512]` float16 token embedding table for O(L) CPU lookup).
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import Sequence

VISION_RESOLUTION_BUCKETS: list[tuple[int, int]] = [
    (480, 672),
    (672, 480),
    (528, 528),
    (672, 960),
    (960, 672),
    (768, 768),
]

TEXT_SEQUENCE_BUCKETS: list[int] = [512]
FP16_SAFE_RMS_PRESCALE: float = 0.0625  # 1/16: keeps (x * alpha)^2 < 65,504 for |x| <= 4,096


def inspect_checkpoint(checkpoint_dir: Path) -> dict:
    config_path = checkpoint_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Checkpoint config.json not found at {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_checkpoint_state(checkpoint_dir: Path) -> dict[str, "torch.Tensor"]:
    from safetensors.torch import load_file

    state: dict[str, "torch.Tensor"] = {}
    for sf in sorted(checkpoint_dir.glob("*.safetensors")):
        shard = load_file(str(sf))
        for k, v in shard.items():
            clean = k[len("model."):] if k.startswith("model.") else k
            state[clean] = v
    return state


def get_tensor(state: dict, *candidates: str):
    for c in candidates:
        if c in state:
            return state[c].float()
    raise KeyError(f"None of {candidates} found in checkpoint state")


def make_ane_safe_rms_norm_cls(torch, nn):
    class ANESafeRMSNorm(nn.Module):
        def __init__(self, dim: int, eps: float = 1e-6, with_scale: bool = True):
            super().__init__()
            self.scaled_eps = float(eps) * (FP16_SAFE_RMS_PRESCALE ** 2)
            self.with_scale = with_scale
            if with_scale:
                self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))

        def forward(self, x):
            scaled = x * FP16_SAFE_RMS_PRESCALE
            inv_rms = torch.rsqrt(torch.mean(scaled * scaled, dim=-1, keepdim=True) + self.scaled_eps)
            normed = scaled * inv_rms
            return normed * self.weight if self.with_scale else normed

    return ANESafeRMSNorm


def precompute_2d_axial_rope(torch, max_grid: int, head_dim: int = 64, base: float = 100.0):
    # Split head_dim=64 into 32 channels for X and 32 channels for Y (each with half=16).
    ch_per_dim = head_dim // 2  # 32
    half = ch_per_dim // 2      # 16
    freq_exponents = (2.0 / float(ch_per_dim)) * torch.arange(half, dtype=torch.float32)
    timescale = torch.pow(torch.tensor(base, dtype=torch.float32), freq_exponents)  # [16]
    pos = torch.arange(max_grid, dtype=torch.float32).unsqueeze(-1)  # [max_grid, 1]
    sinusoid = pos / timescale  # [max_grid, 16]
    cos_1d = torch.cat([torch.cos(sinusoid), torch.cos(sinusoid)], dim=-1)  # [max_grid, 32]
    sin_1d = torch.cat([torch.sin(sinusoid), torch.sin(sinusoid)], dim=-1)  # [max_grid, 32]
    return cos_1d, sin_1d


def precompute_1d_rope(torch, max_seq: int, head_dim: int, base: float):
    half = head_dim // 2
    freq_exponents = (2.0 / float(head_dim)) * torch.arange(half, dtype=torch.float32)
    timescale = torch.pow(torch.tensor(base, dtype=torch.float32), freq_exponents)
    pos = torch.arange(max_seq, dtype=torch.float32).unsqueeze(-1)
    sinusoid = pos / timescale
    cos_1d = torch.cat([torch.cos(sinusoid), torch.cos(sinusoid)], dim=-1)  # [max_seq, head_dim]
    sin_1d = torch.cat([torch.sin(sinusoid), torch.sin(sinusoid)], dim=-1)  # [max_seq, head_dim]
    return cos_1d, sin_1d


def rotate_half(torch, x):
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def export_vision_tower(checkpoint_dir: Path, state: dict, out_dir: Path, compile_modelc: bool) -> Path:
    import coremltools as ct
    import torch
    import torch.nn as nn

    cfg = inspect_checkpoint(checkpoint_dir)
    vcfg = cfg.get("vision_config", {})
    hidden_size = int(vcfg.get("hidden_size", 768))
    intermediate_size = int(vcfg.get("intermediate_size", 3072))
    num_layers = int(vcfg.get("num_hidden_layers", 16))
    num_heads = int(vcfg.get("num_attention_heads", 12))
    head_dim = int(vcfg.get("head_dim", 64))
    text_hidden = int(cfg.get("text_config", {}).get("hidden_size", 512))
    eps = float(vcfg.get("rms_norm_eps", 1e-6))
    rope_theta = float(vcfg.get("rope_parameters", {}).get("rope_theta", 100.0))

    ANESafeRMSNorm = make_ane_safe_rms_norm_cls(torch, nn)
    max_grid = 64  # Covers up to 1024x1024 (max bucket is 960 -> 60 patches)
    cos_1d, sin_1d = precompute_2d_axial_rope(torch, max_grid=max_grid, head_dim=head_dim, base=rope_theta)

    class VisionAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
            self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
            self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
            self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
            self.q_norm = ANESafeRMSNorm(head_dim, eps=eps, with_scale=True)
            self.k_norm = ANESafeRMSNorm(head_dim, eps=eps, with_scale=True)
            self.v_norm = ANESafeRMSNorm(head_dim, eps=eps, with_scale=False)

        def forward(self, x: torch.Tensor, rope_cos: torch.Tensor, rope_sin: torch.Tensor) -> torch.Tensor:
            q = self.q_norm(self.q_proj(x).view(1, -1, num_heads, head_dim)).transpose(1, 2)  # [1, H, N, D]
            k = self.k_norm(self.k_proj(x).view(1, -1, num_heads, head_dim)).transpose(1, 2)
            v = self.v_norm(self.v_proj(x).view(1, -1, num_heads, head_dim)).transpose(1, 2)

            qx, qy = q[..., :32], q[..., 32:]
            kx, ky = k[..., :32], k[..., 32:]
            cos_x, cos_y = rope_cos[..., :32], rope_cos[..., 32:]
            sin_x, sin_y = rope_sin[..., :32], rope_sin[..., 32:]

            q = torch.cat([
                qx * cos_x + rotate_half(torch, qx) * sin_x,
                qy * cos_y + rotate_half(torch, qy) * sin_y,
            ], dim=-1)
            k = torch.cat([
                kx * cos_x + rotate_half(torch, kx) * sin_x,
                ky * cos_y + rotate_half(torch, ky) * sin_y,
            ], dim=-1)

            attn = torch.matmul(q, k.transpose(-2, -1)).softmax(dim=-1)
            out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(1, -1, hidden_size)
            return self.o_proj(out)

    class VisionBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_layernorm = ANESafeRMSNorm(hidden_size, eps=eps)
            self.self_attn = VisionAttention()
            self.post_attention_layernorm = ANESafeRMSNorm(hidden_size, eps=eps)
            self.pre_feedforward_layernorm = ANESafeRMSNorm(hidden_size, eps=eps)
            self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
            self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
            self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
            self.post_feedforward_layernorm = ANESafeRMSNorm(hidden_size, eps=eps)

        def forward(self, x: torch.Tensor, rope_cos: torch.Tensor, rope_sin: torch.Tensor) -> torch.Tensor:
            x = x + self.post_attention_layernorm(self.self_attn(self.input_layernorm(x), rope_cos, rope_sin))
            ff = self.pre_feedforward_layernorm(x)
            act = torch.nn.functional.gelu(self.gate_proj(ff), approximate="tanh") * self.up_proj(ff)
            x = x + self.post_feedforward_layernorm(self.down_proj(act))
            return x

    class EmbeddingGemma2VisionANEWrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.patch_conv = nn.Conv2d(3, hidden_size, kernel_size=16, stride=16, bias=False)
            self.register_buffer("pos_table_x", torch.zeros(1, hidden_size, 1, max_grid))
            self.register_buffer("pos_table_y", torch.zeros(1, hidden_size, max_grid, 1))
            self.register_buffer("cos_x_table", cos_1d.view(1, 1, 1, max_grid, 32))
            self.register_buffer("sin_x_table", sin_1d.view(1, 1, 1, max_grid, 32))
            self.register_buffer("cos_y_table", cos_1d.view(1, 1, max_grid, 1, 32))
            self.register_buffer("sin_y_table", sin_1d.view(1, 1, max_grid, 1, 32))
            self.blocks = nn.ModuleList([VisionBlock() for _ in range(num_layers)])
            self.pool = nn.AvgPool2d(kernel_size=3, stride=3)
            self.embed_norm = ANESafeRMSNorm(hidden_size, eps=eps, with_scale=False)
            self.embed_proj = nn.Linear(hidden_size, text_hidden, bias=False)

        def forward(self, pixels: torch.Tensor) -> torch.Tensor:
            norm_px = 2.0 * (pixels - 0.5)
            patches = self.patch_conv(norm_px)  # [1, 768, H_p, W_p]
            hp = patches.shape[2]
            wp = patches.shape[3]
            patches = patches + self.pos_table_x[:, :, :, :wp] + self.pos_table_y[:, :, :hp, :]
            seq = patches.flatten(2).transpose(1, 2)  # [1, N_patches, 768]

            # Build 2D Axial RoPE tables via 2D broadcast without gather or dynamic view ints
            grid_zero = patches[:, :1, :, :].unsqueeze(-1) * 0.0  # [1, 1, H_p, W_p, 1]
            cx = self.cos_x_table[:, :, :, :wp, :] + grid_zero
            sx = self.sin_x_table[:, :, :, :wp, :] + grid_zero
            cy = self.cos_y_table[:, :, :hp, :, :] + grid_zero
            sy = self.sin_y_table[:, :, :hp, :, :] + grid_zero
            rope_cos = torch.cat([cx, cy], dim=-1).flatten(2, 3)  # [1, 1, N_patches, 64]
            rope_sin = torch.cat([sx, sy], dim=-1).flatten(2, 3)  # [1, 1, N_patches, 64]

            for blk in self.blocks:
                seq = blk(seq, rope_cos, rope_sin)

            grid = seq.transpose(1, 2).reshape(1, hidden_size, hp, wp)
            pooled = self.pool(grid).flatten(2).transpose(1, 2)  # [1, N_soft, 768]
            return self.embed_proj(self.embed_norm(pooled))  # [1, N_soft, 512]

    model = EmbeddingGemma2VisionANEWrapper().eval()

    with torch.no_grad():
        # Patchify in Gemma4VisionPatchEmbedder transposes to [B, Hp, Wp, 16, 16, 3] before Linear(768, 768).
        w_patch = get_tensor(state, "vision_tower.patch_embedder.input_proj.weight")
        model.patch_conv.weight.copy_(w_patch.view(hidden_size, 16, 16, 3).permute(0, 3, 1, 2).contiguous())

        pos_tab = get_tensor(state, "vision_tower.patch_embedder.position_embedding_table")  # [2, 10240, 768]
        model.pos_table_x.copy_(pos_tab[0, :max_grid, :].transpose(0, 1).view(1, hidden_size, 1, max_grid))
        model.pos_table_y.copy_(pos_tab[1, :max_grid, :].transpose(0, 1).view(1, hidden_size, max_grid, 1))

        for i, blk in enumerate(model.blocks):
            p = f"vision_tower.encoder.layers.{i}"
            blk.input_layernorm.weight.copy_(get_tensor(state, f"{p}.input_layernorm.weight"))
            blk.post_attention_layernorm.weight.copy_(get_tensor(state, f"{p}.post_attention_layernorm.weight"))
            blk.pre_feedforward_layernorm.weight.copy_(get_tensor(state, f"{p}.pre_feedforward_layernorm.weight"))
            blk.post_feedforward_layernorm.weight.copy_(get_tensor(state, f"{p}.post_feedforward_layernorm.weight"))

            blk.self_attn.q_proj.weight.copy_(get_tensor(
                state, f"{p}.self_attn.q_proj.linear.weight", f"{p}.self_attn.q_proj.weight"
            ))
            blk.self_attn.k_proj.weight.copy_(get_tensor(
                state, f"{p}.self_attn.k_proj.linear.weight", f"{p}.self_attn.k_proj.weight"
            ))
            blk.self_attn.v_proj.weight.copy_(get_tensor(
                state, f"{p}.self_attn.v_proj.linear.weight", f"{p}.self_attn.v_proj.weight"
            ))
            blk.self_attn.o_proj.weight.copy_(get_tensor(
                state, f"{p}.self_attn.o_proj.linear.weight", f"{p}.self_attn.o_proj.weight"
            ))
            blk.self_attn.q_norm.weight.copy_(get_tensor(state, f"{p}.self_attn.q_norm.weight"))
            blk.self_attn.k_norm.weight.copy_(get_tensor(state, f"{p}.self_attn.k_norm.weight"))

            blk.gate_proj.weight.copy_(get_tensor(
                state, f"{p}.mlp.gate_proj.linear.weight", f"{p}.mlp.gate_proj.weight"
            ))
            blk.up_proj.weight.copy_(get_tensor(
                state, f"{p}.mlp.up_proj.linear.weight", f"{p}.mlp.up_proj.weight"
            ))
            blk.down_proj.weight.copy_(get_tensor(
                state, f"{p}.mlp.down_proj.linear.weight", f"{p}.mlp.down_proj.weight"
            ))

        model.embed_proj.weight.copy_(get_tensor(state, "embed_vision.embedding_projection.weight"))

    example_input = torch.zeros(1, 3, 480, 672, dtype=torch.float32)
    traced = torch.jit.trace(model, example_input)

    enum_shapes = ct.EnumeratedShapes(
        shapes=[(1, 3, h, w) for (h, w) in VISION_RESOLUTION_BUCKETS],
        default=(1, 3, 480, 672),
    )
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="pixels", shape=enum_shapes)],
        outputs=[ct.TensorType(name="soft_tokens")],
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.macOS15,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    pkg_path = out_dir / "EmbeddingGemma2VisionANE.mlpackage"
    mlmodel.save(str(pkg_path))
    if compile_modelc:
        subprocess.run(
            ["xcrun", "coremlcompiler", "compile", str(pkg_path), str(out_dir)],
            check=True,
        )
    return pkg_path


def export_text_backbone(checkpoint_dir: Path, state: dict, out_dir: Path, compile_modelc: bool) -> Path:
    import coremltools as ct
    import torch
    import torch.nn as nn

    cfg = inspect_checkpoint(checkpoint_dir)
    tcfg = cfg.get("text_config", cfg)
    hidden_size = int(tcfg.get("hidden_size", 512))
    num_layers = int(tcfg.get("num_hidden_layers", 24))
    num_heads = int(tcfg.get("num_attention_heads", 4))
    default_kv_heads = int(tcfg.get("num_key_value_heads", 2))
    default_head_dim = int(tcfg.get("head_dim", 256))
    intermediate_size = int(tcfg.get("intermediate_size", 2048))
    ple_size = int(tcfg.get("hidden_size_per_layer_input", 512))
    embedding_dim = int(tcfg.get("embedding_dim", 768))
    eps = float(tcfg.get("rms_norm_eps", 1e-6))
    sliding_window = int(tcfg.get("sliding_window", 512))
    layer_types = tcfg.get("layer_types", ["sliding_attention"] * num_layers)
    per_layer_cfg = tcfg.get("per_layer_config", {})
    rope_params = tcfg.get("rope_parameters", {})

    ANESafeRMSNorm = make_ane_safe_rms_norm_cls(torch, nn)
    max_seq = max(TEXT_SEQUENCE_BUCKETS)

    # Save embed_tokens.weight as a compact float16 binary lookup table for O(L) CPU token gather
    out_dir.mkdir(parents=True, exist_ok=True)
    embed_tok_w = get_tensor(state, "language_model.embed_tokens.weight", "embed_tokens.weight").to(torch.float16)
    tok_bin_path = out_dir / "embed_tokens_fp16.bin"
    tok_bin_path.write_bytes(embed_tok_w.numpy().tobytes())

    class TextLayer(nn.Module):
        def __init__(self, idx: int):
            super().__init__()
            ltype = layer_types[idx]
            lp = f"language_model.layers.{idx}"
            sp = f"layers.{idx}"
            q_w = get_tensor(state, f"{lp}.self_attn.q_proj.weight", f"{sp}.self_attn.q_proj.weight")
            k_w = get_tensor(state, f"{lp}.self_attn.k_proj.weight", f"{sp}.self_attn.k_proj.weight")
            self.num_heads = num_heads
            self.head_dim = int(q_w.shape[0] // self.num_heads)
            self.kv_heads = int(k_w.shape[0] // self.head_dim)
            self.kv_repeats = self.num_heads // self.kv_heads
            self.is_global = (ltype == "full_attention")
            rope_base = float(rope_params.get(ltype, {}).get("rope_theta", 1000000.0 if self.is_global else 10000.0))

            cos_1d, sin_1d = precompute_1d_rope(torch, max_seq=max_seq, head_dim=self.head_dim, base=rope_base)
            self.register_buffer("rope_cos", cos_1d.view(1, 1, max_seq, self.head_dim))
            self.register_buffer("rope_sin", sin_1d.view(1, 1, max_seq, self.head_dim))

            self.input_layernorm = ANESafeRMSNorm(hidden_size, eps=eps)
            self.q_proj = nn.Linear(hidden_size, self.num_heads * self.head_dim, bias=False)
            self.k_proj = nn.Linear(hidden_size, self.kv_heads * self.head_dim, bias=False)
            self.v_proj = nn.Linear(hidden_size, self.kv_heads * self.head_dim, bias=False)
            self.o_proj = nn.Linear(self.num_heads * self.head_dim, hidden_size, bias=False)
            self.q_norm = ANESafeRMSNorm(self.head_dim, eps=eps, with_scale=True)
            self.k_norm = ANESafeRMSNorm(self.head_dim, eps=eps, with_scale=True)
            self.v_norm = ANESafeRMSNorm(self.head_dim, eps=eps, with_scale=False)
            self.post_attention_layernorm = ANESafeRMSNorm(hidden_size, eps=eps)

            self.pre_feedforward_layernorm = ANESafeRMSNorm(hidden_size, eps=eps)
            self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
            self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
            self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
            self.post_feedforward_layernorm = ANESafeRMSNorm(hidden_size, eps=eps)

            self.per_layer_input_gate = nn.Linear(hidden_size, ple_size, bias=False)
            self.per_layer_projection = nn.Linear(ple_size, hidden_size, bias=False)
            self.post_per_layer_input_norm = ANESafeRMSNorm(hidden_size, eps=eps)
            self.ple_layer_proj = nn.Linear(hidden_size, ple_size, bias=False)
            self.ple_layer_norm = ANESafeRMSNorm(ple_size, eps=eps)
            self.ple_scale = 1.0 / math.sqrt(float(hidden_size))
            self.layer_scalar = nn.Parameter(torch.ones(1, dtype=torch.float32))

        def forward(
            self,
            x: torch.Tensor,
            inputs_embeds: torch.Tensor,
            global_bias: torch.Tensor,
        ) -> torch.Tensor:
            normed = self.input_layernorm(x)
            q = self.q_norm(self.q_proj(normed).view(1, max_seq, self.num_heads, self.head_dim)).transpose(1, 2)
            k = self.k_norm(self.k_proj(normed).view(1, max_seq, self.kv_heads, self.head_dim)).transpose(1, 2)
            v = self.v_norm(self.v_proj(normed).view(1, max_seq, self.kv_heads, self.head_dim)).transpose(1, 2)

            q = q * self.rope_cos + rotate_half(torch, q) * self.rope_sin
            k = k * self.rope_cos + rotate_half(torch, k) * self.rope_sin

            # Pure 4D GQA head expansion (avoids 5D unsqueeze/tile ops that ANEC rejects to CPU)
            if self.kv_heads == 2 and self.num_heads == 4:
                k = torch.cat([k[:, 0:1, :, :], k[:, 0:1, :, :], k[:, 1:2, :, :], k[:, 1:2, :, :]], dim=1)
                v = torch.cat([v[:, 0:1, :, :], v[:, 0:1, :, :], v[:, 1:2, :, :], v[:, 1:2, :, :]], dim=1)

            scores = torch.matmul(q, k.transpose(-2, -1)) + global_bias
            attn = torch.softmax(scores, dim=-1)
            attn_out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(1, max_seq, self.num_heads * self.head_dim)
            x = x + self.post_attention_layernorm(self.o_proj(attn_out))

            ff = self.pre_feedforward_layernorm(x)
            act = torch.nn.functional.gelu(self.gate_proj(ff), approximate="tanh") * self.up_proj(ff)
            x = x + self.post_feedforward_layernorm(self.down_proj(act))

            ple_in = self.ple_layer_norm(self.ple_layer_proj(inputs_embeds) * self.ple_scale)
            gated = torch.nn.functional.gelu(self.per_layer_input_gate(x), approximate="tanh") * ple_in
            x = x + self.post_per_layer_input_norm(self.per_layer_projection(gated))
            return x * self.layer_scalar

    class EmbeddingGemma2TextANEWrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([TextLayer(i) for i in range(num_layers)])
            self.norm = ANESafeRMSNorm(hidden_size, eps=eps)
            self.embedding_projection = nn.Linear(hidden_size, embedding_dim, bias=False)

        def forward(self, packed_input: torch.Tensor) -> torch.Tensor:
            # Static-shape tensor: [1, 512, 513] = inputs_embeds [1, 512, 512] || attention_mask [1, 512, 1]
            inputs_embeds = packed_input[:, :, :hidden_size]
            mask_exp = packed_input[:, :, hidden_size:]  # [1, 512, 1]
            # Key padding bias: [1, 1, 1, 512], 0.0 for valid keys, -1e4 for padded keys
            global_bias = (1.0 - mask_exp).permute(0, 2, 1).unsqueeze(1) * (-1e4)

            h = inputs_embeds
            for layer in self.layers:
                h = layer(h, inputs_embeds, global_bias)
            h = self.norm(h)

            denom = torch.clamp(mask_exp.sum(dim=1), min=1.0)
            pooled = (h * mask_exp).sum(dim=1) / denom  # [1, 512]
            proj = self.embedding_projection(pooled)  # [1, 768]
            # Scale-safe L2 normalize
            proj_scaled = proj * FP16_SAFE_RMS_PRESCALE
            norm = torch.rsqrt(torch.sum(proj_scaled * proj_scaled, dim=-1, keepdim=True) + 1e-4)
            return proj_scaled * norm

    model = EmbeddingGemma2TextANEWrapper().eval()

    with torch.no_grad():
        ple_proj_all = get_tensor(
            state,
            "language_model.ple.per_layer_model_projection.weight",
            "ple.per_layer_model_projection.weight",
        )
        ple_norm_w = get_tensor(
            state,
            "language_model.ple.per_layer_projection_norm.weight",
            "ple.per_layer_projection_norm.weight",
        )
        for i, layer in enumerate(model.layers):
            lp = f"language_model.layers.{i}"
            sp = f"layers.{i}"
            layer.ple_layer_proj.weight.copy_(ple_proj_all[i * ple_size : (i + 1) * ple_size, :])
            layer.ple_layer_norm.weight.copy_(ple_norm_w)
            layer.input_layernorm.weight.copy_(get_tensor(state, f"{lp}.input_layernorm.weight", f"{sp}.input_layernorm.weight"))
            layer.post_attention_layernorm.weight.copy_(get_tensor(state, f"{lp}.post_attention_layernorm.weight", f"{sp}.post_attention_layernorm.weight"))
            layer.pre_feedforward_layernorm.weight.copy_(get_tensor(state, f"{lp}.pre_feedforward_layernorm.weight", f"{sp}.pre_feedforward_layernorm.weight"))
            layer.post_feedforward_layernorm.weight.copy_(get_tensor(state, f"{lp}.post_feedforward_layernorm.weight", f"{sp}.post_feedforward_layernorm.weight"))

            layer.q_proj.weight.copy_(get_tensor(state, f"{lp}.self_attn.q_proj.weight", f"{sp}.self_attn.q_proj.weight"))
            layer.k_proj.weight.copy_(get_tensor(state, f"{lp}.self_attn.k_proj.weight", f"{sp}.self_attn.k_proj.weight"))
            layer.v_proj.weight.copy_(get_tensor(state, f"{lp}.self_attn.v_proj.weight", f"{sp}.self_attn.v_proj.weight"))
            layer.o_proj.weight.copy_(get_tensor(state, f"{lp}.self_attn.o_proj.weight", f"{sp}.self_attn.o_proj.weight"))
            layer.q_norm.weight.copy_(get_tensor(state, f"{lp}.self_attn.q_norm.weight", f"{sp}.self_attn.q_norm.weight"))
            layer.k_norm.weight.copy_(get_tensor(state, f"{lp}.self_attn.k_norm.weight", f"{sp}.self_attn.k_norm.weight"))

            layer.gate_proj.weight.copy_(get_tensor(state, f"{lp}.mlp.gate_proj.weight", f"{sp}.mlp.gate_proj.weight"))
            layer.up_proj.weight.copy_(get_tensor(state, f"{lp}.mlp.up_proj.weight", f"{sp}.mlp.up_proj.weight"))
            layer.down_proj.weight.copy_(get_tensor(state, f"{lp}.mlp.down_proj.weight", f"{sp}.mlp.down_proj.weight"))

            layer.per_layer_input_gate.weight.copy_(get_tensor(state, f"{lp}.ple_block.per_layer_input_gate.weight", f"{sp}.ple_block.per_layer_input_gate.weight"))
            layer.per_layer_projection.weight.copy_(get_tensor(state, f"{lp}.ple_block.per_layer_projection.weight", f"{sp}.ple_block.per_layer_projection.weight"))
            layer.post_per_layer_input_norm.weight.copy_(get_tensor(state, f"{lp}.ple_block.post_per_layer_input_norm.weight", f"{sp}.ple_block.post_per_layer_input_norm.weight"))
            layer.layer_scalar.copy_(get_tensor(state, f"{lp}.layer_scalar", f"{sp}.layer_scalar"))

        model.norm.weight.copy_(get_tensor(state, "language_model.norm.weight", "norm.weight"))
        model.embedding_projection.weight.copy_(get_tensor(
            state, "language_model.embedding_projection.weight", "embedding_projection.weight"
        ))
        # Clamp all parameters and buffers to IEEE-754 FP16 finite range [-65500, 65500] so ANEC never sees Inf
        for name, param in model.named_parameters():
            max_abs = float(param.abs().max().item())
            if max_abs > 65500.0:
                print(f"  [FP16 clamp] parameter {name} max_abs={max_abs:.1f} -> clamped to 65500.0")
                param.clamp_(-65500.0, 65500.0)
        for name, buf in model.named_buffers():
            if buf.is_floating_point():
                max_abs = float(buf.abs().max().item())
                if max_abs > 65500.0:
                    print(f"  [FP16 clamp] buffer {name} max_abs={max_abs:.1f} -> clamped to 65500.0")
                    buf.clamp_(-65500.0, 65500.0)

    packed_dim = hidden_size + 1  # 512 embeds + 1 attention_mask
    ex_packed = torch.zeros(1, max_seq, packed_dim, dtype=torch.float32)
    ex_packed[:, :, hidden_size] = 1.0
    traced = torch.jit.trace(model, (ex_packed,))

    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="packed_input", shape=(1, max_seq, packed_dim)),
        ],
        outputs=[ct.TensorType(name="embedding")],
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.macOS15,
    )

    pkg_path = out_dir / "EmbeddingGemma2TextANE.mlpackage"
    mlmodel.save(str(pkg_path))
    if compile_modelc:
        subprocess.run(
            ["xcrun", "coremlcompiler", "compile", str(pkg_path), str(out_dir)],
            check=True,
        )
    return pkg_path


def export_audio_tower(checkpoint_dir: Path, state: dict, out_dir: Path, compile_modelc: bool) -> Path:
    import coremltools as ct
    import torch
    import torch.nn as nn

    cfg = inspect_checkpoint(checkpoint_dir)
    acfg = cfg.get("audio_config", {})
    hidden_size = int(acfg.get("hidden_size", 1024))
    num_layers = int(acfg.get("num_hidden_layers", 12))
    num_heads = int(acfg.get("num_attention_heads", 8))
    head_dim = hidden_size // num_heads  # 128
    out_proj_dims = int(acfg.get("output_proj_dims", 1536))
    text_hidden = int(cfg.get("text_config", {}).get("hidden_size", 512))
    eps = float(acfg.get("rms_norm_eps", 1e-6))
    chunk_size = int(acfg.get("attention_chunk_size", 12))
    context_left = int(acfg.get("attention_context_left", 13))
    context_right = int(acfg.get("attention_context_right", 0))
    past_horizon = context_left - 1  # 12
    context_size = chunk_size + past_horizon + context_right  # 24
    soft_cap = float(acfg.get("attention_logit_cap", 50.0))

    ANESafeRMSNorm = make_ane_safe_rms_norm_cls(torch, nn)

    # Fixed 30s mel input: [1, 2999, 128] -> subsampled T' = 750 -> 63 blocks of 12 (756 padded)
    mel_frames = 2999
    subsampled_len = 750
    block_count = (subsampled_len + chunk_size - 1) // chunk_size  # 63
    padded_len = block_count * chunk_size  # 756

    # Precompute sinusoidal relative positional embeddings [13, 1024]
    half_size = hidden_size // 2
    log_inc = math.log(10000.0) / float(max(half_size - 1, 1))
    inv_timescales = torch.exp(torch.arange(half_size, dtype=torch.float32) * (-log_inc))
    distances = torch.arange(context_size // 2, -1, -1, dtype=torch.float32).unsqueeze(-1)  # [13, 1]
    angles = distances * inv_timescales  # [13, 512]
    rel_pos_embed = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)  # [13, 1024]

    # Precompute visibleKeys additive bias [1, 1, 63, 12, 24]
    row = torch.arange(chunk_size).view(1, chunk_size, 1)
    col = torch.arange(context_size).view(1, 1, context_size)
    key_idx = torch.arange(block_count).view(block_count, 1, 1) * chunk_size - past_horizon + col
    dist = row + past_horizon - col
    valid = (dist >= 0) & (dist <= past_horizon) & (key_idx >= 0) & (key_idx < subsampled_len)
    attn_mask_bias = torch.where(valid, torch.zeros_like(dist, dtype=torch.float32), torch.full_like(dist, -1e4, dtype=torch.float32))
    attn_mask_bias = attn_mask_bias.view(1, 1, block_count, chunk_size, context_size)

    # Precompute static gather indices for overlapping context blocks [63 * 24]
    starts = torch.arange(0, block_count * chunk_size, chunk_size).unsqueeze(-1)
    offsets = torch.arange(context_size).unsqueeze(0)
    ctx_indices = (starts + offsets).flatten().long()

    class ClippableLinear(nn.Module):
        def __init__(self, in_f: int, out_f: int):
            super().__init__()
            self.linear = nn.Linear(in_f, out_f, bias=False)
            self.register_buffer("in_min", torch.tensor(-65500.0))
            self.register_buffer("in_max", torch.tensor(65500.0))
            self.register_buffer("out_min", torch.tensor(-65500.0))
            self.register_buffer("out_max", torch.tensor(65500.0))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = torch.clamp(x, min=self.in_min, max=self.in_max)
            y = self.linear(x)
            return torch.clamp(y, min=self.out_min, max=self.out_max)

    class AudioSubsampleConvLayer(nn.Module):
        def __init__(self, in_ch: int, out_ch: int):
            super().__init__()
            self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False)
            self.norm_weight = nn.Parameter(torch.ones(out_ch))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: [B, C, T, F]
            h = self.conv(x).permute(0, 2, 3, 1)  # [B, T', F', C]
            mean = h.mean(dim=-1, keepdim=True)
            diff = (h - mean) * FP16_SAFE_RMS_PRESCALE
            inv_std = torch.rsqrt(torch.mean(diff * diff, dim=-1, keepdim=True) + eps * (FP16_SAFE_RMS_PRESCALE ** 2))
            normed = (diff * inv_std) * self.norm_weight
            return torch.relu(normed.permute(0, 3, 1, 2))

    class AudioFeedForward(nn.Module):
        def __init__(self):
            super().__init__()
            self.pre_norm = ANESafeRMSNorm(hidden_size, eps=eps)
            self.ffw_layer_1 = ClippableLinear(hidden_size, hidden_size * 4)
            self.ffw_layer_2 = ClippableLinear(hidden_size * 4, hidden_size)
            self.post_norm = ANESafeRMSNorm(hidden_size, eps=eps)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = self.ffw_layer_2(torch.nn.functional.silu(self.ffw_layer_1(self.pre_norm(x))))
            return self.post_norm(h) * 0.5 + x

    class AudioLightConv1d(nn.Module):
        def __init__(self):
            super().__init__()
            self.pre_norm = ANESafeRMSNorm(hidden_size, eps=eps)
            self.linear_start = ClippableLinear(hidden_size, hidden_size * 2)
            self.depthwise_conv1d = nn.Conv1d(hidden_size, hidden_size, kernel_size=5, groups=hidden_size, bias=False)
            self.conv_norm = ANESafeRMSNorm(hidden_size, eps=eps)
            self.linear_end = ClippableLinear(hidden_size, hidden_size)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = self.linear_start(self.pre_norm(x))
            a, b = h[..., :hidden_size], h[..., hidden_size:]
            gated = a * torch.sigmoid(b)  # [B, T, 1024]
            c_in = torch.nn.functional.pad(gated.transpose(1, 2), (4, 0))  # causal left-pad 4
            c_out = self.depthwise_conv1d(c_in).transpose(1, 2)
            return self.linear_end(torch.nn.functional.silu(self.conv_norm(c_out))) + x

    class AudioAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = ClippableLinear(hidden_size, hidden_size)
            self.k_proj = ClippableLinear(hidden_size, hidden_size)
            self.v_proj = ClippableLinear(hidden_size, hidden_size)
            self.post = ClippableLinear(hidden_size, hidden_size)
            self.relative_k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
            self.register_buffer("q_scale_vec", torch.ones(head_dim))
            self.key_scale = math.log(1.0 + math.exp(1.0)) / math.log(2.0)

        def forward(
            self,
            x: torch.Tensor,
            rel_pos: torch.Tensor,
            mask_bias: torch.Tensor,
            gather_idx: torch.Tensor,
        ) -> torch.Tensor:
            q_raw = self.q_proj(x).view(1, subsampled_len, num_heads, head_dim) * self.q_scale_vec
            k_raw = self.k_proj(x).view(1, subsampled_len, num_heads, head_dim) * self.key_scale
            v_raw = self.v_proj(x).view(1, subsampled_len, num_heads, head_dim)

            q_pad = torch.nn.functional.pad(q_raw, (0, 0, 0, 0, 0, padded_len - subsampled_len))
            q_blk = q_pad.view(1, block_count, chunk_size, num_heads, head_dim).permute(0, 3, 1, 2, 4)

            k_pad = torch.nn.functional.pad(k_raw, (0, 0, 0, 0, past_horizon, chunk_size - 1))
            v_pad = torch.nn.functional.pad(v_raw, (0, 0, 0, 0, past_horizon, chunk_size - 1))
            k_blk = k_pad[:, gather_idx, :, :].view(1, block_count, context_size, num_heads, head_dim).permute(0, 3, 1, 2, 4)
            v_blk = v_pad[:, gather_idx, :, :].view(1, block_count, context_size, num_heads, head_dim).permute(0, 3, 1, 2, 4)

            rel_k = self.relative_k_proj(rel_pos).view(13, num_heads, head_dim).permute(1, 2, 0)  # [H, D, 13]
            scores = torch.matmul(q_blk, k_blk.transpose(-2, -1))  # [1, H, 63, 12, 24]

            term_bd = torch.matmul(q_blk.reshape(1, num_heads, padded_len, head_dim), rel_k).view(
                1, num_heads, block_count, chunk_size, 13
            )
            term_bd_pad = torch.nn.functional.pad(term_bd, (0, context_size + 1 - 13)).view(
                1, num_heads, block_count, chunk_size * (context_size + 1)
            )
            term_bd_shift = term_bd_pad[:, :, :, : chunk_size * context_size].view(
                1, num_heads, block_count, chunk_size, context_size
            )

            scores = torch.tanh((scores + term_bd_shift) / soft_cap) * soft_cap + mask_bias
            weights = torch.softmax(scores, dim=-1)
            attended = torch.matmul(weights, v_blk).permute(0, 2, 3, 1, 4).reshape(1, padded_len, hidden_size)
            return self.post(attended[:, :subsampled_len, :])

    class AudioConformerLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.feed_forward1 = AudioFeedForward()
            self.norm_pre_attn = ANESafeRMSNorm(hidden_size, eps=eps)
            self.self_attn = AudioAttention()
            self.norm_post_attn = ANESafeRMSNorm(hidden_size, eps=eps)
            self.lconv1d = AudioLightConv1d()
            self.feed_forward2 = AudioFeedForward()
            self.norm_out = ANESafeRMSNorm(hidden_size, eps=eps)

        def forward(
            self,
            x: torch.Tensor,
            rel_pos: torch.Tensor,
            mask_bias: torch.Tensor,
            gather_idx: torch.Tensor,
        ) -> torch.Tensor:
            h = self.feed_forward1(x)
            h = h + self.norm_post_attn(self.self_attn(self.norm_pre_attn(h), rel_pos, mask_bias, gather_idx))
            h = self.lconv1d(h)
            h = self.feed_forward2(h)
            return self.norm_out(h)

    class EmbeddingGemma2AudioANEWrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.sub_layer0 = AudioSubsampleConvLayer(1, 128)
            self.sub_layer1 = AudioSubsampleConvLayer(128, 32)
            self.input_proj_linear = nn.Linear(1024, hidden_size, bias=False)
            self.register_buffer("rel_pos", rel_pos_embed)
            self.register_buffer("mask_bias", attn_mask_bias)
            self.register_buffer("gather_idx", ctx_indices)
            self.layers = nn.ModuleList([AudioConformerLayer() for _ in range(num_layers)])
            self.output_proj = nn.Linear(hidden_size, out_proj_dims, bias=True)
            self.embed_norm = ANESafeRMSNorm(out_proj_dims, eps=eps, with_scale=False)
            self.embed_proj = nn.Linear(out_proj_dims, text_hidden, bias=False)

        def forward(self, mel: torch.Tensor) -> torch.Tensor:
            # mel: [1, 2999, 128]
            x = mel.unsqueeze(1)  # [1, 1, 2999, 128]
            x = self.sub_layer1(self.sub_layer0(x))  # [1, 32, 750, 32]
            x = x.permute(0, 2, 3, 1).reshape(1, subsampled_len, 1024)
            x = self.input_proj_linear(x)
            for layer in self.layers:
                x = layer(x, self.rel_pos, self.mask_bias, self.gather_idx)
            x = self.output_proj(x)
            return self.embed_proj(self.embed_norm(x))  # [1, 750, 512]

    model = EmbeddingGemma2AudioANEWrapper().eval()

    def copy_clippable(target: ClippableLinear, prefix: str):
        target.linear.weight.copy_(get_tensor(state, f"{prefix}.linear.weight", f"{prefix}.weight"))
        for attr, key in [
            ("in_min", "input_min"),
            ("in_max", "input_max"),
            ("out_min", "output_min"),
            ("out_max", "output_max"),
        ]:
            full_k = f"{prefix}.{key}"
            if full_k in state:
                val = float(state[full_k].float().item())
                getattr(target, attr).fill_(max(-65500.0, min(65500.0, val)))

    with torch.no_grad():
        w0 = get_tensor(state, "audio_tower.subsample_conv_projection.layer0.conv.weight")
        if w0.shape[-1] == 1:
            w0 = w0.permute(0, 3, 1, 2).contiguous()
        model.sub_layer0.conv.weight.copy_(w0)
        model.sub_layer0.norm_weight.copy_(get_tensor(state, "audio_tower.subsample_conv_projection.layer0.norm.weight"))

        w1 = get_tensor(state, "audio_tower.subsample_conv_projection.layer1.conv.weight")
        if w1.shape[-1] == 128:
            w1 = w1.permute(0, 3, 1, 2).contiguous()
        model.sub_layer1.conv.weight.copy_(w1)
        model.sub_layer1.norm_weight.copy_(get_tensor(state, "audio_tower.subsample_conv_projection.layer1.norm.weight"))

        model.input_proj_linear.weight.copy_(get_tensor(state, "audio_tower.subsample_conv_projection.input_proj_linear.weight"))

        q_base_scale = (head_dim ** -0.5) / math.log(2.0)
        for i, layer in enumerate(model.layers):
            p = f"audio_tower.layers.{i}"
            for ff_name in ["feed_forward1", "feed_forward2"]:
                ff = getattr(layer, ff_name)
                ff.pre_norm.weight.copy_(get_tensor(state, f"{p}.{ff_name}.pre_layer_norm.weight"))
                ff.post_norm.weight.copy_(get_tensor(state, f"{p}.{ff_name}.post_layer_norm.weight"))
                copy_clippable(ff.ffw_layer_1, f"{p}.{ff_name}.ffw_layer_1")
                copy_clippable(ff.ffw_layer_2, f"{p}.{ff_name}.ffw_layer_2")

            layer.norm_pre_attn.weight.copy_(get_tensor(state, f"{p}.norm_pre_attn.weight"))
            layer.norm_post_attn.weight.copy_(get_tensor(state, f"{p}.norm_post_attn.weight"))
            layer.norm_out.weight.copy_(get_tensor(state, f"{p}.norm_out.weight"))

            copy_clippable(layer.self_attn.q_proj, f"{p}.self_attn.q_proj")
            copy_clippable(layer.self_attn.k_proj, f"{p}.self_attn.k_proj")
            copy_clippable(layer.self_attn.v_proj, f"{p}.self_attn.v_proj")
            copy_clippable(layer.self_attn.post, f"{p}.self_attn.post")
            layer.self_attn.relative_k_proj.weight.copy_(get_tensor(state, f"{p}.self_attn.relative_k_proj.weight"))
            per_dim = get_tensor(state, f"{p}.self_attn.per_dim_scale")
            layer.self_attn.q_scale_vec.copy_(q_base_scale * torch.nn.functional.softplus(per_dim))

            lc = layer.lconv1d
            lc.pre_norm.weight.copy_(get_tensor(state, f"{p}.lconv1d.pre_layer_norm.weight"))
            lc.conv_norm.weight.copy_(get_tensor(state, f"{p}.lconv1d.conv_norm.weight"))
            copy_clippable(lc.linear_start, f"{p}.lconv1d.linear_start")
            copy_clippable(lc.linear_end, f"{p}.lconv1d.linear_end")
            dw = get_tensor(state, f"{p}.lconv1d.depthwise_conv1d.weight")
            if dw.shape[-1] == 1 and dw.shape[1] == 5:
                dw = dw.permute(0, 2, 1).contiguous()
            lc.depthwise_conv1d.weight.copy_(dw)

        model.output_proj.weight.copy_(get_tensor(state, "audio_tower.output_proj.weight"))
        model.output_proj.bias.copy_(get_tensor(state, "audio_tower.output_proj.bias"))
        model.embed_proj.weight.copy_(get_tensor(state, "embed_audio.embedding_projection.weight"))

    ex_mel = torch.zeros(1, mel_frames, 128, dtype=torch.float32)
    traced = torch.jit.trace(model, ex_mel)

    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="mel_features", shape=(1, mel_frames, 128))],
        outputs=[ct.TensorType(name="soft_tokens")],
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.macOS15,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    pkg_path = out_dir / "EmbeddingGemma2AudioANE.mlpackage"
    mlmodel.save(str(pkg_path))
    if compile_modelc:
        subprocess.run(
            ["xcrun", "coremlcompiler", "compile", str(pkg_path), str(out_dir)],
            check=True,
        )
    return pkg_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export EmbeddingGemma 2 towers to CoreML ANE bundles")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("/Users/fredbliss/Storage/llms/google/embeddinggemma-2-MLX-BF16"),
        help="Path to embeddinggemma-2-MLX-BF16 directory",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for .mlpackage / .mlmodelc bundles (defaults to <checkpoint>/coreml)",
    )
    parser.add_argument(
        "--components",
        nargs="+",
        choices=["vision", "audio", "text"],
        default=["vision", "audio", "text"],
        help="Which EmbeddingGemma 2 towers to export",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Also run `xcrun coremlcompiler compile` to produce .mlmodelc bundles",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect checkpoint config and print planned ANE shape buckets without running coremltools",
    )
    args = parser.parse_args(argv)

    cfg = inspect_checkpoint(args.checkpoint)
    out_dir = args.out_dir or (args.checkpoint / "coreml")

    if args.dry_run:
        print(json.dumps({
            "checkpoint": str(args.checkpoint),
            "out_dir": str(out_dir),
            "model_type": cfg.get("model_type"),
            "vision_buckets_hw": VISION_RESOLUTION_BUCKETS,
            "text_sequence_buckets": TEXT_SEQUENCE_BUCKETS,
            "fp16_safe_rms_prescale": FP16_SAFE_RMS_PRESCALE,
        }, indent=2))
        return 0

    print(f"Loading checkpoint weights from {args.checkpoint}...")
    state = load_checkpoint_state(args.checkpoint)

    if "vision" in args.components:
        print("Exporting Vision Tower (EmbeddingGemma2VisionANE)...")
        pkg = export_vision_tower(args.checkpoint, state, out_dir, args.compile)
        print(f"  -> {pkg}")

    if "audio" in args.components:
        print("Exporting Audio Tower (EmbeddingGemma2AudioANE)...")
        pkg = export_audio_tower(args.checkpoint, state, out_dir, args.compile)
        print(f"  -> {pkg}")

    if "text" in args.components:
        print("Exporting Text & Fusion Backbone (EmbeddingGemma2TextANE + embed_tokens_fp16.bin)...")
        pkg = export_text_backbone(args.checkpoint, state, out_dir, args.compile)
        print(f"  -> {pkg}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
