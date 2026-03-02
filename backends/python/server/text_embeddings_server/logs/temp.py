import torch
import torch_npu
import json
from pathlib import Path
from torch import nn
import torch.nn.functional as F
from typing import List, Union, Optional
from safetensors import safe_open
from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.qwen3 import Qwen3Config
from opentelemetry import trace
from text_embeddings_server.models import Model
from text_embeddings_server.models.pooling import DefaultPooling
from text_embeddings_server.models.types import FlashBatch, Embedding, PaddedBatch
from text_embeddings_server.utils.flash_attn import attention
from loguru import logger
import datetime

tracer = trace.get_tracer(__name__)

_LOGGED_ONCE_KEYS = set()


def log_once(level: str, key: str, message: str):
    if key in _LOGGED_ONCE_KEYS:
        return
    _LOGGED_ONCE_KEYS.add(key)
    if level == "warning":
        logger.warning(message)
    elif level == "error":
        logger.error(message)
    else:
        logger.info(message)

# 关闭调试日志，避免性能开销
DEBUG_LOGGING_ENABLED = False

def log_tensor_stats(name: str, tensor: torch.Tensor, module_name: str = "FlashQwen3"):
    if not DEBUG_LOGGING_ENABLED or tensor is None:
        return
    # 注释掉所有tensor统计信息计算和日志输出，避免性能开销
    # timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    # shape = tuple(tensor.shape)
    # dtype = str(tensor.dtype)
    # device = str(tensor.device)
    # if tensor.numel() > 0:
    #     mean_val = tensor.float().mean().item()
    #     std_val = tensor.float().std().item() if tensor.numel() > 1 else 0.0
    #     min_val = tensor.float().min().item()
    #     max_val = tensor.float().max().item()
    #     logger.info(f"[{timestamp}] [{module_name}] {name} | shape={shape}, dtype={dtype}, device={device}, mean={mean_val:.6f}, std={std_val:.6f}, min={min_val:.6f}, max={max_val:.6f}")
    # else:
    #     logger.info(f"[{timestamp}] [{module_name}] {name} | shape={shape}, dtype={dtype}, device={device}, (empty tensor)")

def log_separator(title: str, module_name: str = "FlashQwen3"):
    if not DEBUG_LOGGING_ENABLED:
        return
    # 注释掉分隔符日志，避免性能开销
    # timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    # logger.info(f"[{timestamp}] [{module_name}] {'='*20} {title} {'='*20}")


def load_weight(model_path, weight_map, name, dtype, device):
    """
    Helper function to load a weight tensor from safetensors.
    Supports both sharded (with index file) and single file models.
    """
    if weight_map is None:
        # Single file model - directly load from model.safetensors
        with safe_open(f"{model_path}/model.safetensors", framework="pt") as f:
            return f.get_tensor(name).to(dtype).to(device)
    else:
        # Sharded model - use weight_map to find the file
        target_file = weight_map[name]
        with safe_open(f"{model_path}/{target_file}", framework="pt") as f:
            return f.get_tensor(name).to(dtype).to(device)

def _generate_attn_mask(max_seq_len, dtype):
    # Construct lower triangle matrix.
    mask_flag = torch.ones((max_seq_len, max_seq_len),
                           dtype=torch.bool).tril_()
    # Create upper triangle matrix used to mark mask positions.
    mask_flag = ~mask_flag
    # Currently for fp16 dtype, the mask value should be set to -inf.
    # TODO: Eliminate this part in the future.
    mask_value = float('-inf') if dtype == torch.float16 else 1
    attn_mask = torch.zeros(size=(max_seq_len, max_seq_len), dtype=dtype) \
        .masked_fill_(mask_flag, mask_value)
    return attn_mask

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


NPU_ROTARY_USE_NATIVE_MATMUL = False


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """
    Apply rotary position embedding.
    
    Two implementation modes:
    1. Native NPU matrix multiplication (default, better performance for TND format)
    2. torch_npu.npu_rotary_mul (requires 4D input, has layout constraints)
    
    Performance considerations:
    - npu_rotary_mul requires 4D BNSD layout, causing dimension conversion overhead for TND
    - Native matmul works directly with 3D TND format without conversion
    - Layout mismatch between TND input and 11SD cos/sin causes suboptimal performance
    """
    if NPU_ROTARY_USE_NATIVE_MATMUL:
        return _apply_rotary_pos_emb_native(q, k, cos, sin, unsqueeze_dim)
    else:
        return _apply_rotary_pos_emb_npu_rotary(q, k, cos, sin, unsqueeze_dim)


def _apply_rotary_pos_emb_native(q, k, cos, sin, unsqueeze_dim=1):
    """
    Native implementation using direct matrix multiplication.
    Optimal for TND format - no dimension conversion needed.
    
    Formula: output = x * cos + rotate_half(x) * sin
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    
    return q_embed, k_embed


def _apply_rotary_pos_emb_npu_rotary(q, k, cos, sin, unsqueeze_dim=1):
    """
    Implementation using torch_npu.npu_rotary_mul.
    Requires 4D input in BNSD format for jit_compile=True scenario.
    
    Note: This implementation has performance overhead due to:
    1. Dimension conversion (TND -> BNSD -> TND)
    2. Layout constraints requiring specific cos/sin shapes
    """
    q_orig_dim = q.dim()

    if q.shape[-1] % 2 != 0:
        log_once("warning", "rope_last_dim_not_even", "[RoPE] npu_rotary_mul skipped: last dim is not even, keep q/k unchanged")
        return q, k

    def _normalize_coeff_for_tnd(coeff: torch.Tensor, t: int, n: int, d: int) -> Optional[torch.Tensor]:
        # target preferred layout: T1D
        if coeff.dim() == 2:
            if coeff.shape == (t, d):
                return coeff.unsqueeze(1)
            return None

        if coeff.dim() == 3:
            # [1, T, D] -> [T, 1, D]
            if coeff.shape[0] == 1 and coeff.shape[1] == t and coeff.shape[2] == d:
                return coeff.squeeze(0).unsqueeze(1)
            # already [T, 1, D]
            if coeff.shape[0] == t and coeff.shape[1] == 1 and coeff.shape[2] == d:
                return coeff
            # also allow [T, N, D] (TND)
            if coeff.shape[0] == t and coeff.shape[1] == n and coeff.shape[2] == d:
                return coeff

        return None

    def _normalize_coeff_for_bnsd(coeff: torch.Tensor, b: int, n: int, s: int, d: int) -> Optional[torch.Tensor]:
        # preferred layout: B1SD or 11SD
        if coeff.dim() == 3:
            # [B, S, D] -> [B, 1, S, D]
            if coeff.shape[0] == b and coeff.shape[1] == s and coeff.shape[2] == d:
                return coeff.unsqueeze(1)
            # [1, S, D] -> [1, 1, S, D]
            if coeff.shape[0] == 1 and coeff.shape[1] == s and coeff.shape[2] == d:
                return coeff.unsqueeze(1)

        if coeff.dim() == 4:
            # already [B, 1, S, D] / [1, 1, S, D] / [B, N, S, D]
            if coeff.shape[-2] == s and coeff.shape[-1] == d:
                if coeff.shape[0] in (1, b) and coeff.shape[1] in (1, n):
                    return coeff

        return None

    try:
        if q_orig_dim == 3:
            # TND input path
            t, n, d = q.shape
            cos_npu = _normalize_coeff_for_tnd(cos, t, n, d)
            sin_npu = _normalize_coeff_for_tnd(sin, t, n, d)
            if cos_npu is None or sin_npu is None:
                log_once("warning", "rope_unsupported_tnd_coeff_layout", "[RoPE] npu_rotary_mul skipped: unsupported TND cos/sin layout, keep q/k unchanged")
                return q, k

            cos_npu = cos_npu.to(device=q.device, dtype=q.dtype)
            sin_npu = sin_npu.to(device=q.device, dtype=q.dtype)

            q_embed = torch_npu.npu_rotary_mul(q, cos_npu, sin_npu, rotary_mode="half")
            k_embed = torch_npu.npu_rotary_mul(k, cos_npu, sin_npu, rotary_mode="half")
            return q_embed, k_embed

        if q_orig_dim == 4:
            # BNSD input path
            b, n, s, d = q.shape
            cos_npu = _normalize_coeff_for_bnsd(cos, b, n, s, d)
            sin_npu = _normalize_coeff_for_bnsd(sin, b, n, s, d)
            if cos_npu is None or sin_npu is None:
                log_once("warning", "rope_unsupported_bnsd_coeff_layout", "[RoPE] npu_rotary_mul skipped: unsupported BNSD cos/sin layout, keep q/k unchanged")
                return q, k

            cos_npu = cos_npu.to(device=q.device, dtype=q.dtype)
            sin_npu = sin_npu.to(device=q.device, dtype=q.dtype)

            q_embed = torch_npu.npu_rotary_mul(q, cos_npu, sin_npu, rotary_mode="half")
            k_embed = torch_npu.npu_rotary_mul(k, cos_npu, sin_npu, rotary_mode="half")
            return q_embed, k_embed

        log_once("warning", "rope_unsupported_q_rank", "[RoPE] npu_rotary_mul skipped: unsupported q rank, keep q/k unchanged")
        return q, k

    except RuntimeError as error:
        log_once("warning", "rope_runtime_error", f"[RoPE] npu_rotary_mul failed: {error}; keep q/k unchanged")
        return q, k

    return q, k


def compute_default_rope_parameters(
    config: Qwen3Config,
    device: torch.device,
) -> tuple["torch.Tensor", float]:
    base = config.rope_theta
    partial_rotary_factor = (
        config.partial_rotary_factor
        if hasattr(config, "partial_rotary_factor")
        else 1.0
    )
    head_dim = (
        getattr(config, "head_dim", None)
        or config.hidden_size // config.num_attention_heads
    )
    dim = int(head_dim * partial_rotary_factor)
    attention_factor = 1.0

    inv_freq = 1.0 / (
        base
        ** (
            torch.arange(0, dim, 2, dtype=torch.int64).to(
                device=device, dtype=torch.float
            )
            / dim
        )
    )
    return inv_freq, attention_factor


class Qwen3RMSNorm:
    def __init__(
        self,
        model_path,
        weight_map,
        name,
        device,
        dtype,
        eps=1e-6,
    ):
        self.weight = load_weight(model_path, weight_map, name, dtype, device)
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        if hidden_states.device.type == "hpu":
            from habana_frameworks.torch.hpex.normalization import (
                FusedRMSNorm as FusedRMSNorm,
            )

            hidden_states = FusedRMSNorm.apply(
                hidden_states, self.weight, self.variance_epsilon
            )
            return hidden_states
        else:
            input_dtype = hidden_states.dtype
            # hidden_states = hidden_states.to(torch.float32)
            # variance = hidden_states.pow(2).mean(-1, keepdim=True)
            # hidden_states = hidden_states * torch.rsqrt(
            #     variance + self.variance_epsilon
            # )
            # return self.weight * hidden_states.to(input_dtype)
            return torch_npu.npu_rms_norm(hidden_states.to(input_dtype), self.weight, epsilon=self.variance_epsilon)[0]


class Qwen3Attention:
    def __init__(
        self,
        model_path,
        weight_map,
        device,
        dtype,
        config: Qwen3Config,
        layer_idx: Optional[int] = None,
    ):
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.softmax_scale = self.head_dim**-0.5
        self.q_proj_weight = load_weight(
            model_path,
            weight_map,
            f"layers.{layer_idx}.self_attn.q_proj.weight",
            dtype,
            device,
        )
        self.k_proj_weight = load_weight(
            model_path,
            weight_map,
            f"layers.{layer_idx}.self_attn.k_proj.weight",
            dtype,
            device,
        )
        self.v_proj_weight = load_weight(
            model_path,
            weight_map,
            f"layers.{layer_idx}.self_attn.v_proj.weight",
            dtype,
            device,
        )
        self.o_proj_weight = load_weight(
            model_path,
            weight_map,
            f"layers.{layer_idx}.self_attn.o_proj.weight",
            dtype,
            device,
        )
        self.q_norm = Qwen3RMSNorm(
            model_path,
            weight_map,
            f"layers.{layer_idx}.self_attn.q_norm.weight",
            device,
            dtype,
            eps=config.rms_norm_eps,
        )
        self.k_norm = Qwen3RMSNorm(
            model_path,
            weight_map,
            f"layers.{layer_idx}.self_attn.k_norm.weight",
            device,
            dtype,
            eps=config.rms_norm_eps,
        )

    def forward(
        self, hidden_states, position_embeddings, cu_seqlens, max_s, attn_mask=None
    ):
        # 注释掉性能开销大的日志
        # log_separator(f"Qwen3Attention Layer Start")
        # log_tensor_stats("Attention.input_hidden_states", hidden_states, "FlashQwen3.Attention")
        
        is_tnd = hidden_states.dim() == 2

        if is_tnd:
            input_shape = hidden_states.shape[:-1]
            q_proj_out = F.linear(hidden_states, self.q_proj_weight)
            q = self.q_norm.forward(
                q_proj_out.view(*input_shape, self.num_heads, self.head_dim)
            )
            # log_tensor_stats("Attention.q_after_q_norm", q, "FlashQwen3.Attention")

            k_proj_out = F.linear(hidden_states, self.k_proj_weight)
            k = self.k_norm.forward(
                k_proj_out.view(*input_shape, self.num_key_value_heads, self.head_dim)
            )
            # log_tensor_stats("Attention.k_after_k_norm", k, "FlashQwen3.Attention")

            v = F.linear(hidden_states, self.v_proj_weight).view(
                *input_shape, self.num_key_value_heads, self.head_dim
            )

            cos, sin = position_embeddings
            if cos.dim() == 3 and cos.shape[0] == 1:
                cos = cos.squeeze(0)
                sin = sin.squeeze(0)
            # log_tensor_stats("Attention.q_before_rope", q, "FlashQwen3.Attention")
            # log_tensor_stats("Attention.k_before_rope", k, "FlashQwen3.Attention")
            
            q, k = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)
            # log_tensor_stats("Attention.q_after_rope", q, "FlashQwen3.Attention")
            # log_tensor_stats("Attention.k_after_rope", k, "FlashQwen3.Attention")

            if self.num_key_value_groups > 1:
                k = k.repeat_interleave(self.num_key_value_groups, dim=1)
                v = v.repeat_interleave(self.num_key_value_groups, dim=1)

            attn_output = torch.empty_like(q)
        else:
            input_shape = hidden_states.shape[:-1]
            hidden_shape_q = (*input_shape, -1, self.head_dim)
            hidden_shape_kv = (*input_shape, self.num_key_value_heads, self.head_dim)

            q_proj_out = F.linear(hidden_states, self.q_proj_weight)
            q = self.q_norm.forward(q_proj_out.view(hidden_shape_q)).transpose(1, 2)
            # log_tensor_stats("Attention.q_after_q_norm", q, "FlashQwen3.Attention")

            k_proj_out = F.linear(hidden_states, self.k_proj_weight)
            k = self.k_norm.forward(k_proj_out.view(hidden_shape_kv)).transpose(1, 2)
            # log_tensor_stats("Attention.k_after_k_norm", k, "FlashQwen3.Attention")

            v = F.linear(hidden_states, self.v_proj_weight).view(hidden_shape_kv).transpose(1, 2)

            cos, sin = position_embeddings

            q, k = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)
            # log_tensor_stats("Attention.q_after_rope", q, "FlashQwen3.Attention")
            # log_tensor_stats("Attention.k_after_rope", k, "FlashQwen3.Attention")

            if self.num_key_value_groups > 1:
                k = k.repeat_interleave(self.num_key_value_groups, dim=1)
                v = v.repeat_interleave(self.num_key_value_groups, dim=1)

            attn_output = torch.empty_like(q)

        attention(
            q,
            k,
            v,
            self.num_heads,
            attn_output,
            cu_seqlens,
            max_s,
            self.softmax_scale,
            is_causal=True,
            attn_mask=attn_mask,
        )

        if is_tnd:
            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        else:
            attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
        attn_output = F.linear(attn_output, self.o_proj_weight, bias=None)
        # log_tensor_stats("Attention.attn_output_final", attn_output, "FlashQwen3.Attention")
        # log_separator(f"Qwen3Attention Layer End")

        return attn_output


class Qwen3MLP:
    def __init__(
        self,
        model_path,
        weight_map,
        device,
        dtype,
        config: Qwen3Config,
        layer_idx: Optional[int] = None,
    ):
        self.gate_proj_weight = load_weight(
            model_path,
            weight_map,
            f"layers.{layer_idx}.mlp.gate_proj.weight",
            dtype,
            device,
        )
        self.up_proj_weight = load_weight(
            model_path,
            weight_map,
            f"layers.{layer_idx}.mlp.up_proj.weight",
            dtype,
            device,
        )
        self.down_proj_weight = load_weight(
            model_path,
            weight_map,
            f"layers.{layer_idx}.mlp.down_proj.weight",
            dtype,
            device,
        )
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_state):
        gated_hidden_states = F.linear(hidden_state, self.gate_proj_weight)
        uped_hidden_states = F.linear(hidden_state, self.up_proj_weight)
        activated = self.act_fn(gated_hidden_states)
        multiplied = activated * uped_hidden_states
        output = F.linear(multiplied, self.down_proj_weight)
        return output


class Qwen3DecoderLayer:
    def __init__(
        self,
        model_path,
        weight_map,
        device,
        dtype,
        config: Qwen3Config,
        layer_idx: Optional[int] = None,
    ):
        self._layer_idx = layer_idx
        self.attention = Qwen3Attention(
            model_path, weight_map, device, dtype, config, layer_idx
        )
        self.mlp = Qwen3MLP(model_path, weight_map, device, dtype, config, layer_idx)
        self.input_layernorm = Qwen3RMSNorm(
            model_path,
            weight_map,
            f"layers.{layer_idx}.input_layernorm.weight",
            device,
            dtype,
            eps=config.rms_norm_eps,
        )
        self.post_attention_layernorm = Qwen3RMSNorm(
            model_path,
            weight_map,
            f"layers.{layer_idx}.post_attention_layernorm.weight",
            device,
            dtype,
            eps=config.rms_norm_eps,
        )

    def forward(
        self, hidden_states, position_embeddings, cu_seqlens, max_s, attn_mask=None
    ):
        # 注释掉性能开销大的日志
        # log_separator(f"DecoderLayer {self._layer_idx if hasattr(self, '_layer_idx') else ''} Start")
        
        residual = hidden_states
        hidden_states = self.input_layernorm.forward(hidden_states)
        
        hidden_states = self.attention.forward(
            hidden_states, position_embeddings, cu_seqlens, max_s, attn_mask
        )
        
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm.forward(hidden_states)
        
        hidden_states = self.mlp.forward(hidden_states)
        
        hidden_states = residual + hidden_states
        # log_tensor_stats("DecoderLayer.output_hidden_states", hidden_states, "FlashQwen3.DecoderLayer")
        # log_separator(f"DecoderLayer {self._layer_idx if hasattr(self, '_layer_idx') else ''} End")

        return hidden_states


class Qwen3RotaryEmbedding(nn.Module):
    def __init__(self, config: Qwen3Config, device=None):
        super().__init__()
        inv_freq, self.attention_scaling = compute_default_rope_parameters(
            config, device
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x, position_ids):
        # 注释掉性能开销大的日志
        # log_tensor_stats("RotaryEmbedding.position_ids", position_ids, "FlashQwen3.RotaryEmbedding")
        
        if position_ids.dim() == 1:
            position_ids = position_ids.unsqueeze(0)

        inv_freq_expanded = (
            self.inv_freq[None, :, None]
            .float()
            .expand(position_ids.shape[0], -1, 1)
            .to(x.device)
        )
        
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = (
            x.device.type
            if isinstance(x.device.type, str) and x.device.type != "mps"
            else "npu"
        )
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (
                inv_freq_expanded.float() @ position_ids_expanded.float()
            ).transpose(1, 2)
            
            emb = torch.cat((freqs, freqs), dim=-1)
            
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling
            # 注释掉性能开销大的日志
            # log_tensor_stats("RotaryEmbedding.cos_output", cos, "FlashQwen3.RotaryEmbedding")
            # log_tensor_stats("RotaryEmbedding.sin_output", sin, "FlashQwen3.RotaryEmbedding")

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class FlashQwen3Model:
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`MistralDecoderLayer`]

    Args:
        config: MistralConfig
    """

    def __init__(self, model_path, weight_map, device, dtype, config: Qwen3Config):
        # 使用 weight_map（可能是 None）
        self.word_embeddings_weight = load_weight(
            model_path,
            weight_map,  # 直接传递 weight_map，None 表示单文件
            "embed_tokens.weight",
            dtype,
            device,
        )
        self.layers = [
            Qwen3DecoderLayer(
                model_path,
                weight_map,  # 同样传递 weight_map
                device,
                dtype,
                config,
                layer_idx,
            )
            for layer_idx in range(config.num_hidden_layers)
        ]
        self.rotary_emb = Qwen3RotaryEmbedding(config=config, device=device)
        self.norm = Qwen3RMSNorm(
            model_path,
            weight_map,  # 这里也要改成 weight_map
            f"norm.weight",
            device,
            dtype,
            eps=config.rms_norm_eps,
        )

    def forward(
        self,
        input_ids,
        position_ids,
        cu_seqlens,
        max_s,
        mask=None,
        attn_mask=None,
    ):
        # 注释掉性能开销大的日志
        # log_separator("FlashQwen3Model Forward Start")
        # log_tensor_stats("Model.position_ids", position_ids, "FlashQwen3.Model")
        # log_tensor_stats("Model.cu_seqlens", cu_seqlens, "FlashQwen3.Model")
        # logger.info(f"[FlashQwen3.Model] max_s={max_s}")
        
        inputs_embeds = nn.functional.embedding(input_ids, self.word_embeddings_weight)
        
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        
        for idx, layer in enumerate(self.layers):
            # 保留关键的层处理日志，便于调试但不计算统计信息
            # logger.info(f"[FlashQwen3.Model] Processing layer {idx}")
            hidden_states = layer.forward(
                hidden_states, position_embeddings, cu_seqlens, max_s, attn_mask
            )
        
        hidden_states = self.norm.forward(hidden_states)
        # log_tensor_stats("Model.final_hidden_states", hidden_states, "FlashQwen3.Model")
        # log_separator("FlashQwen3Model Forward End")
        
        return BaseModelOutputWithPast(last_hidden_state=hidden_states)


class FlashQwen3(Model):
    def __init__(
        self,
        model_path: Path,
        device: torch.device,
        dtype: torch.dtype,
        pool: str = "last",
        trust_remote: bool = False,
    ):
        config = Qwen3Config.from_pretrained(model_path)

        logger.info(f"Model config: hidden_size={config.hidden_size}, "
            f"intermediate_size={config.intermediate_size}, "
            f"num_layers={config.num_hidden_layers}")

        if hasattr(config, "max_seq_length"):
            self.max_input_length = config.max_seq_length
        else:
            self.max_input_length = config.max_position_embeddings

        # 检测是单文件还是分片模型
        index_file = model_path / "model.safetensors.index.json"
        if index_file.exists():
            # 分片模型
            logger.info(f"----------------------- FlshQwen3 load 分片模型权重")
            with open(index_file, "r") as f:
                index_data = json.load(f)
            weight_map = index_data["weight_map"]
        else:
            # 单文件模型
            logger.info(f"----------------------- FlshQwen3 load 单文件模型")
            weight_map = None

        model = FlashQwen3Model(model_path, weight_map, device, dtype, config)
        self.hidden_size = config.hidden_size
        self.pooling = DefaultPooling(self.hidden_size, pooling_mode=pool)
        self.device = device
        self.dtype = dtype

        super(FlashQwen3, self).__init__(model=model, dtype=dtype, device=device)

    @property
    def batch_type(self) -> Union[FlashBatch, PaddedBatch]:
        # for hpu devices, we use PaddedBatch as we do not have real varlen fwd yet
        return FlashBatch if self.device.type != "hpu" else PaddedBatch

    @tracer.start_as_current_span("embed")
    def embed(self, batch: Union[FlashBatch, PaddedBatch]) -> List[Embedding]:
        # 注释掉性能开销大的日志
        # log_separator("FlashQwen3.embed Start")
        
        if isinstance(batch, PaddedBatch):
            # log_separator("Processing PaddedBatch")
            # log_tensor_stats("Embed.position_ids", batch.position_ids, "FlashQwen3.Embed")
            
            input_lens = batch.attention_mask.cumsum(-1)[:, -1].to(torch.int32)
            max_input_lens = int(input_lens.max().item())
            cu_seqlens = torch.cat(
                (input_lens.new_tensor([0]), input_lens.cumsum(-1).int())
            )
            # log_tensor_stats("Embed.cu_seqlens", cu_seqlens, "FlashQwen3.Embed")
            
            mask = batch.attention_mask.bool()
            _, tgt_len = mask.size()
            attn_mask = _generate_attn_mask(tgt_len, self.dtype).unsqueeze(0).unsqueeze(0)
            
            pooling_attention_mask = batch.attention_mask
            
            output = self.model.forward(
                input_ids=batch.input_ids,
                position_ids=batch.position_ids,
                cu_seqlens=cu_seqlens,
                max_s=max_input_lens,
                mask=mask,
                attn_mask=attn_mask,
            )

            # 增加模块来查看npu每个时刻在计算哪个算子
            # experimental_config = torch_npu.profiler._ExperimentalConfig(profiler_level=torch_npu.profiler.ProfilerLevel.Level2)
            # with torch_npu.profiler.profile(
            #         with_stack=False,  # 采集torch 算子的函数调用栈的开关，该参数选填，默认关闭
            #         record_shapes=False,  # 采集torch 算子的input shape和input type的开关，该参数选填，默认关闭
            #         profile_memory=False,  # 采集memory相关数据的开关，该参数选填，默认关闭
            #         schedule=torch_npu.profiler.schedule(wait=0, active=1),  # warmup默认为0，老版本torch_npu包该参数为必填项
            #         experimental_config=experimental_config,  # 该参数选填，默认为Level0
            #         on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("/tmp/profiling")
            #         # 导出tensorboard可呈现的数据形式，可指定worker_name，默认为：{host名称}_{进程id}
            # ) as prof:
            #     with torch.no_grad():
            #         # output = self.model(**kwargs)
            #         output = self.model.forward(
            #         input_ids=batch.input_ids,
            #         position_ids=batch.position_ids,
            #         cu_seqlens=cu_seqlens,
            #         max_s=max_input_lens,
            #         mask=mask,
            #         attn_mask=attn_mask,
            #         )
            #     prof.step()
            
            hidden_states = output.last_hidden_state
            
            if hidden_states.dim() == 2:
                hidden_states = hidden_states.unsqueeze(1)
            output = BaseModelOutputWithPast(last_hidden_state=hidden_states)
            
        elif isinstance(batch, FlashBatch):
            # log_separator("Processing FlashBatch")
            # log_tensor_stats("Embed.cu_seqlens", batch.cu_seqlens, "FlashQwen3.Embed")

            cu_seqlens = batch.cu_seqlens
            input_ids = batch.input_ids
            position_ids = batch.position_ids

            # Guardrail for true TND path: flatten accidental 2D inputs.
            if input_ids.dim() == 2:
                input_ids = input_ids.reshape(-1)
            if position_ids.dim() == 2:
                position_ids = position_ids.reshape(-1)

            mask = None
            attn_mask = None
            max_input_lens = batch.max_s
            
            seq_lens = (cu_seqlens[1:] - cu_seqlens[:-1]).cpu().tolist()
            seq_lens = [int(s) for s in seq_lens]
            # logger.info(f"[FlashQwen3.Embed] seq_lens={seq_lens}, max_s={max_input_lens}")
            
            output = self.model.forward(
                input_ids=input_ids,
                position_ids=position_ids,
                cu_seqlens=cu_seqlens,
                max_s=max_input_lens,
                mask=mask,
                attn_mask=attn_mask,
            )
            
            hidden_states = output.last_hidden_state
            
            batch_hidden_states = torch.zeros(
                (batch.size, batch.max_s, self.hidden_size),
                dtype=hidden_states.dtype,
                device=hidden_states.device
            )
            
            pooling_attention_mask = torch.zeros(
                (batch.size, batch.max_s),
                dtype=torch.long,
                device=self.device
            )
            
            offset = 0
            for i, seq_len in enumerate(seq_lens):
                batch_hidden_states[i, :seq_len, :] = hidden_states[offset:offset + seq_len, :]
                pooling_attention_mask[i, :seq_len] = 1
                offset += seq_len
            
            output = BaseModelOutputWithPast(last_hidden_state=batch_hidden_states)

        # log_separator("Pooling Start")
        embedding = self.pooling.forward(output, pooling_attention_mask)
        # log_tensor_stats("Embed.embedding_after_pooling", embedding, "FlashQwen3.Embed")
        # log_separator("Pooling End")
        
        cpu_results = embedding.view(-1).tolist()
        # log_separator("FlashQwen3.embed End")

        return [
            Embedding(
                values=cpu_results[i * self.hidden_size : (i + 1) * self.hidden_size]
            )
            for i in range(len(batch))
        ]