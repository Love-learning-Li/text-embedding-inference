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
SHAPE_LOG_LEVEL = "warning"
_NPU_ROTARY_TND_AVAILABLE = True


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


def log_shape_once(key: str, message: str):
    log_once(SHAPE_LOG_LEVEL, key, message)

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
    return _apply_rotary_pos_emb_npu_rotary(q, k, cos, sin, unsqueeze_dim)


def _apply_rotary_pos_emb_native(q, k, cos, sin, unsqueeze_dim=1):
    """
    Native implementation using direct matrix multiplication.
    Optimal for TND format - no dimension conversion needed.
    
    Formula: output = x * cos + rotate_half(x) * sin
    """
    if q.dim() != 3 or k.dim() != 3:
        return q, k

    t, n, d = q.shape

    def _normalize_coeff_for_tnd_native(coeff: torch.Tensor) -> Optional[torch.Tensor]:
        if coeff.dim() == 2 and coeff.shape == (t, d):
            return coeff.unsqueeze(1)
        if coeff.dim() == 3:
            if coeff.shape[0] == 1 and coeff.shape[1] == t and coeff.shape[2] == d:
                return coeff.squeeze(0).unsqueeze(1)
            if coeff.shape[0] == t and coeff.shape[1] in (1, n) and coeff.shape[2] == d:
                return coeff
        return None

    cos_native = _normalize_coeff_for_tnd_native(cos)
    sin_native = _normalize_coeff_for_tnd_native(sin)

    if cos_native is None or sin_native is None:
        log_once("warning", "rope_native_bad_coeff_layout", f"[RoPE] native fallback skipped: unsupported coeff shape, cos={tuple(cos.shape)}, sin={tuple(sin.shape)}, q={tuple(q.shape)}")
        return q, k

    cos_native = cos_native.to(device=q.device, dtype=q.dtype)
    sin_native = sin_native.to(device=q.device, dtype=q.dtype)

    q_embed = (q * cos_native) + (rotate_half(q) * sin_native)
    k_embed = (k * cos_native) + (rotate_half(k) * sin_native)

    return q_embed, k_embed


def _apply_rotary_pos_emb_npu_rotary(q, k, cos, sin, unsqueeze_dim=1):
    """
    Implementation using torch_npu.npu_rotary_mul.
    For TND input, convert to 4D BNSD to satisfy RotaryMul constraints:
    - q/k: [T, N, D] -> [1, N, T, D]
    - cos/sin: [T, 1, D] -> [1, 1, T, D]
    Then convert outputs back to TND.
    """
    global _NPU_ROTARY_TND_AVAILABLE

    if not _NPU_ROTARY_TND_AVAILABLE:
        return _apply_rotary_pos_emb_native(q, k, cos, sin, unsqueeze_dim)

    if q.dim() != 3 or k.dim() != 3:
        log_once("warning", "rope_non_tnd_input", "[RoPE] npu_rotary_mul skipped: only TND 3D input is supported in this model")
        return q, k

    t, n, d = q.shape
    if d % 2 != 0:
        log_once("warning", "rope_last_dim_not_even", "[RoPE] npu_rotary_mul skipped: last dim is not even, keep q/k unchanged")
        return q, k

    def _normalize_coeff_for_bnsd(coeff: torch.Tensor) -> Optional[torch.Tensor]:
        if coeff.dim() == 3 and coeff.shape[0] == 1 and coeff.shape[1] == t and coeff.shape[2] == d:
            coeff_t1d = coeff.squeeze(0).unsqueeze(1)
        elif coeff.dim() == 2 and coeff.shape[0] == t and coeff.shape[1] == d:
            coeff_t1d = coeff.unsqueeze(1)
        elif coeff.dim() == 3 and coeff.shape[0] == t and coeff.shape[1] == 1 and coeff.shape[2] == d:
            coeff_t1d = coeff
        else:
            return None

        # [T, 1, D] -> [1, 1, T, D] for BNSD input
        return coeff_t1d.permute(1, 0, 2).unsqueeze(0).contiguous()

    cos_bnsd = _normalize_coeff_for_bnsd(cos)
    sin_bnsd = _normalize_coeff_for_bnsd(sin)

    if cos_bnsd is None:
        log_once("warning", "rope_bad_cos_layout", f"[RoPE] unsupported cos shape={tuple(cos.shape)} for q={tuple(q.shape)}")
        return q, k
    if sin_bnsd is None:
        log_once("warning", "rope_bad_sin_layout", f"[RoPE] unsupported sin shape={tuple(sin.shape)} for q={tuple(q.shape)}")
        return q, k

    q_bnsd = q.permute(1, 0, 2).unsqueeze(0).contiguous()
    k_bnsd = k.permute(1, 0, 2).unsqueeze(0).contiguous()

    log_shape_once(
        "rope_tnd_to_bnsd_shape_once",
        f"[RoPE][shape] TND->BNSD: q={tuple(q.shape)}->{tuple(q_bnsd.shape)}, k={tuple(k.shape)}->{tuple(k_bnsd.shape)}, cos={tuple(cos.shape)}->{tuple(cos_bnsd.shape)}, sin={tuple(sin.shape)}->{tuple(sin_bnsd.shape)}",
    )

    try:
        q_bnsd = q_bnsd.to(device=q.device, dtype=q.dtype)
        k_bnsd = k_bnsd.to(device=k.device, dtype=k.dtype)
        cos_bnsd = cos_bnsd.to(device=q.device, dtype=q.dtype)
        sin_bnsd = sin_bnsd.to(device=q.device, dtype=q.dtype)

        q_embed_bnsd = torch_npu.npu_rotary_mul(q_bnsd, cos_bnsd, sin_bnsd)
        k_embed_bnsd = torch_npu.npu_rotary_mul(k_bnsd, cos_bnsd, sin_bnsd)

        q_embed = q_embed_bnsd.squeeze(0).permute(1, 0, 2).contiguous()
        k_embed = k_embed_bnsd.squeeze(0).permute(1, 0, 2).contiguous()
        log_once("info", "use_npu_rotary_mul_once", "_apply_rotary_pos_emb_npu_rotary")
        return q_embed, k_embed
    except RuntimeError as error:
        _NPU_ROTARY_TND_AVAILABLE = False
        log_once("warning", "rope_runtime_error_disable_npu", f"[RoPE] npu_rotary_mul failed and disabled for this process: {error}; switch to native TND RoPE")
        return _apply_rotary_pos_emb_native(q, k, cos, sin, unsqueeze_dim)

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
        self.layer_idx = layer_idx
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
        
        if hidden_states.dim() != 2:
            raise ValueError(f"[Qwen3Attention] TND required: hidden_states must be 2D [T,H], got {tuple(hidden_states.shape)}")

        input_shape = hidden_states.shape[:-1]
        q_proj_out = F.linear(hidden_states, self.q_proj_weight)
        q = self.q_norm.forward(
            q_proj_out.view(*input_shape, self.num_heads, self.head_dim)
        )

        k_proj_out = F.linear(hidden_states, self.k_proj_weight)
        k = self.k_norm.forward(
            k_proj_out.view(*input_shape, self.num_key_value_heads, self.head_dim)
        )

        v = F.linear(hidden_states, self.v_proj_weight).view(
            *input_shape, self.num_key_value_heads, self.head_dim
        )

        cos, sin = position_embeddings
        if self.layer_idx == 0:
            log_shape_once(
                "attn_layer0_tnd_shapes",
                f"[Attention][shape] layer0 TND shapes: hidden={tuple(hidden_states.shape)}, q={tuple(q.shape)}, k={tuple(k.shape)}, v={tuple(v.shape)}, cos={tuple(cos.shape)}, sin={tuple(sin.shape)}, max_s={max_s}",
            )

        q, k = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)

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

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
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
        self.pool_mode = getattr(getattr(self.pooling, "pooling", None), "pooling_mode", "last")
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
        
        if isinstance(batch, FlashBatch):
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

            if input_ids.dim() != 1 or position_ids.dim() != 1:
                raise ValueError(
                    f"[FlashQwen3.embed] TND required: input_ids/position_ids must be 1D after flatten, got input_ids={tuple(input_ids.shape)}, position_ids={tuple(position_ids.shape)}"
                )

            log_shape_once(
                "embed_tnd_input_shapes",
                f"[Embed][shape] FlashBatch shapes: input_ids={tuple(input_ids.shape)}, position_ids={tuple(position_ids.shape)}, cu_seqlens={tuple(cu_seqlens.shape)}, max_s={batch.max_s}, pool={self.pool_mode}",
            )

            mask = None
            attn_mask = None
            max_input_lens = batch.max_s
            
            output = self.model.forward(
                input_ids=input_ids,
                position_ids=position_ids,
                cu_seqlens=cu_seqlens,
                max_s=max_input_lens,
                mask=mask,
                attn_mask=attn_mask,
            )
            
            hidden_states = output.last_hidden_state

            seq_lens_tensor = cu_seqlens[1:] - cu_seqlens[:-1]
            if self.pool_mode in ("last", "last_token") and torch.all(seq_lens_tensor > 0):
                last_indices = (cu_seqlens[1:] - 1).to(device=hidden_states.device, dtype=torch.long)
                embedding = hidden_states.index_select(0, last_indices)
                cpu_results = embedding.view(-1).tolist()
                return [
                    Embedding(
                        values=cpu_results[i * self.hidden_size : (i + 1) * self.hidden_size]
                    )
                    for i in range(len(batch))
                ]

            batch_hidden_states = torch.zeros(
                (batch.size, batch.max_s, self.hidden_size),
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )

            pooling_attention_mask = torch.zeros(
                (batch.size, batch.max_s),
                dtype=torch.long,
                device=self.device,
            )

            offset = 0
            seq_lens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
            for i, seq_len in enumerate(seq_lens):
                batch_hidden_states[i, :seq_len, :] = hidden_states[offset:offset + seq_len, :]
                pooling_attention_mask[i, :seq_len] = 1
                offset += seq_len

            output = BaseModelOutputWithPast(last_hidden_state=batch_hidden_states)
        else:
            raise TypeError(
                f"[FlashQwen3.embed] FlashBatch(TND) is required on NPU, got {type(batch).__name__}"
            )

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