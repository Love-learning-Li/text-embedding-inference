import torch
import torch_npu
import json
import os
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

tracer = trace.get_tracer(__name__)
from loguru import logger

ENABLE_ONCE_INFO_LOG = False
_ONCE_INFO_KEYS = set()


def logger_info_once(key: str, message: str):
    if not ENABLE_ONCE_INFO_LOG:
        return
    if key in _ONCE_INFO_KEYS:
        return
    _ONCE_INFO_KEYS.add(key)
    logger.info(message)

def load_weight(model_path, weight_map, name, dtype, device):
    """
    Helper function to load a weight tensor from safetensors.
    """
    if weight_map is None:
        with safe_open(f"{model_path}/model.safetensors", framework="pt") as f:
            return f.get_tensor(name).to(dtype).to(device)
    else:
        target_file = weight_map[name]
        with safe_open(f"{model_path}/{target_file}", framework="pt") as f:
            return f.get_tensor(name).to(dtype).to(device)


# def apply_rotary_pos_emb_npu1(q, k, cos, sin, unsqueeze_dim=1):
#     t, n, d = q.shape
    
#     def _normalize_coeff_for_basnd(coeff: torch.Tensor) -> Optional[torch.Tensor]:
#         if coeff.dim() == 3 and coeff.shape[0] == 1 and coeff.shape[1] == t and coeff.shape[2] == d:
#             coeff_t1d = coeff.squeeze(0).unsqueeze(1)
#         elif coeff.dim() == 2 and coeff.shape[0] == t and coeff.shape[1] == d:
#             coeff_t1d = coeff.unsqueeze(1)
#         elif coeff.dim() == 3 and coeff.shape[0] == t and coeff.shape[1] == 1 and coeff.shape[2] == d:
#             coeff_t1d = coeff
#         else:    
#             return None
#         return coeff_t1d.permute(1, 0, 2).unsqueeze(0).contiguous()
    
#     cos_bnsd = _normalize_coeff_for_basnd(cos)
#     sin_bnsd = _normalize_coeff_for_basnd(sin)
#     if cos_bnsd is None or sin_bnsd is None:
#         return q, k
    
#     q_bnsd = q.permute(1, 0, 2).unsqueeze(0).contiguous()
#     k_bnsd = k.permute(1, 0, 2).unsqueeze(0).contiguous()
#     q_bnsd = q_bnsd.to(device=q.device, dtype=q.dtype)
#     k_bnsd = k_bnsd.to(device=q.device, dtype=q.dtype)
    
#     cos_bnsd = cos_bnsd.to(device=q.device, dtype=q.dtype)
#     sin_bnsd = sin_bnsd.to(device=q.device, dtype=q.dtype)
    
#     q_embed_bnsd = torch_npu.npu_rotary_mul(q_bnsd, cos_bnsd, sin_bnsd)
#     k_embed_bnsd = torch_npu.npu_rotary_mul(k_bnsd, cos_bnsd, sin_bnsd)
    
#     q_embed = q_embed_bnsd.squeeze(0).permute(1, 0, 2).contiguous()
#     k_embed = k_embed_bnsd.squeeze(0).permute(1, 0, 2).contiguous()   
#     return q_embed, k_embed
        

def apply_rotary_pos_emb_npu(q, k, cos, sin, unsqueeze_dim=1):
    
    enable_fp32_compute = False
    
    def _pre_process(
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Size, torch.dtype]:
            origin_shape = x.shape
            if len(origin_shape) == 3:
                # x: [seq_len, num_heads, head_size]
                x = x.unsqueeze(0)

            origin_dtype = x.dtype
            if enable_fp32_compute:
                x = x.float()
                cos = cos.float()
                sin = sin.float()

            return x, cos, sin, origin_shape, origin_dtype
        
    def _post_process(
        output: torch.Tensor,
        origin_shape: torch.Size,
        origin_dtype: torch.dtype,
    ) -> torch.Tensor:
        if len(origin_shape) == 3:
            output = output.squeeze(0)
        if enable_fp32_compute:
            output = output.to(origin_dtype)
        return output
    
    # q_ [1, seq_len, num_heads, head_size]
    # k_ [1, seq_len, num_heads // 2, head_size]
    q_, cos, sin, q_origin_shape, q_origin_dtype = _pre_process(q, cos, sin)
    k_, cos, sin, k_origin_shape, k_origin_dtype = _pre_process(k, cos, sin)
    
    head_dim = q_.shape[-1]

    # cos, sin: [1, seq_len, 1, head_dim]
    cos = cos.reshape(1, -1, 1, head_dim)
    sin = sin.reshape(1, -1, 1, head_dim)
    logger_info_once(
        "rope_npu_coeff_shape",
        f"[RoPE] coeff shape: cos={tuple(cos.shape)}, sin={tuple(sin.shape)}",
    )
    logger_info_once(
        "rope_npu_qk_shape",
        f"[RoPE] q/k shape before npu_rotary_mul: q={tuple(q_.shape)}, k={tuple(k_.shape)}",
    )
    output_q = torch_npu.npu_rotary_mul(q_, cos, sin)
    output_k = torch_npu.npu_rotary_mul(k_, cos, sin)

    output_q = _post_process(output_q, q_origin_shape, q_origin_dtype)
    output_k = _post_process(output_k, k_origin_shape, k_origin_dtype)

    return output_q, output_k
    
def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    
    if q.dim() != 3 or k.dim !=3:
        return q, k 
    t, n, d = q.shape
    if cos.dim() == 3 and cos.shape[0] == 1 and cos.shape[1] == t and cos.shape[2] == d:
        cos_npu = cos.squeeze(0).unsqueeze(1)
    elif cos.dim == 2 and cos.shape[0] == t and cos.shape[1] == d:
        cos_npu = cos.unsqueeze(1)
    elif cos.dim() == 3 and cos.shape[0] == t and cos.shape[1] in (1, n) and cos.shape[2] == d:
        cos_npu = cos
    else:
        return q, k
    
    if sin.dim() == 3 and sin.shape[0] == 1 and sin.shape[1] == t and sin.shape[2] == d:
        sin_npu = sin.squeeze(0).unsqueeze(1)
    elif sin.dim == 2 and sin.shape[0] == t and sin.shape[1] == d:
        sin_npu = sin.unsqueeze(1)
    elif sin.dim() == 3 and sin.shape[0] == t and sin.shape[1] in (1, n) and sin.shape[2] == d:
        sin_npu = sin
    else:
        return q, k

    cos_npu = cos_npu.to(device =q.device, dtype = q.dtype)
    sin_npu = sin_npu.to(device =q.device, dtype = q.dtype)
    q_embed = torch_npu.npu_rotray_mul(q, cos_npu, sin_npu)
    k_embed = torch_npu.npu_rotray_mul(k, cos_npu, sin_npu)
    
    return q_embed, k_embed


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
            return torch_npu.npu_rms_norm(hidden_states.to(input_dtype), self.weight, epsilon = self.variance_epsilon)[0]


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
        
        self.layer_idx = layer_idx

    def forward(
        self, hidden_states, position_embeddings, cu_seqlens, max_s, attn_mask=None
    ):
        input_shape = hidden_states.shape[:-1]
		
        q = self.q_norm.forward(
            torch_npu.npu_linear(hidden_states, self.q_proj_weight, bias = None).view(*input_shape, self.num_heads, self.head_dim)
        )
		
        k = self.k_norm.forward(
            torch_npu.npu_linear(hidden_states, self.k_proj_weight, bias = None).view(*input_shape, self.num_key_value_heads, self.head_dim)
        )
		
        v = torch_npu.npu_linear(
			hidden_states, self.v_proj_weight, bias = None).view(*input_shape, self.num_key_value_heads, self.head_dim
		)
		
        cos, sin = position_embeddings
        if self.layer_idx == 0:
            logger_info_once(
                "attn_layer0_qkvcos_shape",
                f"[Attention][layer0] q={tuple(q.shape)}, k={tuple(k.shape)}, v={tuple(v.shape)}, cos={tuple(cos.shape)}, sin={tuple(sin.shape)}",
            )
        q, k = apply_rotary_pos_emb_npu(q, k, cos, sin, unsqueeze_dim=1)

        if self.num_key_value_groups > 1:
            k = k.repeat_interleave(self.num_key_value_groups, dim=1)
            v= v.repeat_interleave(self.num_key_value_groups, dim=1)
           
        if self.layer_idx == 0:
            logger_info_once(
                "attn_layer0_qkv_after_gqa",
                f"[Attention][layer0] q/k/v after gqa: q={tuple(q.shape)}, k={tuple(k.shape)}, v={tuple(v.shape)}",
            )
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
        if self.layer_idx == 0:
            logger_info_once(
                "attn_layer0_output_shape",
                f"[Attention][layer0] attn_output shape={tuple(attn_output.shape)}",
            )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = torch_npu.npu_linear(attn_output, self.o_proj_weight, bias = None)

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
        self.gate_up_proj_weight = torch.cat([self.gate_proj_weight, self.up_proj_weight], dim=0)
        self.down_proj_weight = load_weight(
            model_path,
            weight_map,
            f"layers.{layer_idx}.mlp.down_proj.weight",
            dtype,
            device,
        )
        # self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_state):
		# gated_hidden_states = torch_npu.npu_linear(hidden_state, self.gate_proj_weight, bias = None)
		# uped_hidden_states = torch_npu.npu_linear(hidden_state, self.up_proj_weight, bias = None)
        		
        #return torch_npu.npu_linear(
		#	self.act_fn(gated_hidden_states) * uped_hidden_states,
		#	self.down_proj_weight, 
		#	bias = None
		#)
        gate_up_states = torch_npu.npu_linear(hidden_state, self.gate_up_proj_weight, bias=None)
        hidden_states = torch_npu.npu_swiglu(gate_up_states, dim=-1)

        return torch_npu.npu_linear(
            hidden_states,
            self.down_proj_weight, 
            bias = None
        )



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
        residual = hidden_states
        hidden_states = self.input_layernorm.forward(hidden_states)
        # Self Attention
        hidden_states = self.attention.forward(
            hidden_states, position_embeddings, cu_seqlens, max_s, attn_mask
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm.forward(hidden_states)
        hidden_states = self.mlp.forward(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class Qwen3RotaryEmbedding(nn.Module):
    def __init__(self, config: Qwen3Config, device=None):
        super().__init__()
        inv_freq, self.attention_scaling = compute_default_rope_parameters(
            config, device
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x, position_ids):
        if position_ids.dim() == 1:
            position_ids = position_ids.unsqueeze(0)
        # inv_freq     [1, head_dim // 2]
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
            else "cpu"
        )
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (
                inv_freq_expanded.float() @ position_ids_expanded.float()
            ).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class FlashQwen3Model:
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`MistralDecoderLayer`]

    Args:
        config: MistralConfig
    """

    def __init__(self, model_path, weight_map, device, dtype, config: Qwen3Config):
        self.word_embeddings_weight = load_weight(
            model_path,
            weight_map,
            "embed_tokens.weight",
            dtype,
            device,
        )
        self.layers = [
            Qwen3DecoderLayer(
                model_path,
                weight_map,
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
            weight_map,
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
        inputs_embeds = nn.functional.embedding(input_ids, self.word_embeddings_weight)
        hidden_states = inputs_embeds
        logger_info_once("model_position_ids_shape", f"[Model] position_ids shape={tuple(position_ids.shape)}")
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for layer in self.layers:
            hidden_states = layer.forward(
                hidden_states, position_embeddings, cu_seqlens, max_s, attn_mask
            )
        hidden_states = self.norm.forward(hidden_states)
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

        if hasattr(config, "max_seq_length"):
            self.max_input_length = config.max_seq_length
        else:
            self.max_input_length = config.max_position_embeddings

        index_file = model_path / "model.safetensors.index.json"
        if index_file.exists():
            with open(index_file, "r") as f:
                index_data = json.load(f)
            weight_map = index_data["weight_map"]
        else:
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
        if isinstance(batch, PaddedBatch):
            logger_info_once("embed_padded_input_ids_shape", f"[Embed] PaddedBatch input_ids shape={tuple(batch.input_ids.shape)}")
            input_lens = batch.attention_mask.cumsum(-1)[:, -1].to(torch.int32)
            max_input_lens = 0
            cu_seqlens = torch.cat(
                (input_lens.new_tensor([0]), input_lens.cumsum(-1).int())
            )
            mask = batch.attention_mask.bool()
            bsz, tgt_len = mask.size()
            min_val = torch.finfo(self.dtype).min
            attn_mask = torch.full(
                [bsz, 1, tgt_len, tgt_len],
                fill_value=min_val,
                device=self.device,
                dtype=self.dtype,
            )
            expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, tgt_len)
            attn_mask = attn_mask.masked_fill(expanded_mask, 0.0)
        elif isinstance(batch, FlashBatch):
            cu_seqlens = batch.cu_seqlens
            mask = None
            attn_mask = None
            # mask_flag = torch.ones((2048, 2048), dtype=torch.bool).tril_()
            # mask_flag = ~mask_flag
            # mask_value = float("-inf") if self.dtype == torch.float16 else 1
            # attn_mask = torch.zeros(size=(2048, 2048), dtype=torch.bool).masked_fill_(mask_flag, mask_value).to(self.device)
            # attn_mask = torch.triu(torch.ones((2048, 2048), dtype=torch.bool, device=self.device), diagonal=1)
            max_input_lens = batch.max_s

        output = self.model.forward(
            input_ids=batch.input_ids,
            position_ids=batch.position_ids,
            cu_seqlens=cu_seqlens,
            max_s=max_input_lens,
            mask=mask,
            attn_mask=attn_mask,
        )
        
        last_token_indices = cu_seqlens[1:] - 1
        hidden_states = output.last_hidden_state.cpu()
        logger_info_once("embed_hidden_states_shape", f"[Embed] hidden_states shape={tuple(hidden_states.shape)}")
        embedding = hidden_states[last_token_indices]
        logger_info_once("embed_cu_seqlens", f"[Embed] cu_seqlens={cu_seqlens.tolist()}")
        logger_info_once("embed_embedding_shape", f"[Embed] embedding shape={tuple(embedding.shape)}")
        # embedding = self.pooling.forward(output, None)
        cpu_results = embedding.view(-1).tolist()

        return [
            Embedding(
                values=cpu_results[i * self.hidden_size : (i + 1) * self.hidden_size]
            )
            for i in range(len(batch))
        ]
