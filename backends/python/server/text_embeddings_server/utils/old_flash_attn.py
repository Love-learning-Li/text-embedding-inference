import os
from typing import Optional

import torch
import torch_npu
from loguru import logger

from text_embeddings_server.utils.device import is_hpu, use_ipex, is_npu

if os.getenv("USE_FLASH_ATTENTION", "").lower() == "false":
    raise ImportError("`USE_FLASH_ATTENTION` is false.")

HAS_FLASH_ATTN = False
HAS_FLASH_ATTN_V2 = False

is_hpu = is_hpu()
use_ipex = use_ipex()
is_npu = is_npu()


def _actual_seq_lengths_for_tnd_fia(
    cu_seqlens: torch.Tensor,
    total_tokens: int,
    tensor_name: str,
) -> list[int]:
    actual_seq_lens = cu_seqlens.tolist()
    if actual_seq_lens and actual_seq_lens[0] == 0:
        actual_seq_lens = actual_seq_lens[1:]

    if actual_seq_lens and actual_seq_lens[-1] != total_tokens:
        raise ValueError(
            f"For TND layout, {tensor_name} actual_seq_lengths must be cumulative "
            f"and end at total tokens. Got last={actual_seq_lens[-1]}, "
            f"total_tokens={total_tokens}, raw_cu_seqlens={cu_seqlens.tolist()}"
        )

    return actual_seq_lens


if is_npu:
    if not hasattr(torch_npu, "npu_fused_infer_attention_score"):
        raise ImportError(
            "NPU attention requires torch_npu.npu_fused_infer_attention_score."
        )
    HAS_FLASH_ATTN = True
elif use_ipex or is_hpu:
    HAS_FLASH_ATTN_V2 = True
else:
    if not torch.cuda.is_available():
        raise ImportError("CUDA is not available")

    major, minor = torch.cuda.get_device_capability()
    is_sm75 = major == 7 and minor == 5
    is_sm8x = major == 8 and minor >= 0
    is_sm90 = major == 9 and minor == 0

    try:
        try:
            import flash_attn_2_cuda
        except ImportError:
            raise ImportError(
                "Flash Attention V2 is not installed.\n"
                "Use the official Docker image (ghcr.io/huggingface/text-embeddings-inference:cuda-latest) "
                "or install flash attention v2 with `cd server && make install install-flash-attention-v2`"
            )
        if not (is_sm8x or is_sm90):
            raise ImportError(
                f"GPU with CUDA capability {major} {minor} is not supported for "
                "Flash Attention V2"
            )
        HAS_FLASH_ATTN_V2 = True
    except ImportError as e:
        try:
            import flash_attn_cuda
        except ImportError:
            raise ImportError(
                "Flash Attention is not installed.\n"
                "Use the official Docker image (ghcr.io/huggingface/text-embeddings-inference:cuda-latest) "
                "or install flash attention with `cd server && make install install-flash-attention`"
            ) from e

        if not (is_sm75 or is_sm8x or is_sm90):
            raise ImportError(
                f"GPU with CUDA capability {major} {minor} is not supported"
            ) from e
        logger.warning(f"Unable to use Flash Attention V2: {e}")
        HAS_FLASH_ATTN = True


def hpu_attn(
    q,
    k,
    v,
    out,
    attn_mask,
    seqlen_q,
    seqlen_k,
    max_seqlen_q,
    max_seqlen_k,
    softmax_scale,
    is_causal=False,
):
    from habana_frameworks.torch.hpex.kernels import FusedSDPA

    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    if is_causal:
        attn_mask = None

    out_ = FusedSDPA.apply(
        q, k, v, attn_mask, 0.0, is_causal, softmax_scale, "fast", False
    )
    out_ = out_.transpose(1, 2)
    out.copy_(out_)
    return out

def npu_attn(
    q,
    k,
    v,
    num_heads,
    out,
    _attn_mask,
    seqlen_q,
    seqlen_k,
    _max_seqlen_q,
    _max_seqlen_k,
    softmax_scale,
    is_causal=False,
    num_key_value_heads: Optional[int] = None,
):
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    attn_mask_npu = None
    sparse_mode = 0
    if is_causal:
        attn_mask_npu = torch.triu(
            torch.ones((2048, 2048), dtype=torch.bool, device=q.device),
            diagonal=1,
        ).contiguous()
        sparse_mode = 3

    actual_seq_lengths = _actual_seq_lengths_for_tnd_fia(
        seqlen_q,
        q.shape[0],
        "query",
    )
    actual_seq_lengths_kv = _actual_seq_lengths_for_tnd_fia(
        seqlen_k,
        k.shape[0],
        "key/value",
    )
    num_key_value_heads = (
        num_heads if num_key_value_heads is None else num_key_value_heads
    )

    try:
        out_ = torch_npu.npu_fused_infer_attention_score(
            query=q,
            key=k,
            value=v,
            num_heads=num_heads,
            num_key_value_heads=num_key_value_heads,
            input_layout="TND",
            scale=softmax_scale,
            actual_seq_lengths=actual_seq_lengths,
            actual_seq_lengths_kv=actual_seq_lengths_kv,
            sparse_mode=sparse_mode,
            atten_mask=attn_mask_npu,
        )[0]
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "npu_fused_infer_attention_score failed with "
            f"q_shape={tuple(q.shape)}, k_shape={tuple(k.shape)}, "
            f"v_shape={tuple(v.shape)}, num_heads={num_heads}, "
            f"num_key_value_heads={num_key_value_heads}, "
            f"actual_seq_lengths={actual_seq_lengths}, "
            f"actual_seq_lengths_kv={actual_seq_lengths_kv}, "
            f"is_causal={is_causal}, sparse_mode={sparse_mode}"
        ) from exc

    out.copy_(out_)
    return out


def attention(
    q,
    k,
    v,
    num_heads,
    out,
    cu_seqlens,
    max_s,
    softmax_scale,
    is_causal=False,
    attn_mask=None,
    num_key_value_heads: Optional[int] = None,
):
    if HAS_FLASH_ATTN_V2:
        if use_ipex:
            import intel_extension_for_pytorch as ipex

            return ipex.llm.functional.varlen_attention(
                q.contiguous() if q.device.type == "xpu" else q,
                k.contiguous() if k.device.type == "xpu" else k,
                v.contiguous() if v.device.type == "xpu" else v,
                out,
                cu_seqlens,
                cu_seqlens,
                None,
                max_s,
                max_s,
                0,
                softmax_scale,
                zero_tensors=False,
                is_causal=False,
                return_softmax=False,
                gen_=None,
                )

        elif is_hpu:
            return hpu_attn(
                q,
                k,
                v,
                out,
                attn_mask,
                cu_seqlens,
                cu_seqlens,
                max_s,
                max_s,
                softmax_scale,
                is_causal,
            )

        else:
            return flash_attn_2_cuda.varlen_fwd(
                q,
                k,
                v,
                out,
                cu_seqlens,
                cu_seqlens,
                max_s,
                max_s,
                0.0,
                softmax_scale,
                False,
                is_causal,
                -1,
                -1,
                False,
                None,
            )

    if HAS_FLASH_ATTN:
        if is_npu:
            return npu_attn(
                q,
                k,
                v,
                num_heads,
                out,
                attn_mask,
                cu_seqlens,
                cu_seqlens,
                max_s,
                max_s,
                softmax_scale,
                is_causal,
                num_key_value_heads=num_key_value_heads,
            )

        else:
            return flash_attn_cuda.fwd(
                q,
                k,
                v,
                out,
                cu_seqlens,
                cu_seqlens,
                max_s,
                max_s,
                0.0,
                softmax_scale,
                False,
                is_causal,
                False,
                0,
                None,
            )

    raise NotImplementedError("flash attention is not installed")

