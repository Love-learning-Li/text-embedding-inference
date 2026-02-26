import os

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

if is_npu:
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
    attn_mask,
    seqlen_q,
    seqlen_k,
    max_seqlen_q,
    max_seqlen_k,
    softmax_scale,
    is_causal=False,
    ):
    # npu_fusion_attention for FlashBatch
    # Handle 4D input [batch, seq, heads, dim] -> convert to 3D [total_seq, heads, dim]
    
    # logger.info(f"[NPU Attention] Input shapes - q: {q.shape}, k: {k.shape}, v: {v.shape}")
    # logger.info(f"[NPU Attention] num_heads: {num_heads}, scale: {softmax_scale}")
    # logger.info(f"[NPU Attention] seqlen_q: {seqlen_q}, seqlen_k: {seqlen_k}")
    # logger.info(f"[NPU Attention] max_seqlen_q: {max_seqlen_q}, is_causal: {is_causal}")
    
    # Convert 4D to 3D if needed
    if q.dim() == 4:
        batch_size, seq_len, heads, head_dim = q.shape
        q = q.reshape(batch_size * seq_len, heads, head_dim)
        k = k.reshape(batch_size * seq_len, heads, head_dim)
        
        # v might already be 3D [batch, heads, dim] or 4D [batch, seq, heads, dim]
        if v.dim() == 4:
            v = v.reshape(batch_size * seq_len, heads, head_dim)
        
        # logger.info(f"[NPU Attention] Reshaped to 3D - q: {q.shape}, k: {k.shape}, v: {v.shape}")
    elif v.dim() == 3 and q.dim() == 3:
        # Both q, k, v are already 3D, ensure they're in TND format [total_seq, heads, dim]
        # logger.info(f"[NPU Attention] Already 3D - q: {q.shape}, k: {k.shape}, v: {v.shape}")
        pass
    
    # For varlen/FlashBatch, use TND format
    # sparse_mode=2 for causal (rightDownCausal) attention
    if is_causal:
        sparse_mode = 2  
    else:
        sparse_mode = 0
    
    try:
        out_ = torch_npu.npu_fusion_attention(
            query=q,
            key=k,
            value=v,
            head_num=num_heads,
            input_layout="TND",
            scale=softmax_scale,
            actual_seq_qlen=seqlen_q.tolist(),
            actual_seq_kvlen=seqlen_k.tolist(),
            sparse_mode=sparse_mode,
            atten_mask=None,
            pre_tockens=2147483647,
            next_tockens=0 if is_causal else 2147483647,
        )[0]
        out.copy_(out_)
        # logger.info(f"[NPU Attention] TND format succeeded, output shape: {out_.shape}")
        return out
    except RuntimeError as e:
        logger.warning(f"[NPU Attention] TND format failed: {e}")
        
        # Fallback to BSND format: [batch, seq, heads, dim]
        try:
            q_bsnd = q.unsqueeze(0)
            k_bsnd = k.unsqueeze(0)
            v_bsnd = v.unsqueeze(0)
            
            out_ = torch_npu.npu_fusion_attention(
                query=q_bsnd,
                key=k_bsnd,
                value=v_bsnd,
                head_num=num_heads,
                input_layout="BSND",
                scale=softmax_scale,
                sparse_mode=sparse_mode,
                atten_mask=None,
                pre_tockens=2147483647,
                next_tockens=0 if is_causal else 2147483647,
            )[0]
            out.copy_(out_.squeeze(0))
            logger.info(f"[NPU Attention] BSND format succeeded, output shape: {out.shape}")
            return out
        except RuntimeError as e2:
            logger.warning(f"[NPU Attention] BSND format also failed: {e2}")
            raise RuntimeError(f"NPU attention failed: TND failed with {e}, BSND failed with {e2}")


def attention(
    q, k, v, num_heads, out, cu_seqlens, max_s, softmax_scale, is_causal=False, attn_mask=None
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
