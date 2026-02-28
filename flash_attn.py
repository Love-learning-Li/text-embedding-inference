import os

import torch
import torch_npu
from loguru import logger

from text_embeddings_server.utils.device import is_hpu, use_ipex, is_npu

if os.getenv("USE_FLASH_ATTENTION", "").lower() == "false":
    raise ImportError("`USE_FLASH_ATTENTION` is false.")

HAS_FLASH_ATTN = False
HAS_FLASH_ATTN_V2 = False
NPU_COMPRESSED_MASK_SIZE = 2048

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


def _get_npu_compressed_causal_mask(device: torch.device) -> torch.Tensor:
    return torch.triu(
        torch.ones(
            (NPU_COMPRESSED_MASK_SIZE, NPU_COMPRESSED_MASK_SIZE),
            dtype=torch.bool,
            device=device,
        ),
        diagonal=1,
    )


def _eager_varlen_tnd_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    actual_seq_qlen: list,
    softmax_scale: float,
    is_causal: bool,
) -> torch.Tensor:
    out = torch.empty_like(q)
    start = 0
    for end in actual_seq_qlen:
        seq_end = int(end)
        if seq_end <= start:
            continue

        q_i = q[start:seq_end].transpose(0, 1)  # [H, S, D]
        k_i = k[start:seq_end].transpose(0, 1)
        v_i = v[start:seq_end].transpose(0, 1)

        scores = torch.matmul(q_i, k_i.transpose(-1, -2)) * softmax_scale
        if is_causal:
            s = seq_end - start
            causal = torch.triu(
                torch.ones((s, s), dtype=torch.bool, device=q.device), diagonal=1
            )
            scores = scores.masked_fill(causal.unsqueeze(0), float("-inf"))

        probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        out_i = torch.matmul(probs, v_i).transpose(0, 1).contiguous()  # [S, H, D]
        out[start:seq_end].copy_(out_i)
        start = seq_end

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
    
    orig_q_shape = q.shape
    orig_is_4d = q.dim() == 4
    input_was_bhsd = False

    # Convert 4D to 3D if needed
    if q.dim() == 4:
        # Normalize to BSND before flattening.
        # Some callers may pass BHSD.
        if q.shape[1] == num_heads and q.shape[2] != num_heads:
            q = q.transpose(1, 2).contiguous()
            k = k.transpose(1, 2).contiguous()
            v = v.transpose(1, 2).contiguous()
            input_was_bhsd = True

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
    
    sparse_mode = 3 if is_causal else 0

    # npu_fusion_attention expects cumulative lengths without the leading 0.
    if isinstance(seqlen_q, torch.Tensor):
        seqlen_q_list = seqlen_q.tolist()
    else:
        seqlen_q_list = list(seqlen_q)
    if isinstance(seqlen_k, torch.Tensor):
        seqlen_k_list = seqlen_k.tolist()
    else:
        seqlen_k_list = list(seqlen_k)

    actual_seq_qlen = seqlen_q_list[1:] if len(seqlen_q_list) > 0 and seqlen_q_list[0] == 0 else seqlen_q_list
    actual_seq_kvlen = seqlen_k_list[1:] if len(seqlen_k_list) > 0 and seqlen_k_list[0] == 0 else seqlen_k_list

    npu_attn_mask = attn_mask
    if npu_attn_mask is not None:
        while npu_attn_mask.dim() > 2:
            npu_attn_mask = npu_attn_mask.squeeze(0)
        npu_attn_mask = npu_attn_mask.to(torch.bool)

    pre_tockens = 2147483647
    next_tockens = 2147483647
    if is_causal:
        npu_attn_mask = _get_npu_compressed_causal_mask(q.device)
        next_tockens = 0
    
    try:
        out_ = torch_npu.npu_fusion_attention(
            query=q,
            key=k,
            value=v,
            head_num=num_heads,
            input_layout="TND",
            scale=softmax_scale,
            actual_seq_qlen=actual_seq_qlen,
            actual_seq_kvlen=actual_seq_kvlen,
            sparse_mode=sparse_mode,
            atten_mask=npu_attn_mask,
            pre_tockens=pre_tockens,
            next_tockens=next_tockens,
        )[0]
        if orig_is_4d:
            out_view = out_.view(orig_q_shape[0], orig_q_shape[2] if input_was_bhsd else orig_q_shape[1], num_heads, out_.shape[-1])
            if input_was_bhsd:
                out_view = out_view.transpose(1, 2).contiguous()
            out.copy_(out_view)
        else:
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
                atten_mask=npu_attn_mask,
                pre_tockens=pre_tockens,
                next_tockens=next_tockens,
            )[0]
            out_bsnd = out_.squeeze(0)
            if orig_is_4d:
                if input_was_bhsd:
                    out.copy_(out_bsnd.transpose(1, 2).contiguous())
                else:
                    out.copy_(out_bsnd)
            else:
                out.copy_(out_bsnd)
            logger.info(f"[NPU Attention] BSND format succeeded, output shape: {out.shape}")
            return out
        except RuntimeError as e2:
            logger.warning(f"[NPU Attention] BSND format also failed: {e2}")
            logger.warning("[NPU Attention] Falling back to eager varlen attention on NPU")
            out_eager = _eager_varlen_tnd_attn(
                q=q,
                k=k,
                v=v,
                actual_seq_qlen=actual_seq_qlen,
                softmax_scale=softmax_scale,
                is_causal=is_causal,
            )
            if orig_is_4d:
                out_view = out_eager.view(orig_q_shape[0], orig_q_shape[2] if input_was_bhsd else orig_q_shape[1], num_heads, out_eager.shape[-1])
                if input_was_bhsd:
                    out_view = out_view.transpose(1, 2).contiguous()
                out.copy_(out_view)
            else:
                out.copy_(out_eager)
            return out


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
                is_causal=is_causal,
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
