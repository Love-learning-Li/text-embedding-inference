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
_NPU_COMPRESSED_MASK_CACHE = {}
_NPU_WARN_ONCE_KEYS = set()

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
    #q,
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
    device_key = f"{device.type}:{device.index}"
    cached = _NPU_COMPRESSED_MASK_CACHE.get(device_key)
    if cached is not None:
        return cached

    mask = torch.triu(
        torch.ones(
            (NPU_COMPRESSED_MASK_SIZE, NPU_COMPRESSED_MASK_SIZE),
            dtype=torch.bool,
            device=device,
        ),
        diagonal=1,
    )
    _NPU_COMPRESSED_MASK_CACHE[device_key] = mask
    return mask


def _npu_warn_once(key: str, message: str):
    if key in _NPU_WARN_ONCE_KEYS:
        return
    _NPU_WARN_ONCE_KEYS.add(key)
    logger.warning(message)


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
    if q.dim() != 3 or k.dim() != 3 or v.dim() != 3:
        raise ValueError(
            f"[NPU Attention] Strict TND is required: q/k/v must be 3D [T,N,D], got q={tuple(q.shape)}, k={tuple(k.shape)}, v={tuple(v.shape)}"
        )
    
    sparse_mode = 3 if is_causal else 0

    # npu_fusion_attention expects cumulative lengths without the leading 0.
    if isinstance(seqlen_q, torch.Tensor):
        seq_q_tensor = seqlen_q[1:] if seqlen_q.numel() > 0 and int(seqlen_q[0]) == 0 else seqlen_q
        actual_seq_qlen = seq_q_tensor.tolist()
    else:
        seq_q_list = list(seqlen_q)
        actual_seq_qlen = seq_q_list[1:] if len(seq_q_list) > 0 and seq_q_list[0] == 0 else seq_q_list

    if isinstance(seqlen_k, torch.Tensor):
        seq_k_tensor = seqlen_k[1:] if seqlen_k.numel() > 0 and int(seqlen_k[0]) == 0 else seqlen_k
        actual_seq_kvlen = seq_k_tensor.tolist()
    else:
        seq_k_list = list(seqlen_k)
        actual_seq_kvlen = seq_k_list[1:] if len(seq_k_list) > 0 and seq_k_list[0] == 0 else seq_k_list

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
        out.copy_(out_)
        return out
    except RuntimeError as e:
        _npu_warn_once("npu_tnd_fallback_eager", f"[NPU Attention] TND fusion failed once: {e}; fallback to eager varlen attention")
        out_eager = _eager_varlen_tnd_attn(
            q=q,
            k=k,
            v=v,
            actual_seq_qlen=actual_seq_qlen,
            softmax_scale=softmax_scale,
            is_causal=is_causal,
        )
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
