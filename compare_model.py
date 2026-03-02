import torch
import torch_npu
import torch.nn.functional as F
from pathlib import Path
from typing import List, Optional
import sys
import os
from loguru import logger
import datetime

sys.path.insert(0, str(Path(__file__).parent / "backends" / "python" / "server"))

from transformers import AutoTokenizer

def setup_logging():
    logger.remove()
    log_format = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    logger.add(sys.stdout, format=log_format, level="INFO")
    
    log_dir = Path(__file__).parent / "logs"
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"model_comparison_{timestamp}.log"
    logger.add(log_file, format=log_format, level="INFO")
    logger.info(f"Log file: {log_file}")
    return log_file

def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    a_flat = a.flatten().float()
    b_flat = b.flatten().float()
    return F.cosine_similarity(a_flat.unsqueeze(0), b_flat.unsqueeze(0)).item()

def last_token_pooling(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    last_indices = attention_mask.sum(dim=1, keepdim=True) - 1
    last_indices = last_indices.clamp(min=0)
    last_indices_expanded = last_indices.unsqueeze(-1).expand(-1, -1, hidden_states.shape[-1])
    return hidden_states.gather(1, last_indices_expanded).squeeze(1)

def test_flash_qwen3_model(model_path: str, input_text: str, device: torch.device, dtype: torch.dtype):
    from text_embeddings_server.models.flash_qwen3 import FlashQwen3
    from text_embeddings_server.models.types import FlashBatch
    
    logger.info("="*80)
    logger.info("Testing FlashQwen3 Model")
    logger.info("="*80)
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
    
    input_ids_2d = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    seq_len = input_ids_2d.shape[1]
    # FlashBatch/TND expects flattened token stream: [total_tokens]
    input_ids = input_ids_2d.reshape(-1)
    position_ids = torch.arange(seq_len, device=device, dtype=torch.int32)
    # Keep cu_seqlens on CPU to match TEI FlashBatch convention
    cu_seqlens = torch.tensor([0, seq_len], device="cpu", dtype=torch.int32)
    
    logger.info(f"Input text: '{input_text}'")
    logger.info(f"Input IDs shape (2D): {input_ids_2d.shape}")
    logger.info(f"Input IDs: {input_ids_2d.tolist()}")
    logger.info(f"Attention mask: {attention_mask.tolist()}")
    logger.info(f"Position IDs (TND): {position_ids.tolist()}")
    logger.info(f"cu_seqlens (CPU): {cu_seqlens.tolist()}")
    
    model = FlashQwen3(
        model_path=Path(model_path),
        device=device,
        dtype=dtype,
        pool="last",
        trust_remote=True
    )
    
    token_type_ids = torch.zeros_like(input_ids, dtype=torch.int32)
    
    batch = FlashBatch(
        input_ids=input_ids,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        cu_seqlens=cu_seqlens,
        max_s=seq_len,
        size=1
    )
    
    with torch.no_grad():
        embeddings = model.embed(batch)
    
    embedding_tensor = torch.tensor(embeddings[0].values, device=device, dtype=dtype)
    logger.info(f"FlashQwen3 embedding shape: {embedding_tensor.shape}")
    logger.info(f"FlashQwen3 embedding mean: {embedding_tensor.mean().item():.6f}")
    logger.info(f"FlashQwen3 embedding std: {embedding_tensor.std().item():.6f}")
    
    return embedding_tensor

def test_default_model(model_path: str, input_text: str, device: torch.device, dtype: torch.dtype):
    from text_embeddings_server.utils.default_model_logger import DefaultModelWrapper
    
    logger.info("="*80)
    logger.info("Testing Default Model (AutoModel)")
    logger.info("="*80)
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
    
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    position_ids = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)
    
    logger.info(f"Input text: '{input_text}'")
    logger.info(f"Input IDs shape: {input_ids.shape}")
    logger.info(f"Input IDs: {input_ids.tolist()}")
    logger.info(f"Attention mask: {attention_mask.tolist()}")
    logger.info(f"Position IDs: {position_ids.tolist()}")
    
    model_wrapper = DefaultModelWrapper(
        model_path=model_path,
        device=device,
        dtype=dtype,
        trust_remote=True
    )
    
    with torch.no_grad():
        output = model_wrapper.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids
        )
    
    hidden_states = output.last_hidden_state
    logger.info(f"Default model hidden states shape: {hidden_states.shape}")
    logger.info(f"Default model hidden states mean: {hidden_states.mean().item():.6f}")
    logger.info(f"Default model hidden states std: {hidden_states.std().item():.6f}")
    
    embedding = last_token_pooling(hidden_states, attention_mask)
    logger.info(f"Default model embedding shape: {embedding.shape}")
    logger.info(f"Default model embedding mean: {embedding.mean().item():.6f}")
    logger.info(f"Default model embedding std: {embedding.std().item():.6f}")
    
    model_wrapper.remove_hooks()
    
    return embedding.squeeze(0)

def compare_models():
    model_path = "/home/HwHiAiUser/model/Qwen3-Embedding-0.6B"
    input_text = "The capital of China is Beijing."
    device = torch.device("npu")
    dtype = torch.float32
    
    log_file = setup_logging()
    
    logger.info("="*80)
    logger.info("Starting Model Comparison Test")
    logger.info(f"Model path: {model_path}")
    logger.info(f"Input text: '{input_text}'")
    logger.info(f"Device: {device}")
    logger.info(f"Dtype: {dtype}")
    logger.info("="*80)
    
    flash_embedding = test_flash_qwen3_model(model_path, input_text, device, dtype)
    
    logger.info("\n" + "="*80 + "\n")
    
    default_embedding = test_default_model(model_path, input_text, device, dtype)
    
    logger.info("\n" + "="*80)
    logger.info("Comparison Results")
    logger.info("="*80)
    
    similarity = cosine_similarity(flash_embedding, default_embedding)
    
    logger.info(f"Cosine Similarity: {similarity:.6f}")
    logger.info(f"FlashQwen3 embedding norm: {torch.norm(flash_embedding).item():.6f}")
    logger.info(f"Default model embedding norm: {torch.norm(default_embedding).item():.6f}")
    
    diff = (flash_embedding - default_embedding).abs()
    logger.info(f"Mean absolute difference: {diff.mean().item():.6f}")
    logger.info(f"Max absolute difference: {diff.max().item():.6f}")
    
    if similarity > 0.99:
        logger.success(f"Models are highly similar! (similarity={similarity:.6f})")
    elif similarity > 0.9:
        logger.warning(f"Models have moderate similarity (similarity={similarity:.6f})")
    else:
        logger.error(f"Models have low similarity! (similarity={similarity:.6f})")
    
    logger.info(f"\nFull log saved to: {log_file}")
    
    return similarity

if __name__ == "__main__":
    compare_models()

