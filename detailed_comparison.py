import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Tuple
import sys
import os
from loguru import logger
import datetime
import json

sys.path.insert(0, str(Path(__file__).parent / "backends" / "python" / "server"))

from transformers import AutoTokenizer, AutoModel

def setup_detailed_logging():
    logger.remove()
    log_format = "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {message}"
    logger.add(sys.stdout, format=log_format, level="INFO")
    
    log_dir = Path(__file__).parent / "logs"
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    flash_log = log_dir / f"flash_model_{timestamp}.log"
    default_log = log_dir / f"default_model_{timestamp}.log"
    comparison_log = log_dir / f"comparison_{timestamp}.log"
    
    return flash_log, default_log, comparison_log

def extract_tensor_stats(tensor: torch.Tensor) -> Dict:
    if tensor is None:
        return {"error": "None tensor"}
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "mean": float(tensor.float().mean().item()),
        "std": float(tensor.float().std().item()) if tensor.numel() > 1 else 0.0,
        "min": float(tensor.float().min().item()),
        "max": float(tensor.float().max().item()),
        "norm": float(torch.norm(tensor.float()).item()),
    }

def compare_tensors(t1: torch.Tensor, t2: torch.Tensor, name: str) -> Dict:
    if t1 is None or t2 is None:
        return {"name": name, "error": "One or both tensors are None"}
    
    if t1.shape != t2.shape:
        return {
            "name": name,
            "error": f"Shape mismatch: {t1.shape} vs {t2.shape}",
            "t1_shape": list(t1.shape),
            "t2_shape": list(t2.shape),
        }
    
    diff = (t1.float() - t2.float()).abs()
    cos_sim = F.cosine_similarity(t1.flatten().unsqueeze(0), t2.flatten().unsqueeze(0)).item()
    
    return {
        "name": name,
        "cosine_similarity": float(cos_sim),
        "mean_abs_diff": float(diff.mean().item()),
        "max_abs_diff": float(diff.max().item()),
        "t1_stats": extract_tensor_stats(t1),
        "t2_stats": extract_tensor_stats(t2),
    }

class DetailedFlashModelLogger:
    def __init__(self, model_path: str, device: torch.device, dtype: torch.dtype):
        from text_embeddings_server.models.flash_qwen3 import FlashQwen3
        
        self.device = device
        self.dtype = dtype
        self.intermediate_outputs: Dict[str, torch.Tensor] = {}
        
        self.model_wrapper = FlashQwen3(
            model_path=Path(model_path),
            device=device,
            dtype=dtype,
            pool="last",
            trust_remote=True
        )
        self.model = self.model_wrapper.model
        
    def capture_embedding_output(self, tensor: torch.Tensor):
        self.intermediate_outputs["inputs_embeds"] = tensor.clone()
        
    def capture_layer_output(self, layer_idx: int, tensor: torch.Tensor, stage: str):
        key = f"layer_{layer_idx}_{stage}"
        self.intermediate_outputs[key] = tensor.clone()
        
    def capture_final_hidden(self, tensor: torch.Tensor):
        self.intermediate_outputs["final_hidden_states"] = tensor.clone()
        
    def forward(self, input_ids: torch.Tensor, position_ids: torch.Tensor, 
                cu_seqlens: torch.Tensor, max_s: int) -> Dict[str, torch.Tensor]:
        import torch.nn.functional as F
        from text_embeddings_server.models.flash_qwen3_SignleSafetensors import log_tensor_stats, log_separator
        
        with torch.no_grad():
            inputs_embeds = torch.nn.functional.embedding(input_ids, self.model.word_embeddings_weight)
            self.capture_embedding_output(inputs_embeds)
            
            hidden_states = inputs_embeds
            position_embeddings = self.model.rotary_emb(hidden_states, position_ids)
            self.intermediate_outputs["position_embeddings_cos"] = position_embeddings[0].clone()
            self.intermediate_outputs["position_embeddings_sin"] = position_embeddings[1].clone()
            
            for idx, layer in enumerate(self.model.layers):
                residual = hidden_states
                hidden_states = layer.input_layernorm.forward(hidden_states)
                self.capture_layer_output(idx, hidden_states, "after_input_ln")
                
                hidden_states = layer.attention.forward(
                    hidden_states, position_embeddings, cu_seqlens, max_s, None
                )
                self.capture_layer_output(idx, hidden_states, "after_attn")
                
                hidden_states = residual + hidden_states
                self.capture_layer_output(idx, hidden_states, "after_residual1")
                
                residual = hidden_states
                hidden_states = layer.post_attention_layernorm.forward(hidden_states)
                self.capture_layer_output(idx, hidden_states, "after_post_ln")
                
                hidden_states = layer.mlp.forward(hidden_states)
                self.capture_layer_output(idx, hidden_states, "after_mlp")
                
                hidden_states = residual + hidden_states
                self.capture_layer_output(idx, hidden_states, "after_residual2")
            
            hidden_states = self.model.norm.forward(hidden_states)
            self.capture_final_hidden(hidden_states)
            
        return self.intermediate_outputs

class DetailedDefaultModelLogger:
    def __init__(self, model_path: str, device: torch.device, dtype: torch.dtype):
        self.device = device
        self.dtype = dtype
        self.intermediate_outputs: Dict[str, torch.Tensor] = {}
        self.handles: List = []
        
        self.model = AutoModel.from_pretrained(
            model_path, 
            trust_remote_code=True
        ).to(dtype).to(device).eval()
        
        self._register_hooks()
        
    def _make_hook(self, name: str):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                self.intermediate_outputs[name] = output.detach().clone()
            elif isinstance(output, tuple) and len(output) > 0 and isinstance(output[0], torch.Tensor):
                self.intermediate_outputs[name] = output[0].detach().clone()
        return hook
    
    def _register_hooks(self):
        if hasattr(self.model, 'embed_tokens'):
            self.handles.append(
                self.model.embed_tokens.register_forward_hook(self._make_hook("inputs_embeds"))
            )
        
        if hasattr(self.model, 'layers'):
            for idx, layer in enumerate(self.model.layers):
                if hasattr(layer, 'input_layernorm'):
                    self.handles.append(
                        layer.input_layernorm.register_forward_hook(
                            self._make_hook(f"layer_{idx}_after_input_ln")
                        )
                    )
                if hasattr(layer, 'post_attention_layernorm'):
                    self.handles.append(
                        layer.post_attention_layernorm.register_forward_hook(
                            self._make_hook(f"layer_{idx}_after_post_ln")
                        )
                    )
                if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'o_proj'):
                    self.handles.append(
                        layer.self_attn.o_proj.register_forward_hook(
                            self._make_hook(f"layer_{idx}_after_attn")
                        )
                    )
                if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'down_proj'):
                    self.handles.append(
                        layer.mlp.down_proj.register_forward_hook(
                            self._make_hook(f"layer_{idx}_after_mlp")
                        )
                    )
        
        if hasattr(self.model, 'norm'):
            self.handles.append(
                self.model.norm.register_forward_hook(self._make_hook("final_hidden_states"))
            )
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, 
                position_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        self.intermediate_outputs.clear()
        
        with torch.no_grad():
            output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids
            )
        
        return self.intermediate_outputs
    
    def remove_hooks(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()

def run_detailed_comparison():
    model_path = "/home/HwHiAiUser/model/Qwen3-Embedding-0.6B"
    input_text = "The capital of China is Beijing."
    device = torch.device("cpu")
    dtype = torch.float32
    
    flash_log, default_log, comparison_log = setup_detailed_logging()
    
    logger.info("="*80)
    logger.info("Detailed Model Comparison")
    logger.info(f"Model: {model_path}")
    logger.info(f"Input: '{input_text}'")
    logger.info(f"Device: {device}, Dtype: {dtype}")
    logger.info("="*80)
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
    
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    seq_len = input_ids.shape[1]
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    cu_seqlens = torch.tensor([0, seq_len], device=device, dtype=torch.int32)
    
    logger.info(f"\nInput IDs: {input_ids.tolist()}")
    logger.info(f"Decoded: {tokenizer.decode(input_ids[0])}")
    
    logger.info("\n" + "="*80)
    logger.info("Running FlashQwen3 Model...")
    logger.info("="*80)
    
    flash_logger = DetailedFlashModelLogger(model_path, device, dtype)
    flash_outputs = flash_logger.forward(input_ids, position_ids, cu_seqlens, seq_len)
    
    logger.info("\n" + "="*80)
    logger.info("Running Default Model...")
    logger.info("="*80)
    
    default_logger = DetailedDefaultModelLogger(model_path, device, dtype)
    default_outputs = default_logger.forward(input_ids, attention_mask, position_ids)
    
    logger.info("\n" + "="*80)
    logger.info("Comparing Intermediate Outputs")
    logger.info("="*80)
    
    comparison_results = []
    
    common_keys = set(flash_outputs.keys()) & set(default_outputs.keys())
    all_keys = set(flash_outputs.keys()) | set(default_outputs.keys())
    
    for key in sorted(all_keys):
        if key in flash_outputs and key in default_outputs:
            result = compare_tensors(flash_outputs[key], default_outputs[key], key)
            comparison_results.append(result)
            
            status = "✓" if result.get("cosine_similarity", 0) > 0.99 else "✗"
            logger.info(f"{status} {key}:")
            if "error" not in result:
                logger.info(f"    Cosine Similarity: {result['cosine_similarity']:.6f}")
                logger.info(f"    Mean Abs Diff: {result['mean_abs_diff']:.6f}")
                logger.info(f"    Max Abs Diff: {result['max_abs_diff']:.6f}")
            else:
                logger.info(f"    Error: {result['error']}")
        elif key in flash_outputs:
            logger.info(f"⚠ {key}: Only in FlashQwen3")
        else:
            logger.info(f"⚠ {key}: Only in Default Model")
    
    output_dir = Path(__file__).parent / "comparison_results"
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    with open(output_dir / f"comparison_{timestamp}.json", "w") as f:
        json.dump(comparison_results, f, indent=2, default=str)
    
    logger.info(f"\nResults saved to: {output_dir / f'comparison_{timestamp}.json'}")
    
    default_logger.remove_hooks()
    
    return comparison_results

if __name__ == "__main__":
    run_detailed_comparison()
