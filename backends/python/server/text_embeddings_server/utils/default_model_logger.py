import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Callable
from loguru import logger
import datetime
from contextlib import contextmanager

DEBUG_LOGGING_ENABLED = True

def log_tensor_stats_default(name: str, tensor: torch.Tensor, module_name: str = "DefaultModel"):
    if not DEBUG_LOGGING_ENABLED or tensor is None:
        return
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    shape = tuple(tensor.shape)
    dtype = str(tensor.dtype)
    device = str(tensor.device)
    if tensor.numel() > 0:
        mean_val = tensor.float().mean().item()
        std_val = tensor.float().std().item() if tensor.numel() > 1 else 0.0
        min_val = tensor.float().min().item()
        max_val = tensor.float().max().item()
        logger.info(f"[{timestamp}] [{module_name}] {name} | shape={shape}, dtype={dtype}, device={device}, mean={mean_val:.6f}, std={std_val:.6f}, min={min_val:.6f}, max={max_val:.6f}")
    else:
        logger.info(f"[{timestamp}] [{module_name}] {name} | shape={shape}, dtype={dtype}, device={device}, (empty tensor)")

def log_separator_default(title: str, module_name: str = "DefaultModel"):
    if not DEBUG_LOGGING_ENABLED:
        return
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    logger.info(f"[{timestamp}] [{module_name}] {'='*20} {title} {'='*20}")


class DefaultModelLogger:
    def __init__(self, model: nn.Module):
        self.model = model
        self.handles: List = []
        self.layer_outputs: Dict[str, torch.Tensor] = {}
        self._is_logging = False
        
    def _make_hook(self, name: str, log_input: bool = False, log_output: bool = True) -> Callable:
        def hook(module, input, output):
            if not self._is_logging:
                return
            log_separator_default(f"{name} Start")
            if log_input and input:
                for i, inp in enumerate(input):
                    if isinstance(inp, torch.Tensor):
                        log_tensor_stats_default(f"{name}.input_{i}", inp, "DefaultModel.Hook")
            if log_output:
                if isinstance(output, torch.Tensor):
                    log_tensor_stats_default(f"{name}.output", output, "DefaultModel.Hook")
                    self.layer_outputs[name] = output.detach().clone() if output.requires_grad else output.clone()
                elif isinstance(output, tuple):
                    for i, out in enumerate(output):
                        if isinstance(out, torch.Tensor):
                            log_tensor_stats_default(f"{name}.output_{i}", out, "DefaultModel.Hook")
                            if i == 0:
                                self.layer_outputs[name] = out.detach().clone() if out.requires_grad else out.clone()
            log_separator_default(f"{name} End")
        return hook
    
    def register_hooks(self):
        self._register_embedding_hooks()
        self._register_layer_hooks()
        self._register_norm_hooks()
        
    def _register_embedding_hooks(self):
        if hasattr(self.model, 'embed_tokens'):
            handle = self.model.embed_tokens.register_forward_hook(
                self._make_hook("embed_tokens", log_input=True, log_output=True)
            )
            self.handles.append(handle)
            logger.info("[DefaultModelLogger] Registered hook for embed_tokens")
    
    def _register_layer_hooks(self):
        if hasattr(self.model, 'layers'):
            for idx, layer in enumerate(self.model.layers):
                layer_prefix = f"layer_{idx}"
                self._register_single_layer_hooks(layer, layer_prefix, idx)
                
    def _register_single_layer_hooks(self, layer: nn.Module, prefix: str, layer_idx: int):
        if hasattr(layer, 'input_layernorm'):
            handle = layer.input_layernorm.register_forward_hook(
                self._make_hook(f"{prefix}.input_layernorm", log_input=True, log_output=True)
            )
            self.handles.append(handle)
            
        if hasattr(layer, 'self_attn'):
            self._register_attention_hooks(layer.self_attn, prefix, layer_idx)
            
        if hasattr(layer, 'post_attention_layernorm'):
            handle = layer.post_attention_layernorm.register_forward_hook(
                self._make_hook(f"{prefix}.post_attention_layernorm", log_input=True, log_output=True)
            )
            self.handles.append(handle)
            
        if hasattr(layer, 'mlp'):
            self._register_mlp_hooks(layer.mlp, prefix)
    
    def _register_attention_hooks(self, attn_module: nn.Module, prefix: str, layer_idx: int):
        if hasattr(attn_module, 'q_proj'):
            handle = attn_module.q_proj.register_forward_hook(
                self._make_hook(f"{prefix}.attention.q_proj", log_input=False, log_output=True)
            )
            self.handles.append(handle)
            
        if hasattr(attn_module, 'k_proj'):
            handle = attn_module.k_proj.register_forward_hook(
                self._make_hook(f"{prefix}.attention.k_proj", log_input=False, log_output=True)
            )
            self.handles.append(handle)
            
        if hasattr(attn_module, 'v_proj'):
            handle = attn_module.v_proj.register_forward_hook(
                self._make_hook(f"{prefix}.attention.v_proj", log_input=False, log_output=True)
            )
            self.handles.append(handle)
            
        if hasattr(attn_module, 'q_norm'):
            handle = attn_module.q_norm.register_forward_hook(
                self._make_hook(f"{prefix}.attention.q_norm", log_input=True, log_output=True)
            )
            self.handles.append(handle)
            
        if hasattr(attn_module, 'k_norm'):
            handle = attn_module.k_norm.register_forward_hook(
                self._make_hook(f"{prefix}.attention.k_norm", log_input=True, log_output=True)
            )
            self.handles.append(handle)
            
        if hasattr(attn_module, 'o_proj'):
            handle = attn_module.o_proj.register_forward_hook(
                self._make_hook(f"{prefix}.attention.o_proj", log_input=True, log_output=True)
            )
            self.handles.append(handle)
    
    def _register_mlp_hooks(self, mlp_module: nn.Module, prefix: str):
        if hasattr(mlp_module, 'gate_proj'):
            handle = mlp_module.gate_proj.register_forward_hook(
                self._make_hook(f"{prefix}.mlp.gate_proj", log_input=True, log_output=True)
            )
            self.handles.append(handle)
            
        if hasattr(mlp_module, 'up_proj'):
            handle = mlp_module.up_proj.register_forward_hook(
                self._make_hook(f"{prefix}.mlp.up_proj", log_input=True, log_output=True)
            )
            self.handles.append(handle)
            
        if hasattr(mlp_module, 'down_proj'):
            handle = mlp_module.down_proj.register_forward_hook(
                self._make_hook(f"{prefix}.mlp.down_proj", log_input=True, log_output=True)
            )
            self.handles.append(handle)
    
    def _register_norm_hooks(self):
        if hasattr(self.model, 'norm'):
            handle = self.model.norm.register_forward_hook(
                self._make_hook("final_norm", log_input=True, log_output=True)
            )
            self.handles.append(handle)
            logger.info("[DefaultModelLogger] Registered hook for final norm")
    
    def remove_hooks(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        logger.info("[DefaultModelLogger] All hooks removed")
        
    @contextmanager
    def logging_context(self):
        self._is_logging = True
        try:
            yield self
        finally:
            self._is_logging = False
            
    def forward_with_logging(self, *args, **kwargs) -> Any:
        with self.logging_context():
            log_separator_default("DefaultModel Forward Start")
            if args:
                for i, arg in enumerate(args):
                    if isinstance(arg, torch.Tensor):
                        log_tensor_stats_default(f"Forward.arg_{i}", arg, "DefaultModel")
            if kwargs:
                for key, value in kwargs.items():
                    if isinstance(value, torch.Tensor):
                        log_tensor_stats_default(f"Forward.kwarg_{key}", value, "DefaultModel")
            
            output = self.model(*args, **kwargs)
            
            if isinstance(output, torch.Tensor):
                log_tensor_stats_default("Forward.output", output, "DefaultModel")
            elif hasattr(output, 'last_hidden_state'):
                log_tensor_stats_default("Forward.last_hidden_state", output.last_hidden_state, "DefaultModel")
            
            log_separator_default("DefaultModel Forward End")
            return output
    
    def get_layer_outputs(self) -> Dict[str, torch.Tensor]:
        return self.layer_outputs.copy()
    
    def clear_layer_outputs(self):
        self.layer_outputs.clear()


class DefaultModelWrapper:
    def __init__(self, model_path: str, device: torch.device, dtype: torch.dtype, trust_remote: bool = True):
        from transformers import AutoModel
        import sys
        default_model_path = "c:\\Users\\l50056623\\Desktop\\text-embeddings-inference\\backends\\python\\server\\text_embeddings_server\\Default_Model"
        if default_model_path not in sys.path:
            sys.path.insert(0, default_model_path)
        
        self.model = (
            AutoModel.from_pretrained(model_path, trust_remote_code=trust_remote)
            .to(dtype)
            .to(device)
            .eval()
        )
        self.device = device
        self.dtype = dtype
        self.logger = DefaultModelLogger(self.model)
        self.logger.register_hooks()
        
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, position_ids: Optional[torch.Tensor] = None) -> Any:
        return self.logger.forward_with_logging(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
    def get_layer_outputs(self) -> Dict[str, torch.Tensor]:
        return self.logger.get_layer_outputs()
    
    def remove_hooks(self):
        self.logger.remove_hooks()
