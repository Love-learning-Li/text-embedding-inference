import os
import torch

from loguru import logger
from pathlib import Path
from typing import Optional
from transformers import AutoConfig
from transformers.models.bert import BertConfig

from text_embeddings_server.models.model import Model
from text_embeddings_server.models.masked_model import MaskedLanguageModel
from text_embeddings_server.models.default_model import DefaultModel
from text_embeddings_server.models.classification_model import ClassificationModel
# from text_embeddings_server.models.qwen3_rerank_model import Qwen3RerankModel
# from text_embeddings_server.models.unixcoder_model import UniXcoderModel
from text_embeddings_server.models.flash_qwen3 import FlashQwen3, FlashQwen3Model

from text_embeddings_server.utils.device import get_device, use_ipex

__all__ = ["Model"]

TRUST_REMOTE_CODE = os.getenv("TRUST_REMOTE_CODE", "false").lower() in ["true", "1"]
# Disable gradients
torch.set_grad_enabled(False)

FLASH_ATTENTION = True
try:
    from text_embeddings_server.models.flash_bert import FlashBert
    from text_embeddings_server.models.flash_roberta import FlashRoberta
except ImportError as e:
    logger.warning(f"Could not import Flash Attention enabled models: {e}")
    FLASH_ATTENTION = False

if FLASH_ATTENTION:
    __all__.append(FlashBert)
    __all__.append(FlashRoberta)
    __all__.append(FlashQwen3)


def wrap_model_if_hpu(model_handle, device):
    """Wrap the model in HPU graph if the device is HPU."""
    if device.type == "hpu":
        from habana_frameworks.torch.hpu import wrap_in_hpu_graph

        model_handle.model = wrap_in_hpu_graph(
            model_handle.model, disable_tensor_cache=DISABLE_TENSOR_CACHE
        )
    return model_handle


def create_model(model_class, model_path, device, datatype, pool="cls"):
    """Create a model instance and wrap it if needed."""
    model_handle = model_class(
        model_path,
        device,
        datatype,
        pool,
        trust_remote=TRUST_REMOTE_CODE,
    )
    return wrap_model_if_hpu(model_handle, device)


def get_model(model_path: Path, dtype: Optional[str], pool: str):
    if dtype == "float32":
        datatype = torch.float32
    elif dtype == "float16":
        datatype = torch.float16
    elif dtype == "bfloat16":
        datatype = torch.bfloat16
    else:
        raise RuntimeError(f"Unknown dtype {dtype}")


    enable_boost = os.getenv("ENABLE_BOOST", "True")
    if enable_boost not in("True", "False"):
        raise ValueError("env ENABLE_BOOST value must be True or False")
    
    if enable_boost == "True":
        dtype == torch.float16
        
    device = get_device()
    logger.info(f"__init__.py -> get device -> backend device: {device} {device.type}")

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=TRUST_REMOTE_CODE)

    if (
        hasattr(config, "auto_map")
        and isinstance(config.auto_map, dict)
        and "AutoModel" in config.auto_map
        and config.auto_map["AutoModel"]
        == "jinaai/jina-bert-v2-qk-post-norm--modeling_bert.JinaBertModel"
    ):
        # Add specific offline modeling for model "jinaai/jina-embeddings-v2-base-code" which uses "autoMap" to reference code in other repository
        return create_model(FlashJinaBert, model_path, device, datatype)

    if config.model_type == "bert":
        config: BertConfig
        if (
            use_ipex()
            or device.type in ["cuda", "hpu", "npu"]
            and config.position_embedding_type == "absolute"
            and datatype in [torch.float16, torch.bfloat16]
            and FLASH_ATTENTION
        ):
            if pool != "cls":
                if config.architectures[0].endswith("ForMaskedLM") and pool == "splade":
                    return create_model(
                        MaskedLanguageModel, model_path, device, datatype, pool
                    )
                return create_model(DefaultModel, model_path, device, datatype, pool)

            try:
                logger.info(f"----------------------- use flash bert")
                return create_model(FlashBert, model_path, device, datatype)
            except FileNotFoundError:
                logger.info(
                    "Do not have safetensors file for this model, use default transformers model path instead"
                )
                return create_model(DefaultModel, model_path, device, datatype, pool)

        if config.architectures[0].endswith("Classification"):
            logger.info(f"----------------------- use flash robert")
            return create_model(RobertaForSequenceClassification, model_path, device, datatype)
        elif os.getenv("IS_RERANK", None):
            return create_model(Qwen3RerankModel, model_path, device, datatype)
        elif config.architectures[0].endswith("ForMaskedLM") and pool == "splade":
            return create_model(MaskedLanguageModel, model_path, device, datatype)
        elif str(model_path).endswith("unixcoder-base"):
            return create_model(UniXcoderModel, model_path, device, datatype)
        else:
            return create_model(DefaultModel, model_path, device, datatype, pool)

    if config.model_type == "mistral" and device.type == "hpu":
        try:
            return create_model(FlashMistral, model_path, device, datatype, pool)
        except FileNotFoundError:
            return create_model(DefaultModel, model_path, device, datatype, pool)

    if config.model_type == "qwen3" and device.type == "hpu":
        try:
            return create_model(FlashQwen3, model_path, device, datatype, pool)
        except FileNotFoundError:
            return create_model(DefaultModel, model_path, device, datatype, pool)
    
    if config.model_type == "qwen3" and device.type == "npu":
        try:
            logger.info(f"----------------------- use npu in my modified flash qwen3 model")
            return create_model(FlashQwen3, model_path, device, datatype, pool)
        except FileNotFoundError as e:
            logger.warning(f"FlashQwen3 failed with FileNotFoundError: {e}")  # 添加详细错误信息
            logger.info(f"----------------------- use npu in default flash qwen3 model")
            return create_model(DefaultModel, model_path, device, datatype, pool)
        except Exception as e:
            logger.error(f"FlashQwen3 failed with unexpected error: {type(e).__name__}: {e}")  # 捕获其他异常
            raise

    # Default case
    if config.architectures[0].endswith("Classification"):
        logger.info(f"----------------------- use flash robert")
        return create_model(FlashRoberta, model_path, device, datatype)
        # return create_model(ClassificationModel, model_path, device, datatype)
    elif config.architectures[0].endswith("ForMaskedLM") and pool == "splade":
        return create_model(MaskedLanguageModel, model_path, device, datatype)
    else:
        return create_model(DefaultModel, model_path, device, datatype, pool)
