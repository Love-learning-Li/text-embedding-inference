from loguru import logger
import torch
import torch_npu
from pathlib import Path
from transformers import AutoModelForSequenceClassification,AutoTokenizer, AutoConfig
from text_embeddings_server.models.flash_roberta import FlashRoberta
from text_embeddings_server.models.types import FlashBatch, PaddedBatch, Embedding, Score

    
def calc_softmax(scores):
    scores_tensor = torch.tensor(scores, dtype=torch.float)
    if len(scores) > 1:
        return torch.softmax(scores_tensor, dim=0).tolist()
    else:
        return torch.sigmoid(scores_tensor).tolist()
        

def create_position_ids_from_input_ids(input_ids, padding_idx, past_key_values_length=0):
    """
    Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols
    are ignored. This is modified from fairseq's `utils.make_positions`.

    Args:
        x: torch.Tensor x:

    Returns: torch.Tensor
    """
    # The series of casts and type-conversions here are carefully balanced to both work with ONNX export and XLA.
    mask = input_ids.ne(padding_idx).int()
    incremental_indices = (torch.cumsum(mask, dim=0).type_as(mask) + past_key_values_length) * mask
    return incremental_indices.long() + padding_idx
    
    
if __name__ == "__main__":
    device_id = 0

    device = torch.device(f"npu:{device_id}")
    torch_npu.npu.set_device(f"npu:{device_id}")

    model_path = "/home/HwHiAiUser/model/bge-reranker-large"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    config = AutoConfig.from_pretrained(model_path)
    model = FlashRoberta(Path(model_path), device, torch.float16)
    
    query = "What is Deep Learning?"
    # docs  = ["Deep learning is a subset of machine learning that uses multi-layered artificial neural networks to learn complex patterns from large datasets. Inspired by the human brain, these models automatically extract features from raw data, enabling end-to-end learning without manual feature engineering.",
    #          "A deep neural network consists of an input layer, multiple hidden layers applying nonlinear transformations, and an output layer producing predictions. Training involves forward propagation to compute outputs and backpropagation with gradient descent to adjust weights and biases, minimizing prediction error."]
    docs = ["Deep Learning is not...", "Deep learning is..."]
    # docs = ["Deep learning is a subset of machine learning", "A deep neural network consists of an input layer"]
    sentence_pairs = [[query, doc] for  doc in docs]
    
    cu_seq_lengths = []
    cu_seq_lengths.append(0)
    
    sum = 0
    max_length = 0
    all_input_ids = []
    all_position_ids=[]
    for sentence_pair in sentence_pairs:
        encode_pairs = tokenizer([sentence_pair], padding=True, truncation=True, max_length=512)
        input_ids = encode_pairs['input_ids']
        logger.info(f"--------------input_ids:{input_ids}")
        sum += len(input_ids[0])
        cu_seq_lengths.append(sum)
        all_input_ids.extend(input_ids[0])
        
        input_ids_tensor = torch.tensor(input_ids[0], dtype=torch.int32)
        position_ids = create_position_ids_from_input_ids(input_ids_tensor, config.pad_token_id).to(torch.int32).tolist()
        all_position_ids.extend(position_ids)
        
    logger.info(f"-------------input_ids:{all_input_ids}")
    input_ids_tensor = torch.tensor(all_input_ids, dtype=torch.int32)
    batch_input_ids = torch.tensor(all_input_ids, dtype=torch.int32, device=device)
    batch_token_type_ids = torch.zeros_like(input_ids_tensor, dtype=torch.int32, device=device)
    batch_position_ids = torch.tensor(all_position_ids, dtype=torch.int32, device=device)
    cu_seqlens = torch.tensor(cu_seq_lengths, dtype=torch.int32, device="cpu")

    logger.info(f"-------------------batch_position_ids:{all_position_ids}")

    flashBatch = FlashBatch(
            input_ids=batch_input_ids,
            token_type_ids=batch_token_type_ids,
            position_ids=batch_position_ids,
            cu_seqlens=cu_seqlens,
            max_s=max_length,
            size=len(cu_seqlens) - 1,
        )
    
    res=model.predict(flashBatch)
    
    values = [r.values[0] for r in res]
    logger.info(f"000000000000 res:{values}")
    # logger.info(f"000000000000 res:{res}")
    s = calc_softmax(values)

    logger.info(f"000000000000 res:{s}")
    