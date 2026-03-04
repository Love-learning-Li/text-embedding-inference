from loguru import logger
import torch
import torch_npu
from pathlib import Path
from transformers import AutoTokenizer, AutoConfig
from text_embeddings_server.models.flash_bert import FlashBert
from text_embeddings_server.models.types import FlashBatch, PaddedBatch, Embedding, Score

    
def create_position_ids_from_input_ids(input_ids, max_position_embeddings, past_key_values_length=0):
    position_ids_tmp = torch.arange(max_position_embeddings)
    seq_length = input_ids.size()[-1]
    position_ids = position_ids_tmp[past_key_values_length : seq_length + past_key_values_length]
    return position_ids
    
    
if __name__ == "__main__":
    device_id = 0

    device = torch.device(f"npu:{device_id}")
    torch_npu.npu.set_device(f"npu:{device_id}")

    model_path = "/home/HwHiAiUser/model/bge-large-zh-v1.5/"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    config = AutoConfig.from_pretrained(model_path)
    model = FlashBert(Path(model_path), device, torch.float16)
    
    # docs  = ["Deep learning is a subset of machine learning that uses multi-layered artificial neural networks to learn complex patterns from large datasets. Inspired by the human brain, these models automatically extract features from raw data, enabling end-to-end learning without manual feature engineering.",
    #          "A deep neural network consists of an input layer, multiple hidden layers applying nonlinear transformations, and an output layer producing predictions. Training involves forward propagation to compute outputs and backpropagation with gradient descent to adjust weights and biases, minimizing prediction error."]
    docs = ["Deep Learning is not...", "Deep learning is..."]
 
    
    cu_seq_lengths = []
    cu_seq_lengths.append(0)
    
    sum = 0
    max_length = 0
    all_input_ids = []
    all_position_ids=[]
    for doc in docs:
        encode_pairs = tokenizer([doc], padding=True, truncation=True, max_length=512)
        input_ids = encode_pairs['input_ids']
        logger.info(f"--------------input_ids:{input_ids}")
        sum += len(input_ids[0])
        cu_seq_lengths.append(sum)
        all_input_ids.extend(input_ids[0])
        
        input_ids_tensor = torch.tensor(input_ids[0], dtype=torch.int32)
        position_ids = create_position_ids_from_input_ids(input_ids_tensor, config.max_position_embeddings).tolist()
        all_position_ids.extend(position_ids)
        
    logger.info(f"-------------input_ids:{all_input_ids}")
    logger.info(f"-------------input_ids:{all_position_ids}")
       
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
    
    res=model.embed(flashBatch)
    
    values = [r.values for r in res]
    logger.info(f"000000000000 res:{values}")
    # logger.info(f"000000000000 res:{res}")

    