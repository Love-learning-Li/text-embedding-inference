import inspect
import torch
import os

from pathlib import Path
from typing import Type, List
from transformers import AutoModelForSequenceClassification, Qwen3ForCausalLM, AutoTokenizer, AutoModelForCausalLM
from opentelemetry import trace

from text_embeddings_server.models import Model
from text_embeddings_server.models.types import PaddedBatch, Embedding, Score

tracer = trace.get_tracer(__name__)


class Qwen3RerankModel(Model):
    def __init__(
        self,
        model_path: Path,
        device: torch.device,
        dtype: torch.dtype,
        pool: str = "cls",
        trust_remote: bool = False,
    ):

        # Check environment variable to decide reranker mode
        self.qwen3_mode = os.environ.get("IS_RERANK", "0") == "1"
        position_offset = 0
        # Load tokenizer (for both modes)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left",
                                                        trust_remote_code=trust_remote)
        # 为 Qwen3 模型设置 pad_token 以支持 batch 推理
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # -------------------------------
        #  Qwen3-Reranker 初始化逻辑
        # -------------------------------
        self.model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=trust_remote)
        self.model = self.model.to(dtype).to(device).eval()
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        # 用于从 logits 提取 "yes" 和 "no" 的得分
        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
        prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.prefix_tokens = self.tokenizer.encode(prefix, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(suffix, add_special_tokens=False)

        if hasattr(self.model.config, "max_seq_length"):
            self.max_input_length = self.model.config.max_seq_length
        else:
            self.max_input_length = (
                    self.model.config.max_position_embeddings - position_offset
            )

        super(Qwen3RerankModel, self).__init__(
            model=self.model, dtype=dtype, device=device
        )


    @property
    def batch_type(self) -> Type[PaddedBatch]:
        return PaddedBatch

    @tracer.start_as_current_span("embed")
    def embed(self, batch: PaddedBatch) -> List[Embedding]:
        pass

    @tracer.start_as_current_span("predict")
    def predict(self, batch: PaddedBatch) -> List[Score]:
        kwargs = {"input_ids": batch.input_ids, "attention_mask": batch.attention_mask}
        # Qwen3-Reranker 的打分逻辑（prompt 生成 + yes/no logits）
        input_ids = []
        for ele in batch.input_ids:
            ids = self.prefix_tokens + ele.tolist() + self.suffix_tokens
            input_ids.append(ids)

        # 注意：传入的是 list 而非 tensor
        tokenized = self.tokenizer.pad(
            {"input_ids": input_ids},
            padding=True,
            return_tensors="pt",
            max_length=self.max_input_length
         )
        inputs = {k: v.to(self.model.device) for k, v in tokenized.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[:, -1, :]  # 最后一个 token 的预测分布

            # 提取 "yes" 和 "no" 的概率，作为是否相关的判断
            true_logits = logits[:, self.token_true_id]
            false_logits = logits[:, self.token_false_id]
            logit_diff = true_logits - false_logits
        return [Score(values=[p.item()]) for p in logit_diff]


