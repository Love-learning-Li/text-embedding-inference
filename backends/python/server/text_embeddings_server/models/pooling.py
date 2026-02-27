from abc import ABC, abstractmethod

import torch
from opentelemetry import trace
from torch import Tensor

tracer = trace.get_tracer(__name__)


class _Pooling(ABC):
    @abstractmethod
    def forward(self, model_output, attention_mask) -> Tensor:
        pass


class MeanPooling(_Pooling):
    def __init__(self, hidden_size, pooling_mode="mean") -> None:
        self.pooling_mode = pooling_mode

    @tracer.start_as_current_span("pooling")
    def forward(self, model_output, attention_mask) -> Tensor:
        token_embeddings = model_output[0]
        attention_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * attention_mask_expanded, dim=1)
        sum_mask = torch.clamp(attention_mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask


class MaxPooling(_Pooling):
    def __init__(self, hidden_size, pooling_mode="max") -> None:
        self.pooling_mode = pooling_mode

    @tracer.start_as_current_span("pooling")
    def forward(self, model_output, attention_mask) -> Tensor:
        token_embeddings = model_output[0]
        attention_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        token_embeddings[attention_mask_expanded == 0] = -1e9
        return torch.max(token_embeddings, dim=1).values


class CLSPooling(_Pooling):
    def __init__(self, hidden_size, pooling_mode="cls") -> None:
        self.pooling_mode = pooling_mode

    @tracer.start_as_current_span("pooling")
    def forward(self, model_output, attention_mask) -> Tensor:
        token_embeddings = model_output[0]
        return token_embeddings[:, 0, :]


class LastTokenPooling(_Pooling):
    def __init__(self, hidden_size, pooling_mode="last") -> None:
        self.pooling_mode = pooling_mode
        self.hidden_size = hidden_size

    @tracer.start_as_current_span("pooling")
    def forward(self, model_output, attention_mask) -> Tensor:
        token_embeddings = model_output[0]
        attention_mask = attention_mask.to(dtype=torch.bool)
        
        batch_size = token_embeddings.shape[0]
        seq_length = token_embeddings.shape[1]
        
        last_indices = attention_mask.sum(dim=1, keepdim=True) - 1
        last_indices = last_indices.clamp(min=0)
        
        last_indices_expanded = last_indices.unsqueeze(-1).expand(-1, -1, token_embeddings.shape[-1])
        
        last_token_embeddings = token_embeddings.gather(1, last_indices_expanded).squeeze(1)
        
        return last_token_embeddings


class DefaultPooling(_Pooling):
    def __init__(self, hidden_size, pooling_mode="last") -> None:
        assert (
            pooling_mode != "splade"
        ), "Splade pooling is not supported for DefaultPooling"
        
        if pooling_mode == "mean":
            self.pooling = MeanPooling(hidden_size, pooling_mode)
        elif pooling_mode == "max":
            self.pooling = MaxPooling(hidden_size, pooling_mode)
        elif pooling_mode == "cls":
            self.pooling = CLSPooling(hidden_size, pooling_mode)
        elif pooling_mode == "last" or pooling_mode == "last_token":
            self.pooling = LastTokenPooling(hidden_size, pooling_mode)
        else:
            self.pooling = LastTokenPooling(hidden_size, "last")

    @tracer.start_as_current_span("pooling")
    def forward(self, model_output, attention_mask) -> Tensor:
        return self.pooling.forward(model_output, attention_mask)


class SpladePooling(_Pooling):
    @tracer.start_as_current_span("pooling")
    def forward(self, model_output, attention_mask) -> Tensor:
        hidden_states = torch.relu(model_output[0])
        hidden_states = (1 + hidden_states).log()
        hidden_states = torch.mul(hidden_states, attention_mask.unsqueeze(-1))
        return hidden_states.max(dim=1).values
