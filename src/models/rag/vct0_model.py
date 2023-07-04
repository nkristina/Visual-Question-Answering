import torch
from torch import nn

from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from transformers import AutoModelForSeq2SeqLM, T5ForConditionalGeneration

class MLP(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def __init__(self, sizes: Tuple[int, ...], bias=True, act=nn.Tanh):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)

class VCT0Model(nn.Module):
    def __init__(
        self,
        prefix_length: int,
        clip_length: Optional[int] = None,
        prefix_size: int = 768,
        num_layers: int = 8,
        mapping_type: str = "mlp",
        model_version: str = "t5_large",
    ):
        super(VCT0Model, self).__init__()
        self.prefix_length = prefix_length
        self.lm = T5ForConditionalGeneration.from_pretrained(model_version)
        self.lm_embedding_size = self.lm.model_dim # dimesnion of hidden state of lm model !
        print("\n\n Using MLP \n\n")
        self.clip_project = MLP(
            (
                prefix_size,
                (self.lm_embedding_size * prefix_length) // 2,
                self.lm_embedding_size * prefix_length,
            )
        )
        print(self.clip_project)

class VCT0Prefix(VCT0Model):
    def parameters(self, recurse: bool = True):
        return self.clip_project.parameters()