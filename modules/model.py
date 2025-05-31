import os
from typing import TYPE_CHECKING
import torch
import torch.nn as nn
from transformers import DebertaV2Model
from transformers.models.deberta_v2 import modeling_deberta_v2


@torch.jit.script
def make_log_bucket_position(relative_pos, bucket_size: int, max_position: int):
    sign = torch.sign(relative_pos)
    mid = bucket_size // 2
    abs_pos = torch.where(
        (relative_pos < mid) & (relative_pos > -mid),
        torch.tensor(mid - 1, device=relative_pos.device).type_as(relative_pos),
        torch.abs(relative_pos),
    )
    log_pos = (
        torch.ceil(torch.log(abs_pos / mid) / torch.log(torch.tensor((max_position - 1) / mid, device=relative_pos.device)) * (mid - 1)) + mid
    )
    bucket_pos = torch.where(abs_pos <= mid, relative_pos.type_as(log_pos), log_pos * sign)
    return bucket_pos
modeling_deberta_v2.make_log_bucket_position = make_log_bucket_position


@torch.jit.script
def scaled_size_sqrt(query_layer: torch.Tensor, scale_factor: int):
    return torch.sqrt(torch.tensor(query_layer.size(-1), device=query_layer.device, dtype=torch.float) * scale_factor)
modeling_deberta_v2.scaled_size_sqrt = scaled_size_sqrt


class AuxiliaryDeberta(nn.Module):
    NUM_AUXILIARY_TASKS = 2

    def __init__(self, state_dict = None, auxiliary_tasks = [False] * NUM_AUXILIARY_TASKS):
        super(AuxiliaryDeberta, self).__init__()

        self.encoder = DebertaV2Model.from_pretrained('microsoft/deberta-v3-large')

        self.front_block = nn.Sequential(
            nn.Linear(self.encoder.config.hidden_size, 256),
            nn.GELU(approximate="tanh"),
            nn.Dropout(self.encoder.config.pooler_dropout),
        )
        # Main Task: AI 여부 예측 (0~1 확률)
        self.main_block = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(approximate="tanh"),
            nn.Dropout(self.encoder.config.pooler_dropout),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

        for i, enabled in enumerate(auxiliary_tasks):
            if enabled:
                block = nn.Sequential(
                    nn.Linear(256, 128),
                    nn.GELU(approximate="tanh"),
                    nn.Dropout(self.encoder.config.pooler_dropout),
                    nn.Linear(128, 1),
                )
                self.add_module(f'aux_block{i}', block)

        if state_dict is not None:
            self.load_state_dict(state_dict)

    @property
    def device(self) -> torch.device:
        iter = self.parameters()
        parameter = next(iter, None)
        if parameter is None:
            return torch.get_default_device()
        for v in iter:
            if v.device != parameter.device:
                raise ValueError("Cannot determine device because parameters are on different devices.")
        return parameter.device

    @property
    def dtype(self) -> torch.dtype:
        iter = self.parameters()
        parameter = next(iter, None)
        if parameter is None:
            return torch.get_default_dtype()
        for v in iter:
            if v.dtype != parameter.dtype:
                raise ValueError("Cannot determine dtype because parameters are of different dtypes.")
        return parameter.dtype

    @classmethod
    def from_pretrained(cls, path: os.PathLike, device: torch.device = torch.device("cpu"), dtype: torch.dtype = torch.float32, compile: bool = False):
        state_dict = torch.load(path, map_location=device)
        auxiliary_tasks = [False] * cls.NUM_AUXILIARY_TASKS
        for key in state_dict.keys():
            for i in range(cls.NUM_AUXILIARY_TASKS):
                if auxiliary_tasks[i]:
                    continue
                if key.startswith(f'aux_block{i}.'):
                    auxiliary_tasks[i] = True
        model = cls(state_dict, auxiliary_tasks)
        model = model.to(device=device, dtype=dtype)
        if compile and not TYPE_CHECKING:
            model = torch.compile(model)
        return model

    def has_auxiliary_task(self, index: int) -> bool:
        if not hasattr(self, f'aux_block{index}'):
            return False
        if getattr(self, f'aux_block{index}') is None:
            return False
        return True

    def forward(self, input_ids, attention_mask):
        encoder_output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = encoder_output.last_hidden_state[:, 0]
        features = self.front_block(last_hidden_state)

        output = {
            'main': self.main_block(features).squeeze(-1),
        }

        if self.training:
            aux = [None] * self.NUM_AUXILIARY_TASKS
            for i in range(self.NUM_AUXILIARY_TASKS):
                module = getattr(self, f'aux_block{i}', None)
                if module is None:
                    continue
                aux[i] = module(features).squeeze(-1)
            output['aux'] = aux

        return output
