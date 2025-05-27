import torch.nn as nn
from transformers import DebertaV2Model


class AuxiliaryDeberta(nn.Module):
    NUM_AUXILIARY_TASKS = 2

    def __init__(self, model_name = 'microsoft/deberta-v3-large', auxiliary_tasks = [False] * NUM_AUXILIARY_TASKS):
        super(AuxiliaryDeberta, self).__init__()

        self.encoder = DebertaV2Model.from_pretrained(model_name)
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

    @classmethod
    def from_state_dict(cls, state_dict):
        auxiliary_tasks = [False] * cls.NUM_AUXILIARY_TASKS
        for key in state_dict.keys():
            for i in range(cls.NUM_AUXILIARY_TASKS):
                if auxiliary_tasks[i]:
                    continue
                if key.startswith(f'aux_block{i}.'):
                    auxiliary_tasks[i] = True
        model = cls(auxiliary_tasks=auxiliary_tasks)
        model.load_state_dict(state_dict)
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
