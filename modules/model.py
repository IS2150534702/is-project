import torch
import torch.nn as nn
from transformers import DebertaV2Model


class AuxiliaryDeberta(nn.Module):
    def __init__(self, model_name = 'microsoft/deberta-v3-large', auxiliary_tasks = (False, False,)):
        super(AuxiliaryDeberta, self).__init__()

        self.encoder = DebertaV2Model.from_pretrained(model_name)
        self.auxiliary_tasks = auxiliary_tasks

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
        # Auxiliary Task 1: func_word_count
        if auxiliary_tasks[0]:
            self.aux1_block = nn.Sequential(
                nn.Linear(256, 128),
                nn.GELU(approximate="tanh"),
                nn.Dropout(self.encoder.config.pooler_dropout),
                nn.Linear(128, 1),
            )
        # Auxiliary Task 2: token_repetition_count
        if auxiliary_tasks[1]:
            self.aux2_block = nn.Sequential(
                nn.Linear(256, 128),
                nn.GELU(approximate="tanh"),
                nn.Dropout(self.encoder.config.pooler_dropout),
                nn.Linear(128, 1),
            )

    def forward(self, input_ids, attention_mask):
        encoder_output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = encoder_output.last_hidden_state[:, 0]
        features = self.front_block(last_hidden_state)

        output = {
            'main': self.main_block(features).squeeze(-1),
        }

        if self.training:
            if self.auxiliary_tasks[0]:
                output['aux1'] = self.aux1_block(features).squeeze(-1)
            if self.auxiliary_tasks[1]:
                output['aux2'] = self.aux2_block(features).squeeze(-1)

        return output
