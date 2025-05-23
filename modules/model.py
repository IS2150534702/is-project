import torch
import torch.nn as nn
from transformers import DebertaV2Model


class AuxiliaryDeberta(nn.Module):
    def __init__(self, model_name='microsoft/deberta-v3-large'):
        super(AuxiliaryDeberta, self).__init__()

        self.encoder = DebertaV2Model.from_pretrained(model_name)
        self.shared = nn.Sequential(
            nn.Linear(self.encoder.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(self.encoder.config.pooler_dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(self.encoder.config.pooler_dropout),
        )
        # Main Task: AI 여부 예측 (0~1 확률)
        self.main_output = nn.Sequential(
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        # Auxiliary Task 1: 쉼표 개수 (회귀)
        #self.aux1_output = nn.Linear(128, 1)
        # Auxiliary Task 2: 문장 길이 (회귀)
        #self.aux2_output = nn.Linear(128, 1)
        # Auxiliary Task 3: 인용 여부 (0~1 확률)
        #self.aux3_output = nn.Sequential(
        #    nn.Linear(128, 1),
        #    nn.Sigmoid()
        #)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        encoder_output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state: torch.Tensor = encoder_output.last_hidden_state[:, 0]
        features: torch.Tensor = self.shared(last_hidden_state)

        batch_size = last_hidden_state.size(0)
        dummy_tensor = features.new_zeros(batch_size)

        return {
            'main': self.main_output(features).squeeze(-1),
            'aux1': dummy_tensor, # self.aux1_output(features).squeeze(-1),
            'aux2': dummy_tensor, # self.aux2_output(features).squeeze(-1),
            'aux3': dummy_tensor, # self.aux3_output(features).squeeze(-1)
        }
