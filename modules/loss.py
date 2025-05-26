from typing import Tuple, List
import torch
import torch.nn as nn

# loss modify
class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_pos=2.0, gamma_neg=0.5):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        eps = 1e-8
        pred = pred.clamp(min=eps, max=1 - eps)
        pos_loss = target * torch.log(pred) * ((1 - pred) ** self.gamma_pos)
        neg_loss = (1 - target) * torch.log(1 - pred) * (pred ** self.gamma_neg)
        return -torch.mean(pos_loss + neg_loss)

def compute_multitask_loss(outputs, labels, weights=(1.0, 0.3, 0.3)) -> Tuple[torch.Tensor, List[float]]:
    main_loss_fn = AsymmetricLoss()
    aux_loss_fn = nn.MSELoss()

    has_aux1 = 'aux1' in outputs and 'aux1' in labels
    has_aux2 = 'aux2' in outputs and 'aux2' in labels

    loss_main: torch.Tensor = main_loss_fn(outputs['main'], labels['main'])

    loss = [loss_main.item()]
    total_loss = weights[0] * loss_main

    if has_aux1:
        loss_aux1: torch.Tensor = aux_loss_fn(outputs['aux1'], labels['aux1'])
        loss.append(loss_aux1.item())
        total_loss += weights[1] * loss_aux1
    if has_aux2:
        loss_aux2: torch.Tensor = aux_loss_fn(outputs['aux2'], labels['aux2'])
        loss.append(loss_aux2.item())
        total_loss += weights[2] * loss_aux2

    return total_loss, loss
