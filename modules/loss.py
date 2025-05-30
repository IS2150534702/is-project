from typing import Tuple, List
import torch
import torch.nn as nn

# loss modify
class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_pos=0.0, gamma_neg=0.0):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        eps = 1e-8
        pred = pred.clamp(min=eps, max=1 - eps)
        pos_loss = target * torch.log(pred) * ((1 - pred) ** self.gamma_pos)
        neg_loss = (1 - target) * torch.log(1 - pred) * (pred ** self.gamma_neg)
        return -torch.mean(pos_loss + neg_loss)

def compute_multitask_loss(outputs, labels, weights) -> Tuple[torch.Tensor, List[float]]:
    main_loss_fn = AsymmetricLoss(2.0, 4.0) # ? 4.0
    aux_loss_fn = nn.MSELoss()

    loss_main: torch.Tensor = main_loss_fn(outputs['main'], labels['main'])

    loss = [loss_main.item()]
    total_loss = weights[0] * loss_main
    for i, aux in enumerate(outputs['aux']):
        if aux is None:
            continue
        loss_aux: torch.Tensor = aux_loss_fn(aux, labels[f'aux{i}'])
        loss.append(loss_aux.item())
        total_loss += weights[i + 1] * loss_aux

    return total_loss, loss
