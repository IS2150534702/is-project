from typing import Tuple
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

def compute_multitask_loss(outputs, labels, weights=(1.0, 0.0, 0.0, 0.0)) -> Tuple[torch.Tensor, Tuple[float, float, float, float]]:
    main_loss_fn = AsymmetricLoss()
    return main_loss_fn(outputs['main'], labels['main']), (0.0, 0.0, 0.0, 0.0)
    #aux_loss_fn = nn.MSELoss()
    #aux_cls_loss_fn = nn.BCELoss()

    #loss_main = main_loss_fn(outputs['main'], labels['main'])
    #loss_aux1 = aux_loss_fn(outputs['aux1'], labels['aux1'])
    #loss_aux2 = aux_loss_fn(outputs['aux2'], labels['aux2'])
    #loss_aux3 = aux_cls_loss_fn(outputs['aux3'], labels['aux3'])

    #total_loss = (weights[0] * loss_main + weights[1] * loss_aux1 +
    #              weights[2] * loss_aux2 + weights[3] * loss_aux3)
    #return total_loss, (loss_main.item(), loss_aux1.item(), loss_aux2.item(), loss_aux3.item())
