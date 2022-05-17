import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from ..models import LocalityLevel
from collections import defaultdict


def byol_loss_fn(x, y):
    """
    x and y are flattened along last dim embeddings
    """
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)

    return 2 - 2 * (x * y).sum(dim=-1)


def byol_loss(target_embeddings_one, target_embeddings_two, online_predictions_one, online_predictions_two):
    """
    inputs output of byol model
    """
    loss_one = byol_loss_fn(target_embeddings_one, online_predictions_two)
    loss_two = byol_loss_fn(target_embeddings_two, online_predictions_one)
    loss = loss_one + loss_two

    return {'byol_loss': loss.mean()}


