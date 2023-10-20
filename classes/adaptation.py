import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Adaptation(nn.Module):
    """
    Model that is responsible for adapting the extracted features.
    """
    def __init__(self, in_dim):
        super().__init__()
        self.weight_matrix = nn.Linear(in_dim, 1024, bias=False)

    def forward(self, x):
        x = self.weight_matrix(x)
        x = torch.relu(x)
        return x
    

def construct_model(name : str, in_dim : int):
    """Construct the appropriate adaptation module from the passed string."""
    assert name in ("original", "peterson")
    in_dim = int(in_dim)
    assert in_dim > 0

    if name == "original":
        return Adaptation(in_dim)
    # if name == "peterson":
    #     return AdaptationPeterson(in_dim)

def sim_matrix(a, b, eps=1e-8):
    """
    Calculate the similarity matrix given two lists of embeddings.
    added eps for numerical stability.
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))

    return sim_mt

def new_sim_matrix(a, b, eps=1e-8):
    """
    Calculate the similarity matrix given two lists of embeddings.
    added eps for numerical stability.
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm((a_norm + 1) ** 1.5, (b_norm + 1).transpose(0, 1) ** 1.5)
    return sim_mt

def calc_loss(left, right, temp, device):
    """
    Our loss function that is used in training.
    """
    sim1 = sim_matrix(left, right)
    sim2 = sim1.t()
    loss_left2right = F.cross_entropy(
        sim1 * temp, torch.arange(len(sim1)).long().to(device)
    )
    loss_right2left = F.cross_entropy(
        sim2 * temp, torch.arange(len(sim2)).long().to(device)
    )
    loss = loss_left2right * 0.5 + loss_right2left * 0.5

    return loss


def inference_loss(left, right, temp, device):
    """
    Our loss function that is used in training.
    """
    sim1 = new_sim_matrix(left, right)
    sim2 = sim1.t()

    num_classes = 20
    one_hot_encoded_list = torch.tensor(np.array([np.eye(num_classes)[i - 1] for i in range(1, num_classes + 1)]))
    # print(torch.arange(len(sim1)).long().to(device))
    loss_left2right = F.cross_entropy(
        sim1 * temp, one_hot_encoded_list
    )
    loss_right2left = F.cross_entropy(
        sim2 * temp, one_hot_encoded_list
    )
    loss = loss_left2right * 0.5 + loss_right2left * 0.5

    return loss