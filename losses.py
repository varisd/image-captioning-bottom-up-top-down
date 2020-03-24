import torch

from torch import nn
from torch.distributions.categorical import Categorical

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


MAX_SIZE = 50


class ClusterLoss(torch.nn.Module):
    """TODO"""

    def __init__(self, margin=1.0):
        super(ClusterLoss, self).__init__()
        self.margin = 1.0

    def forward(self, x, indices):
        # Flatten
        indices = indices.view(-1)
        x = x.view(-1, x.shape[-1])

        # Remove padding and sample random subset
        # TODO: change uniform sampling to frequency-based sampling?
        indices_gather = (~indices.eq(0)).nonzero().long().view(-1)

        indices = indices[indices_gather]
        x = x[indices_gather]

        # WORKAROUND due to memory problems
        indices = indices[:MAX_SIZE]
        x = x[:MAX_SIZE, :]

        # Normalize
        x_norm = torch.norm(x, p=2, dim=-1, keepdim=True).detach()
        x = x.div(x_norm)

        # Compute Similarity
        sim = torch.matmul(x, x.transpose(1, 0))

        # Compute Mask
        indices_expand = indices.view(-1, 1)
        mask_same = indices_expand.eq(indices_expand.transpose(1, 0)).float()
        mask_diff = 1 - mask_same
        mask = mask_same.unsqueeze(2) * mask_diff.unsqueeze(1)

        # Compute Loss
        loss = (self.margin - sim.unsqueeze(2) + sim.unsqueeze(1)) * mask
        loss = torch.max(loss, torch.zeros_like(loss))

        return torch.sum(loss) / torch.sum(mask)


class PerceptualLoss(torch.nn.Module):
    """TODO"""

    def __init__(self):
        super(PerceptualLoss, self).__init__()
        pass

    def forward(self, x, y, indices):
        # Flatten
        indices = indices.view(-1)
        x = x.view(-1, x.shape[-1])
        y = y.view(-1, y.shape[-1])

        # Remove padding and sample random subset
        # TODO: change uniform sampling to frequency-based sampling?
        indices_gather = (~indices.eq(0)).nonzero().long().view(-1)

        indices = indices[indices_gather]
        x = x[indices_gather]
        y = y[indices_gather]

        # WORKAROUND due to memory problems
        indices = indices[:MAX_SIZE]
        x = x[:MAX_SIZE, :]
        y = y[:MAX_SIZE, :]

        # Normalize
        x_norm = torch.norm(x, p=2, dim=-1, keepdim=True).detach()
        x = x.div(x_norm)
        y_norm = torch.norm(y, p=2, dim=-1, keepdim=True).detach()
        y = y.div(y_norm)

        # Compute Mask
        indices_expand = indices.view(-1, 1)
        mask_same = indices_expand.eq(indices_expand.transpose(1, 0)).float()
        mask_diff = 1 - mask_same

        # Compute Similarity
        sim_x = torch.matmul(x, x.transpose(1, 0))
        sim_y = torch.matmul(y, y.transpose(1, 0))
        
        # Compute mean
        sim_x_mean = torch.sum(sim_x * mask_diff) / torch.sum(mask_diff)
        sim_y_mean = torch.sum(sim_y * mask_diff) / torch.sum(mask_diff)

        # Mean center and square
        sim_x_c2 = (sim_x - sim_x_mean) * (sim_x - sim_x_mean)
        sim_y_c2 = (sim_y - sim_y_mean) * (sim_y - sim_y_mean)

        # Compute sigma
        sim_x_sigma = torch.sqrt(
            torch.sum(sim_x_c2 * mask_diff) / torch.sum(mask_diff))
        sim_y_sigma = torch.sqrt(
            torch.sum(sim_y_c2 * mask_diff) / torch.sum(mask_diff))

        return 1 - (
            (torch.sum(sim_x * sim_y * mask_diff) / torch.sum(mask_diff)) /
            (sim_x_sigma * sim_y_sigma))
