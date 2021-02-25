import numpy as np
import torch
import torch.nn as nn

class NT_XentLoss:
    def __init__(self, batch_size, temperature, device):
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.mask = self._make_mask(batch_size)

    def _make_mask(self, batch_size):
        "Make mask for correlated samples"
        mask = torch.ones((2*batch_size, 2*batch_size), dtype=bool).to(self.device)
        mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size+i] = 0
            mask[batch_size+i, i] = 0
    
    def __call__(self, zi, zj):
        criterion = nn.CrossEntropyLoss()
        # Calculate sim
        N = zi.shape[0]
        z = torch.cat((zi, zj))
        cos = nn.CosineSimilarity(dim=2)
        sim = cos(z.unsqueeze(0), z.unsqueeze(1)) # Shape: 2N * 2N

        # Calculate loss
        sim = sim / self.temperature
        sim_ij = torch.diag(sim, N)
        sim_ji = torch.diag(sim, -N)
        positive_samples = torch.cat((sim_ij, sim_ji)).reshape(2*N, 1)
        negative_samples = sim[self.mask].reshape(2*N, -1)

        target = torch.zeros(2*N).to(self.device).long()
        input = torch.cat((positive_samples, negative_samples), dim=1)
        loss = criterion(input, target) / (N * 2)
        return loss