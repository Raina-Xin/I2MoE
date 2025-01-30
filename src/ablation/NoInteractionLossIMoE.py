import os
import sys

sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))

import torch
import torch.nn as nn
from copy import deepcopy

from src.imoe.InteractionMoE import MLPReWeighting


class NoInteractionLossIMoE(nn.Module):
    def __init__(
        self,
        num_modalities=3,
        fusion_model=None,
        fusion_sparse=False,
        hidden_dim=256,
        hidden_dim_rw=256,
        num_layer_rw=2,
        temperature_rw=1,
    ):
        super(NoInteractionLossIMoE, self).__init__()
        num_branches = num_modalities + 1 + 1  # uni + syn + red
        self.num_modalities = num_modalities
        self.reweight = MLPReWeighting(
            num_modalities,
            num_branches,
            hidden_dim=hidden_dim,
            hidden_dim_rw=hidden_dim_rw,
            num_layers=num_layer_rw,
            temperature=temperature_rw,
        )
        self.experts = nn.ModuleList(
            [deepcopy(fusion_model) for _ in range(num_branches)]
        )
        self.fusion_sparse = fusion_sparse

    def forward(self, inputs):

        expert_outputs = []
        if self.fusion_sparse:
            gate_losses = []
        for expert in self.experts:
            expert_outputs.append(expert(inputs))
            if self.fusion_sparse:
                gate_losses.append(expert.gate_loss())

        all_logits = torch.stack(expert_outputs, dim=1)

        expert_weights = self.reweight(inputs)  # Get interaction weights
        weights_transposed = expert_weights.unsqueeze(2)
        weighted_logits = (all_logits * weights_transposed).sum(dim=1)

        if self.fusion_sparse:
            return (expert_outputs, expert_weights, weighted_logits, gate_losses)
        else:
            return (expert_outputs, expert_weights, weighted_logits)
