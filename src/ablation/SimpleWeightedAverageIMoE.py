import os
import sys

sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))

import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from src.common.modules.common import MLP
import random
from scipy.special import softmax

from src.imoe.InteractionMoE import InteractionExpert


class WeightedAverageSimple(nn.Module):
    def __init__(self, num_logits=4):
        super(WeightedAverageSimple, self).__init__()
        self.weights = nn.Parameter(
            torch.ones(num_logits) / num_logits
        )  # Initialize weights
        self.softmax = nn.Softmax(dim=0)

    def forward(self, logits):
        normalized_weights = self.softmax(self.weights)
        transposed_weights = normalized_weights.unsqueeze(0).unsqueeze(-1)
        weighted_average = torch.sum(
            logits * transposed_weights, dim=1
        )  # Weighted average along the logit dimension
        return weighted_average, normalized_weights


class SimpleWeightedAverageIMoE(nn.Module):
    def __init__(
        self,
        num_modalities=3,
        fusion_model=None,
        fusion_sparse=True,
        hidden_dim=256,
        hidden_dim_rw=256,
        num_layer_rw=2,
        temperature_rw=1,
    ):
        super(SimpleWeightedAverageIMoE, self).__init__()
        num_branches = num_modalities + 1 + 1  # uni + syn + red
        self.num_modalities = num_modalities
        self.reweight = WeightedAverageSimple(num_logits=num_branches)
        self.interaction_experts = nn.ModuleList(
            [
                InteractionExpert(deepcopy(fusion_model), fusion_sparse)
                for _ in range(num_branches)
            ]
        )
        self.fusion_sparse = fusion_sparse

    def uniqueness_loss_single(self, anchor, pos, neg):
        triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2, eps=1e-7)
        return triplet_loss(anchor, pos, neg)

    def synergy_loss(self, anchor, negatives):
        total_syn_loss = 0
        anchor_normalized = F.normalize(anchor, p=2, dim=1)
        for negative in negatives:
            negative_normalized = F.normalize(negative, p=2, dim=1)
            cosine_sim = torch.sum(anchor_normalized * negative_normalized, dim=1)
            total_syn_loss += torch.mean(cosine_sim)
        total_syn_loss = total_syn_loss / len(negatives)

        return total_syn_loss  # Synergy loss

    def redundancy_loss(self, anchor, positives):
        total_redundancy_loss = 0
        anchor_normalized = F.normalize(anchor, p=2, dim=1)
        for positive in positives:
            positive_normalized = F.normalize(positive, p=2, dim=1)
            cosine_sim = torch.sum(anchor_normalized * positive_normalized, dim=1)
            total_redundancy_loss += torch.mean(1 - cosine_sim)
        total_redundancy_loss = total_redundancy_loss / len(positives)
        return total_redundancy_loss  # Redundancy loss

    def forward(self, inputs):

        expert_outputs = []

        if self.fusion_sparse:
            expert_gate_losses = []

            for expert in self.interaction_experts:
                expert_output, expert_gate_loss = expert.forward_multiple(inputs)
                expert_outputs.append(expert_output)
                expert_gate_losses.append(expert_gate_loss)
        else:
            for expert in self.interaction_experts:
                expert_outputs.append(expert.forward_multiple(inputs))

        ###### Define interaction losses ######
        # First n experts are uniqueness interaction expert
        uniqueness_losses = []
        for i in range(self.num_modalities):
            uniqueness_loss = 0
            outputs = expert_outputs[i]
            anchor = outputs[0]
            neg = outputs[i + 1]
            positives = outputs[1 : i + 1] + outputs[i + 2 :]
            for pos in positives:
                uniqueness_loss += self.uniqueness_loss_single(anchor, pos, neg)
            uniqueness_losses.append(uniqueness_loss / len(positives))

        # One Synergy Expert
        synergy_output = expert_outputs[-2]
        synergy_anchor = synergy_output[0]
        synergy_negatives = torch.stack(synergy_output[1:])
        synergy_loss = self.synergy_loss(synergy_anchor, synergy_negatives)

        # One Redundacy Expert
        redundancy_output = expert_outputs[-1]
        redundancy_anchor = redundancy_output[0]
        redundancy_positives = torch.stack(redundancy_output[1:])
        redundancy_loss = self.redundancy_loss(redundancy_anchor, redundancy_positives)

        interaction_losses = uniqueness_losses + [synergy_loss] + [redundancy_loss]

        all_logits = torch.stack([output[0] for output in expert_outputs], dim=1)

        ###### MLP reweighting the experts output ######

        weighted_logits, interaction_weights = self.reweight(all_logits)

        if self.fusion_sparse:
            return (
                expert_outputs,
                interaction_weights,
                weighted_logits,
                interaction_losses,
                expert_gate_losses,
            )

        return (
            expert_outputs,
            interaction_weights,
            weighted_logits,
            interaction_losses,
        )

    def inference(self, inputs):
        # Get outputs for each interaction type
        expert_outputs = []
        if self.fusion_sparse:
            for expert in self.interaction_experts:
                expert_output, _ = expert.forward(inputs)
                expert_outputs.append(expert_output)
        else:
            for expert in self.interaction_experts:
                expert_outputs.append(expert.forward(inputs))

        all_logits = torch.stack(expert_outputs, dim=1)

        weighted_logits, interaction_weights = self.reweight(all_logits)

        return expert_outputs, interaction_weights, weighted_logits
