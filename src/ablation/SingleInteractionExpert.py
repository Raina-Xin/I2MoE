import os
import sys

sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))

import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

from src.imoe.InteractionMoE import InteractionExpert, MLPReWeighting


class SingleInterctionExpert(nn.Module):
    def __init__(
        self,
        num_modalities=3,
        fusion_model=None,
        fusion_sparse=True,
        hidden_dim=256,
        hidden_dim_rw=256,
        num_layer_rw=2,
        temperature_rw=1,
        num_purturb=2,
        type_of_interaction="red",
    ):
        super(SingleInterctionExpert, self).__init__()
        num_branches = 1
        self.num_modalities = num_modalities

        self.interaction_expert = InteractionExpert(
            deepcopy(fusion_model), fusion_sparse
        )
        self.fusion_sparse = fusion_sparse
        self.type_of_interaction = type_of_interaction

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

            expert_output, expert_gate_loss = self.interaction_expert.forward_multiple(
                inputs
            )
            expert_outputs.append(expert_output)
            expert_gate_losses.append(expert_gate_loss)
        else:

            expert_outputs.append(self.interaction_expert.forward_multiple(inputs))

        if self.type_of_interaction == "syn":
            synergy_output = expert_outputs[0]
            synergy_anchor = synergy_output[0]
            synergy_negatives = torch.stack(synergy_output[1:])
            synergy_loss = self.synergy_loss(synergy_anchor, synergy_negatives)
            interaction_loss = synergy_loss

        elif self.type_of_interaction == "red":
            redundancy_output = expert_outputs[0]
            redundancy_anchor = redundancy_output[0]
            redundancy_positives = torch.stack(redundancy_output[1:])
            redundancy_loss = self.redundancy_loss(
                redundancy_anchor, redundancy_positives
            )
            interaction_loss = redundancy_loss

        else:
            modality_idx = int(self.type_of_interaction[-1]) - 1
            uniqueness_loss = 0
            outputs = expert_outputs[0]
            anchor = outputs[0]
            neg = outputs[modality_idx + 1]
            positives = outputs[1 : modality_idx + 1] + outputs[modality_idx + 2 :]
            for pos in positives:
                uniqueness_loss += self.uniqueness_loss_single(anchor, pos, neg)
            interaction_loss = uniqueness_loss

        logits = expert_output[0]

        if self.fusion_sparse:
            return (
                logits,
                interaction_loss,
                expert_gate_loss,
            )

        return (
            logits,
            interaction_loss,
        )

    def inference(self, inputs):
        # Get outputs for each interaction type
        if self.fusion_sparse:
            logits, _ = self.interaction_expert.forward(inputs)
        else:
            logits = self.interaction_expert.forward(inputs)

        return logits
