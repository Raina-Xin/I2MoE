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

from src.imoe.InteractionMoE import InteractionExpert, MLPReWeighting


class LessPerturbedForwardPassIMoE(nn.Module):
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
    ):
        super(LessPerturbedForwardPassIMoE, self).__init__()
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
        self.interaction_experts = nn.ModuleList(
            [
                InteractionExpert(deepcopy(fusion_model), fusion_sparse)
                for _ in range(num_branches)
            ]
        )
        self.fusion_sparse = fusion_sparse
        self.num_perturb = min(num_purturb, num_modalities)

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
        perturbed_modality_idxs = random.sample(
            list(range(self.num_modalities)), self.num_perturb
        )

        if self.fusion_sparse:
            expert_gate_losses = []

            for expert in self.interaction_experts:
                single_expert_outputs = []
                single_expert_gate_losses = []

                expert_output, expert_gate_loss = expert.forward(inputs)
                single_expert_outputs.append(expert_output)
                single_expert_gate_losses.append(expert_gate_loss)

                for i in perturbed_modality_idxs:
                    expert_output, expert_gate_loss = expert.forward_with_replacement(
                        inputs, i
                    )
                    single_expert_outputs.append(expert_output)
                    single_expert_gate_losses.append(expert_gate_loss)

                expert_outputs.append(single_expert_outputs)
                expert_gate_losses.append(single_expert_gate_losses)
        else:
            for expert in self.interaction_experts:
                single_expert_outputs = []

                expert_output = expert.forward(inputs)
                single_expert_outputs.append(expert_output)

                for i in perturbed_modality_idxs:
                    expert_output = expert.forward_with_replacement(inputs, i)
                    single_expert_outputs.append(expert_output)

                expert_outputs.append(single_expert_outputs)

        ###### Define interaction losses ######
        # First n experts are uniqueness interaction expert
        # Changes: only update uniqueness expert with purturbed
        perturbed_modality_to_idx = {
            v: k for k, v in enumerate(perturbed_modality_idxs)
        }
        uniqueness_losses = []
        for e in range(self.num_modalities):
            if e in perturbed_modality_idxs:
                uniqueness_loss = 0
                outputs = expert_outputs[e]
                anchor = outputs[0]
                i = perturbed_modality_to_idx[e]
                neg = outputs[i + 1]
                positives = outputs[1 : i + 1] + outputs[i + 2 :]
                for pos in positives:
                    uniqueness_loss += self.uniqueness_loss_single(anchor, pos, neg)
                uniqueness_losses.append(uniqueness_loss / len(positives))
            else:
                uniqueness_losses.append(torch.tensor(0).to("cuda:0"))

        # One Synergy Expert - No Change
        synergy_output = expert_outputs[-2]
        synergy_anchor = synergy_output[0]
        synergy_negatives = torch.stack(synergy_output[1:])
        synergy_loss = self.synergy_loss(synergy_anchor, synergy_negatives)

        # One Redundacy Expert - No Change
        redundancy_output = expert_outputs[-1]
        redundancy_anchor = redundancy_output[0]
        redundancy_positives = torch.stack(redundancy_output[1:])
        redundancy_loss = self.redundancy_loss(redundancy_anchor, redundancy_positives)

        interaction_losses = uniqueness_losses + [synergy_loss] + [redundancy_loss]

        all_logits = torch.stack([output[0] for output in expert_outputs], dim=1)

        ###### MLP reweighting the experts output ######
        interaction_weights = self.reweight(inputs)  # Get interaction weights
        weights_transposed = interaction_weights.unsqueeze(2)
        weighted_logits = (all_logits * weights_transposed).sum(dim=1)

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

        interaction_weights = self.reweight(inputs)  # Get interaction weights
        weights_transposed = interaction_weights.unsqueeze(2)
        weighted_logits = (all_logits * weights_transposed).sum(dim=1)

        return expert_outputs, interaction_weights, weighted_logits
