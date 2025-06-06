import os
import sys

sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))

import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from src.imoe.InteractionMoE import MLPReWeighting


class LatentContrastiveInteractionExpert(nn.Module):
    """
    Interaction Expert with contrastive loss applied on latent embedding.
    """

    def __init__(self, fusion_model, fusion_sparse):
        super(LatentContrastiveInteractionExpert, self).__init__()
        self.fusion_model = fusion_model
        self.fusion_sparse = fusion_sparse

    def forward(self, inputs):
        """
        Forward pass with all modalities present.
        """
        return self._forward_with_replacement(inputs, replace_index=None)

    def forward_with_replacement(self, inputs, replace_index):
        """
        Forward pass with one modality replaced by a random vector.

        Args:
            inputs (list of tensors): List of modality inputs.
            replace_index (int): Index of the modality to replace. If None, no modality is replaced.
        """
        return self._forward_with_replacement(inputs, replace_index=replace_index)

    def _forward_with_replacement(self, inputs, replace_index=None):
        """
        Internal function to handle forward pass with optional modality replacement.
        """
        # Replace specified modality with a random vector
        if replace_index is not None:
            random_vector = torch.randn_like(inputs[replace_index])
            inputs = (
                inputs[:replace_index] + [random_vector] + inputs[replace_index + 1 :]
            )

        x, latent = self.fusion_model(inputs, return_latent=True)
        if self.fusion_sparse:
            return x, latent, self.fusion_model.gate_loss()

        return x, latent

    def forward_multiple(self, inputs):
        """
        Perform (1 + n) forward passes: one with all modalities and one for each modality replaced.

        Args:
            inputs (list of tensors): List of modality inputs.

        Returns:
            List of outputs from the forward passes.
        """
        outputs = []
        latents = []
        if self.fusion_sparse:
            gate_losses = []

            output, latent, gate_loss = self.forward(inputs)
            outputs.append(output)
            latents.append(latent)
            gate_losses.append(gate_loss)

            for i in range(len(inputs)):
                output, latent, gate_loss = self.forward_with_replacement(
                    inputs, replace_index=i
                )
                outputs.append(output)
                latents.append(latent)
                gate_losses.append(gate_loss)

            return outputs, latents, gate_losses
        else:
            output, latent = self.forward(inputs)
            outputs.append(output)
            latents.append(latent)

        # Forward passes with each modality replaced
        for i in range(len(inputs)):
            output, latent = self.forward_with_replacement(inputs, replace_index=i)
            outputs.append(output)
            latents.append(latent)

        return outputs, latents


class LatentContrastiveInteractionMoE(nn.Module):
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
        super(LatentContrastiveInteractionMoE, self).__init__()
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
                LatentContrastiveInteractionExpert(
                    deepcopy(fusion_model), fusion_sparse
                )
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
        latent_outputs = []

        if self.fusion_sparse:
            expert_gate_losses = []

            for expert in self.interaction_experts:
                expert_output, latent_output, expert_gate_loss = (
                    expert.forward_multiple(inputs)
                )
                expert_outputs.append(expert_output)
                latent_outputs.append(latent_output)
                expert_gate_losses.append(expert_gate_loss)

        else:
            for expert in self.interaction_experts:
                expert_output, latent_output = expert.forward_multiple(inputs)
                expert_outputs.append(expert_output)
                latent_outputs.append(latent_output)

        ###### Define interaction losses ######
        # First n experts are uniqueness interaction expert
        uniqueness_losses = []
        for i in range(self.num_modalities):
            uniqueness_loss = 0
            outputs = latent_outputs[i]
            anchor = outputs[0]
            neg = outputs[i + 1]
            positives = outputs[1 : i + 1] + outputs[i + 2 :]
            for pos in positives:
                uniqueness_loss += self.uniqueness_loss_single(anchor, pos, neg)
            uniqueness_losses.append(uniqueness_loss / len(positives))

        # One Synergy Expert
        synergy_output = latent_outputs[-2]
        synergy_anchor = synergy_output[0]
        synergy_negatives = torch.stack(synergy_output[1:])
        synergy_loss = self.synergy_loss(synergy_anchor, synergy_negatives)

        # One Redundacy Expert
        redundancy_output = latent_outputs[-1]
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
                expert_output, _, _ = expert.forward(inputs)
                expert_outputs.append(expert_output)
        else:
            for expert in self.interaction_experts:
                expert_output, _ = expert.forward(inputs)
                expert_outputs.append(expert_output)

        all_logits = torch.stack(expert_outputs, dim=1)

        interaction_weights = self.reweight(inputs)  # Get interaction weights
        weights_transposed = interaction_weights.unsqueeze(2)
        weighted_logits = (all_logits * weights_transposed).sum(dim=1)

        return expert_outputs, interaction_weights, weighted_logits
