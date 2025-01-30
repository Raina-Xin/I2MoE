import os
import sys

sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))

import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from src.common.modules.common import MLP


class MLPReWeighting(nn.Module):
    """Use MLP to re-weight all interaction experts."""

    def __init__(
        self,
        num_modalities,
        num_branches,
        hidden_dim=256,
        hidden_dim_rw=256,
        num_layers=2,
        temperature=1,
    ):
        """args:
        hidden_dim: hidden dimension of input embeddings.
        hidden_dim_rw: hidden dimension of the re-weighting model.
        """
        super(MLPReWeighting, self).__init__()
        self.temperature = temperature
        self.mlp = MLP(
            hidden_dim * num_modalities,
            hidden_dim_rw,
            num_branches,
            num_layers,
            activation=nn.ReLU(),
            dropout=0.5,
        )

    def temperature_scaled_softmax(self, logits):
        logits = logits / self.temperature
        return torch.softmax(logits, dim=1)

    def forward(self, inputs):
        if inputs[0].dim() == 3:
            x = [item.mean(dim=1) for item in inputs]
            x = torch.cat(x, dim=1)
        else:
            x = torch.cat(inputs, dim=1)
        x = self.mlp(x)
        return self.temperature_scaled_softmax(x)


class InteractionExpert(nn.Module):
    """
    Interaction Expert.
    """

    def __init__(self, fusion_model, fusion_sparse):
        super(InteractionExpert, self).__init__()
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

        x = self.fusion_model(inputs)
        if self.fusion_sparse:
            return x, self.fusion_model.gate_loss()

        return x

    def forward_multiple(self, inputs):
        """
        Perform (1 + n) forward passes: one with all modalities and one for each modality replaced.

        Args:
            inputs (list of tensors): List of modality inputs.

        Returns:
            List of outputs from the forward passes.
        """
        outputs = []
        if self.fusion_sparse:
            gate_losses = []

            output, gate_loss = self.forward(inputs)
            outputs.append(output)
            gate_losses.append(gate_loss)

            for i in range(len(inputs)):
                output, gate_loss = self.forward_with_replacement(
                    inputs, replace_index=i
                )
                outputs.append(output)
                gate_losses.append(gate_loss)

            return outputs, gate_losses
        else:
            outputs.append(self.forward(inputs))

        # Forward passes with each modality replaced
        for i in range(len(inputs)):
            outputs.append(self.forward_with_replacement(inputs, replace_index=i))

        return outputs


class InteractionMoERegression(nn.Module):
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
        super(InteractionMoERegression, self).__init__()
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

    def uniqueness_loss_single(self, anchor, pos, neg):
        mse_loss = nn.MSELoss()
        return mse_loss(anchor, pos) - mse_loss(anchor, neg)

    def synergy_loss(self, anchor, negatives):
        mse_loss = nn.MSELoss()
        total_syn_loss = 0
        for negative in negatives:
            total_syn_loss += mse_loss(anchor, negative)
        return total_syn_loss / len(negatives)

    def redundancy_loss(self, anchor, positives):
        mse_loss = nn.MSELoss()
        total_redundancy_loss = 0
        for positive in positives:
            total_redundancy_loss += mse_loss(anchor, positive)
        return total_redundancy_loss / len(positives)

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
        # First n experts are uniqueness interaction experts
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

        # One Redundancy Expert
        redundancy_output = expert_outputs[-1]
        redundancy_anchor = redundancy_output[0]
        redundancy_positives = torch.stack(redundancy_output[1:])
        redundancy_loss = self.redundancy_loss(redundancy_anchor, redundancy_positives)

        interaction_losses = uniqueness_losses + [synergy_loss] + [redundancy_loss]

        all_predictions = torch.stack([output[0] for output in expert_outputs], dim=1)

        ###### MLP reweighting the experts' outputs ######
        interaction_weights = self.reweight(inputs)  # Get interaction weights
        weights_transposed = interaction_weights.unsqueeze(2)
        weighted_predictions = (all_predictions * weights_transposed).sum(dim=1)

        if self.fusion_sparse:
            return (
                expert_outputs,
                interaction_weights,
                weighted_predictions,
                interaction_losses,
                expert_gate_losses,
            )

        return (
            expert_outputs,
            interaction_weights,
            weighted_predictions,
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

        all_predictions = torch.stack(expert_outputs, dim=1)

        interaction_weights = self.reweight(inputs)  # Get interaction weights
        weights_transposed = interaction_weights.unsqueeze(2)
        weighted_predictions = (all_predictions * weights_transposed).sum(dim=1)

        return expert_outputs, interaction_weights, weighted_predictions
