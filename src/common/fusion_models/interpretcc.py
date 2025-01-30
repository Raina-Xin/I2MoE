import torch
import torch.nn as nn
import torch.nn.functional as F


class GumbelSigmoid(nn.Module):
    """
    Gumbel-Sigmoid layer for feature gating.
    """

    def __init__(self, tau=1.0, hard=False, threshold=0.5):
        super(GumbelSigmoid, self).__init__()
        self.tau = tau
        self.hard = hard
        self.threshold = threshold

    def forward(self, logits):
        gumbels = -torch.empty_like(logits).exponential_().log()  # ~Gumbel(0,1)
        gumbels = (logits + gumbels) / self.tau  # Apply temperature
        y_soft = gumbels.sigmoid()

        if self.hard:
            y_hard = (y_soft > self.threshold).float()
            return y_hard - y_soft.detach() + y_soft
        return y_soft


class InterpretCC(nn.Module):
    """
    InterpretCC Model for Classification.
    """

    def __init__(
        self,
        num_classes=3,
        num_modality=4,
        input_dim=256,
        dropout=0.5,
        tau=1.0,
        hard=True,
        threshold=0.5,
    ):
        """
        Args:
            num_classes (int): Number of output classes.
            num_modality (int): Number of input modalities.
            input_dim (int): Dimension of the input embeddings for each modality.
            dropout (float): Dropout rate for regularization.
        """
        super(InterpretCC, self).__init__()

        self.num_modality = num_modality
        self.gumbel_sigmoid = GumbelSigmoid(tau=tau, hard=hard, threshold=threshold)
        self.modality_gates = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(input_dim, input_dim),
                    nn.ReLU(),
                    nn.Linear(
                        input_dim, 1
                    ),  # Produces a gating score for each modality
                )
                for _ in range(num_modality)
            ]
        )

        self.fusion_layer = nn.Sequential(
            nn.Linear(num_modality * input_dim, input_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, num_classes),
        )

    def forward(self, inputs):
        """
        Forward pass through the model.

        Args:
            inputs (list[Tensor]): List of tensors, one for each modality,
                                   each with shape [batch_size, input_dim].

        Returns:
            logits (Tensor): Class logits of shape [batch_size, num_classes].
        """
        gated_inputs = []
        for i, x in enumerate(inputs):
            gate_score = self.modality_gates[i](x)  # [batch_size, 1]
            gate_mask = self.gumbel_sigmoid(gate_score)  # Apply gating
            gated_input = x * gate_mask  # Mask the input
            gated_inputs.append(gated_input)

        # Concatenate gated inputs and apply fusion layer
        fused_input = torch.cat(
            gated_inputs, dim=1
        )  # [batch_size, num_modality * input_dim]
        fusion_output = self.fusion_layer(fused_input)

        # Classify fused representation
        logits = self.classifier(fusion_output)
        return logits
