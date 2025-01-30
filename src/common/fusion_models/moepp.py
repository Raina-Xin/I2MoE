import os
import sys

sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))

from src.common.modules.moepp_layer import *
import numpy as np
from copy import deepcopy


class MoEPlusPlusTransformer(nn.Module):
    """
    baseline fusion module:
    No interaction loss and no ensemble.
    """

    def __init__(
        self,
        num_modalities,
        num_patches,
        hidden_dim,
        output_dim,
        num_layers,
        num_experts,
        num_heads=2,
        dropout=0.5,
    ):
        super(MoEPlusPlusTransformer, self).__init__()
        self.layers = nn.ModuleList(
            [
                MoEPlusPlusEncoderLayer(
                    num_experts, hidden_dim, num_heads, num_modalities, dropout=dropout
                )
                for _ in range(num_layers)
            ]
        )
        self.classification_head = Linear(
            hidden_dim * num_modalities, output_dim
        ).cuda()

        self.pos_embed = nn.Parameter(
            torch.zeros(1, np.sum([num_patches] * num_modalities), hidden_dim)
        )

    def forward(self, inputs, return_latent=False):
        chunk_size = [input.shape[1] for input in inputs]
        x = torch.cat(inputs, dim=1)
        if self.pos_embed != None:
            x += self.pos_embed

        x = torch.split(x, chunk_size, dim=1)

        gate_residual = None
        for idx, layer in enumerate(self.layers):
            layer_outputs, gate_residual = layer(x, gate_residual=gate_residual)
        x = [item.mean(dim=1) for item in layer_outputs]
        x = torch.cat(x, dim=1)
        if return_latent:
            latent = deepcopy(x)
        x = self.classification_head(x)
        if return_latent:
            return x, latent
        return x


class Linear(torch.nn.Module):
    """Linear Layer with Xavier Initialization, and 0 Bias."""

    def __init__(self, indim, outdim, xavier_init=False):
        """Initialize Linear Layer w/ Xavier Init.

        Args:
            indim (int): Input Dimension
            outdim (int): Output Dimension
            xavier_init (bool, optional): Whether to apply Xavier Initialization to Layer. Defaults to False.

        """
        super(Linear, self).__init__()
        self.fc = nn.Linear(indim, outdim)
        if xavier_init:
            nn.init.xavier_normal_(self.fc.weight)
            self.fc.bias.data.fill_(0.0)

    def forward(self, x):
        """Apply Linear Layer to Input.

        Args:
            x (torch.Tensor): Input Tensor

        Returns:
            torch.Tensor: Output Tensor

        """
        return self.fc(x)
