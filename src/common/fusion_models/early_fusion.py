import torch
from torch import nn
from torch.nn import functional as F
import pdb
from torch.autograd import Variable

class ConcatEarly(nn.Module):
    """Concatenation of input data on dimension 2."""

    def __init__(self):
        """Initialize ConcatEarly Module."""
        super(ConcatEarly, self).__init__()

    def forward(self, modalities):
        """
        Forward Pass of ConcatEarly.

        :param modalities: An iterable of modalities to combine
        """
        return torch.cat(modalities, dim=2)
    
class EarlyFusionTransformer(nn.Module):
    """Implements a Transformer with Early Fusion."""

    embed_dim = 9

    def __init__(self, hidden_dim=256, num_modalities=2, output_dim=3):
        """Initialize EarlyFusionTransformer Object.

        Args:
            n_features (int): Number of features in input.

        """
        super().__init__()
        self.concat = ConcatEarly()
        self.conv = nn.Conv1d(
            hidden_dim*num_modalities, self.embed_dim, kernel_size=1, padding=0, bias=False
        )
        layer = nn.TransformerEncoderLayer(d_model=self.embed_dim, nhead=3)
        self.transformer = nn.TransformerEncoder(layer, num_layers=3)
        self.linear = nn.Linear(self.embed_dim, output_dim)

    def forward(self, x):
        """Apply EarlyFusion with a Transformer Encoder to input.

        Args:
            x (torch.Tensor): Input Tensor

        Returns:
            torch.Tensor: Layer Output
        """
        x = self.concat(x)
        x = self.conv(x.permute([0, 2, 1]))
        x = x.permute([2, 0, 1])
        x = self.transformer(x)[-1]
        return self.linear(x)