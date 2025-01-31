import torch
from torch import nn
from torch.nn import functional as F
import pdb
from torch.autograd import Variable

class Concat(nn.Module):
    """Concatenation of input data on dimension 1."""

    def __init__(self):
        """Initialize Concat Module."""
        super(Concat, self).__init__()

    def forward(self, modalities):
        """
        Forward Pass of Concat.

        :param modalities: An iterable of modalities to combine
        """
        flattened = []
        for modality in modalities:
            flattened.append(torch.flatten(modality, start_dim=1))
        return torch.cat(flattened, dim=1)
    
class LateFusionTransformer(nn.Module):
    """Implements a Transformer with Late Fusion."""

    def __init__(self, embed_dim=9, output_dim=3):
        """Initialize LateFusionTransformer Layer.

        Args:
            embed_dim (int, optional): Size of embedding layer. Defaults to 9.
        """
        super().__init__()
        self.concat = Concat()
        self.embed_dim = embed_dim

        self.conv = nn.Conv1d(1, self.embed_dim, kernel_size=1, padding=0, bias=False)
        layer = nn.TransformerEncoderLayer(d_model=self.embed_dim, nhead=2)
        self.transformer = nn.TransformerEncoder(layer, num_layers=2)
        self.linear = nn.Linear(self.embed_dim, output_dim)
    
    def forward(self, x):
        """Apply LateFusionTransformer Layer to input.

        Args:
            x (torch.Tensor): Layer input

        Returns:
            torch.Tensor: Layer output
        """
        x = self.concat(x)
        x = self.conv(x.view(x.size(0), 1, -1))
        x = x.permute(2, 0, 1)
        x = self.transformer(x)[-1]
        return self.linear(x)