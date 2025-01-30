import os
import sys

sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from itertools import combinations
from torchvision import models as tmodels
from scipy.special import softmax
import math

from src.common.modules.moe import *


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


class PatchEmbeddings(nn.Module):
    """
    Image to Patch Embedding.

    """

    def __init__(self, feature_size, num_patches, embed_dim, dropout=0.25):
        super().__init__()
        patch_size = math.ceil(feature_size / num_patches)
        pad_size = num_patches * patch_size - feature_size
        self.pad_size = pad_size
        self.num_patches = num_patches
        self.feature_size = feature_size
        self.patch_size = patch_size
        self.projection = nn.Linear(patch_size, embed_dim)

    def forward(self, x):
        x = F.pad(x, (0, self.pad_size)).view(
            x.shape[0], self.num_patches, self.patch_size
        )
        # x = F.normalize(x, dim=-1)
        x = self.projection(x)
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        num_experts,
        num_routers,
        d_model,
        num_head,
        dropout=0.1,
        activation=nn.GELU,
        hidden_times=2,
        mlp_sparse=False,
        self_attn=True,
        top_k=2,
        gate="GShardGate",
        **kwargs,
    ) -> None:
        super(TransformerEncoderLayer, self).__init__()

        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = activation()
        self.attn = Attention(
            d_model,
            num_heads=num_head,
            qkv_bias=False,
            attn_drop=dropout,
            proj_drop=dropout,
        )

        self.mlp_sparse = mlp_sparse
        self.self_attn = self_attn

        if gate == "SwitchGate":
            top_k = 1
        if self.mlp_sparse:
            self.mlp = FMoETransformerMLP(
                num_expert=num_experts,
                n_router=num_routers,
                d_model=d_model,
                d_hidden=d_model * hidden_times,
                activation=nn.GELU(),
                top_k=top_k,
                gate=gate,
                **kwargs,
            )
        else:
            self.mlp = MLP(
                input_dim=d_model,
                hidden_dim=d_model * hidden_times,
                output_dim=d_model,
                num_layers=2,
                activation=nn.GELU(),
                dropout=dropout,
            )

    def forward(self, x):
        if self.self_attn:
            chunk_size = [item.shape[1] for item in x]
            x = self.norm1(torch.cat(x, dim=1))
            kv = x
            x = self.attn(x, kv)
            x = x + self.dropout1(x)
            x = torch.split(x, chunk_size, dim=1)
            x = [item for item in x]

            for i in range(len(chunk_size)):
                x[i] = x[i] + self.dropout2(self.mlp(self.norm2(x[i])))
        else:
            chunk_size = [item.shape[1] for item in x]
            x = [item for item in x]
            for i in range(len(chunk_size)):
                other_m = [x[j] for j in range(len(chunk_size)) if j != i]
                other_m = torch.cat([x[i], *other_m], dim=1)
                x[i] = self.attn(x[i], other_m)
            x = [x[i] + self.dropout1(x[i]) for i in range(len(chunk_size))]

            for i in range(len(chunk_size)):
                x[i] = x[i] + self.dropout2(self.mlp(self.norm2(x[i])))

        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.head_dim = head_dim
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        # self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(head_dim * self.num_heads, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, kv, attn_mask=None):
        # attn_mask: (B, N+1, N+1) input-dependent
        eps = 1e-6

        Bx, Nx, Cx = x.shape
        B, N, C = kv.shape
        q = self.q(x).reshape(Bx, Nx, self.num_heads, Cx // self.num_heads)
        q = q.permute(0, 2, 1, 3)
        kv = self.kv(kv)
        kv = kv.reshape(B, N, 2, self.num_heads, C // self.num_heads)
        kv = kv.permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)

        attn = (
            q @ k.transpose(-2, -1)
        ) * self.scale  # (B, H, N+1, C/H) @ (B, H, C/H, N+1) -> (B, H, N+1, N+1)

        attn = attn.softmax(dim=-1)

        attn = self.attn_drop(attn)
        x = attn @ v

        x = x.transpose(1, 2).reshape(
            Bx, Nx, -1
        )  # (B, H, N+1, N+1) * (B, H, N+1, C/H) -> (B, H, N+1, C/H) -> (B, N+1, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class MLP(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        num_layers,
        activation=nn.ReLU(),
        dropout=0.5,
    ):
        super(MLP, self).__init__()
        layers = []
        self.drop = nn.Dropout(dropout)
        if num_layers == 1:
            layers.append(nn.Linear(input_dim, output_dim))
        else:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(activation)
            layers.append(self.drop)
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(activation)
                layers.append(self.drop)
            layers.append(nn.Linear(hidden_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class Maxout(nn.Module):
    """Implements Maxout module."""

    def __init__(self, d, m, k):
        """Initialize Maxout object.

        Args:
            d (int): (Unused)
            m (int): Number of features remeaining after Maxout.
            k (int): Pool Size
        """
        super(Maxout, self).__init__()
        self.d_in, self.d_out, self.pool_size = d, m, k
        self.lin = nn.Linear(d, m * k)

    def forward(self, inputs):
        """Apply Maxout to inputs.

        Args:
            inputs (torch.Tensor): Layer Input

        Returns:
            torch.Tensor: Layer Output
        """
        shape = list(inputs.size())
        shape[-1] = self.d_out
        shape.append(self.pool_size)
        max_dim = len(shape) - 1
        out = self.lin(inputs)
        m, _ = out.view(*shape).max(dim=max_dim)
        return m


class MaxOut_MLP(nn.Module):
    """Implements Maxout w/ MLP."""

    def __init__(
        self,
        num_outputs,
        first_hidden=64,
        number_input_feats=300,
        second_hidden=None,
        linear_layer=True,
    ):
        """Instantiate MaxOut_MLP Module.

        Args:
            num_outputs (int): Output dimension
            first_hidden (int, optional): First hidden layer dimension. Defaults to 64.
            number_input_feats (int, optional): Input dimension. Defaults to 300.
            second_hidden (_type_, optional): Second hidden layer dimension. Defaults to None.
            linear_layer (bool, optional): Whether to include an output hidden layer or not. Defaults to True.
        """
        super(MaxOut_MLP, self).__init__()

        if second_hidden is None:
            second_hidden = first_hidden
        self.op0 = nn.BatchNorm1d(number_input_feats, 1e-4)
        self.op1 = Maxout(number_input_feats, first_hidden, 2)
        self.op2 = nn.Sequential(nn.BatchNorm1d(first_hidden), nn.Dropout(0.3))
        # self.op2 = nn.BatchNorm1d(first_hidden)
        # self.op3 = Maxout(first_hidden, first_hidden * 2, 5)
        self.op3 = Maxout(first_hidden, second_hidden, 2)
        self.op4 = nn.Sequential(nn.BatchNorm1d(second_hidden), nn.Dropout(0.3))
        # self.op4 = nn.BatchNorm1d(second_hidden)

        # The linear layer that maps from hidden state space to output space
        if linear_layer:
            self.hid2val = nn.Linear(second_hidden, num_outputs)
        else:
            self.hid2val = None

    def forward(self, x):
        """Apply module to layer input

        Args:
            x (torch.Tensor): Layer Input

        Returns:
            torch.Tensor: Layer Output
        """
        o0 = self.op0(x)
        o1 = self.op1(o0)
        o2 = self.op2(o1)
        o3 = self.op3(o2)
        o4 = self.op4(o3)
        if self.hid2val is None:
            return o4
        o5 = self.hid2val(o4)

        return o5


class FusionMLP(nn.Module):
    def __init__(self, total_input_dim, hidden_dim, output_dim, num_layers):
        super(FusionMLP, self).__init__()
        layers = []
        layers.append(nn.Linear(total_input_dim, hidden_dim))
        layers.append(nn.ReLU())
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, *inputs):
        x = torch.cat(inputs, dim=1)
        return self.network(x)


class Custom3DCNN(nn.Module):
    # Architecture provided by: End-To-End Alzheimer's Disease Diagnosis and Biomarker Identification
    def __init__(self, hidden_dim=128):
        super(Custom3DCNN, self).__init__()
        self.conv1 = nn.Conv3d(1, 32, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.conv2 = nn.Conv3d(32, 32, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.pool1 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2)
        self.dropout1 = nn.Dropout3d(0.2)

        self.conv3 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.conv4 = nn.Conv3d(64, 64, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.pool2 = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=3)
        self.dropout2 = nn.Dropout3d(0.2)

        self.conv5 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.conv6 = nn.Conv3d(
            128, hidden_dim, kernel_size=(3, 3, 3), stride=1, padding=1
        )
        self.pool3 = nn.MaxPool3d(kernel_size=(4, 4, 4))
        self.dropout3 = nn.Dropout3d(0.2)

        # Flatten the output and add a fully connected layer to reduce to hidden_dim
        self.fc = nn.Linear(hidden_dim * 3 * 3 * 4, hidden_dim)

        # self.fc1 = nn.Linear(128*6*6, 512)
        # self.fc2 = nn.Linear(512, 256)
        # self.fc3 = nn.Linear(256, num_classes)
        # self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.dropout1(self.pool1(x))

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.dropout2(self.pool2(x))

        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.dropout3(self.pool3(x))

        x = x.view(x.size(0), -1)  # Flatten the output
        x = self.fc(x)  # Apply the fully connected layer

        # x = x.view(x.size(0), -1)
        # x = F.relu(self.fc1(x))
        # x = self.dropout(x)
        # x = F.relu(self.fc2(x))
        # x = self.dropout(x)
        # x = self.fc3(x)

        return x


class WeightedAverageMLP(nn.Module):
    def __init__(
        self, num_modalities, hidden_dim, num_layers_weight_mlp=2, temperature=1
    ):
        super(WeightedAverageMLP, self).__init__()
        self.temperature = temperature
        self.mlp = MLP(
            hidden_dim * num_modalities,
            hidden_dim,
            4,
            num_layers_weight_mlp,
            activation=nn.ReLU(),
            dropout=0.5,
        )

    def temperature_scaled_softmax(self, logits):
        logits = logits / self.temperature
        return torch.softmax(logits, dim=1)

    def forward(self, inputs):
        x = [item.mean(dim=1) for item in inputs]
        x = torch.cat(x, dim=1)
        x = self.mlp(x)
        return self.temperature_scaled_softmax(x)


class WeightedAverageSimple(nn.Module):
    def __init__(self, num_logits=4):
        super(WeightedAverageSimple, self).__init__()
        self.weights = nn.Parameter(
            torch.ones(num_logits) / num_logits
        )  # Initialize weights
        self.softmax = nn.Softmax(dim=0)

    def forward(self, logits):
        normalized_weights = self.softmax(self.weights)
        normalized_weights = normalized_weights.unsqueeze(0).unsqueeze(-1)
        weighted_average = torch.sum(
            logits * normalized_weights, dim=1
        )  # Weighted average along the logit dimension
        return weighted_average
