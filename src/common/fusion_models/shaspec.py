import os
import sys

sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from src.common.modules.common import MLP, Linear


class SharedEncoder(nn.Module):
    def __init__(self, input_dim, shared_dim):
        super(SharedEncoder, self).__init__()
        self.fc = nn.Linear(input_dim, shared_dim)

    def forward(self, x):
        return self.fc(x)


class FeatureProjection(nn.Module):
    def __init__(self, shared_dim, specific_dim, fused_dim):
        super(FeatureProjection, self).__init__()
        self.fc = nn.Linear(shared_dim + specific_dim, fused_dim)

    def forward(self, shared, specific):
        combined = torch.cat((shared, specific), dim=-1)
        return self.fc(combined)


class Decoder(nn.Module):
    def __init__(self, fused_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(fused_dim, output_dim)

    def forward(self, x):
        return self.fc(x)


class ShaSpec(nn.Module):
    def __init__(
        self,
        num_classes=2,
        num_modalities=2,
        shared_dim=256,
        specific_dim=256,
        fused_dim=256,
    ):
        super(ShaSpec, self).__init__()
        self.shared_encoder = SharedEncoder(specific_dim, shared_dim)
        self.feature_projection = FeatureProjection(shared_dim, specific_dim, fused_dim)
        self.num_modalities = num_modalities  # Number of modalities
        ########### Classification Head
        self.classification_head = Linear(fused_dim, num_classes).cuda()

    def forward(self, x_list):
        specific_features = []
        available_shared_features = []

        for i, feat in enumerate(x_list):
            if x_list[i] is not None:
                specific = x_list[i]
                shared = self.shared_encoder(specific)
                specific_features.append(specific)
                available_shared_features.append(shared)
            else:
                specific_features.append(None)
                available_shared_features.append(None)

        combined_shared = torch.stack(
            [sf for sf in available_shared_features if sf is not None], dim=1
        ).mean(dim=1)

        for i, shared in enumerate(available_shared_features):
            if shared is None:
                available_shared_features[i] = combined_shared

        fused_features = [
            self.feature_projection(shared, specific)
            for shared, specific in zip(available_shared_features, specific_features)
            if specific is not None
        ]
        combined_features = torch.stack(fused_features, dim=1).mean(dim=1)
        out = self.classification_head(combined_features)

        return out


import torch
import torch.nn as nn


class SharedEncoder_gentle_push(nn.Module):
    def __init__(self, input_dim, shared_dim):
        super(SharedEncoder, self).__init__()
        self.rnn = nn.GRU(
            input_dim, shared_dim, batch_first=True
        )  # Use GRU for sequence processing

    def forward(self, x):
        # x: (batch, sequence, input_dim)
        output, _ = self.rnn(x)  # output: (batch, sequence, shared_dim)
        return output[
            :, -1, :
        ]  # Return the last hidden state as the shared representation


class FeatureProjection_gentle_push(nn.Module):
    def __init__(self, shared_dim, specific_dim, fused_dim):
        super(FeatureProjection, self).__init__()
        self.fc = nn.Linear(shared_dim + specific_dim, fused_dim)

    def forward(self, shared, specific):
        # shared: (batch, shared_dim)
        # specific: (batch, sequence, specific_dim)
        shared = shared.unsqueeze(1).repeat(
            1, specific.size(1), 1
        )  # Repeat shared for each timestep
        combined = torch.cat(
            (shared, specific), dim=-1
        )  # combined: (batch, sequence, shared_dim + specific_dim)
        return self.fc(combined)  # output: (batch, sequence, fused_dim)


class Decoder_gentle_push(nn.Module):
    def __init__(self, fused_dim, output_dim):
        super(Decoder, self).__init__()
        self.rnn = nn.GRU(
            fused_dim, output_dim, batch_first=True
        )  # Use GRU for decoding

    def forward(self, x):
        # x: (batch, sequence, fused_dim)
        output, _ = self.rnn(x)
        return output  # output: (batch, sequence, output_dim)
