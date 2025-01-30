import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn.parameter import Parameter


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features).float())
        if bias:
            self.bias = Parameter(torch.Tensor(out_features).float())
        else:
            self.register_parameter("bias", None)
        self.initialize_parameters()

    def initialize_parameters(self):
        std = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-std, std)
        if self.bias is not None:
            self.bias.data.uniform_(-std, std)

    def forward(self, adj, x):
        y = torch.matmul(x.float(), self.weight.float())
        output = torch.matmul(adj.float(), y.float())
        if self.bias is not None:
            return output + self.bias.float()
        else:
            return output


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_heads, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.pos_encoder = PositionalEncoding(input_dim, dropout)
        encoder_layers = TransformerEncoderLayer(
            input_dim, num_heads, hidden_dim, dropout
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
        self.input_dim = input_dim

    def forward(self, src, src_key_padding_mask=None):
        src = src * torch.sqrt(torch.tensor(self.input_dim, dtype=torch.float))
        src = self.pos_encoder(src)
        output = self.transformer_encoder(
            src, src_key_padding_mask=src_key_padding_mask
        )
        return output


class SimilarityKernel(nn.Module):
    def __init__(self, input_dim, output_dim, task="default", sigma=1.0):
        """
        input_dim: The dimensionality of the input embeddings.
        output_dim: The dimensionality of the output similarity matrix.
        task: Task name for which this kernel is designed (e.g., 'notes', 'codes', 'labs', 'images').
        sigma: The bandwidth parameter for the RBF kernel.
        """
        super(SimilarityKernel, self).__init__()
        self.task = task
        self.fc = nn.Linear(
            input_dim, output_dim
        )  # Linear layer for projecting embeddings if needed
        self.sigma = sigma  # Bandwidth for RBF kernel

    def forward(self, x1, x2):
        """
        x1, x2: Input embeddings for which similarity needs to be computed.
        Output: Similarity matrix of size (batch_size, batch_size).
        """
        # Compute projected representations for x1 and x2 (task-guided)
        x1_proj = self.fc(x1)
        x2_proj = self.fc(x2)

        # Compute pairwise Euclidean distances
        diff = x1_proj - x2_proj
        dist = torch.norm(diff, dim=-1)

        # Compute RBF similarity kernel
        similarity = torch.exp(-(dist**2) / (2 * (self.sigma + 1e-6) ** 2))
        return similarity


class M3Care(nn.Module):
    def __init__(
        self,
        num_classes=2,
        num_modality=2,
        input_dim=128,
        gcn_dim=256,
        dropout=0.5,
        num_heads=4,
    ):
        """
        input_dims: Dictionary specifying the input dimension for each modality.
        """
        super(M3Care, self).__init__()

        self.num_modality = num_modality
        for i in range(num_modality):
            # Task-specific similarity kernels (task-guided deep kernels)
            setattr(
                self,
                f"similarity_kernel_{i}",
                SimilarityKernel(input_dim, gcn_dim, task=str(i)).cuda(),
            )
        # self.similarity_kernel_notes = SimilarityKernel(input_dim, gcn_dim, task='notes')
        # self.similarity_kernel_codes = SimilarityKernel(input_dim, gcn_dim, task='codes')
        # self.similarity_kernel_labs = SimilarityKernel(input_dim, gcn_dim, task='labs')

        # Graph convolution layers for each modality
        setattr(
            self,
            f"GCN",
            nn.ModuleList(
                [GraphConvolution(input_dim, gcn_dim) for _ in range(num_modality)]
            ).cuda(),
        )
        setattr(
            self,
            f"GCN2",
            nn.ModuleList(
                [GraphConvolution(gcn_dim, gcn_dim) for _ in range(num_modality)]
            ).cuda(),
        )
        # self.GCN = nn.ModuleList([GraphConvolution(input_dim, gcn_dim) for _ in range(3)])
        # self.GCN2 = nn.ModuleList([GraphConvolution(gcn_dim, gcn_dim) for _ in range(3)])

        # Attention mechanism for adaptive fusion
        self.attention = nn.MultiheadAttention(
            embed_dim=gcn_dim, num_heads=num_heads, dropout=dropout
        )

        # Fully connected layers for final classification
        # self.fc_final = nn.Linear(num_modality * gcn_dim, final_dim)  # 4 modalities (notes, codes, labs, images)
        # self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # Learnable threshold for adjacency matrix creation
        self.threshold = nn.Parameter(torch.tensor(0.75))  # Learnable threshold

        # Stability loss weight (for regularization)
        self.lambda_stability = 0.1
        self.classification_head = nn.Sequential(
            nn.Linear(gcn_dim * num_modality, 128),  # Combined embedding size
            nn.ReLU(),
            nn.Linear(128, num_classes),  # Output the class logits
        ).cuda()

    def forward(self, x_list, mask=None):

        gcns = []
        for i in range(self.num_modality):
            sim = getattr(self, f"similarity_kernel_{i}")(
                x_list[i].unsqueeze(1), x_list[i].unsqueeze(0)
            )
            # Compute task-guided similarities and adjacency matrix
            # notes_sim = self.similarity_kernel_notes(x_list[0].unsqueeze(1), x_list[0].unsqueeze(0))
            # codes_sim = self.similarity_kernel_codes(x_list[1].unsqueeze(1), x_list[1].unsqueeze(0))
            # labs_sim = self.similarity_kernel_labs(x_list[2].unsqueeze(1), x_list[2].unsqueeze(0))
            adj = torch.where(
                sim > torch.sigmoid(self.threshold), sim, torch.zeros_like(sim)
            )
            # notes_adj = torch.where(notes_sim > torch.sigmoid(self.threshold), notes_sim, torch.zeros_like(notes_sim))
            # codes_adj = torch.where(codes_sim > torch.sigmoid(self.threshold), codes_sim, torch.zeros_like(codes_sim))
            # labs_adj = torch.where(labs_sim > torch.sigmoid(self.threshold), labs_sim, torch.zeros_like(labs_sim))

            # Use these adj matrices for each modality-specific GCN
            gcn_layer = getattr(self, "GCN")[i]
            gcn_layer2 = getattr(self, "GCN2")[i]

            gcn = gcn_layer(adj.squeeze(-1), x_list[i].squeeze(1))
            gcn = gcn_layer2(adj.squeeze(-1), gcn)
            gcns.append(gcn)

        # notes_gcn = self.GCN[0](notes_adj.squeeze(-1), x_list[0].squeeze(1))
        # notes_gcn = self.GCN2[0](notes_adj.squeeze(-1), notes_gcn)

        # codes_gcn = self.GCN[1](codes_adj.squeeze(-1), x_list[1].squeeze(1))
        # codes_gcn = self.GCN2[1](codes_adj.squeeze(-1), codes_gcn)

        # labs_gcn = self.GCN[2](labs_adj.squeeze(-1), x_list[2].squeeze(1))
        # labs_gcn = self.GCN2[2](labs_adj.squeeze(-1), labs_gcn)

        # Attention-based adaptive fusion
        modality_embeddings = torch.stack(gcns, dim=1)
        combined_emb, _ = self.attention(
            modality_embeddings, modality_embeddings, modality_embeddings
        )

        # Combine embeddings from all modalities and apply FC layers
        # print(combined_emb.shape)
        combined_emb = combined_emb.view(
            combined_emb.size(0), -1
        )  # (batch_size, 3 * gcn_dim)
        output = self.classification_head(combined_emb)

        # return torch.sigmoid(output)
        return output


class M3Care_enrico(nn.Module):
    def __init__(
        self,
        hidden_dim,
        gcn_dim,
        final_dim,
        num_heads,
        num_layers,
        num_gcn_layers,
        dropout,
    ):
        """
        input_dims: Dictionary specifying the input dimension for each modality.
        """
        super(M3Care_enrico, self).__init__()

        # Task-specific similarity kernels (task-guided deep kernels)
        self.similarity_kernel_notes = SimilarityKernel(16, gcn_dim, task="notes")
        self.similarity_kernel_codes = SimilarityKernel(16, gcn_dim, task="codes")

        # Graph convolution layers for each modality
        self.GCN = nn.ModuleList(
            [GraphConvolution(16, gcn_dim) for _ in range(2)]
        )  # 2 modalities
        self.GCN2 = nn.ModuleList(
            [GraphConvolution(gcn_dim, gcn_dim) for _ in range(2)]
        )  # 2 modalities

        # Attention mechanism for adaptive fusion
        self.attention = nn.MultiheadAttention(
            embed_dim=gcn_dim, num_heads=4, dropout=dropout
        )

        # Fully connected layers for final classification
        self.fc_final = nn.Linear(2 * gcn_dim, final_dim)  # 2 modalities (notes, codes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(final_dim, 1)  # Assuming binary classification

        # Learnable threshold for adjacency matrix creation
        self.threshold = nn.Parameter(torch.tensor(0.75))

        # Stability loss weight (for regularization)
        self.lambda_stability = 0.1

    def forward(self, x_list, mask=None):
        # Compute task-guided similarities and adjacency matrix
        notes_sim = self.similarity_kernel_notes(
            x_list[0].unsqueeze(1), x_list[0].unsqueeze(0)
        )
        codes_sim = self.similarity_kernel_codes(
            x_list[1].unsqueeze(1), x_list[1].unsqueeze(0)
        )

        notes_adj = torch.where(
            notes_sim > torch.sigmoid(self.threshold),
            notes_sim,
            torch.zeros_like(notes_sim),
        )
        codes_adj = torch.where(
            codes_sim > torch.sigmoid(self.threshold),
            codes_sim,
            torch.zeros_like(codes_sim),
        )

        # Use these adj matrices for each modality-specific GCN
        notes_gcn = self.GCN[0](
            notes_adj.squeeze(-1), x_list[0].squeeze(1)
        )  # Use x_list directly
        notes_gcn = self.GCN2[0](notes_adj.squeeze(-1), notes_gcn)

        codes_gcn = self.GCN[1](
            codes_adj.squeeze(-1), x_list[1].squeeze(1)
        )  # Use x_list directly
        codes_gcn = self.GCN2[1](codes_adj.squeeze(-1), codes_gcn)

        # Attention-based adaptive fusion
        modality_embeddings = torch.stack(
            [notes_gcn, codes_gcn], dim=1
        )  # Stack only 2 modalities
        combined_emb, _ = self.attention(
            modality_embeddings, modality_embeddings, modality_embeddings
        )

        # Combine embeddings from all modalities and apply FC layers
        combined_emb = combined_emb.view(
            combined_emb.size(0), -1
        )  # (batch_size, 2 * gcn_dim)

        return combined_emb
