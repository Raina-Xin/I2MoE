# pylint: disable=E1101
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
from src.common.modules.sparse_moe import MoE, MoEConfig
from src.common.modules.hme import HierarchicalMoE


class Outer(nn.Module):
    def __init__(
        self, inp1_size: int = 128, inp2_size: int = 128, n_neurons: int = 128
    ):
        super(Outer, self).__init__()
        self.inp1_size = inp1_size
        self.inp2_size = inp2_size
        self.feedforward = nn.Sequential(
            nn.Linear((inp1_size + 1) * (inp2_size + 1), n_neurons),
            nn.ReLU(),
            nn.Linear(n_neurons, n_neurons),
            nn.ReLU(),
        )

    def forward(self, inp1, inp2):
        batch_size = inp1.size(0)
        append = torch.ones((batch_size, 1)).type_as(inp1)
        inp1 = torch.cat([inp1, append], dim=-1)
        inp2 = torch.cat([inp2, append], dim=-1)
        fusion = torch.zeros(
            (batch_size, self.inp1_size + 1, self.inp2_size + 1)
        ).type_as(inp1)
        for i in range(batch_size):
            fusion[i] = torch.outer(inp1[i], inp2[i])
        fusion = fusion.flatten(1)

        return self.feedforward(fusion)


# class MAGGate(nn.Module):
#     def __init__(self, inp1_size, inp2_size, dropout):
#         super(MAGGate, self).__init__()

#         self.fc1 = nn.Linear(inp1_size + inp2_size, 1)
#         self.fc3 = nn.Linear(inp2_size, inp1_size)
#         self.beta = nn.Parameter(torch.randn((1,)))
#         self.norm = nn.LayerNorm(inp1_size)
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, inp1, inp2):
#         w2 = torch.sigmoid(self.fc1(torch.cat([inp1, inp2], -1)))
#         adjust = self.fc3(w2 * inp2)
#         one = torch.tensor(1).type_as(adjust)
#         alpha = torch.min(torch.norm(inp1) / torch.norm(adjust) * self.beta, one)
#         output = inp1 + alpha * adjust
#         output = self.dropout(self.norm(output))
#         return output

# class MAGGate(nn.Module):
#     # https://github.com/WasifurRahman/BERT_multimodal_transformer/blob/master/modeling.py
#     def __init__(self, modality1_dim, modality2_dim, modality3_dim, modality4_dim, device,
#                  beta_shift=0.5, dropout_prob=0.1):
#         super(MAGGate, self).__init__()

#         self.device = device

#         self.W_hm2 = nn.Linear(modality2_dim + modality1_dim, modality1_dim)
#         self.W_hm3 = nn.Linear(modality3_dim + modality1_dim, modality1_dim)
#         self.W_hm4 = nn.Linear(modality4_dim + modality1_dim, modality1_dim)

#         self.W_m2 = nn.Linear(modality2_dim, modality1_dim)
#         self.W_m3 = nn.Linear(modality3_dim, modality1_dim)
#         self.W_m4 = nn.Linear(modality4_dim, modality1_dim)

#         self.beta_shift = beta_shift

#         self.LayerNorm = nn.LayerNorm(modality1_dim)
#         self.dropout = nn.Dropout(dropout_prob)

#     def forward(self, modality1, modality2, modality3, modality4):
#         eps = 1e-6
#         weight_m2 = F.relu(self.W_hm2(torch.cat((modality2, modality1), dim=-1)))
#         weight_m3 = F.relu(self.W_hm3(torch.cat((modality3, modality1), dim=-1)))
#         weight_m4 = F.relu(self.W_hm4(torch.cat((modality4, modality1), dim=-1)))

#         h_m = (weight_m2 * self.W_m2(modality2) +
#                weight_m3 * self.W_m3(modality3) +
#                weight_m4 * self.W_m4(modality4))

#         em_norm = modality1.norm(2, dim=-1)
#         hm_norm = h_m.norm(2, dim=-1)

#         hm_norm_ones = torch.ones(hm_norm.shape, requires_grad=True).to(self.device)
#         hm_norm = torch.where(hm_norm == 0, hm_norm_ones, hm_norm)

#         thresh_hold = (em_norm / (hm_norm + eps)) * self.beta_shift

#         ones = torch.ones(thresh_hold.shape, requires_grad=True).to(self.device)

#         alpha = torch.min(thresh_hold, ones)
#         alpha = alpha.unsqueeze(dim=-1)

#         multimodal_embedding = alpha * h_m

#         embedding_output = self.dropout(
#             self.LayerNorm(multimodal_embedding)
#         )

#         return embedding_output


class MAGGate(nn.Module):
    # https://github.com/WasifurRahman/BERT_multimodal_transformer/blob/master/modeling.py
    def __init__(self, modality_dims, device, beta_shift=0.5, dropout_prob=0.1):
        super(MAGGate, self).__init__()

        self.device = device
        self.beta_shift = beta_shift

        # Linear layers for each modality
        self.W_hms = nn.ModuleList(
            [
                nn.Linear(modality_dims[i] + modality_dims[0], modality_dims[0])
                for i in range(1, len(modality_dims))
            ]
        )

        self.W_ms = nn.ModuleList(
            [
                nn.Linear(modality_dims[i], modality_dims[0])
                for i in range(1, len(modality_dims))
            ]
        )

        self.LayerNorm = nn.LayerNorm(modality_dims[0])
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, modalities):
        eps = 1e-6

        modality1 = modalities[0]
        weights_m = []
        h_ms = []

        for i, modality in enumerate(modalities[1:], start=1):
            weight_m = F.relu(
                self.W_hms[i - 1](torch.cat((modality, modality1), dim=-1))
            )
            h_m = weight_m * self.W_ms[i - 1](modality)
            weights_m.append(weight_m)
            h_ms.append(h_m)

        h_m = sum(h_ms)

        em_norm = modality1.norm(2, dim=-1)
        hm_norm = h_m.norm(2, dim=-1)

        hm_norm_ones = torch.ones(hm_norm.shape, requires_grad=True).to(self.device)
        hm_norm = torch.where(hm_norm == 0, hm_norm_ones, hm_norm)

        thresh_hold = (em_norm / (hm_norm + eps)) * self.beta_shift

        ones = torch.ones(thresh_hold.shape, requires_grad=True).to(self.device)

        alpha = torch.min(thresh_hold, ones)
        alpha = alpha.unsqueeze(dim=-1)

        multimodal_embedding = alpha * h_m

        embedding_output = self.dropout(self.LayerNorm(multimodal_embedding))

        return embedding_output


class gateMLP(nn.Module):
    def __init__(self, input_dim, hidden_size, output_dim, dropout=0.1):
        super().__init__()

        self.gate = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_dim),
            nn.Sigmoid(),
        )

        self._initialize()

    def _initialize(self):
        for model in [self.gate]:
            for layer in model:
                if type(layer) in [nn.Linear]:
                    torch.nn.init.xavier_normal_(layer.weight)

    def forward(self, hidden_states):
        gate_logits = self.gate(hidden_states)
        return gate_logits


class TimeSeriesCnnModel(nn.Module):
    def __init__(
        self, input_size, n_filters, filter_size, dropout, length, n_neurons, layers
    ):
        super().__init__()

        padding = int(np.floor(filter_size / 2))
        self.layers = layers
        if layers >= 1:
            self.conv1 = nn.Conv1d(input_size, n_filters, filter_size, padding=padding)
            self.pool1 = nn.MaxPool1d(2, 2)

        if layers >= 2:
            self.conv2 = nn.Conv1d(n_filters, n_filters, filter_size, padding=padding)
            self.pool2 = nn.MaxPool1d(2, 2)

        if layers >= 3:
            self.conv3 = nn.Conv1d(n_filters, n_filters, filter_size, padding=padding)
            self.pool3 = nn.MaxPool1d(2, 2)

        self.fc1 = nn.Linear(int(length * n_filters / (2**layers)), n_neurons)
        self.fc1_drop = nn.Dropout(dropout)

    def forward(self, x):
        if self.layers >= 1:
            x = self.pool1(F.relu(self.conv1(x)))
        if self.layers >= 2:
            x = self.pool2(F.relu(self.conv2(x)))
        if self.layers >= 3:
            x = self.pool3(F.relu(self.conv3(x)))

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1_drop(self.fc1(x)))

        return x


# F.gumbel_softmax(logits, tau=1, hard=True)


class multiTimeAttention(nn.Module):
    "mTAND module"

    def __init__(self, input_dim, nhidden=16, embed_time=16, num_heads=1):
        super(multiTimeAttention, self).__init__()
        assert embed_time % num_heads == 0
        self.embed_time = embed_time
        self.embed_time_k = embed_time // num_heads
        self.h = num_heads
        self.dim = input_dim
        self.nhidden = nhidden
        self.linears = nn.ModuleList(
            [
                nn.Linear(embed_time, embed_time),
                nn.Linear(embed_time, embed_time),
                nn.Linear(input_dim * num_heads, nhidden),
            ]
        )

    def attention(self, query, key, value, mask=None, dropout=None):
        "Compute 'Scaled Dot Product Attention'"

        dim = value.size(-1)
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        scores = scores.unsqueeze(-1).repeat_interleave(dim, dim=-1)
        if mask is not None:
            if len(mask.shape) == 3:
                mask = mask.unsqueeze(-1)

            scores = scores.masked_fill(mask.unsqueeze(-3) == 0, -10000)
        p_attn = F.softmax(scores, dim=-2)
        if dropout is not None:
            p_attn = F.dropout(p_attn, p=dropout, training=self.training)
        return torch.sum(p_attn * value.unsqueeze(-3), -2), p_attn

    def forward(self, query, key, value, mask=None, dropout=0.1):
        "Compute 'Scaled Dot Product Attention'"
        batch, seq_len, dim = value.size()
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        value = value.unsqueeze(1)
        query, key = [
            l(x).view(x.size(0), -1, self.h, self.embed_time_k).transpose(1, 2)
            for l, x in zip(self.linears, (query, key))
        ]
        x, _ = self.attention(query, key, value, mask, dropout)
        x = x.transpose(1, 2).contiguous().view(batch, -1, self.h * dim)
        return self.linears[-1](x)


class MultiheadAttention(nn.Module):
    """Multi-headed attention.
    See "Attention Is All You Need" for more details.
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        attn_dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim**-0.5

        self.in_proj_weight = Parameter(torch.Tensor(3 * embed_dim, embed_dim))
        self.register_parameter("in_proj_bias", None)
        if bias:
            self.in_proj_bias = Parameter(torch.Tensor(3 * embed_dim))
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        if add_bias_kv:
            self.bias_k = Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.0)
            nn.init.constant_(self.out_proj.bias, 0.0)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def forward(self, query, key, value, attn_mask=None):
        """Input shape: Time x Batch x Channel
        Self-attention can be implemented by passing in the same arguments for
        query, key and value. Timesteps can be masked by supplying a T x T mask in the
        `attn_mask` argument. Padding elements can be excluded from
        the key by passing a binary ByteTensor (`key_padding_mask`) with shape:
        batch x src_len, where padding elements are indicated by 1s.
        """

        qkv_same = query.data_ptr() == key.data_ptr() == value.data_ptr()
        kv_same = key.data_ptr() == value.data_ptr()

        # embed_dim can be decomposed into num_heads x head_dim?
        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        assert key.size() == value.size()

        aved_state = None
        if qkv_same:
            # self-attention
            q, k, v = self.in_proj_qkv(query)
        elif kv_same:
            # encoder-decoder attention
            q = self.in_proj_q(query)

            if key is None:
                assert value is None
                k = v = None
            else:
                k, v = self.in_proj_kv(key)
        else:
            q = self.in_proj_q(query)
            k = self.in_proj_k(key)
            v = self.in_proj_v(value)
        q = q * self.scaling

        if self.bias_k is not None:
            assert self.bias_v is not None
            k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat(
                    [attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1
                )

        q = (
            q.contiguous()
            .view(tgt_len, bsz * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )
        if k is not None:
            k = (
                k.contiguous()
                .view(-1, bsz * self.num_heads, self.head_dim)
                .transpose(0, 1)
            )
        if v is not None:
            v = (
                v.contiguous()
                .view(-1, bsz * self.num_heads, self.head_dim)
                .transpose(0, 1)
            )

        # src_len = bsz * num_heads
        src_len = k.size(1)
        if self.add_zero_attn:
            src_len += 1
            k = torch.cat([k, k.new_zeros((k.size(0), 1) + k.size()[2:])], dim=1)
            v = torch.cat([v, v.new_zeros((v.size(0), 1) + v.size()[2:])], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat(
                    [attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1
                )

        attn_weights = torch.bmm(q, k.transpose(1, 2))
        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if attn_mask is not None:
            try:
                attn_weights += attn_mask.unsqueeze(0)
            except:
                print(attn_weights.shape)
                print(attn_mask.unsqueeze(0).shape)
                assert False

        attn_weights = F.softmax(attn_weights.float(), dim=-1).type_as(attn_weights)
        # attn_weights = F.relu(attn_weights)
        # attn_weights = attn_weights / torch.max(attn_weights)
        attn_weights = F.dropout(
            attn_weights, p=self.attn_dropout, training=self.training
        )

        attn = torch.bmm(attn_weights, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]

        # embed_dim = self.num_heads * self.head_dim
        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)

        # average attention weights over heads
        attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
        attn_weights = attn_weights.sum(dim=1) / self.num_heads

        return attn, attn_weights

    def in_proj_qkv(self, query):
        return self._in_proj(query).chunk(3, dim=-1)

    def in_proj_kv(self, key):
        return self._in_proj(key, start=self.embed_dim).chunk(2, dim=-1)

    def in_proj_q(self, query, **kwargs):
        return self._in_proj(query, end=self.embed_dim, **kwargs)

    def in_proj_k(self, key):
        return self._in_proj(key, start=self.embed_dim, end=2 * self.embed_dim)

    def in_proj_v(self, value):
        return self._in_proj(value, start=2 * self.embed_dim)

    def _in_proj(self, input, start=0, end=None, **kwargs):
        weight = kwargs.get("weight", self.in_proj_weight)
        bias = kwargs.get("bias", self.in_proj_bias)
        weight = weight[start:end, :]
        if bias is not None:
            bias = bias[start:end]
        return F.linear(input, weight, bias)


class TransformerEncoder(nn.Module):
    """
    Transformer encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.
    Args:
        embed_tokens (torch.nn.Embedding): input embedding
        num_heads (int): number of heads
        layers (int): number of layers
        attn_dropout (float): dropout applied on the attention weights
        relu_dropout (float): dropout applied on the first layer of the residual block
        res_dropout (float): dropout applied on the residual block
        attn_mask (bool): whether to apply mask on the attention weights
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        layers,
        device,
        attn_dropout=0.0,
        relu_dropout=0.0,
        res_dropout=0.0,
        embed_dropout=0.0,
        attn_mask=False,
        learn_embed=True,
        q_seq_len=None,
        kv_seq_len=None,
    ):
        super().__init__()
        self.dropout = embed_dropout  # Embedding dropout
        self.attn_dropout = attn_dropout
        self.embed_dim = embed_dim
        self.embed_scale = math.sqrt(embed_dim)
        self.device = device
        self.q_seq_len = q_seq_len
        self.kv_seq_len = kv_seq_len
        if learn_embed:
            if self.q_seq_len != None:
                self.embed_positions_q = nn.Embedding(
                    self.q_seq_len, embed_dim, padding_idx=0
                )
                nn.init.normal_(self.embed_positions_q.weight, std=0.02)

            if self.kv_seq_len != None:
                self.embed_positions_kv = nn.Embedding(self.kv_seq_len, embed_dim)
                nn.init.normal_(self.embed_positions_kv.weight, std=0.02)

        else:
            raise NotImplementedError
            self.embed_positions = SinusoidalPositionalEmbedding(embed_dim)

        self.attn_mask = attn_mask

        self.layers = nn.ModuleList([])
        for layer in range(layers):
            new_layer = TransformerEncoderLayer(
                embed_dim,
                num_heads=num_heads,
                attn_dropout=attn_dropout,
                relu_dropout=relu_dropout,
                res_dropout=res_dropout,
                attn_mask=attn_mask,
            )
            self.layers.append(new_layer)

        self.normalize = True
        if self.normalize:
            self.layer_norm = LayerNorm(embed_dim)

    def forward(self, x_in, x_in_k=None, x_in_v=None):
        """
        Args:
            x_in (FloatTensor): embedded input of shape `(src_len, batch, embed_dim)`
            x_in_k (FloatTensor): embedded input of shape `(src_len, batch, embed_dim)`
            x_in_v (FloatTensor): embedded input of shape `(src_len, batch, embed_dim)`
        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
        """

        x = x_in
        length_x = x.size(0)  # (length,Batch_size,input_dim)
        x = self.embed_scale * x_in
        if self.q_seq_len is not None:
            position_x = torch.tensor(torch.arange(length_x), dtype=torch.long).to(
                self.device
            )
            x += (self.embed_positions_q(position_x).unsqueeze(0)).transpose(
                0, 1
            )  # Add positional embedding
        x = F.dropout(x, p=self.dropout, training=self.training)

        if x_in_k is not None and x_in_v is not None:
            # embed tokens and positions

            length_kv = x_in_k.size(0)  # (Batch_size,length,input_dim)
            position_kv = torch.tensor(torch.arange(length_kv), dtype=torch.long).to(
                self.device
            )

            x_k = self.embed_scale * x_in_k
            x_v = self.embed_scale * x_in_v
            if self.kv_seq_len is not None:
                x_k += (self.embed_positions_kv(position_kv).unsqueeze(0)).transpose(
                    0, 1
                )  # Add positional embedding
                x_v += (self.embed_positions_kv(position_kv).unsqueeze(0)).transpose(
                    0, 1
                )  # Add positional embedding
            x_k = F.dropout(x_k, p=self.dropout, training=self.training)
            x_v = F.dropout(x_v, p=self.dropout, training=self.training)

        # encoder layers
        intermediates = [x]
        for layer in self.layers:
            if x_in_k is not None and x_in_v is not None:
                x = layer(x, x_k, x_v)
            else:
                x = layer(x)
            intermediates.append(x)

        if self.normalize:
            x = self.layer_norm(x)

        return x

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        if self.embed_positions is None:
            return self.max_source_positions
        return min(self.max_source_positions, self.embed_positions.max_positions())


class TransformerCrossEncoder(nn.Module):
    """
    Transformer encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`TransformerCrossEncoderLayer`.
    Args:
        embed_tokens (torch.nn.Embedding): input embedding
        num_heads (int): number of heads
        layers (int): number of layers
        attn_dropout (float): dropout applied on the attention weights
        relu_dropout (float): dropout applied on the first layer of the residual block
        res_dropout (float): dropout applied on the residual block
        attn_mask (bool): whether to apply mask on the attention weights
    """

    def __init__(
        self,
        args,
        embed_dim,
        num_heads,
        layers,
        device,
        attn_dropout=0.0,
        relu_dropout=0.0,
        res_dropout=0.0,
        embed_dropout=0.0,
        attn_mask=False,
        q_seq_len_1=None,
        q_seq_len_2=None,
        num_modalities=2,
    ):
        super().__init__()
        self.dropout = embed_dropout  # Embedding dropout
        self.attn_dropout = attn_dropout
        self.embed_dim = embed_dim
        self.embed_scale = math.sqrt(embed_dim)
        self.device = device

        self.q_seq_len_1 = q_seq_len_1
        # seq_len_1 is tt_max, the longest sequence length, which is 48 for 48 hrs
        self.q_seq_len_2 = q_seq_len_2
        self.num_modalities = num_modalities
        # self.intermediate=intermediate
        self.embed_positions_q_1 = nn.Embedding(
            self.q_seq_len_1, embed_dim, padding_idx=0
        )
        nn.init.normal_(self.embed_positions_q_1.weight, std=0.02)

        if self.q_seq_len_2 != None:
            self.embed_positions_q_2 = nn.Embedding(
                self.q_seq_len_2, embed_dim, padding_idx=0
            )
            nn.init.normal_(self.embed_positions_q_2.weight, std=0.02)
            self.embed_positions_q = nn.ModuleList(
                [self.embed_positions_q_1, self.embed_positions_q_2]
            )
        else:
            self.embed_positions_q = nn.ModuleList(
                [self.embed_positions_q_1 for _ in range(num_modalities)]
            )

        self.attn_mask = attn_mask
        self.layers = nn.ModuleList([])
        for layer in range(layers):
            new_layer = TransformerCrossEncoderLayer(
                args,
                embed_dim,
                num_heads=num_heads,
                attn_dropout=attn_dropout,
                relu_dropout=relu_dropout,
                res_dropout=res_dropout,
                attn_mask=attn_mask,
                num_modalities=num_modalities,
            )
            self.layers.append(new_layer)

        self.normalize = True
        if self.normalize:
            self.layer_norm = nn.ModuleList(
                [nn.LayerNorm(embed_dim) for _ in range(num_modalities)]
            )

    def forward(self, x_in_list, modality):
        """
        Args:
            x_in_list (list of FloatTensor): embedded input of shape `(src_len, batch, embed_dim)`
        Returns:
            dict:
                - **encoder_out** (Tensor): the list of last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`

        """

        # x_in_list contains ts and clinical notes tensors
        x_list = x_in_list
        lengths, positions = [], []
        for i in range(self.num_modalities):
            lengths.append(x_list[i].size(0))
        x_list = [self.embed_scale * x_in for x_in in x_in_list]
        if self.q_seq_len_1 is not None:
            for length in lengths:
                positions.append(
                    torch.tensor(torch.arange(length), dtype=torch.long).to(self.device)
                )
            x_list = [
                l(position_x).unsqueeze(0).transpose(0, 1) + x
                for l, x, position_x in zip(self.embed_positions_q, x_list, positions)
            ]
            # Add positional embedding
            x_list = [
                F.dropout(x, p=self.dropout, training=self.training) for x in x_list
            ]
        # encoder layers
        for layer in self.layers:
            x_list = layer(x_list, modality)  # proj_x_txt, proj_x_ts
            if x_list is None:
                return None

        if self.normalize:
            x_list = [l(x) for l, x in zip(self.layer_norm, x_list)]
        return x_list


class TransformerCrossEncoderLayer(nn.Module):
    def __init__(
        self,
        args,
        embed_dim,
        num_heads=4,
        attn_dropout=0.1,
        relu_dropout=0.1,
        res_dropout=0.1,
        attn_mask=False,
        num_modalities=2,
    ):
        super().__init__()
        self.args = args
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_modalities = num_modalities
        self.pre_self_attn_layer_norm = nn.ModuleList(
            [nn.LayerNorm(self.embed_dim) for _ in range(num_modalities)]
        )

        self.self_attns = nn.ModuleList(
            [
                MultiheadAttention(
                    embed_dim=self.embed_dim,
                    num_heads=self.num_heads,
                    attn_dropout=attn_dropout,
                )
                for _ in range(num_modalities)
            ]
        )

        self.post_self_attn_layer_norm = nn.ModuleList(
            [nn.LayerNorm(self.embed_dim) for _ in range(num_modalities)]
        )
        self.pre_encoder_attn_layer_norm = nn.ModuleList(
            [nn.LayerNorm(self.embed_dim) for _ in range(num_modalities)]
        )

        self.cross_attn_1 = MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            attn_dropout=attn_dropout,
        )

        self.cross_attn_2 = MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            attn_dropout=attn_dropout,
        )

        self.post_encoder_attn_layer_norm = nn.ModuleList(
            [nn.LayerNorm(self.embed_dim) for _ in range(num_modalities)]
        )

        self.attn_mask = attn_mask

        self.relu_dropout = relu_dropout
        self.res_dropout = res_dropout
        self.normalize_before = True

        self.pre_ffn_layer_norm = nn.ModuleList(
            [nn.LayerNorm(self.embed_dim) for _ in range(num_modalities)]
        )
        self.fc1 = nn.ModuleList(
            [
                nn.Linear(self.embed_dim, 4 * self.embed_dim)
                for _ in range(num_modalities)
            ]
        )  # The "Add & Norm" part in the paper
        self.fc2 = nn.ModuleList(
            [
                nn.Linear(4 * self.embed_dim, self.embed_dim)
                for _ in range(num_modalities)
            ]
        )
        self.pre_ffn_layer_norm = nn.ModuleList(
            [nn.LayerNorm(self.embed_dim) for _ in range(num_modalities)]
        )

        if args.cross_method == "moe" or args.cross_method == "LiMOE":
            moe_config = MoEConfig(
                num_experts=args.num_of_experts[0],
                moe_input_size=args.tt_max * args.embed_dim * num_modalities,
                moe_hidden_size=args.hidden_size,
                moe_output_size=args.tt_max * args.embed_dim * num_modalities,
                top_k=args.top_k[0],
                router_type=args.router_type,
                num_modalities=args.num_modalities,
                gating=args.gating_function[0],
            )
            self.moe = MoE(moe_config)
            self.moe = self.moe.to("cuda:0")
        elif args.cross_method == "hme":
            moe_config = MoEConfig(
                num_experts=args.num_of_experts,
                moe_input_size=args.tt_max * args.embed_dim * num_modalities,
                moe_hidden_size=args.hidden_size,
                moe_output_size=args.tt_max * args.embed_dim * num_modalities,
                top_k=args.top_k,
                router_type=args.router_type,
                num_modalities=args.num_modalities,
                gating=args.gating_function,
            )
            self.moe = HierarchicalMoE(moe_config)
            self.moe = self.moe.to("cuda:0")

    def forward(self, x_list, modality):
        """
        Args:
            x (List of Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.
        Returns:
            list of encoded output of shape `(batch, src_len, embed_dim)`
        """
        # TODO: figure out how many layers of this is required?
        residual = x_list
        seq_len, bs = x_list[0].shape[0], x_list[0].shape[1]

        x_list = [l(x) for l, x in zip(self.pre_self_attn_layer_norm, x_list)]

        output = [l(query=x, key=x, value=x) for l, x in zip(self.self_attns, x_list)]
        # attn: output[0][0].shape -> [48, 3, 128]; attn_weights: output[0][1].shape -> [3, 48, 48]
        # filter out attn_weights
        x_list = [x for x, _ in output]
        x_list = [
            F.dropout(x, p=self.res_dropout, training=self.training) for x in x_list
        ]
        x_list = [r + x for r, x in zip(residual, x_list)]

        # moe or cross attn
        residual = x_list
        x_list = [l(x) for l, x in zip(self.pre_encoder_attn_layer_norm, x_list)]
        if self.args.cross_method in ["moe", "hme"]:
            x_mod_in = [torch.reshape(x, (bs, -1)) for x in x_list]
            embd_len_list = [0] + list(np.cumsum([x.shape[1] for x in x_mod_in]))
            embeddings = torch.concat(x_mod_in, dim=1)
            if torch.isnan(embeddings).any():
                return None
            # just replace this with hierarchical moe
            moe_out, balance_loss = self.moe(x_mod_in, modalities=modality)
            x_mod_out = [
                moe_out[:, embd_len_list[i] : embd_len_list[i + 1]]
                for i in range(len(embd_len_list) - 1)
            ]
            x_allmod_output = [torch.reshape(x, (seq_len, bs, -1)) for x in x_mod_out]
            moe_output = [
                F.dropout(x, p=self.res_dropout, training=self.training)
                for x in x_allmod_output
            ]
            x_list = [r + x for r, x in zip(residual, moe_output)]

        # pay attention to how the text and patch embeddings are concated in LIMOE
        # LIMOE just concat? add modality type embeddings
        # sparse attention combined with dense attention
        if self.args.cross_method == "self_cross":
            assert (
                self.num_modalities == 2
            ), "Input modality should be 2 if using cross attention method."
            x_txt, x_ts = x_list  # proj_x_txt, proj_x_ts
            x_ts_to_txt, _ = self.cross_attn_1(query=x_txt, key=x_ts, value=x_ts)
            x_txt_to_ts, _ = self.cross_attn_2(query=x_ts, key=x_txt, value=x_txt)

            x_ts_to_txt = F.dropout(
                x_ts_to_txt, p=self.res_dropout, training=self.training
            )
            x_txt_to_ts = F.dropout(
                x_txt_to_ts, p=self.res_dropout, training=self.training
            )
            x_list = [r + x for r, x in zip(residual, (x_ts_to_txt, x_txt_to_ts))]

        # FNN
        residual = x_list
        x_list = [l(x) for l, x in zip(self.pre_ffn_layer_norm, x_list)]
        x_list = [F.relu(l(x)) for l, x in zip(self.fc1, x_list)]
        x_list = [
            F.dropout(x, p=self.relu_dropout, training=self.training) for x in x_list
        ]
        x_list = [l(x) for l, x in zip(self.fc2, x_list)]
        x_list = [
            F.dropout(x, p=self.res_dropout, training=self.training) for x in x_list
        ]
        x_list = [r + x for r, x in zip(residual, x_list)]
        return x_list


class TransformerEncoderLayer(nn.Module):
    """Encoder layer block.
    In the original paper each operation (multi-head attention or FFN) is
    postprocessed with: `dropout -> add residual -> layernorm`. In the
    tensor2tensor code they suggest that learning is more robust when
    preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.encoder_normalize_before* to ``True``.
    Args:
        embed_dim: Embedding dimension
    """

    def __init__(
        self,
        embed_dim,
        num_heads=4,
        attn_dropout=0.1,
        relu_dropout=0.1,
        res_dropout=0.1,
        attn_mask=False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.self_attn = MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            attn_dropout=attn_dropout,
        )
        self.attn_mask = attn_mask

        self.relu_dropout = relu_dropout
        self.res_dropout = res_dropout
        self.normalize_before = True

        self.fc1 = Linear(
            self.embed_dim, 4 * self.embed_dim
        )  # The "Add & Norm" part in the paper
        self.fc2 = Linear(4 * self.embed_dim, self.embed_dim)
        self.layer_norms = nn.ModuleList([LayerNorm(self.embed_dim) for _ in range(2)])

    def forward(self, x, x_k=None, x_v=None):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.
            x_k (Tensor): same as x
            x_v (Tensor): same as x
        Returns:bpbpp
            encoded output of shape `(batch, src_len, embed_dim)`
        """

        residual = x
        x = self.maybe_layer_norm(0, x, before=True)
        mask = buffered_future_mask(x, x_k) if self.attn_mask else None
        if x_k is None and x_v is None:
            x, _ = self.self_attn(query=x, key=x, value=x, attn_mask=mask)
        else:
            x_k = self.maybe_layer_norm(0, x_k, before=True)
            x_v = self.maybe_layer_norm(0, x_v, before=True)
            x, _ = self.self_attn(query=x, key=x_k, value=x_v, attn_mask=mask)
        x = F.dropout(x, p=self.res_dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(0, x, after=True)

        residual = x
        x = self.maybe_layer_norm(1, x, before=True)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.relu_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.res_dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(1, x, after=True)
        return x

    def maybe_layer_norm(self, i, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return self.layer_norms[i](x)
        else:
            return x


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    #     nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m


def LayerNorm(embedding_dim):
    m = nn.LayerNorm(embedding_dim)

    return m


def fill_with_neg_inf(t):
    """FP16-compatible function that fills a tensor with -inf."""
    return t.float().fill_(float("-inf")).type_as(t)


def buffered_future_mask(tensor, tensor2=None):
    dim1 = dim2 = tensor.size(0)
    if tensor2 is not None:
        dim2 = tensor2.size(0)
    future_mask = torch.triu(
        fill_with_neg_inf(torch.ones(dim1, dim2)), 1 + abs(dim2 - dim1)
    )
    if tensor.is_cuda:
        future_mask = future_mask.cuda()
    return future_mask[:dim1, :dim2]


class PatchEmbeddings(nn.Module):
    def __init__(self, feature_size, num_patches, embed_dim, dropout=0.25):
        super().__init__()
        patch_size = math.ceil(feature_size / num_patches)
        pad_size = num_patches * patch_size - feature_size
        self.pad_size = pad_size
        self.num_patches = num_patches
        self.feature_size = feature_size
        self.patch_size = patch_size
        self.projection = nn.Sequential(
            nn.Linear(patch_size, embed_dim),
            nn.ELU(),
            nn.AlphaDropout(p=dropout, inplace=False),
        )

    def forward(self, x):
        x = F.pad(x, (0, self.pad_size)).view(
            x.shape[0], self.num_patches, self.patch_size
        )
        x = self.projection(x)
        return x


class ResBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm3d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm3d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class ResNet3DEmbedding(nn.Module):
    def __init__(self, in_channels, seq_len, embedding_dim):
        super(ResNet3DEmbedding, self).__init__()
        self.layer0 = nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
        )
        self.layer1 = self._make_layer(64, 64, stride=1)
        self.layer2 = self._make_layer(64, 128, stride=2)
        self.layer3 = self._make_layer(128, 256, stride=2)
        self.layer4 = self._make_layer(256, 512, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512, seq_len * embedding_dim)
        self.seq_len = seq_len
        self.embedding_dim = embedding_dim

    def _make_layer(self, in_channels, out_channels, stride):
        return nn.Sequential(
            ResBlock3D(in_channels, out_channels, stride),
            ResBlock3D(out_channels, out_channels, 1),
        )

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = x.view(-1, self.seq_len, self.embedding_dim)
        return x


# class FlexibleMULTModel(nn.Module):
#     # https://github.com/yaohungt/Multimodal-Transformer/blob/master/src/models.py
#     def __init__(self, num_modalities, modalities_dims, output_dim, main_modality=None,
#                  num_heads=8, layers=6,
#                  attn_dropout=0.1, relu_dropout=0.1, res_dropout=0.1, out_dropout=0.1, embed_dropout=0.25,
#                  attn_mask=None):
#         """
#         Construct a flexible MulT model for k modalities, supporting both main_modality and partial_mode.
#         """
#         super(FlexibleMULTModel, self).__init__()
#         self.num_modalities = num_modalities
#         self.modalities_dims = modalities_dims
#         self.output_dim = output_dim
#         self.main_modality = main_modality
#         self.num_heads = num_heads
#         self.layers = layers
#         self.attn_dropout = attn_dropout
#         self.relu_dropout = relu_dropout
#         self.res_dropout = res_dropout
#         self.out_dropout = out_dropout
#         self.embed_dropout = embed_dropout
#         self.attn_mask = attn_mask

#         # Initialize projection layers for each modality
#         self.proj_layers = nn.ModuleList([
#             nn.Conv1d(in_channels=dim, out_channels=30, kernel_size=1, padding=0, bias=False)
#             for dim in self.modalities_dims
#         ])

#         # Initialize cross-modal and self-attention networks
#         self.cross_modal_nets = nn.ModuleDict()
#         self.self_attention_nets = nn.ModuleList()
#         modalities = [f'modality{i+1}' for i in range(self.num_modalities)]
#         for i in range(self.num_modalities):
#             for j in range(self.num_modalities):
#                 if i != j:
#                     key = f'{modalities[i]}_with_{modalities[j]}'
#                     self.cross_modal_nets[key] = self.get_network(30)

#             # Self-attention for combined features of each modality
#             self.self_attention_nets.append(self.get_network(30 * (self.num_modalities - 1)))

#         # Projection layers for output
#         self.proj1 = nn.Linear(30 * self.num_modalities, 30 * self.num_modalities)
#         self.proj2 = nn.Linear(30 * self.num_modalities, 30 * self.num_modalities)
#         self.out_layer = nn.Linear(30 * self.num_modalities, self.output_dim)

#     def get_network(self, embed_dim):
#         return TransformerEncoder(
#             embed_dim=embed_dim,
#             num_heads=self.num_heads,
#             layers=self.layers,
#             attn_dropout=self.attn_dropout,
#             relu_dropout=self.relu_dropout,
#             res_dropout=self.res_dropout,
#             embed_dropout=self.embed_dropout,
#             attn_mask=self.attn_mask
#         )

#     def forward(self, *inputs):
#         assert len(inputs) == self.num_modalities, "Number of inputs must match the number of modalities."

#         # Project and transpose modalities
#         projected = [self.proj_layers[i](input.transpose(1, 2)).permute(2, 0, 1) for i, input in enumerate(inputs)]

#         if self.main_modality is not None:
#             # Use main_modality to guide the feature fusion
#             combined_features = []
#             modalities = [f'modality{i+1}' for i in range(self.num_modalities)]
#             for i in range(self.num_modalities):
#                 if i == self.main_modality:
#                     continue
#                 key = f'{modalities[self.main_modality]}_with_{modalities[i]}'
#                 combined_features.append(self.cross_modal_nets[key](projected[self.main_modality], projected[i], projected[i]))

#             main_features = torch.cat(combined_features, dim=2)
#             output_features = self.self_attention_nets[self.main_modality](main_features)

#         else:
#             # Partial mode processing, integrate features from all modalities independently
#             outputs = []
#             for i in range(self.num_modalities):
#                 cross_features = [self.cross_modal_nets[f'{modalities[i]}_with_{modalities[j]}'](projected[i], projected[j], projected[j])
#                                   for j in range(self.num_modalities) if i != j]
#                 cross_features = torch.cat(cross_features, dim=2)
#                 outputs.append(self.self_attention_nets[i](cross_features))

#             output_features = torch.cat(outputs, dim=1)  # Concatenate all modality outputs

#         # Final output processing
#         last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(output_features)), p=self.out_dropout, training=self.training))
#         last_hs_proj += output_features
#         output = self.out_layer(last_hs_proj)
#         return output, last_hs_proj


class vExpert(nn.Module):
    def __init__(self, expert_id, input_size, output_size):
        super().__init__()
        self.expert_id = expert_id
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.fc(x)


class FlexMoE(nn.Module):
    def __init__(self, num_classes=2, num_experts=4, input_size=128, output_size=128):
        super().__init__()
        self.num_experts = num_experts
        self.input_size = input_size
        self.output_size = output_size
        self.vexperts = nn.ModuleList(
            [vExpert(i, input_size, output_size) for i in range(num_experts)]
        )
        self.gate = nn.Linear(input_size, num_experts)
        self.classification_head = Linear(output_size, num_classes).cuda()

    def forward(self, x):
        # Gate computation
        x = torch.stack(x)

        gate_logits = self.gate(x)
        gate_probs = torch.softmax(gate_logits, dim=-1)

        k = 2
        top_k_probs, top_k_indices = torch.topk(gate_probs, k, dim=-1)

        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

        expert_outputs = self.dispatch_to_experts(x, top_k_indices)

        outputs = (expert_outputs.unsqueeze(-2) * top_k_probs.unsqueeze(-1)).sum(dim=-2)

        outputs = outputs.transpose(0, 1)

        outputs.reshape(outputs.shape[0], -1)

        outputs = torch.mean(outputs, 1)

        outputs = self.classification_head(outputs)

        return outputs

    def dispatch_to_experts(self, inputs, expert_indices):

        outputs = torch.zeros_like(inputs).to(inputs.dtype)

        for i in range(self.num_experts):
            mask = (expert_indices == i).any(dim=-1)

            if mask.any():
                expert_inputs = inputs[mask]
                expert_outputs = self.vexperts[i](expert_inputs)
                outputs[mask] = expert_outputs.to(inputs.dtype)

        return outputs


class Scheduler:
    def __init__(self, flexmoe, threshold=0.8):
        self.flexmoe = flexmoe
        self.threshold = threshold
        self.expert_loads = {i: 0 for i in range(flexmoe.num_experts)}

    def monitor_and_adjust(self, token_assignment):
        self.update_expert_loads(token_assignment)
        balance_ratio = self.calculate_balance_ratio()
        if balance_ratio < self.threshold:
            self.trigger_adjustment()

    def update_expert_loads(self, token_assignment):
        for expert_id, load in token_assignment.items():
            self.expert_loads[expert_id] += load

    def calculate_balance_ratio(self):
        max_load = max(self.expert_loads.values())
        avg_load = sum(self.expert_loads.values()) / len(self.expert_loads)
        return avg_load / max_load if max_load > 0 else 1.0

    def trigger_adjustment(self):
        policy_maker = PolicyMaker(self.flexmoe)
        plan = policy_maker.make_scheduling_plan(self.expert_loads)
        self.apply_plan(plan)

    def apply_plan(self, plan):
        for operation, expert_id in plan:
            if operation == "Expand":
                self.expand_expert(expert_id)
            elif operation == "Shrink":
                self.shrink_expert(expert_id)

    def expand_expert(self, expert_id):
        new_expert = vExpert(
            len(self.flexmoe.vexperts),
            self.flexmoe.input_size,
            self.flexmoe.output_size,
        )
        new_expert.fc.weight.data = self.flexmoe.vexperts[
            expert_id
        ].fc.weight.data.clone()
        new_expert.fc.bias.data = self.flexmoe.vexperts[expert_id].fc.bias.data.clone()
        self.flexmoe.vexperts.append(new_expert)
        self.expert_loads[len(self.flexmoe.vexperts) - 1] = (
            self.expert_loads[expert_id] // 2
        )
        self.expert_loads[expert_id] //= 2

    def shrink_expert(self, expert_id):
        if len(self.flexmoe.vexperts) > self.flexmoe.num_experts:
            self.flexmoe.vexperts = nn.ModuleList(
                [e for e in self.flexmoe.vexperts if e.expert_id != expert_id]
            )
            del self.expert_loads[expert_id]


class PolicyMaker:
    def __init__(self, flexmoe):
        self.flexmoe = flexmoe

    def make_scheduling_plan(self, expert_loads):
        plan = []
        max_load_expert = max(expert_loads, key=expert_loads.get)
        min_load_expert = min(expert_loads, key=expert_loads.get)

        if expert_loads[max_load_expert] > 2 * expert_loads[min_load_expert]:
            plan.append(("Expand", max_load_expert))

        if len(self.flexmoe.vexperts) > self.flexmoe.num_experts and expert_loads[
            min_load_expert
        ] < 0.5 * (sum(expert_loads.values()) / len(expert_loads)):
            plan.append(("Shrink", min_load_expert))

        return plan
