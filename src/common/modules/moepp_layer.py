import typing
from collections.abc import Callable
from collections import defaultdict
from typing import Any, Dict, TYPE_CHECKING, Optional, Tuple, List

import torch
import copy

from torch import Tensor
from torch.nn import Module
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    Base = Module[Tensor]
else:
    Base = Module


MOE_TOP_K = 2
Constant = 2


class CopyExpert(torch.nn.Module):
    def __init__(self, expert):
        super(CopyExpert, self).__init__()
        pass

    def forward(self, inputs):
        return inputs


class ZeroExpert(torch.nn.Module):
    def __init__(self, expert):
        super(ZeroExpert, self).__init__()
        pass

    def forward(self, inputs):
        return torch.zeros_like(inputs).to(inputs.dtype).to(inputs.device)


class ConstantExpert(torch.nn.Module):
    def __init__(self, expert):
        super(ConstantExpert, self).__init__()
        self.constant = torch.nn.Parameter(torch.empty((expert.hidden_size)))
        torch.nn.init.normal_(self.constant)

        self.wg = torch.nn.Linear(expert.hidden_size, 2, bias=False)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, inputs):
        # print(inputs.size())
        weight = self.wg(inputs)
        weight = self.softmax(weight)
        return torch.einsum(
            "b,bd->bd", [weight[:, 0].type_as(inputs), inputs]
        ) + torch.einsum(
            "b,d->bd", [weight[:, 1].type_as(inputs), self.constant.type_as(inputs)]
        )


def gating(
    logits: Tensor,
    moe_use_mixtral_gating=False,
    moe_use_logits_norm=False,
    moe_gate_norm_std=1.0,
) -> Dict[int, List[Tuple[int, float]]]:
    # gates shape [num_tokens, num_experts]
    num_experts = logits.size(1)
    if moe_use_mixtral_gating:
        if moe_use_logits_norm:
            target_std = moe_gate_norm_std
            logits_std = logits.std(dim=1, keepdim=True)
            logits = logits / (logits_std / target_std)
        gates, indices = torch.topk(logits, k=MOE_TOP_K, dim=1)
        gates = F.softmax(gates, dim=1)
    else:
        target_std = moe_gate_norm_std
        if moe_use_logits_norm:
            logits_std = logits.std(dim=1, keepdim=True)
            gates = F.softmax(logits / (logits_std / target_std), dim=1)
        else:
            gates = F.softmax(logits, dim=1)
        # gates shape [num_tokens, MOE_TOP_K]
        # indices shape [num_tokens, MOE_TOP_K]
        gates, indices = torch.topk(gates, k=MOE_TOP_K, dim=1)
        gates = torch.where(
            indices == (num_experts - 1),
            torch.zeros_like(gates).to(gates.dtype).to(gates.device),
            gates,
        )
        gates /= torch.sum(gates, dim=1, keepdim=True)

    expert_info = defaultdict(list)
    for expert_id in range(num_experts):
        token_ids, score_ids = torch.nonzero(indices == expert_id, as_tuple=True)
        expert_info[expert_id] = [token_ids, gates[token_ids, score_ids]]

    return expert_info


class Router(Module):
    def __init__(
        self,
        model_dim: int,
        num_experts: int,
        moe_use_mixtral_gating: bool,
        moe_2layer_gate: bool,
        moe_use_logits_norm: bool,
        moe_gate_norm_std: float,
    ) -> None:
        super().__init__()

        if moe_2layer_gate:
            self.wg = torch.nn.Sequential(
                torch.nn.Linear(model_dim, num_experts * 8, bias=False).float(),
                torch.nn.Tanh(),
                torch.nn.Linear(num_experts * 8, num_experts, bias=False).float(),
            ).float()
        else:
            self.wg = torch.nn.Linear(model_dim, num_experts, bias=False).float()

        self.gate_map = torch.nn.Linear(num_experts, num_experts, bias=False)

        self.gate = gating
        self.moe_use_mixtral_gating = moe_use_mixtral_gating
        self.moe_use_logits_norm = moe_use_logits_norm
        self.moe_gate_norm_std = moe_gate_norm_std

    def forward(
        self, input: torch.Tensor, gate_residual=None
    ) -> Dict[int, List[Tuple[int, float]]]:
        if isinstance(self.wg, torch.nn.Linear):
            if self.wg.weight.dtype != torch.float32:
                self.wg = self.wg.float()
                setattr(self.wg.weight, "router", True)
        else:
            if self.wg[0].weight.dtype != torch.float32:
                self.wg = self.wg.float()
                setattr(self.wg[0].weight, "router", True)
                setattr(self.wg[2].weight, "router", True)
        input_fp32 = input.float()
        logits = self.wg(input_fp32)

        if gate_residual is not None:
            gate_residual = self.gate_map(gate_residual.to(self.gate_map.weight.dtype))
            logits += gate_residual

        gate_output = self.gate(
            logits,
            self.moe_use_mixtral_gating,
            self.moe_use_logits_norm,
            self.moe_gate_norm_std,
        )

        return gate_output, logits


class Experts(torch.nn.Module):
    def __init__(self, expert, num_local_experts=1):
        super(Experts, self).__init__()

        self.experts = torch.nn.ModuleList(
            [copy.deepcopy(expert) for _ in range(num_local_experts - 2 - Constant)]
            + [ConstantExpert(expert) for _ in range(Constant)]
            + [CopyExpert(expert), ZeroExpert(expert)]
        )

    def forward(self, inputs):
        raise NotImplementedError


class MOELayer(Base):
    def __init__(
        self,
        gate: Module,
        experts: Module,
        ep_size,
        num_local_experts: int,
        moe_use_mixtral_gating: bool,
        moe_feature_no_mul_topk: bool,
    ) -> None:
        super().__init__()
        self.gate = gate
        self.experts = experts
        self.ep_size = ep_size
        self.num_local_experts = num_local_experts
        self.moe_use_mixtral_gating = moe_use_mixtral_gating
        self.moe_feature_no_mul_topk = moe_feature_no_mul_topk

    def forward(self, *input: Tensor, gate_residual=None, **kwargs: Any) -> Tensor:
        d_model = input[0].shape[-1]
        reshaped_input = input[0].reshape(-1, d_model)
        output = torch.zeros_like(reshaped_input)
        expert_info, gate_residual = self.gate(reshaped_input, gate_residual)
        if not (self.moe_use_mixtral_gating or self.moe_feature_no_mul_topk):
            reshaped_input *= MOE_TOP_K
        for expert, token_indices_and_gates in expert_info.items():
            indices, gating = token_indices_and_gates
            gating = gating.unsqueeze(-1)
            tokens = reshaped_input.index_select(dim=0, index=indices)
            expert_output = self.experts.experts[expert](tokens)
            expert_output *= gating
            output.index_add_(dim=0, index=indices, source=expert_output)
        output = output.reshape(input[0].shape)

        return output, gate_residual


class MOE(torch.nn.Module):
    def __init__(
        self,
        hidden_size,
        expert,
        num_experts=1,
        ep_size=1,
        moe_use_mixtral_gating=False,
        moe_2layer_gate=True,
        moe_use_logits_norm=False,
        moe_gate_norm_std=1.0,
        moe_feature_no_mul_topk=False,
    ):
        super(MOE, self).__init__()

        self.ep_size = ep_size
        self.num_experts = num_experts
        self.num_local_experts = num_experts // self.ep_size
        self.moe_use_mixtral_gating = moe_use_mixtral_gating
        self.moe_2layer_gate = moe_2layer_gate
        self.moe_use_logits_norm = moe_use_logits_norm
        self.moe_gate_norm_std = moe_gate_norm_std
        self.moe_feature_no_mul_topk = moe_feature_no_mul_topk

        experts = Experts(expert, self.num_local_experts)
        self.moe = MOELayer(
            Router(
                hidden_size,
                num_experts,
                self.moe_use_mixtral_gating,
                self.moe_2layer_gate,
                self.moe_use_logits_norm,
                self.moe_gate_norm_std,
            ),
            experts,
            self.ep_size,
            self.num_local_experts,
            self.moe_use_mixtral_gating,
            self.moe_feature_no_mul_topk,
        )

    def forward(self, hidden_states, used_token=None, gate_residual=None):
        output, gate_residual = self.moe(
            hidden_states, used_token, gate_residual=gate_residual
        )
        return output, gate_residual


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


class MLPMoE(nn.Module):
    def __init__(self, hidden_size=128, intermediate_size=64):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.up_proj = nn.Linear(
            self.hidden_size, self.intermediate_size * 2, bias=False
        )
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

        def swiglu(x):
            x = torch.chunk(x, 2, dim=-1)
            return F.silu(x[0]) * x[1]

        self.act_fn = swiglu

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.up_proj(x)))

        return down_proj


class MoEPlusPlusEncoderLayer(nn.Module):
    def __init__(
        self,
        num_experts,
        d_model,
        num_head,
        num_modalities,
        dropout=0.1,
        activation=nn.GELU,
    ):
        super(MoEPlusPlusEncoderLayer, self).__init__()
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
        self.mlp = MOE(
            d_model,
            MLPMoE(hidden_size=d_model, intermediate_size=64),
            num_experts=num_experts,
            moe_use_mixtral_gating=False,
            moe_2layer_gate=True,
            moe_use_logits_norm=False,
            moe_gate_norm_std=1.0,
            moe_feature_no_mul_topk=False,
        )

    def forward(self, x, gate_residual=None):
        chunk_size = [item.shape[1] for item in x]

        residual = torch.cat(x, dim=1)
        normed_x = self.norm1(torch.cat(x, dim=1))
        kv = normed_x
        attended_x = self.attn(normed_x, kv)
        hidden_states = residual + self.dropout1(attended_x)

        # Fully Connected
        residual = hidden_states.clone()
        hidden_states, gate_residual = self.mlp(
            self.norm2(hidden_states), gate_residual=gate_residual
        )

        hidden_states = residual + self.dropout1(attended_x)

        updated_x = torch.split(hidden_states, chunk_size, dim=1)
        updated_x = [item for item in updated_x]

        return updated_x, gate_residual
