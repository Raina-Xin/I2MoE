import math

from torch import Tensor
from torch.nn.init import xavier_uniform_

import torch
import torch.nn.functional as F
import torch.nn as nn

# from timm.models.layers import trunc_normal_

from fmoe.transformer import _Expert
from fmoe.layers import FMoE, _fmoe_general_global_forward, mark_module_parallel_comm
from fmoe.functions import ensure_comm, Slice, AllGather
from fmoe.gates import NaiveGate
import fmoe_cuda as fmoe_native
from fmoe.functions import count_by_gate

import tree

from fmoe.gates import NoisyGate
import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


class FixedFMoE(nn.Module):
    def __init__(
        self,
        num_expert=32,
        d_model=1024,
        world_size=1,
        mp_group=None,
        slice_group=None,
        moe_group=None,
        top_k=2,
        gate=NaiveGate,
        expert=None,
        gate_hook=None,
        mask=None,
        mask_dict=None,
    ):
        super().__init__()
        self.num_expert = num_expert
        self.d_model = d_model
        self.world_size = world_size
        self.slice_group = slice_group
        if mp_group is not None:
            print("[Warning] mp_group is being deprecated")
            self.slice_group = mp_group
        if self.slice_group is None:
            self.slice_size = 1
            self.slice_rank = 0
        else:
            self.slice_size = self.slice_group.size()
            self.slice_rank = self.slice_group.rank()
        self.top_k = top_k
        if type(expert) is list:
            self.experts = nn.ModuleList([e(d_model) for e in expert])
            self.experts_fused = False
            self.num_expert = num_expert = len(expert)
        elif expert is not None:
            self.experts = nn.ModuleList([expert(d_model) for _ in range(num_expert)])
            self.experts_fused = False
        else:
            self.experts_fused = True
        self.gate = gate(d_model, num_expert, world_size, top_k)
        self.gate_hook = gate_hook
        self.mask = mask
        self.mask_dict = mask_dict
        self.moe_group = moe_group

    def expert_fn(self, inp, fwd_expert_count):
        if self.experts_fused:
            return self.experts(inp, fwd_expert_count)
        if isinstance(fwd_expert_count, torch.Tensor):
            fwd_expert_count = fwd_expert_count.cpu().numpy()
        outputs = []
        base_idx = 0
        for i in range(self.num_expert):
            batch_size = fwd_expert_count[i]
            inp_slice = inp[base_idx : base_idx + batch_size]
            outputs.append(self.experts[i](inp_slice))
            base_idx += batch_size
        return torch.cat(outputs, dim=0)

    def mark_parallel_comm(self, expert_dp_comm="none"):
        if self.experts is not None:
            comm = expert_dp_comm
            if isinstance(self.experts, list):
                for e in self.experts:
                    mark_module_parallel_comm(e, comm)
            else:
                mark_module_parallel_comm(self.experts, comm)
        mark_module_parallel_comm(self.gate, "gate")

    def forward(self, moe_inp):
        r"""
        The FMoE module first computes gate output, and then conduct MoE forward
        according to the gate.  The score of the selected gate given by the
        expert is multiplied to the experts' output tensors as a weight.
        """
        moe_inp_batch_size = tree.flatten(
            tree.map_structure(lambda tensor: tensor.shape[0], moe_inp)
        )
        assert all(
            [batch_size == moe_inp_batch_size[0] for batch_size in moe_inp_batch_size]
        ), "MoE inputs must have the same batch size"

        if self.world_size > 1:

            def ensure_comm_func(tensor):
                ensure_comm(tensor, self.moe_group)

            tree.map_structure(ensure_comm_func, moe_inp)

        if self.slice_size > 1:

            def slice_func(tensor):
                return Slice.apply(
                    tensor, self.slice_rank, self.slice_size, self.slice_group
                )

            moe_inp = tree.map_structure(slice_func, moe_inp)

        # gate_top_k_idx, gate_score = self.gate(moe_inp, expert_indices)
        gate_top_k_idx, gate_score = self.gate(moe_inp)

        # # Reshape gate_top_k_idx to be 2-dimensional
        # gate_top_k_idx = gate_top_k_idx.view(moe_inp.shape[0], self.top_k)

        if self.gate_hook is not None:
            self.gate_hook(gate_top_k_idx, gate_score, None)
        # self.gate.set_topk_indicates(gate_top_k_idx)

        if self.mask is not None and self.mask_dict is not None:

            def delete_mask_func(tensor):
                tensor = tensor[self.mask == 0, :]
                return tensor

            mask = self.mask.view(-1)
            moe_inp = tree.map_structure(delete_mask_func, moe_inp)
            gate_top_k_idx = gate_top_k_idx[mask == 0, :]

        fwd = _fmoe_general_global_forward(
            moe_inp,
            gate_top_k_idx,
            self.expert_fn,
            self.num_expert,
            self.world_size,
            experts=self.experts,
        )

        if self.mask is not None and self.mask_dict is not None:

            def recover_func(tensor):
                dim = tensor.shape[-1]
                tensor = tensor.view(-1, self.top_k, dim)
                x = torch.zeros(
                    mask.shape[0],
                    self.top_k,
                    dim,
                    device=tensor.device,
                    dtype=tensor.dtype,
                )
                x[mask == 0] = tensor
                for k, v in self.mask_dict.items():
                    x[mask == k] = v
                return x

            moe_outp = tree.map_structure(recover_func, fwd)
        else:

            def view_func(tensor):
                dim = tensor.shape[-1]
                tensor = tensor.view(-1, self.top_k, dim)
                return tensor

            moe_outp = tree.map_structure(view_func, fwd)

        gate_score = gate_score.view(-1, 1, self.top_k)

        def bmm_func(tensor):
            dim = tensor.shape[-1]
            tensor = torch.bmm(gate_score, tensor).reshape(-1, dim)
            return tensor

        moe_outp = tree.map_structure(bmm_func, moe_outp)

        if self.slice_size > 1:

            def all_gather_func(tensor):
                return AllGather.apply(
                    tensor, self.slice_rank, self.slice_size, self.slice_group
                )

            moe_outp = tree.map_structure(all_gather_func, moe_outp)

        moe_outp_batch_size = tree.flatten(
            tree.map_structure(lambda tensor: tensor.shape[0], moe_outp)
        )
        assert all(
            [batch_size == moe_outp_batch_size[0] for batch_size in moe_outp_batch_size]
        ), "MoE outputs must have the same batch size"
        return moe_outp


class FMoETransformerMLP(FixedFMoE):
    r"""
    A complete MoE MLP module in a Transformer block.
    * `activation` is the activation function to be used in MLP in each expert.
    * `d_hidden` is the dimension of the MLP layer.
    """

    def __init__(
        self,
        num_expert=32,
        d_model=1024,
        d_hidden=4096,
        activation=torch.nn.GELU(),
        expert_dp_comm="none",
        expert_rank=0,
        n_router=1,
        gate="AddtionalNoisyGate",  # NaiveGate
        world_size=1,
        top_k=2,
        **kwargs,
    ):
        if type(gate) is str:
            gate = eval(gate)
        super().__init__(
            num_expert=num_expert,
            d_model=d_model,
            gate=gate,
            world_size=world_size,
            top_k=top_k,
            **kwargs,
        )
        self.experts = _Expert(
            num_expert, d_model, d_hidden, activation, rank=expert_rank
        )
        self.n_router = n_router
        self.all_gates = nn.ModuleDict(
            {
                f"{i}": gate(d_model, num_expert, world_size, top_k)
                for i in range(n_router)
            }
        )
        self.gate = self.all_gates[f"{0}"]

        self.mark_parallel_comm(expert_dp_comm)

    def forward(self, inp: torch.Tensor):
        r"""
        This module wraps up the FMoE module with reshape, residual and layer
        normalization.
        """
        original_shape = inp.shape
        inp = inp.reshape(-1, self.d_model)

        output = super().forward(inp)

        return output.reshape(original_shape)


class AddtionalNoisyGate(NoisyGate):
    def __init__(self, d_model, num_expert, world_size, top_k=2):
        super().__init__(d_model, num_expert, world_size, top_k)
        self.topk_logits = []
        self.indicates = None
        self.is_full_modality = False

    def set_topk_logit(self, logit):
        self.topk_logits.append(logit)

    def reset_topk_logit(self):
        self.topk_logits = []

    def get_topk_logit(self, clear=True):
        topk_logit = self.topk_logits
        if clear:
            self.topk_logits = None
        return topk_logit

    def set_topk_indicates(self, indicate):
        self.indicates = indicate

    def get_topk_indicate(self, clear=True):
        topk_indicate = self.indicates
        if clear:
            self.indicates = None
        return topk_indicate

    def set_loss(self, loss):
        if self.loss is None:
            self.loss = loss
        else:
            self.loss += loss

    def forward(self, inp):
        # Clean logits for gate
        clean_logits = inp @ self.w_gate

        # Add noise to logits
        raw_noise_stddev = inp @ self.w_noise
        noise_stddev = (
            self.softplus(raw_noise_stddev) + self.noise_epsilon
        ) * self.training
        noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)

        logits = noisy_logits
        loss = 0

        # Calculate topk + 1 that will be needed for the noisy gates
        top_logits, top_indices = logits.topk(
            min(self.top_k + 1, self.tot_expert), dim=1
        )
        top_k_logits = top_logits[:, : self.top_k]
        top_k_indices = top_indices[:, : self.top_k]
        top_k_gates = self.softmax(top_k_logits)

        # Save top-k expert indices for logging or debugging
        self.set_topk_logit(top_k_indices)

        # Create gates (distribution of weights over top-k experts)
        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        # Load balance calculation (optional depending on your MoE setup)
        if self.top_k < self.tot_expert and self.training:
            load = self._prob_in_top_k(
                clean_logits, noisy_logits, noise_stddev, top_logits
            )
        else:
            load = self._gates_to_load(gates)

        # Calculate importance and load for loss computation
        load = load.sum(0) if self.training else load
        importance = gates.sum(0) if self.training else gates.sum(0)
        loss += self.cv_squared(importance) + self.cv_squared(load)

        # Store the computed loss
        self.set_loss(loss)

        # Return selected top-k indices and gates
        return top_k_indices, top_k_gates


class GShardGate(NaiveGate):
    def __init__(
        self,
        d_model,
        num_expert,
        world_size,
        topk=2,
        capacity=(1.2, 2.4),
        random_routing=True,
        gate_bias=True,
    ):
        assert topk == 2, "topk should be 2 in gshard"
        super().__init__(d_model, num_expert, world_size, top_k=2, gate_bias=gate_bias)
        self.capacity = capacity
        self.random_routing = random_routing
        self.topk_logits = []

    def reset_topk_logit(self):
        self.topk_logits = []

    def forward(self, x):

        naive_outs = super().forward(x, return_all_scores=True)
        topk_idx, topk_val, gate_score = naive_outs

        S = gate_score.shape[0]
        top1_idx = topk_idx.view((-1, self.top_k))[:, 0]
        c_e = (
            torch.scatter_add(
                torch.zeros(self.tot_expert, device=top1_idx.device),
                0,
                top1_idx,
                torch.ones_like(top1_idx, dtype=torch.float),
            )
            / S
        )
        m_e = torch.mean(F.softmax(gate_score, dim=1), dim=0)
        loss = torch.mean(c_e * m_e) * (self.num_expert**2)
        self.set_loss(loss)

        cap_rate = self.capacity[0 if self.training else 1]
        capacity = math.ceil(cap_rate * x.shape[0])
        capacity = capacity * self.top_k // (self.world_size * self.num_expert)
        capacity = (
            torch.ones(
                self.num_expert * self.world_size,
                dtype=torch.int32,
                device=topk_idx.device,
            )
            * capacity
        )
        topk_idx = fmoe_native.prune_gate_by_capacity(
            topk_idx, capacity, self.num_expert, self.world_size
        )

        if self.random_routing:
            rand_routing_prob = torch.rand(gate_score.size(0), device=x.device)
            mask = 2 * topk_val[:, 1] < rand_routing_prob
            topk_idx[:, 1].masked_fill_(mask, -1)
        self.topk_logits.append(topk_idx)

        return topk_idx, topk_val


def limit_by_capacity(topk_idx, num_expert, world_size, capacity):
    with torch.no_grad():
        capacity = (
            torch.ones(num_expert, dtype=torch.int32, device=topk_idx.device) * capacity
        )

        pos, lec, gec = count_by_gate(
            topk_idx, num_expert, world_size, require_pos=False
        )
        new_gec = fmoe_native.limit_by_capacity(gec, capacity, num_expert, world_size)
        if world_size > 1:
            new_lec = fmoe_native.expert_exchange(new_gec, num_expert, world_size)
        else:
            new_lec = new_gec

        topk_idx = fmoe_native.prune_gate_by_capacity(
            topk_idx, new_lec.to(torch.int32), num_expert, world_size
        )
    return new_lec, new_gec, topk_idx


class SwitchGate(NaiveGate):
    r"""
    A switch gate implementation
    """

    def __init__(
        self,
        d_model,
        num_expert,
        world_size,
        topk=1,
        switch_eps=0.1,
        capacity=(1.2, 2.4),
        gate_bias=True,
    ):
        assert topk == 1, "topk should be 1 in switch"
        super().__init__(d_model, num_expert, world_size, top_k=1, gate_bias=gate_bias)
        self.switch_eps = switch_eps
        self.capacity = capacity
        self.topk_logits = []

    def reset_topk_logit(self):
        self.topk_logits = []

    def forward(self, inp):
        r"""
        The switch firstly conduct softmax and then calculates the top-1
        """
        score = self.gate(inp)

        if self.training:
            # random uniform number from [1-eps, 1+eps]
            noise = torch.rand_like(score)
            noise = noise * 2 * self.switch_eps + 1.0 - self.switch_eps
            score += noise

        # fp32 softmax for numerical stability
        score = F.softmax(score.float(), dim=-1)

        top1_score, top1_idx = torch.topk(
            score, k=1, dim=-1, largest=True
        )  # [.. x top_k]
        top1_score = top1_score.to(dtype=inp.dtype)

        cap_rate = self.capacity[0 if self.training else 1]
        capacity = math.ceil(cap_rate * inp.shape[0] / self.num_expert)
        _new_lec, _new_gec, top1_idx = limit_by_capacity(
            top1_idx, self.num_expert, self.world_size, capacity
        )

        valid_idx = top1_idx[top1_idx > -1]
        fraction_expert = (
            torch.scatter_add(
                torch.zeros(self.tot_expert, device=valid_idx.device),
                0,
                valid_idx,
                torch.ones_like(valid_idx, dtype=torch.float),
            )
            / valid_idx.numel()
        )
        prob_expert = score.sum(dim=0) / valid_idx.numel()
        loss = (fraction_expert * prob_expert).sum() * self.tot_expert
        self.set_loss(loss)
        self.topk_logits.append(top1_idx)
        return top1_idx, top1_score


class DCGate(NaiveGate):
    def __init__(
        self,
        d_model,
        num_expert,
        world_size,
        topk=2,
        capacity=(1.2, 2.4),
        random_routing=True,
        gate_bias=True,
    ):
        assert topk == 2, "topk should be 2 in gshard"
        super().__init__(d_model, num_expert, world_size, top_k=2, gate_bias=gate_bias)
        self.capacity = capacity
        self.random_routing = random_routing
        self.topk_logits = []

    def reset_topk_logit(self):
        self.topk_logits = []

    def forward(self, x):
        naive_outs = super().forward(x, return_all_scores=True)
        topk_idx, topk_val, gate_score = naive_outs

        S = gate_score.shape[0]
        top1_idx = topk_idx.view((-1, self.top_k))[:, 0]
        c_e = (
            torch.scatter_add(
                torch.zeros(self.tot_expert, device=top1_idx.device),
                0,
                top1_idx,
                torch.ones_like(top1_idx, dtype=torch.float),
            )
            / S
        )
        m_e = torch.mean(F.softmax(gate_score, dim=1), dim=0)
        loss = torch.mean(c_e * m_e) * (self.num_expert**2)
        self.set_loss(loss)

        cap_rate = self.capacity[0 if self.training else 1]
        capacity = math.ceil(cap_rate * x.shape[0])
        _new_lec, _new_gec, topk_idx = limit_by_capacity(
            topk_idx, self.num_expert, self.world_size, capacity
        )

        if self.random_routing:
            rand_routing_prob = torch.rand(gate_score.size(0), device=x.device)
            mask = 2 * topk_val[:, 1] < rand_routing_prob
            topk_idx[:, 1].masked_fill_(mask, -1)
        self.topk_logits.append(topk_idx)

        return topk_idx, topk_val


class BaseGate(nn.Module):
    def __init__(self, num_expert, world_size):
        super().__init__()
        self.world_size = world_size
        self.num_expert = num_expert
        self.tot_expert = world_size * num_expert
        self.loss = None

    def forward(self, x):
        raise NotImplementedError("Base gate cannot be directly used for fwd")

    def set_loss(self, loss):
        self.loss = loss

    def get_loss(self, clear=True):
        loss = self.loss
        if clear:
            self.loss = None
        return loss

    @property
    def has_loss(self):
        return self.loss is not None


class NaiveGate(BaseGate):
    r"""
    A naive gate implementation that defines the standard behavior of the gate
    which determines which experts the tokens are going to.
    Both the indicies and the score, or confidence, are output to the parent
    module.
    The load-balance strategies are also designed to be implemented within the
    `Gate` module.
    """

    def __init__(self, d_model, num_expert, world_size, top_k=2, gate_bias=True):
        super().__init__(num_expert, world_size)
        self.gate = nn.Linear(d_model, self.tot_expert, bias=gate_bias)
        self.top_k = top_k
        self.topk_logits = []

    def reset_topk_logit(self):
        self.topk_logits = []

    def forward(self, inp, return_all_scores=False):
        r"""
        The naive implementation simply calculates the top-k of a linear layer's
        output.
        """
        gate = self.gate(inp)
        gate_top_k_val, gate_top_k_idx = torch.topk(
            gate, k=self.top_k, dim=-1, largest=True, sorted=False
        )  # [.. x top_k]
        self.topk_logits.append(gate_top_k_idx)
        gate_top_k_val = gate_top_k_val.view(-1, self.top_k)

        # (BxL) x 1 x top_k
        gate_score = F.softmax(gate_top_k_val, dim=-1)

        # dummy loss
        self.set_loss(torch.zeros(1, requires_grad=True).to(inp.device))

        if return_all_scores:
            return gate_top_k_idx, gate_score, gate
        return gate_top_k_idx, gate_score


# https://huggingface.co/deepseek-ai/deepseek-moe-16b-base/blob/main/modeling_deepseek.py
class DeepSeekGate(NaiveGate):
    def __init__(
        self,
        d_model,
        num_expert,
        world_size,
        top_k=2,
        n_routed_experts=2,
        aux_loss_alpha=0.001,
        seq_aux=True,
        norm_topk_prob=False,
        gate_bias=True,
    ):
        super().__init__(d_model, num_expert, world_size, top_k=1, gate_bias=gate_bias)
        self.top_k = top_k
        self.n_routed_experts = n_routed_experts

        self.alpha = aux_loss_alpha
        self.seq_aux = seq_aux

        # topk selection algorithm
        self.norm_topk_prob = norm_topk_prob

        self.topk_logits = []
        self.loss = None

    def forward(self, x):
        naive_outs = super().forward(x)
        topk_idx, scores = naive_outs

        ### norm gate to sum 1
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator

        bsz, seq_len, h = x.unsqueeze(1).shape  # torch.Size([32, 1, 128])

        ### expert-level computation auxiliary loss
        if self.training and self.alpha > 0.0:
            scores_for_aux = scores  # torch.Size([32, 2])
            aux_topk = self.top_k  # 2
            # always compute aux loss based on the naive greedy topk method
            topk_idx_for_aux_loss = topk_idx  # .view(-1)  torch.Size([64]
            if self.seq_aux:
                scores_for_seq_aux = scores_for_aux.view(
                    bsz, seq_len, -1
                )  # torch.Size([32, 1, 2])
                ce = torch.zeros(
                    bsz, self.n_routed_experts, device=x.device
                )  # torch.Size([32, 1]
                ce.scatter_add_(
                    0,
                    topk_idx_for_aux_loss,
                    torch.ones(bsz, seq_len * aux_topk).to(x.device),
                ).div_(seq_len * aux_topk / self.n_routed_experts)
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(
                    dim=1
                ).mean() * self.alpha
            else:
                mask_ce = F.one_hot(
                    topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts
                )
                ce = mask_ce.float().mean(0)
                Pi = scores_for_aux.mean(0)
                fi = ce * self.n_routed_experts
                aux_loss = (Pi * fi).sum() * self.alpha
        else:
            aux_loss = torch.zeros(1, requires_grad=True).to(x.device)

        self.set_loss(aux_loss)
        return topk_idx, scores
