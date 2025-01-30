# Sparsely-Gated Mixture-of-Experts Layers.
# See "Outrageously Large Neural Networks"
# https://arxiv.org/abs/1701.06538
#
# Author: David Rau
#
# The code is based on the TensorFlow implementation:
# https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/utils/expert_utils.py
import os
import sys

sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))

from src.common.modules.hme import ACT2FN
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import numpy as np


class MoEConfig:
    def __init__(
        self,
        num_experts,
        moe_input_size,
        moe_hidden_size,
        moe_output_size,
        router_type,
        gating="softmax",
        num_modalities=1,
        vocab_size=100,
        num_tasks=2,
        top_k=4,
        disjoint_top_k=2,
        noisy_gating=True,
        max_position_embeddings=512,
        type_vocab_size=2,
        modality_type_vocab_size=2,
        hidden_dim=768,
        num_layers=8,
        dropout=0.1,
        hidden_dropout_prob=0.1,
        pre_lnorm=True,
        n_heads=8,
        image_size=224,
        patch_size=16,
        num_channels=3,
        max_image_length=-1,
        layer_norm_eps=1e-5,
        expert_activation=nn.ReLU(),
        task_activation=nn.ReLU(),
        output_activation=nn.Sigmoid(),
        hidden_act="gelu",
        output_attentions=False,
        output_hidden_states=False,
        use_return_dict=True,
        is_decoder=False,
    ):
        # Input
        self.vocab_size = vocab_size
        self.hidden_size = hidden_dim
        self.type_vocab_size = type_vocab_size
        self.modality_type_vocab_size = modality_type_vocab_size

        # MoE
        self.num_experts = num_experts
        self.num_tasks = num_tasks
        self.top_k = top_k
        self.disjoint_top_k = disjoint_top_k
        self.noisy_gating = noisy_gating
        self.moe_input_size = moe_input_size
        self.moe_hidden_size = moe_hidden_size
        self.moe_output_size = moe_output_size
        self.router_type = router_type
        self.num_modalities = num_modalities
        self.gating = gating

        # image
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.max_image_length = max_image_length

        # Transformer
        self.max_position_embeddings = max_position_embeddings
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.hidden_dropout_prob = hidden_dropout_prob
        self.pre_lnorm = pre_lnorm
        self.n_heads = n_heads
        self.d_heads = int(hidden_dim / n_heads)

        # LayerNorm
        self.layer_norm_eps = layer_norm_eps

        # Activations
        self.expert_activation = expert_activation
        self.task_activation = task_activation
        self.output_activation = output_activation
        self.hidden_act = hidden_act

        # Other
        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states
        self.use_return_dict = use_return_dict
        self.is_decoder = is_decoder


class SparseDispatcher(object):
    """Helper for implementing a mixture of experts.
    The purpose of this class is to create input minibatches for the
    experts and to combine the results of the experts to form a unified
    output tensor.
    There are two functions:
    dispatch - take an input Tensor and create input Tensors for each expert.
    combine - take output Tensors from each expert and form a combined output
      Tensor.  Outputs from different experts for the same batch element are
      summed together, weighted by the provided "gates".
    The class is initialized with a "gates" Tensor, which specifies which
    batch elements go to which experts, and the weights to use when combining
    the outputs.  Batch element b is sent to expert e iff gates[b, e] != 0.
    The inputs and outputs are all two-dimensional [batch, depth].
    Caller is responsible for collapsing additional dimensions prior to
    calling this class and reshaping the output to the original shape.
    See common_layers.reshape_like().
    Example use:
    gates: a float32 `Tensor` with shape `[batch_size, num_experts]`
    inputs: a float32 `Tensor` with shape `[batch_size, input_size]`
    experts: a list of length `num_experts` containing sub-networks.
    dispatcher = SparseDispatcher(num_experts, gates)
    expert_inputs = dispatcher.dispatch(inputs)
    expert_outputs = [experts[i](expert_inputs[i]) for i in range(num_experts)]
    outputs = dispatcher.combine(expert_outputs)
    The preceding code sets the output for a particular example b to:
    output[b] = Sum_i(gates[b, i] * experts[i](inputs[b]))
    This class takes advantage of sparsity in the gate matrix by including in the
    `Tensor`s for expert i only the batch elements for which `gates[b, i] > 0`.
    """

    def __init__(self, num_experts, gates, router_type):
        """Create a SparseDispatcher."""
        # each forward pass initialize the sparse dispatcher once
        self._gates = gates
        self._num_experts = num_experts
        self._router_type = router_type
        # sort experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        # drop indices
        _, self._expert_index = sorted_experts.split(1, dim=1)
        # get according batch index for each expert
        # _batch_index: sample index inside a batch that is assigned to particular expert, concatenated
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
        # calculate num samples that each expert gets
        self._part_sizes = (gates > 0).sum(0).tolist()
        # expand gates to match with self._batch_index
        # from e.g., [64, 4] -> [256, 4], collection of samples assigned to each gate
        gates_exp = gates[self._batch_index.flatten()]
        # difference between torch.nonzero(gates) and self._nonzero_gates
        # the first one is index and the second one is actural weights!
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp):
        """Create one input Tensor for each expert.
        The `Tensor` for a expert `i` contains the slices of `inp` corresponding
        to the batch elements `b` where `gates[b, i] > 0`.
        Args:
          inp: a `Tensor` of shape "[batch_size, <extra_input_dims>]`
        Returns:
          a list of `num_experts` `Tensor`s with shapes
            `[expert_batch_size_i, <extra_input_dims>]`.
        """

        # assigns samples to experts whose gate is nonzero

        # expand according to batch index so we can just split by _part_sizes
        if isinstance(inp, list) and self._router_type == "joint":
            inp = torch.concat(inp, dim=1)
        inp_exp = inp[self._batch_index].squeeze(1)
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, multiply_by_gates=True):
        """Sum together the expert output, weighted by the gates.
        The slice corresponding to a particular batch element `b` is computed
        as the sum over all experts `i` of the expert output, weighted by the
        corresponding gate values.  If `multiply_by_gates` is set to False, the
        gate values are ignored.
        Args:
          expert_out: a list of `num_experts` `Tensor`s, each with shape
            `[expert_batch_size_i, <extra_output_dims>]`.
          multiply_by_gates: a boolean
        Returns:
          a `Tensor` with shape `[batch_size, <extra_output_dims>]`.
        """
        # apply exp to expert outputs, so we are not longer in log space
        # concat all sample outputs from each expert
        stitched = torch.cat(expert_out, 0).exp()
        if multiply_by_gates:
            stitched = stitched.mul(self._nonzero_gates)
        zeros = torch.zeros(
            self._gates.size(0),
            expert_out[-1].size(1),
            requires_grad=True,
            device=stitched.device,
        )
        # combine samples that have been processed by the same k experts
        # this is the weighted combination step
        combined = zeros.index_add(0, self._batch_index, stitched.float())
        # add eps to all zero values in order to avoid nans when going back to log space
        combined[combined == 0] = np.finfo(float).eps
        # back to log space
        return combined.log()

    def expert_to_gates(self):
        """Gate values corresponding to the examples in the per-expert `Tensor`s.
        Returns:
          a list of `num_experts` one-dimensional `Tensor`s with type `tf.float32`
              and shapes `[expert_batch_size_i]`
        """
        # split nonzero gates for each expert
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)


class MLP(nn.Module):
    def __init__(
        self, config: MoEConfig, input_size: int, output_size: int, hidden_size: int
    ):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(config.dropout)
        self.activation = ACT2FN[config.hidden_act]
        self.log_soft = nn.LogSoftmax(1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.log_soft(out)
        return out


class MoE(nn.Module):
    """Call a Sparsely gated mixture of experts layer with 1-layer Feed-Forward networks as experts.
    Args:
    input_size: integer - size of the input
    output_size: integer - size of the input
    num_experts: an integer - number of experts
    hidden_size: an integer - hidden size of the experts
    noisy_gating: a boolean
    k: an integer - how many experts to use for each batch element
    """

    def __init__(self, config: MoEConfig):
        super(MoE, self).__init__()
        self.noisy_gating = config.noisy_gating
        self.num_experts = config.num_experts
        self.output_size = config.moe_output_size
        self.input_size = config.moe_input_size
        self.hidden_size = config.moe_hidden_size
        self.k = config.top_k
        self.disjoint_k = config.disjoint_top_k
        self.router_type = config.router_type
        self.num_modalities = config.num_modalities
        self.gating = config.gating

        # instantiate experts
        if self.router_type == "disjoint":
            self.w_gate = [
                nn.Parameter(
                    torch.zeros(
                        self.input_size // self.num_modalities,
                        self.num_experts // self.num_modalities,
                    ),
                    requires_grad=True,
                )
                for _ in range(self.num_modalities)
            ]
            self.w_noise = [
                nn.Parameter(
                    torch.zeros(
                        self.input_size // self.num_modalities,
                        self.num_experts // self.num_modalities,
                    ),
                    requires_grad=True,
                )
                for _ in range(self.num_modalities)
            ]
        elif self.router_type == "permod":
            self.w_gate = [
                nn.Parameter(
                    torch.zeros(
                        self.input_size // self.num_modalities, self.num_experts
                    ),
                    requires_grad=True,
                )
                for _ in range(self.num_modalities)
            ]
            self.w_noise = [
                nn.Parameter(
                    torch.zeros(
                        self.input_size // self.num_modalities, self.num_experts
                    ),
                    requires_grad=True,
                )
                for _ in range(self.num_modalities)
            ]
        else:
            self.w_gate = nn.Parameter(
                torch.zeros(self.input_size, self.num_experts), requires_grad=True
            )
            self.w_noise = nn.Parameter(
                torch.zeros(self.input_size, self.num_experts), requires_grad=True
            )
        if self.router_type == "disjoint":
            self.experts = nn.ModuleList(
                nn.ModuleList(
                    [
                        MLP(
                            config,
                            self.input_size // self.num_modalities,
                            self.output_size,
                            self.hidden_size,
                        )
                        for _ in range(self.num_experts // self.num_modalities)
                    ]
                )
                for _ in range(self.num_modalities)
            )
        elif self.router_type == "permod":
            self.experts = nn.ModuleList(
                [
                    MLP(
                        config,
                        self.input_size // self.num_modalities,
                        self.output_size,
                        self.hidden_size,
                    )
                    for _ in range(self.num_experts)
                ]
            )
        else:
            self.experts = nn.ModuleList(
                [
                    MLP(config, self.input_size, self.output_size, self.hidden_size)
                    for _ in range(self.num_experts)
                ]
            )

        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))

        assert self.k <= self.num_experts

    def cv_squared(self, x):
        """The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability.
        Returns 0 for an empty Tensor.
        Args:
        x: a `Tensor`.
        Returns:
        a `Scalar`.
        """
        eps = 1e-10
        # if only num_experts = 1
        if x.shape[0] == 1:
            return torch.Tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def _gates_to_load(self, gates):
        """Compute the true load per expert, given the gates.
        The load is the number of examples for which the corresponding gate is >0.
        Args:
        gates: a `Tensor` of shape [batch_size, n]
        Returns:
        a float32 `Tensor` of shape [n]
        """
        return (gates > 0).sum(0)

    def _prob_in_top_k(
        self, clean_values, noisy_values, noise_stddev, noisy_top_values
    ):
        """Helper function to NoisyTopKGating.
        Computes the probability that value is in top k, given different random noise.
        This gives us a way of backpropagating from a loss that balances the number
        of times each expert is in the top k experts per example.
        In the case of no noise, pass in None for noise_stddev, and the result will
        not be differentiable.
        Args:
        clean_values: a `Tensor` of shape [batch, n].
        noisy_values: a `Tensor` of shape [batch, n].  Equal to clean values plus
          normally distributed noise with standard deviation noise_stddev.
        noise_stddev: a `Tensor` of shape [batch, n], or None
        noisy_top_values: a `Tensor` of shape [batch, m].
           "values" Output of tf.top_k(noisy_top_values, m).  m >= k+1
        Returns:
        a `Tensor` of shape [batch, n].
        """

        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        if self.router_type == "disjoint":
            threshold_positions_if_in = (
                torch.arange(batch, device=clean_values.device) * m + self.disjoint_k
            )
        else:
            threshold_positions_if_in = (
                torch.arange(batch, device=clean_values.device) * m + self.k
            )
        threshold_if_in = torch.unsqueeze(
            torch.gather(top_values_flat, 0, threshold_positions_if_in), 1
        )
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(
            torch.gather(top_values_flat, 0, threshold_positions_if_out), 1
        )
        # is each value currently in the top k.
        normal = Normal(self.mean, self.std)

        prob_if_in = normal.cdf((clean_values - threshold_if_in) / noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out) / noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def _get_logits(self, x, train, noise_epsilon, idx=None):
        if idx is not None:
            w_gate = self.w_gate[idx].to(x.device)
            w_noise = self.w_noise[idx].to(x.device)
        else:
            w_gate = self.w_gate
            w_noise = self.w_noise
        if self.gating == "softmax":
            clean_logits = x @ w_gate
        elif self.gating == "laplace":
            clean_logits = -torch.cdist(x, torch.t(w_gate))
        elif self.gating == "gaussian":
            clean_logits = -torch.pow(torch.cdist(x, torch.t(w_gate)), 2)

        if self.noisy_gating:
            raw_noise_stddev = x @ w_noise
            noise_stddev = (self.softplus(raw_noise_stddev) + noise_epsilon) * train
            noisy_logits = clean_logits + (
                torch.randn_like(clean_logits) * noise_stddev
            )
            logits = noisy_logits
        else:
            logits = clean_logits
        return logits, clean_logits, noisy_logits, noise_stddev

    def _top_k_gating(self, logits, clean_logits, noisy_logits, noise_stddev, k):
        top_logits, top_indices = logits.topk(min(k + 1, self.num_experts), dim=1)
        top_k_logits = top_logits[:, :k]
        top_k_indices = top_indices[:, :k]
        if self.gating == "softmax":
            top_k_gates = self.softmax(top_k_logits)
        elif self.gating == "laplace" or self.gating == "gaussian":
            top_k_gates = torch.exp(
                top_k_logits - torch.logsumexp(top_k_logits, dim=1, keepdim=True)
            )

        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        if self.noisy_gating and k < self.num_experts:
            load = (
                self._prob_in_top_k(
                    clean_logits, noisy_logits, noise_stddev, top_logits
                )
            ).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2, modalities=None):
        """Multimodal noisy top-k gating.
        See paper: https://arxiv.org/abs/1701.06538.
        Args:
          x: input Tensor with shape [batch_size, input_size]
          train: a boolean - we only add noise at training time.
          noise_epsilon: a float
        Returns:
          gates: a Tensor with shape [batch_size, num_experts]
          load: a Tensor with shape [num_experts]
        """

        if self.router_type == "joint":
            if isinstance(x, list):
                embeddings = torch.concat(x, dim=1)
            else:
                embeddings = x
            all_logits = self._get_logits(embeddings, train, noise_epsilon)
            logits, clean_logits, noisy_logits, noise_stddev = (
                all_logits[0],
                all_logits[1],
                all_logits[2],
                all_logits[3],
            )
            gates, load = self._top_k_gating(
                logits, clean_logits, noisy_logits, noise_stddev, self.k
            )
            return gates, load
        else:
            all_gates, all_loads = [], []
            for i in range(self.num_modalities):
                all_logits = self._get_logits(x[i], train, noise_epsilon, idx=i)
                logits, clean_logits, noisy_logits, noise_stddev = (
                    all_logits[0],
                    all_logits[1],
                    all_logits[2],
                    all_logits[3],
                )
                if self.router_type == "permod":
                    gates, load = self._top_k_gating(
                        logits, clean_logits, noisy_logits, noise_stddev, self.k
                    )
                else:
                    gates, load = self._top_k_gating(
                        logits,
                        clean_logits,
                        noisy_logits,
                        noise_stddev,
                        self.disjoint_k,
                    )
                all_gates.append(gates)
                all_loads.append(load)
            return all_gates, all_loads

    def forward(self, x, train=True, loss_coef=1e-2, modalities=None):
        """Args:
        x: tensor shape [batch_size, input_size]
        gating: type of gating function
        train: a boolean scalar.
        loss_coef: a scalar - multiplier on load-balancing losses
        Returns:
        y: a tensor with shape [batch_size, output_size].
        extra_training_loss: a scalar.  This should be added into the overall
        training loss of the model.  The backpropagation of this loss
        encourages all experts to be approximately equally used across a batch.
        """
        gates, load = self.noisy_top_k_gating(x, train, modalities=modalities)
        # calculate importance loss
        if isinstance(gates, list):
            loss, y, sub_experts = 0, 0, self.num_experts // self.num_modalities
            for g, l in zip(gates, load):
                loss += self.cv_squared(g.sum(0)) + self.cv_squared(l)
            loss *= loss_coef
            for j, g in enumerate(gates):
                dispatcher = SparseDispatcher(self.num_experts, g, self.router_type)
                expert_inputs = dispatcher.dispatch(x[j])
                if self.router_type == "permod":
                    expert_outputs = [
                        self.experts[i](expert_inputs[i])
                        for i in range(self.num_experts)
                    ]
                elif self.router_type == "disjoint":
                    expert_outputs = [
                        self.experts[j][i](expert_inputs[i]) for i in range(sub_experts)
                    ]
                y += dispatcher.combine(expert_outputs)
        else:
            loss = self.cv_squared(gates.sum(0)) + self.cv_squared(load)
            loss *= loss_coef
            dispatcher = SparseDispatcher(self.num_experts, gates, self.router_type)
            expert_inputs = dispatcher.dispatch(x)
            # gates = dispatcher.expert_to_gates() # how is this line be used?
            expert_outputs = [
                self.experts[i](expert_inputs[i]) for i in range(self.num_experts)
            ]
            y = dispatcher.combine(expert_outputs)
        return y, loss
