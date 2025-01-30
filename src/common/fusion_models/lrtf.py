"""Implements LowRankTensorFusion. From MultiBench GitHub """

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable


class LowRankTensorFusion(nn.Module):
    """
    Implementation of Low-Rank Tensor Fusion.

    See https://github.com/Justin1904/Low-rank-Multimodal-Fusion for more information.
    """

    def __init__(
        self, input_dims=[256, 256], ir_dim=256, output_dim=3, rank=2, flatten=True
    ):
        """
        Initialize LowRankTensorFusion object.

        :param input_dims: list or tuple of integers indicating input dimensions of the modalities
        :param output_dim: output dimension
        :param rank: a hyperparameter of LRTF. See link above for details
        :param flatten: Boolean to dictate if output should be flattened or not. Default: True

        """
        super(LowRankTensorFusion, self).__init__()

        # dimensions are specified in the order of audio, video and text
        self.input_dims = input_dims
        self.ir_dim = ir_dim
        self.output_dim = output_dim
        self.rank = rank
        self.flatten = flatten

        # low-rank factors
        self.factors = []
        for input_dim in input_dims:
            factor = nn.Parameter(
                torch.Tensor(self.rank, input_dim + 1, self.ir_dim)
            ).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
            nn.init.xavier_normal_(factor)
            self.factors.append(factor)

        self.fusion_weights = nn.Parameter(torch.Tensor(1, self.rank)).to(
            torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        )
        self.fusion_bias = nn.Parameter(torch.Tensor(1, self.ir_dim)).to(
            torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        )
        # init the fusion weights
        nn.init.xavier_normal_(self.fusion_weights)
        self.fusion_bias.data.fill_(0)
        self.cls_head = nn.Sequential(
            nn.Linear(ir_dim, 128),  # Combined embedding size
            nn.ReLU(),
            nn.Linear(128, output_dim),  # Output the class logits
        ).cuda()

    def forward(self, modalities):
        """
        Forward Pass of Low-Rank TensorFusion.

        :param modalities: An iterable of modalities to combine.
        """
        batch_size = modalities[0].shape[0]
        # next we perform low-rank multimodal fusion
        # here is a more efficient implementation than the one the paper describes
        # basically swapping the order of summation and elementwise product
        fused_tensor = 1
        for modality, factor in zip(modalities, self.factors):
            ones = Variable(
                torch.ones(batch_size, 1).type(modality.dtype), requires_grad=False
            ).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
            if self.flatten:
                modality_withones = torch.cat(
                    (ones, torch.flatten(modality, start_dim=1)), dim=1
                )
            else:
                modality_withones = torch.cat((ones, modality), dim=1)
            modality_factor = torch.matmul(modality_withones, factor)
            fused_tensor = fused_tensor * modality_factor

        output = (
            torch.matmul(self.fusion_weights, fused_tensor.permute(1, 0, 2)).squeeze()
            + self.fusion_bias
        )
        output = output.view(-1, self.ir_dim)
        output = self.cls_head(output)
        return output
