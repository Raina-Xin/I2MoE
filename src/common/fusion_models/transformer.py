import os
import sys

sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))

from src.common.modules.transformer import *
from copy import deepcopy


class Transformer(nn.Module):
    """
    baseline fusion module:
    No interaction loss and no ensemble.
    """

    def __init__(
        self,
        num_modalities,
        num_patches,
        hidden_dim,
        output_dim,
        num_layers,
        num_layers_pred,
        num_experts,
        num_routers,
        top_k,
        num_heads=2,
        dropout=0.5,
        mlp_sparse=False,
        gate="GShardGate",
    ):
        super(Transformer, self).__init__()
        layers = []
        layers.append(
            TransformerEncoderLayer(
                num_experts,
                num_routers,
                hidden_dim,
                num_heads,
                dropout=dropout,
                hidden_times=2,
                mlp_sparse=mlp_sparse,
                top_k=top_k,
                gate=gate,
            )
        )
        for j in range(num_layers - 1):
            tmp = (mlp_sparse) & (j % 2 == 1)
            layers.append(
                TransformerEncoderLayer(
                    num_experts,
                    num_routers,
                    hidden_dim,
                    num_heads,
                    dropout=dropout,
                    hidden_times=2,
                    mlp_sparse=tmp,
                    top_k=top_k,
                    gate=gate,
                )
            )
        layers.append(Linear(hidden_dim * num_modalities, output_dim).cuda())

        self.network = nn.Sequential(*layers)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, np.sum([num_patches] * num_modalities), hidden_dim)
        )

    def forward(self, inputs, return_latent=False):
        chunk_size = [input.shape[1] for input in inputs]
        x = torch.cat(inputs, dim=1)
        if self.pos_embed != None:
            x += self.pos_embed

        x = torch.split(x, chunk_size, dim=1)

        for i in range(len(self.network) - 1):
            x = self.network[i](x)
        x = [item.mean(dim=1) for item in x]
        x = torch.cat(x, dim=1)
        if return_latent:
            latent = x.detach().clone()
        x = self.network[-1](x)
        if return_latent:
            return x, latent
        return x

    def gate_loss(self):
        g_loss = []
        for mn, mm in self.named_modules():
            # print(mn)
            if hasattr(mm, "all_gates"):
                for i in range(len(mm.all_gates)):
                    i_loss = mm.all_gates[f"{i}"].get_loss()
                    if i_loss is None:
                        print(
                            f"[WARN] The gate loss if {mn}, modality: {i} is emtpy, check weather call <get_loss> twice."
                        )
                    else:
                        g_loss.append(i_loss)
        return sum(g_loss)
