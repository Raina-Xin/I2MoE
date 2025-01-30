import os
import sys

sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))

import torch
import numpy as np
import argparse
from pathlib import Path

import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning, message="os.fork()")

from src.common.fusion_models.transformer import Transformer
from src.ablation.train_simple_weighted_average import (
    train_and_evaluate_simple_weighted_average,
)
from src.common.utils import setup_logger, str2bool


# Parse input arguments
def parse_args():
    parser = argparse.ArgumentParser(description="simple_weighted_average-Transformer")
    parser.add_argument("--data", type=str, default="adni")
    parser.add_argument(
        "--modality", type=str, default="IGCB"
    )  # I G C B for ADNI, L N C for MIMIC
    parser.add_argument("--initial_filling", type=str, default="mean")  # None mean
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_runs", type=int, default=1)
    parser.add_argument(
        "--num_workers", type=int, default=4
    )  # Number of workers for DataLoader
    parser.add_argument(
        "--pin_memory", type=str2bool, default=True
    )  # Pin memory in DataLoader
    parser.add_argument(
        "--use_common_ids", type=str2bool, default=True
    )  # Use common ids across modalities
    parser.add_argument(
        "--save", type=str2bool, default=True
    )  # Use common ids across modalities
    parser.add_argument(
        "--debug", type=str2bool, default=False
    )  # Use common ids across modalities

    parser.add_argument("--train_epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--temperature_rw", type=int, default=1
    )  # Temperature of the reweighting model
    parser.add_argument(
        "--hidden_dim_rw", type=int, default=256
    )  # Hidden dimension of the reweighting model
    parser.add_argument(
        "--num_layer_rw", type=int, default=1
    )  # Number of layers of the reweighting model
    parser.add_argument("--interaction_loss_weight", type=float, default=1e-2)

    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument(
        "--num_layers_enc", type=int, default=1
    )  # Number of MLP layers for encoders
    parser.add_argument(
        "--num_layers_fus", type=int, default=1
    )  # Number of MLP layers for fusion model
    parser.add_argument(
        "--num_layers_pred", type=int, default=1
    )  # Number of MLP layers for fusion model
    parser.add_argument("--num_heads", type=int, default=4)  # Number of heads
    parser.add_argument(
        "--patch", type=str2bool, default=True
    )  # Use common ids across modalities
    parser.add_argument(
        "--num_patches", type=int, default=16
    )  # Use common ids across modalities

    parser.add_argument(
        "--fusion_sparse", type=str2bool, default=False
    )  # Whether to include SMoE in Fusion Layer
    parser.add_argument("--gate", type=str, default="None")
    parser.add_argument("--num_experts", type=int, default=16)  # Number of Experts
    parser.add_argument("--num_routers", type=int, default=1)  # Number of Routers
    parser.add_argument("--top_k", type=int, default=2)  # Number of Routers
    parser.add_argument("--dropout", type=float, default=0.5)  # Number of Routers
    parser.add_argument("--gate_loss_weight", type=float, default=1e-2)

    return parser.parse_known_args()


def main():
    args, _ = parse_args()
    logger = setup_logger(
        f"./logs/simple_weighted_average/transformer/{args.data}",
        f"{args.data}",
        f"{args.modality}.txt",
    )
    seeds = np.arange(args.n_runs)  # [0, 1, 2]
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    log_summary = "======================================================================================\n"

    model_kwargs = {
        "model": "simple_weighted_average-Transformer",
        "lr": args.lr,
        "temperature_rw": args.temperature_rw,
        "hidden_dim_rw": args.hidden_dim_rw,
        "num_layer_rw": args.num_layer_rw,
        "interaction_loss_weight": args.interaction_loss_weight,
        "modality": args.modality,
        "data": args.data,
        "gate_loss_weight": args.gate_loss_weight,
        "interaction_loss_weight": args.interaction_loss_weight,
        "train_epochs": args.train_epochs,
        "num_experts": args.num_experts,
        "num_layers_enc": args.num_layers_enc,
        "num_layers_fus": args.num_layers_fus,
        "num_layers_pred": args.num_layers_pred,
        "num_heads": args.num_heads,
        "batch_size": args.batch_size,
        "hidden_dim": args.hidden_dim,
        "num_patches": args.num_patches,
    }

    log_summary += f"Model configuration: {model_kwargs}\n"

    print("Modality:", args.modality)

    data_to_nlabels = {"adni": 3, "mimic": 2, "mmimdb": 23, "enrico": 20, "mosi": 2}
    n_labels = data_to_nlabels[args.data]
    num_modalities = num_modality = len(args.modality)

    val_accs = []
    val_f1s = []
    val_aucs = []
    test_accs = []
    test_f1s = []
    test_aucs = []

    if len(seeds) == 1:
        fusion_model = Transformer(
            num_modalities,
            args.num_patches,
            args.hidden_dim,
            n_labels,
            args.num_layers_fus,
            args.num_layers_pred,
            args.num_experts,
            args.num_routers,
            args.top_k,
            args.num_heads,
            args.dropout,
            args.fusion_sparse,
            args.gate,
        ).to(device)

        val_acc, val_f1, val_auc, test_acc, test_f1, test_auc = (
            train_and_evaluate_simple_weighted_average(
                args, args.seed, fusion_model, "transformer"
            )
        )
        val_accs.append(val_acc)
        val_f1s.append(val_f1)
        val_aucs.append(val_auc)
        test_accs.append(test_acc)
        test_f1s.append(test_f1)
        test_aucs.append(test_auc)
    else:
        for seed in seeds:
            fusion_model = Transformer(
                num_modalities,
                args.num_patches,
                args.hidden_dim,
                n_labels,
                args.num_layers_fus,
                args.num_layers_pred,
                args.num_experts,
                args.num_routers,
                args.top_k,
                args.num_heads,
                args.dropout,
                args.fusion_sparse,
                args.gate,
            ).to(device)

            val_acc, val_f1, val_auc, test_acc, test_f1, test_auc = (
                train_and_evaluate_simple_weighted_average(
                    args, seed, fusion_model, "transformer"
                )
            )

            val_accs.append(val_acc)
            val_f1s.append(val_f1)
            val_aucs.append(val_auc)
            test_accs.append(test_acc)
            test_f1s.append(test_f1)
            test_aucs.append(test_auc)

    val_avg_acc = np.mean(val_accs) * 100
    val_std_acc = np.std(val_accs) * 100
    val_avg_f1 = np.mean(val_f1s) * 100
    val_std_f1 = np.std(val_f1s) * 100
    val_avg_auc = np.mean(val_aucs) * 100
    val_std_auc = np.std(val_aucs) * 100

    test_avg_acc = np.mean(test_accs) * 100
    test_std_acc = np.std(test_accs) * 100
    test_avg_f1 = np.mean(test_f1s) * 100
    test_std_f1 = np.std(test_f1s) * 100
    test_avg_auc = np.mean(test_aucs) * 100
    test_std_auc = np.std(test_aucs) * 100

    log_summary += f"[Val] Average Accuracy: {val_avg_acc:.2f} ± {val_std_acc:.2f} "
    log_summary += f"[Val] Average F1 Score: {val_avg_f1:.2f} ± {val_std_f1:.2f} "
    log_summary += f"[Val] Average AUC: {val_avg_auc:.2f} ± {val_std_auc:.2f} / "
    log_summary += f"[Test] Average Accuracy: {test_avg_acc:.2f} ± {test_std_acc:.2f} "
    log_summary += f"[Test] Average F1 Score: {test_avg_f1:.2f} ± {test_std_f1:.2f} "
    log_summary += f"[Test] Average AUC: {test_avg_auc:.2f} ± {test_std_auc:.2f} "

    print(model_kwargs)
    print(
        f"[Val] Average Accuracy: {val_avg_acc:.2f} ± {val_std_acc:.2f} / Average F1 Score: {val_avg_f1:.2f} ± {val_std_f1:.2f} / Average AUC: {val_avg_auc:.2f} ± {val_std_auc:.2f}"
    )
    print(
        f"[Test] Average Accuracy: {test_avg_acc:.2f} ± {test_std_acc:.2f} / Average F1 Score: {test_avg_f1:.2f} ± {test_std_f1:.2f} / Average AUC: {test_avg_auc:.2f} ± {test_std_auc:.2f}"
    )

    logger.info(log_summary)


if __name__ == "__main__":
    main()
