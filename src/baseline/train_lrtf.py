import sys
import os
from pathlib import Path

sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
import torch
import numpy as np
import argparse
import random
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    mean_absolute_error,
)
from copy import deepcopy
from tqdm import trange
from src.common.fusion_models.lrtf import LowRankTensorFusion
from src.common.datasets.adni import load_and_preprocess_data_adni
from src.common.datasets.mimic import load_and_preprocess_data_mimic
from src.common.datasets.enrico import load_and_preprocess_data_enrico
from src.common.datasets.mmimdb import load_and_preprocess_data_mmimdb
from src.common.datasets.mosi import (
    load_and_preprocess_data_mosi,
    load_and_preprocess_data_mosi_regression,
)
from src.common.datasets.MultiModalDataset import create_loaders
from src.common.utils import (
    setup_logger,
    str2bool,
    seed_everything,
    plot_total_loss_curves,
)
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning, message="os.fork()")
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


# Parse input arguments
def parse_args():
    parser = argparse.ArgumentParser(description="lrtf")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--data", type=str, default="adni")
    parser.add_argument(
        "--modality", type=str, default="IG"
    )  # I G C B for ADNI, L N C for MIMIC
    parser.add_argument("--ir_dim", type=int, default=64)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--initial_filling", type=str, default="mean")
    parser.add_argument("--patch", type=str2bool, default=False)
    parser.add_argument("--train_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden_dim", type=int, default=64)
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
        "--num_workers", type=int, default=4
    )  # Number of workers for DataLoader
    parser.add_argument(
        "--pin_memory", type=str2bool, default=True
    )  # Pin memory in DataLoader
    parser.add_argument(
        "--use_common_ids", type=str2bool, default=True
    )  # Use common ids across modalities
    parser.add_argument("--dropout", type=float, default=0.5)  # Number of Routers
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_runs", type=int, default=3)
    parser.add_argument(
        "--save", type=str2bool, default=False
    )  # Use common ids across modalities
    parser.add_argument("--debug", type=str2bool, default=False)
    return parser.parse_known_args()


def train_and_evaluate(args, seed):
    seed_everything(seed)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    num_modalities = len(args.modality)

    if args.data == "adni":
        (
            data_dict,
            encoder_dict,
            labels,
            train_ids,
            valid_ids,
            test_ids,
            n_labels,
            input_dims,
            transforms,
            masks,
            observed_idx_arr,
            mc_idx_dict,
            num_mc_dict,
        ) = load_and_preprocess_data_adni(args)

    elif args.data == "mimic":
        (
            data_dict,
            encoder_dict,
            labels,
            train_ids,
            valid_ids,
            test_ids,
            n_labels,
            input_dims,
            transforms,
            masks,
            observed_idx_arr,
            mc_idx_dict,
            num_mc_dict,
        ) = load_and_preprocess_data_mimic(args)

    elif args.data == "mosi":
        (
            data_dict,
            encoder_dict,
            labels,
            train_ids,
            valid_ids,
            test_ids,
            n_labels,
            input_dims,
            transforms,
            masks,
            observed_idx_arr,
            mc_idx_dict,
            num_mc_dict,
        ) = load_and_preprocess_data_mosi(args)

    elif args.data == "enrico":
        (
            data_dict,
            encoder_dict,
            labels,
            train_ids,
            valid_ids,
            test_ids,
            n_labels,
            input_dims,
            transforms,
            masks,
            observed_idx_arr,
            mc_idx_dict,
            num_mc_dict,
        ) = load_and_preprocess_data_enrico(args)
        n_labels = 20
    elif args.data == "mosi_regression":
        (
            data_dict,
            encoder_dict,
            labels,
            train_ids,
            valid_ids,
            test_ids,
            n_labels,
            input_dims,
            transforms,
            masks,
            observed_idx_arr,
            mc_idx_dict,
            num_mc_dict,
        ) = load_and_preprocess_data_mosi_regression(args)
        n_labels = 1

    elif args.data == "mmimdb":
        (
            data_dict,
            encoder_dict,
            labels,
            train_ids,
            valid_ids,
            test_ids,
            n_labels,
            input_dims,
            transforms,
            masks,
            observed_idx_arr,
            mc_idx_dict,
            num_mc_dict,
        ) = load_and_preprocess_data_mmimdb(args)

    fusion_model = LowRankTensorFusion(
        input_dims=[args.hidden_dim] * len(args.modality),
        ir_dim=args.ir_dim,
        output_dim=n_labels,
        rank=args.rank,
        flatten=True,
    ).to(device)

    train_loader, val_loader, test_loader = create_loaders(
        data_dict,
        observed_idx_arr,
        labels,
        train_ids,
        valid_ids,
        test_ids,
        args.batch_size,
        args.num_workers,
        args.pin_memory,
        input_dims,
        transforms,
        masks,
        args.use_common_ids,
        dataset=args.data,
    )

    params = list(fusion_model.parameters()) + [
        param for encoder in encoder_dict.values() for param in encoder.parameters()
    ]
    optimizer = torch.optim.Adam(params, lr=args.lr)
    if args.data == "adni" or args.data == "enrico" or args.data == "mosi":
        criterion = torch.nn.CrossEntropyLoss()
    elif args.data == "mimic":
        criterion = torch.nn.CrossEntropyLoss(torch.tensor([0.25, 0.75]).to(device))
    elif args.data == "mmimdb":
        criterion = torch.nn.BCEWithLogitsLoss()
    elif args.data == "mosi_regression":
        criterion = torch.nn.SmoothL1Loss()  # Regression

    if args.data == "mosi_regression":
        best_val_loss = 100000
    elif args.data == "mmimdb":
        best_val_f1 = 0
    else:
        best_val_acc = 0.0
    if args.data == "adni":
        modality_dict = {"image": 0, "genomic": 1, "clinical": 2, "biospecimen": 3}
        char_to_modality = {
            "I": "image",
            "G": "genomic",
            "C": "clinical",
            "B": "biospecimen",
        }

    elif args.data == "mimic":
        modality_dict = {"lab": 0, "note": 1, "code": 2}
    elif args.data in ["mosi", "mosi_regression", "sarcasm", "humor"]:
        modality_dict = {"vision": 0, "audio": 1, "text": 2}
        char_to_modality = {"V": "vision", "A": "audio", "T": "text"}

    elif args.data == "enrico":
        modality_dict = {"screenshot": 0, "wireframe": 1}
        char_to_modality = {"S": "screenshot", "W": "wireframe"}

    elif args.data == "mmimdb":
        modality_dict = {"language": 0, "img": 1}
        char_to_modality = {"L": "language", "I": "img"}

    for epoch in trange(args.train_epochs):

        for encoder in encoder_dict.values():
            encoder.train()

        task_losses = []

        for batch_samples, batch_labels, batch_mcs, batch_observed in train_loader:
            batch_samples = {
                k: v.to(device, non_blocking=True) for k, v in batch_samples.items()
            }
            batch_labels = batch_labels.to(device, non_blocking=True)
            batch_mcs = batch_mcs.to(device, non_blocking=True)
            batch_observed = batch_observed.to(device, non_blocking=True)
            optimizer.zero_grad()
            fusion_input = []
            for i, (modality, samples) in enumerate(batch_samples.items()):
                encoded_samples = encoder_dict[modality](samples).to(device)
                fusion_input.append(encoded_samples)

            outputs = fusion_model(fusion_input)
            if args.data == "mosi_regression":
                task_loss = criterion(outputs, batch_labels.unsqueeze(1))
            else:
                task_loss = criterion(outputs, batch_labels)
            task_losses.append(task_loss.item())

            # print(loss)
            task_loss.backward()
            optimizer.step()

        fusion_model.eval()

        for encoder in encoder_dict.values():
            encoder.eval()

        all_preds = []
        all_labels = []
        all_probs = []
        val_losses = []

        # Need to update!!
        with torch.no_grad():
            for batch_samples, batch_labels, batch_mcs, batch_observed in val_loader:
                batch_samples = {
                    k: v.to(device, non_blocking=True) for k, v in batch_samples.items()
                }
                batch_labels = batch_labels.to(device, non_blocking=True)
                batch_mcs = batch_mcs.to(device, non_blocking=True)
                batch_observed = batch_observed.to(device, non_blocking=True)
                optimizer.zero_grad()

                fusion_input = []
                for i, (modality, samples) in enumerate(batch_samples.items()):
                    encoded_samples = encoder_dict[modality](samples).to(device)
                    fusion_input.append(encoded_samples)

                outputs = fusion_model(fusion_input)

                if args.data == "mosi_regression":
                    # if False:
                    val_loss = criterion(outputs, batch_labels.unsqueeze(1))
                    val_losses.append(val_loss.item())
                    all_preds.extend(outputs.cpu().numpy())
                    all_labels.extend(batch_labels.cpu().numpy())

                else:
                    if args.data == "mmimdb":
                        val_loss = criterion(outputs, batch_labels.float())
                    else:
                        val_loss = criterion(outputs, batch_labels)
                    val_losses.append(val_loss.item())
                    if args.data == "mmimdb":
                        preds = torch.sigmoid(outputs).round()
                    else:
                        _, preds = torch.max(outputs, 1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(batch_labels.cpu().numpy())
                    if args.data in ["mimic", "mosi", "sarcasm", "humor"]:
                        all_probs.extend(
                            torch.nn.functional.softmax(outputs, dim=1)[:, 1]
                            .cpu()
                            .numpy()
                        )
                    else:
                        probs = (
                            torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()
                        )
                        all_probs.extend(probs)
                        if (
                            probs.shape[1] != n_labels
                        ):  # n_labels is the number of classes
                            raise ValueError("Incorrect output shape from the model")

        if args.data == "mosi_regression":
            val_loss = np.mean(val_losses)
            val_acc = accuracy_score(
                (np.array(all_preds) > 0), (np.array(all_labels) > 0)
            )
            print(
                f"[Seed {seed}/{args.n_runs-1}] [Epoch {epoch+1}/{args.train_epochs}] Task Loss: {np.mean(val_losses):.2f} / Val Loss: {val_loss:.2f}"
            )
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_acc = val_acc

                print(
                    f"[(**Best**) [Epoch {epoch+1}/{args.train_epochs}]  Val Loss: {val_loss:.2f}, Val Acc: {val_acc*100:.2f}"
                )

                best_model_fus = deepcopy(fusion_model.state_dict())
                best_model_enc = {
                    modality: deepcopy(encoder.state_dict())
                    for modality, encoder in encoder_dict.items()
                }
                # Move the models to CPU for saving (only state_dict)
                if args.save:
                    best_model_fus_cpu = {k: v.cpu() for k, v in best_model_fus.items()}
                    best_model_enc_cpu = {
                        modality: {k: v.cpu() for k, v in enc_state.items()}
                        for modality, enc_state in best_model_enc.items()
                    }
        else:
            val_acc = accuracy_score(all_labels, all_preds)
            val_f1 = f1_score(all_labels, all_preds, average="macro")
            if args.data == "enrico":
                val_auc = roc_auc_score(
                    np.array(all_labels),
                    np.array(all_probs),
                    multi_class="ovo",
                    labels=list(range(n_labels)),
                )
            elif args.data in ["mimic", "mosi", "sarcasm", "humor"]:
                val_auc = roc_auc_score(all_labels, all_probs)
            elif args.data == "mmimdb":
                val_auc = 0
            elif args.data == "adni":
                val_auc = roc_auc_score(all_labels, all_probs, multi_class="ovr")

            print(
                f"[Seed {seed}/{args.n_runs-1}] [Epoch {epoch+1}/{args.train_epochs}]  Val Loss: {val_loss:.2f}, Val Acc: {val_acc*100:.2f}, Val F1: {val_f1*100:.2f}, Val AUC: {val_auc*100:.2f}"
            )

            if args.data == "mmimdb":
                # if False:
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    best_val_acc = val_acc
                    best_val_auc = val_auc
                    print(
                        f" [(**Best**) Epoch {epoch+1}/{args.train_epochs}] Val Acc: {val_acc*100:.2f}, Val F1: {val_f1*100:.2f}, Val AUC: {val_auc*100:.2f}"
                    )

                    best_model_fus = deepcopy(fusion_model.state_dict())
                    best_model_enc = {
                        modality: deepcopy(encoder.state_dict())
                        for modality, encoder in encoder_dict.items()
                    }

                    if args.save:
                        best_model_fus_cpu = {
                            k: v.cpu() for k, v in best_model_fus.items()
                        }
                        best_model_enc_cpu = {
                            modality: {k: v.cpu() for k, v in enc_state.items()}
                            for modality, enc_state in best_model_enc.items()
                        }
            else:
                if val_acc > best_val_acc:
                    print(
                        f" [(**Best**) Epoch {epoch+1}/{args.train_epochs}] Val Acc: {val_acc*100:.2f}, Val F1: {val_f1*100:.2f}, Val AUC: {val_auc*100:.2f}"
                    )
                    best_val_acc = val_acc
                    best_val_f1 = val_f1
                    best_val_auc = val_auc
                    best_model_fus = deepcopy(fusion_model.state_dict())
                    best_model_enc = {
                        modality: deepcopy(encoder.state_dict())
                        for modality, encoder in encoder_dict.items()
                    }

                    # Move the models to CPU for saving (only state_dict)
                    if args.save:
                        best_model_fus_cpu = {
                            k: v.cpu() for k, v in best_model_fus.items()
                        }
                        best_model_enc_cpu = {
                            modality: {k: v.cpu() for k, v in enc_state.items()}
                            for modality, enc_state in best_model_enc.items()
                        }

    # Save the best model
    if args.save:

        save_path = f"./saves/data_{args.data}_seed_{seed}_modality_{args.modality}_train_epochs_{args.train_epochs}_val_acc_{best_val_acc*100:.2f}.pth"
        torch.save(
            {"fusion_model": best_model_fus_cpu, "encoder_dict": best_model_enc_cpu},
            save_path,
        )

        print(f"Best model saved to {save_path}")

    # Load best model for test evaluation
    for modality, encoder in encoder_dict.items():
        encoder.load_state_dict(best_model_enc[modality])
        encoder.eval()

    fusion_model.load_state_dict(best_model_fus)
    fusion_model.eval()

    all_preds = []
    all_labels = []
    all_probs = []
    test_losses = []

    with torch.no_grad():

        for batch_samples, _, batch_labels, batch_mcs, batch_observed in test_loader:
            batch_samples = {
                k: v.to(device, non_blocking=True) for k, v in batch_samples.items()
            }
            batch_labels = batch_labels.to(device, non_blocking=True)
            batch_mcs = batch_mcs.to(device, non_blocking=True)
            batch_observed = batch_observed.to(device, non_blocking=True)
            optimizer.zero_grad()

            fusion_input = []
            for i, (modality, samples) in enumerate(batch_samples.items()):
                encoded_samples = encoder_dict[modality](samples).to(device)
                fusion_input.append(encoded_samples)

            outputs = fusion_model(fusion_input)

            if args.data == "mosi_regression":
                all_preds.extend(outputs.squeeze().cpu().numpy())
                all_labels.extend(batch_labels.cpu().numpy())

            else:
                if args.data == "mmimdb":
                    preds = torch.sigmoid(outputs).round()
                else:
                    _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch_labels.cpu().numpy())
                if args.data in ["mimic", "mosi", "sarcasm", "humor"]:
                    all_probs.extend(
                        torch.nn.functional.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                    )
                else:
                    all_probs.extend(
                        torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()
                    )

    if args.data == "mosi_regression":
        all_binary_preds = np.array(all_preds) > 0
        all_labels = np.array(all_labels) > 0
        test_acc = accuracy_score(all_binary_preds, all_labels)
        test_mae = mean_absolute_error(all_preds, all_labels)

        return (best_val_loss, best_val_acc, test_acc, test_mae)
    else:
        test_acc = accuracy_score(all_labels, all_preds)
        test_f1 = f1_score(all_labels, all_preds, average="macro")
        test_f1_micro = f1_score(all_labels, all_preds, average="micro")
        if args.data == "enrico":
            test_auc = roc_auc_score(
                np.array(all_labels),
                np.array(all_probs),
                multi_class="ovo",
                labels=list(range(n_labels)),
            )
        elif args.data in ["mimic", "mosi", "sarcasm", "humor"]:
            test_auc = roc_auc_score(all_labels, all_probs)
        elif args.data == "mmimdb":
            test_auc = 0
        elif args.data == "adni":
            test_auc = roc_auc_score(all_labels, all_probs, multi_class="ovr")
        return (
            best_val_acc,
            best_val_f1,
            best_val_auc,
            test_acc,
            test_f1,
            test_f1_micro,
            test_auc,
        )


def main():
    args, _ = parse_args()
    logger = setup_logger(
        f"./logs/baseline/lrtf/{args.data}", f"{args.data}", f"{args.modality}.txt"
    )
    seeds = np.arange(args.n_runs)  # [0, 1, 2]

    log_summary = "======================================================================================\n"

    model_kwargs = {
        "model": "lrtf",
        "modality": args.modality,
        "use_common_ids": args.use_common_ids,
        "train_epochs": args.train_epochs,
        "num_layers_enc": args.num_layers_enc,
        "num_layers_fus": args.num_layers_fus,
        "num_layers_pred": args.num_layers_pred,
        "num_heads": args.num_heads,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "hidden_dim": args.hidden_dim,
    }

    log_summary += f"Model configuration: {model_kwargs}\n"

    print("Modality:", args.modality)

    if args.data == "mosi_regression":
        val_losses = []
        val_accs = []
        test_accs = []
        test_maes = []
        for seed in seeds:
            best_val_loss, test_acc, test_acc, test_mae = train_and_evaluate(args, seed)
            val_losses.append(best_val_loss)
            test_accs.append(test_acc)

        val_loss_mean = np.mean(val_losses)
        test_acc_mean = np.mean(test_accs) * 100
        val_loss_std = np.std(val_losses)
        test_acc_std = np.std(test_accs) * 100

        log_summary += f"Val loss: {val_loss_mean:.2f} ± {val_loss_std:.2f} "
        log_summary += f"Test Acc: {test_acc_mean:.2f} ± {test_acc_std:.2f} "
        logger.info(log_summary)
    else:
        val_accs = []
        val_f1s = []
        val_aucs = []
        test_accs = []
        test_f1s = []
        test_f1_micros = []
        test_aucs = []

        for seed in seeds:
            val_acc, val_f1, val_auc, test_acc, test_f1, test_f1_micro, test_auc = (
                train_and_evaluate(args, seed)
            )
            val_accs.append(val_acc)
            val_f1s.append(val_f1)
            val_aucs.append(val_auc)
            test_accs.append(test_acc)
            test_f1s.append(test_f1)
            test_f1_micros.append(test_f1_micro)
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
        test_avg_f1_micro = np.mean(test_f1_micros) * 100
        test_std_f1_micro = np.std(test_f1_micros) * 100
        test_avg_auc = np.mean(test_aucs) * 100
        test_std_auc = np.std(test_aucs) * 100

        log_summary += f"[Val] Average Accuracy: {val_avg_acc:.2f} ± {val_std_acc:.2f} "
        log_summary += f"[Val] Average F1 Score: {val_avg_f1:.2f} ± {val_std_f1:.2f} "
        log_summary += f"[Val] Average AUC: {val_avg_auc:.2f} ± {val_std_auc:.2f} / "
        log_summary += (
            f"[Test] Average Accuracy: {test_avg_acc:.2f} ± {test_std_acc:.2f} "
        )
        log_summary += (
            f"[Test] Average F1 (Macro) Score: {test_avg_f1:.2f} ± {test_std_f1:.2f} "
        )
        log_summary += f"[Test] Average F1 (Micro) Score: {test_avg_f1_micro:.2f} ± {test_std_f1_micro:.2f} "

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
