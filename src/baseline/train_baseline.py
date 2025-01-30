import os
import torch
import numpy as np
from tqdm import trange
from pathlib import Path
from copy import deepcopy
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from src.common.utils import seed_everything, plot_total_loss_curves
from src.common.datasets.MultiModalDataset import create_loaders
from src.common.datasets.adni import load_and_preprocess_data_adni
from src.common.datasets.mimic import load_and_preprocess_data_mimic
from src.common.datasets.enrico import load_and_preprocess_data_enrico
from src.common.datasets.mmimdb import load_and_preprocess_data_mmimdb
from src.common.datasets.mosi import (
    load_and_preprocess_data_mosi,
    load_and_preprocess_data_mosi_regression,
)


def load_data(args):
    """Load and preprocess dataset."""
    loaders = {
        "adni": "load_and_preprocess_data_adni",
        "mimic": "load_and_preprocess_data_mimic",
        "enrico": "load_and_preprocess_data_enrico",
        "mmimdb": "load_and_preprocess_data_mmimdb",
        "mosi": "load_and_preprocess_data_mosi",
        "mosi_regression": "load_and_preprocess_data_mosi_regression",
    }
    if args.data not in loaders:
        raise ValueError(f"Dataset {args.data} is not supported.")
    loader = eval(loaders[args.data])
    return loader(args)


def get_criterion(args, device):
    """Return the appropriate loss function based on the dataset."""
    if args.data in ["adni", "enrico", "mosi"]:
        return torch.nn.CrossEntropyLoss()
    elif args.data == "mimic":
        return torch.nn.CrossEntropyLoss(torch.tensor([0.25, 0.75]).to(device))
    elif args.data == "mosi_regression":
        return torch.nn.SmoothL1Loss()  # Regression task
    elif args.data == "mmimdb":
        return torch.nn.BCEWithLogitsLoss()
    else:
        raise ValueError(f"No loss function defined for dataset: {args.data}")


def save_model(model_state, encoder_states, args, seed, metrics, model_name="baseline"):
    """
    Save the best model based on validation metrics.

    Args:
        model_state (dict): State dict of the fusion model.
        encoder_states (dict): State dicts of the encoders.
        args (argparse.Namespace): Arguments containing configurations.
        seed (int): Random seed for reproducibility.
        metrics (dict): Best metrics from validation.
        model_name (str): Name of the baseline model (default: "baseline").
    """
    # Define save directory
    save_dir = Path(f"./saves/baseline/{model_name}/{args.data}")
    save_dir.mkdir(parents=True, exist_ok=True)

    # Determine filename based on task
    if args.data == "mosi_regression":
        metric_name = "val_loss"
        metric_value = metrics.get("val_loss", float("inf"))
        filename = f"seed_{seed}_modality_{args.modality}_train_epochs_{args.train_epochs}_{metric_name}_{metric_value:.4f}.pth"
    else:
        metric_name = "val_acc"
        metric_value = metrics.get("val_acc", 0.0)
        filename = f"seed_{seed}_modality_{args.modality}_train_epochs_{args.train_epochs}_{metric_name}_{metric_value:.4f}.pth"

    # Full save path
    save_path = save_dir / filename

    # Save the model
    torch.save(
        {
            "fusion_model": model_state,
            "encoder_dict": encoder_states,
            "args": vars(args),
        },
        save_path,
    )

    print(f"Model saved to {save_path}")


def train_and_evaluate(args, seed, fusion_model, fusion_name):
    """Train and evaluate a baseline model."""
    seed_everything(seed)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    num_modalities = len(args.modality)

    # Load dataset
    data = load_data(args)
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
        _,
        _,
    ) = data

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

    # Get criterion dynamically
    criterion = get_criterion(args, device)

    optimizer = torch.optim.Adam(
        list(fusion_model.parameters())
        + [
            param for encoder in encoder_dict.values() for param in encoder.parameters()
        ],
        lr=args.lr,
    )

    best_metrics = (
        {
            "acc": 0.0,
            "f1": 0.0,
            "auc": 0.0,
        }
        if args.data != "mosi_regression"
        else {"loss": float("inf"), "acc": 0.0}
    )
    best_model = None
    plotting_total_losses = {"task": []}

    results_log = []

    for epoch in trange(args.train_epochs):
        fusion_model.train()
        for encoder in encoder_dict.values():
            encoder.train()

        batch_task_losses = []
        for batch_samples, batch_labels, *_ in train_loader:
            batch_samples = {k: v.to(device) for k, v in batch_samples.items()}
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad()

            fusion_input = [
                encoder_dict[modality](samples).to(device)
                for modality, samples in batch_samples.items()
            ]
            outputs = fusion_model(fusion_input)
            dataset = args.data
            if dataset == "mmimdb":
                task_loss = criterion(outputs, batch_labels.float())
            elif dataset == "mosi_regression":
                task_loss = criterion(outputs, batch_labels.unsqueeze(1))
            else:
                task_loss = criterion(outputs, batch_labels)
            task_loss.backward()
            optimizer.step()
            batch_task_losses.append(task_loss.item())

        avg_train_loss = np.mean(batch_task_losses)
        plotting_total_losses["task"].append(avg_train_loss)
        print(f"[Epoch {epoch+1}/{args.train_epochs}] Train Loss: {avg_train_loss:.4f}")

        # Validation
        val_metrics = evaluate_model(
            fusion_model,
            encoder_dict,
            val_loader,
            criterion,
            device,
            n_labels,
            args.data,
            mode="val",
        )
        val_metric_to_compare = (
            val_metrics["loss"]
            if args.data == "mosi_regression"
            else val_metrics["acc"]
        )

        if (
            args.data == "mosi_regression"
            and val_metric_to_compare < best_metrics.get("loss", float("inf"))
        ) or (
            args.data != "mosi_regression"
            and val_metric_to_compare > best_metrics.get("acc", 0.0)
        ):
            best_metrics.update(val_metrics)
            best_model_fus = deepcopy(fusion_model.state_dict())
            best_model_enc = {
                modality: deepcopy(encoder.state_dict())
                for modality, encoder in encoder_dict.items()
            }
            print(f"[**Best Model**] Updated for epoch {epoch+1}: {val_metrics}")

        # Log validation metrics for this epoch
        epoch_result = {"epoch": epoch + 1, "val_metrics": val_metrics}
        results_log.append(epoch_result)

    # Save best model
    if args.save:
        best_model_fus_cpu = {k: v.cpu() for k, v in best_model_fus.items()}
        best_model_enc_cpu = {
            modality: {k: v.cpu() for k, v in enc_state.items()}
            for modality, enc_state in best_model_enc.items()
        }
        save_model(best_model_fus_cpu, best_model_enc_cpu, args, seed, best_metrics)

    # Test evaluation
    test_metrics = evaluate_model(
        fusion_model,
        encoder_dict,
        test_loader,
        criterion,
        device,
        n_labels,
        args.data,
        mode="test",
    )
    print(f"[Test] {test_metrics}")
    plot_total_loss_curves(
        args, plotting_total_losses, framework="baseline", fusion=fusion_name
    )

    return best_metrics, test_metrics


def evaluate_model(
    model, encoder_dict, loader, criterion, device, n_labels, dataset, mode="val"
):
    """Evaluate the model on validation or test data."""
    model.eval()
    for encoder in encoder_dict.values():
        encoder.eval()

    all_preds, all_labels, all_probs = [], [], []
    if dataset == "mosi_regression":
        total_loss = []

    with torch.no_grad():
        for batch_samples, batch_labels, *_ in loader:
            batch_samples = {k: v.to(device) for k, v in batch_samples.items()}
            batch_labels = batch_labels.to(device)

            fusion_input = [
                encoder_dict[modality](samples).to(device)
                for modality, samples in batch_samples.items()
            ]
            outputs = model(fusion_input)

            # Calculate loss
            if dataset == "mosi_regression":
                loss = criterion(outputs, batch_labels.unsqueeze(1))
                total_loss.append(loss.item())

            # Collect predictions and probabilities
            if dataset == "mosi_regression":
                preds = outputs.squeeze().cpu().numpy()
            elif dataset == "mmimdb":
                preds = torch.sigmoid(outputs).round().cpu().numpy()
            else:
                _, preds = torch.max(outputs, 1)
                preds = preds.cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(batch_labels.cpu().numpy())
            if dataset in ["mimic", "mosi", "sarcasm", "humor"]:
                all_probs.extend(
                    torch.nn.functional.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                )
            else:
                probs = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()
                all_probs.extend(probs)
    if dataset == "mosi_regression":
        metrics = {"loss": np.mean(total_loss)}
    else:
        metrics = dict()
    if dataset == "mosi_regression":
        metrics["acc"] = accuracy_score(
            (np.array(all_preds) > 0), (np.array(all_labels) > 0)
        )
    else:
        metrics["acc"] = accuracy_score(all_labels, all_preds)
        metrics["f1"] = f1_score(all_labels, all_preds, average="macro")
        if dataset == "enrico":
            metrics["auc"] = roc_auc_score(
                np.array(all_labels),
                np.array(all_probs),
                multi_class="ovo",
                labels=list(range(n_labels)),
            )
        elif dataset in ["mimic", "mosi"]:
            metrics["auc"] = roc_auc_score(all_labels, all_probs)
        elif dataset == "mmimdb":
            metrics["auc"] = 0
        elif dataset == "adni":
            metrics["auc"] = roc_auc_score(all_labels, all_probs, multi_class="ovr")
    print(f"[{mode.capitalize()}] Metrics: {metrics}")
    return metrics
