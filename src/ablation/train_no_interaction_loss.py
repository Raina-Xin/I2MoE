import torch
from tqdm import trange
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from copy import deepcopy

from src.common.datasets.adni import load_and_preprocess_data_adni
from src.common.datasets.mimic import load_and_preprocess_data_mimic
from src.common.datasets.enrico import load_and_preprocess_data_enrico
from src.common.datasets.mmimdb import load_and_preprocess_data_mmimdb
from src.common.datasets.mosi import load_and_preprocess_data_mosi
from src.common.datasets.MultiModalDataset import create_loaders

from src.common.utils import (
    seed_everything,
    plot_total_loss_curves,
    visualize_sample_weights,
    visualize_expert_logits,
)

from src.ablation.NoInteractionLossIMoE import NoInteractionLossIMoE


def train_and_evaluate_no_interaction_loss(args, seed, fusion_model, fusion):
    """Train and evaluate interaction MoE withoug interaction loss.

    Args:
        args (argparser.args): argument
        seed (int): random seed
        fusion (str): name of fusion method

    Raises:
        ValueError

    Returns:
        tuple: (best_val_acc, best_val_f1, best_val_auc, test_acc, test_f1, test_auc)
    """
    seed_everything(seed)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    print(device)
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
    elif args.data == "sarcasm":
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
        ) = load_and_preprocess_data_sarcasm(args)
    elif args.data == "humor":
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
        ) = load_and_preprocess_data_humor(args)
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

    ensemble_model = NoInteractionLossIMoE(
        num_modalities=num_modalities,
        fusion_model=fusion_model,
        fusion_sparse=args.fusion_sparse,
        hidden_dim=args.hidden_dim,
        hidden_dim_rw=args.hidden_dim_rw,
        num_layer_rw=args.num_layer_rw,
        temperature_rw=args.temperature_rw,
    ).to(device)

    params = list(ensemble_model.parameters()) + [
        param for encoder in encoder_dict.values() for param in encoder.parameters()
    ]
    optimizer = torch.optim.Adam(params, lr=args.lr)
    if args.data in ["adni", "enrico", "mosi", "sarcasm", "humor"]:
        criterion = torch.nn.CrossEntropyLoss()
    elif args.data == "mimic":
        criterion = torch.nn.CrossEntropyLoss(torch.tensor([0.25, 0.75]).to(device))
    elif args.data == "mosi_regression":
        criterion = torch.nn.SmoothL1Loss()  # Regression
    elif args.data == "mmimdb":
        criterion = torch.nn.BCEWithLogitsLoss()

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

    if args.fusion_sparse:
        plotting_total_losses = {"task": [], "gate": []}
    else:
        plotting_total_losses = {"task": []}

    for epoch in trange(args.train_epochs):

        ensemble_model.train()

        for encoder in encoder_dict.values():
            encoder.train()

        batch_task_losses = []
        if args.fusion_sparse:
            batch_gate_losses = []

        minibatch_count = len(train_loader)

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
                encoded_samples = encoder_dict[modality](samples)
                fusion_input.append(encoded_samples)

            if args.fusion_sparse:
                _, _, outputs, gate_losses = ensemble_model(fusion_input)
            else:
                _, _, outputs = ensemble_model(fusion_input)

            if args.data == "mosi_regression":
                task_loss = criterion(outputs, batch_labels.unsqueeze(1))
            else:
                task_loss = criterion(outputs, batch_labels)

            if args.fusion_sparse:
                gate_loss = torch.mean(torch.tensor(gate_losses))
                loss = task_loss + args.gate_loss_weight * gate_loss
            else:
                loss = task_loss

            loss.backward()
            optimizer.step()

            batch_task_losses.append(task_loss.item())
            if args.fusion_sparse:
                batch_gate_losses.append(gate_loss.item())

            if args.data == "enrico":
                torch.nn.utils.clip_grad_norm_(params, 1.0)

        plotting_total_losses["task"].append(np.mean(batch_task_losses))
        if args.fusion_sparse:
            plotting_total_losses["gate"].append(np.mean(batch_gate_losses))

        ensemble_model.eval()
        for encoder in encoder_dict.values():
            encoder.eval()

        all_preds = []
        all_labels = []
        all_probs = []
        val_losses = []

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
                    encoded_samples = encoder_dict[modality](samples)
                    fusion_input.append(encoded_samples)

                _, _, outputs = ensemble_model(fusion_input)

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
            # if False:
            val_loss = np.mean(val_losses)
            print(
                f"[Seed {seed}/{args.n_runs-1}] [Epoch {epoch+1}/{args.train_epochs}] Task Loss: {np.mean(val_losses):.2f} / Val Loss: {val_loss:.2f}"
            )
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

                    best_model_fus = deepcopy(ensemble_model.state_dict())
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
                    best_model_fus = deepcopy(ensemble_model.state_dict())
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
    plot_total_loss_curves(
        args,
        plotting_total_losses=plotting_total_losses,
        framework="no-interaction-loss",
        fusion=fusion,
    )

    # Save the best model
    if args.save:
        Path("./saves").mkdir(exist_ok=True, parents=True)
        Path(f"./saves/no-interaction-loss/{fusion}/{args.data}").mkdir(
            exist_ok=True, parents=True
        )

        if args.data == "mmimdb":
            save_path = f"./saves/no-interaction-loss/{fusion}/{args.data}/seed_{seed}_modality_{args.modality}_train_epochs_{args.train_epochs}_val_f1_{best_val_f1:.2f}.pth"
        else:
            save_path = f"./saves/no-interaction-loss/{fusion}/{args.data}/seed_{seed}_modality_{args.modality}_train_epochs_{args.train_epochs}_val_acc_{best_val_acc:.2f}.pth"
        torch.save(
            {"ensemble_model": best_model_fus_cpu, "encoder_dict": best_model_enc_cpu},
            save_path,
        )

        print(f"Best model saved to {save_path}")

    # Load best model for test evaluation
    for modality, encoder in encoder_dict.items():
        encoder.load_state_dict(best_model_enc[modality])
        encoder.eval()

    ensemble_model.load_state_dict(best_model_fus)
    ensemble_model.eval()

    all_preds = []
    all_labels = []
    all_probs = []
    test_losses = []
    all_routing_weights = []

    with torch.no_grad():

        for batch_samples, batch_labels, batch_mcs, batch_observed in test_loader:
            batch_samples = {
                k: v.to(device, non_blocking=True) for k, v in batch_samples.items()
            }
            batch_labels = batch_labels.to(device, non_blocking=True)
            batch_mcs = batch_mcs.to(device, non_blocking=True)
            batch_observed = batch_observed.to(device, non_blocking=True)
            optimizer.zero_grad()

            fusion_input = []
            for i, (modality, samples) in enumerate(batch_samples.items()):
                encoded_samples = encoder_dict[modality](samples)
                fusion_input.append(encoded_samples)

            expert_outputs, routing_weights, outputs = ensemble_model(fusion_input)

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
                all_routing_weights.extend(routing_weights.cpu().numpy())
                if args.data in ["mimic", "mosi", "sarcasm", "humor"]:
                    all_probs.extend(
                        torch.nn.functional.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                    )
                else:
                    all_probs.extend(
                        torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()
                    )

    visualize_expert_logits(
        expert_outputs,
        routing_weights,
        outputs,
        args,
        framework="no-interaction-loss",
        fusion=fusion,
    )

    visualize_sample_weights(
        all_routing_weights, args, framework="no-interaction-loss", fusion=fusion
    )

    if args.data == "mosi_regression":
        all_preds = np.array(all_preds) > 0
        all_labels = np.array(all_labels) > 0
        test_acc = accuracy_score(all_preds, all_labels)
        return best_val_loss, test_acc
    else:
        test_acc = accuracy_score(all_labels, all_preds)
        test_f1 = f1_score(all_labels, all_preds, average="macro")
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
        return best_val_acc, best_val_f1, best_val_auc, test_acc, test_f1, test_auc
