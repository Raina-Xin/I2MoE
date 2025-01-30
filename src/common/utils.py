from pathlib import Path
import os
import logging
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import random
import numpy as np
import torch
from itertools import combinations
from datetime import datetime

# Define a consistent color map for experts
COLORS = [
    "#005f73",
    "#0a9396",
    "#94d2bd",
    "#e9d8a6",
    "#ee9b00",
    "#ca6702",
    "#bb3e03",
    "#ae2012",
    "#9b2226",
]  # 001219


def set_style():
    """set_style"""
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["font.family"] = "sans-serif"
    sns.set(context="paper", style="ticks")
    mpl.rcParams["text.color"] = "black"
    mpl.rcParams["axes.labelcolor"] = "black"
    mpl.rcParams["axes.edgecolor"] = "black"
    mpl.rcParams["axes.labelcolor"] = "black"
    mpl.rcParams["xtick.color"] = "black"
    mpl.rcParams["ytick.color"] = "black"
    mpl.rcParams["xtick.major.size"] = 2.5
    mpl.rcParams["ytick.major.size"] = 2.5
    mpl.rcParams["xtick.major.width"] = 0.45
    mpl.rcParams["ytick.major.width"] = 0.45
    mpl.rcParams["axes.edgecolor"] = "black"
    mpl.rcParams["axes.linewidth"] = 0.45
    mpl.rcParams["font.size"] = 9
    mpl.rcParams["axes.labelsize"] = 9
    mpl.rcParams["axes.titlesize"] = 9
    mpl.rcParams["figure.titlesize"] = 9
    mpl.rcParams["figure.titlesize"] = 9
    mpl.rcParams["legend.fontsize"] = 6
    mpl.rcParams["legend.title_fontsize"] = 9
    mpl.rcParams["xtick.labelsize"] = 6
    mpl.rcParams["ytick.labelsize"] = 6


def set_size(w, h, ax=None):
    """w, h: width, height in inches
    ​
        Resize the axis to have exactly these dimensions

    """
    if not ax:
        ax = plt.gca()
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    figw = float(w) / (r - l)
    figh = float(h) / (t - b)
    ax.figure.set_size_inches(figw, figh)


def get_modality_combinations(modalities):
    # full modality index is 0.
    all_combinations = []
    for i in range(len(modalities), 0, -1):
        comb = list(combinations(modalities, i))
        all_combinations.extend(comb)

    # Create a mapping dictionary
    combination_to_index = {
        "".join(sorted(comb)): idx for idx, comb in enumerate(all_combinations)
    }
    return combination_to_index


def setup_logger(log_path, log_name, file_name):
    Path("./logs").mkdir(exist_ok=True, parents=True)
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.INFO)
    Path(log_path).mkdir(exist_ok=True, parents=True)
    file_handler = logging.FileHandler(os.path.join(log_path, file_name))
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def seed_everything(seed=0):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def str2bool(s):
    if s not in {"False", "True", "false", "true"}:
        raise ValueError("Not a valid boolean string")
    return (s == "True") or (s == "true")


def plot_interaction_loss_curves(
    args, plotting_interaction_losses=[], framework="ensemble", fusion="transformer"
):

    epochs = range(1, args.train_epochs + 1)

    plt.figure(figsize=(10, 6))  # Adjust figure size as needed

    for name, losses in plotting_interaction_losses.items():
        plt.plot(epochs, losses, label=name)

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    # plt.title(f"Training Interaction Loss Curves ({args.data}-{framework}-{fusion})")
    plt.legend()
    plt.grid(True)

    save_dir = Path(f"./figures/{framework}/{fusion}/loss_curves/interaction/")
    save_dir.mkdir(exist_ok=True, parents=True)

    now = datetime.now()
    plt.savefig(
        str(save_dir)
        + f"/{args.data}_epochs_{args.train_epochs}_{args.modality}_seed_{args.seed}-{now.strftime('%Y-%m-%d_%H:%M:%S')}.pdf",
        dpi=300,
    )

    return


def plot_total_loss_curves(
    args, plotting_total_losses=[], framework="ensemble", fusion="transformer"
):
    """Plot the loss curves of models without interaction loss."""
    epochs = range(1, args.train_epochs + 1)

    plt.figure(figsize=(10, 6))  # Adjust figure size as needed

    for name, losses in plotting_total_losses.items():
        plt.plot(epochs, losses, label=name)

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    # plt.title(f"Training Total Loss Curves ({args.data}-{framework}-{fusion})")
    plt.legend()
    plt.grid(True)

    save_dir = Path(f"./figures/{framework}/{fusion}/loss_curves/total/")
    save_dir.mkdir(exist_ok=True, parents=True)
    now = datetime.now()
    plt.savefig(
        str(save_dir)
        + f"/{args.data}_epochs_{args.train_epochs}_{args.modality}_seed_{args.seed}-{now.strftime('%Y-%m-%d_%H:%M:%S')}.pdf",
        dpi=300,
    )

    return


def visualize_expert_logits(
    expert_outputs, routing_weights, output, args, framework=None, fusion=None
):
    """
    Visualizes the logits output by multiple experts for each sample in the minibatch.

    Args:
        expert_outputs (list of torch.Tensor): List of logits from experts,
                                               where each tensor has shape (batch_size, num_classes).
    """
    num_experts = len(expert_outputs)
    batch_size, num_classes = expert_outputs[0].shape
    num_samples_to_plot = min(4, batch_size)  # Plot only the first 8 samples
    num_subplots_per_sample = (
        num_classes + 3
    )  # Subplots: logits for each class, routing weights, final output

    # Convert tensors to NumPy for visualization
    expert_outputs_np = [expert.cpu().detach().numpy() for expert in expert_outputs]
    routing_weights_np = routing_weights.cpu().detach().numpy()
    output_np = output.cpu().detach().numpy()

    # Create figure: Each row corresponds to one sample
    fig, axes = plt.subplots(
        num_samples_to_plot,
        num_subplots_per_sample,
        figsize=(4 * num_subplots_per_sample, 3 * num_samples_to_plot),
        squeeze=False,
    )

    for idx, sample_idx in enumerate(
        random.sample(list(range(batch_size)), num_samples_to_plot)
    ):
        # Plot expert logits for each class
        for class_idx in range(num_classes):
            ax_logits = axes[idx, class_idx]
            logits_for_class = [
                expert_outputs_np[expert_idx][sample_idx, class_idx]
                for expert_idx in range(num_experts)
            ]
            for expert_idx, logit in enumerate(logits_for_class):
                ax_logits.bar(
                    [expert_idx],
                    [logit],
                    color=COLORS[expert_idx],
                    alpha=0.7,
                    label=f"Expert {expert_idx + 1}",
                )
            ax_logits.set_title(
                f"Sample {sample_idx + 1}, Class {class_idx + 1} - Logits"
            )
            ax_logits.set_xlabel("Experts")
            ax_logits.set_ylabel("Logits")
            ax_logits.set_xticks(range(num_experts))
            ax_logits.set_xticklabels([f"Exp {i + 1}" for i in range(num_experts)])
            if sample_idx == 0 and class_idx == 0:  # Add legend only once
                ax_logits.legend()

        # Plot routing weights
        ax_weights = axes[idx, num_classes]
        weights = routing_weights_np[sample_idx]  # Shape: (num_experts,)
        ax_weights.bar(range(num_experts), weights, color=COLORS, alpha=0.7)
        ax_weights.set_title(f"Sample {sample_idx + 1} - Routing Weights")
        ax_weights.set_xlabel("Experts")
        ax_weights.set_ylabel("Weights")
        ax_weights.set_xticks(range(num_experts))
        ax_weights.set_xticklabels([f"Exp {i + 1}" for i in range(num_experts)])

        # Plot final output with weighted contributions
        ax_output = axes[idx, num_classes + 1]
        for class_idx in range(num_classes):
            # Compute weighted contribution of each expert
            contributions = [
                routing_weights_np[sample_idx, expert_idx]
                * expert_outputs_np[expert_idx][sample_idx, class_idx]
                for expert_idx in range(num_experts)
            ]
            for expert_idx, contribution in enumerate(contributions):
                ax_output.bar(
                    [class_idx],
                    [contribution],
                    bottom=np.sum(contributions[:expert_idx]),
                    color=COLORS[expert_idx],
                    alpha=0.7,
                )
        ax_output.set_title(f"Sample {sample_idx + 1} - Final Output Contributions")
        ax_output.set_xlabel("Classes")
        ax_output.set_ylabel("Weighted Contributions")

        # Plot final output
        ax_output = axes[idx, num_classes + 2]
        final_output = output_np[sample_idx]  # Shape: (num_classes,)
        ax_output.bar(range(num_classes), final_output, color=COLORS, alpha=0.7)
        ax_output.set_title(f"Sample {sample_idx + 1} - Final Output")
        ax_output.set_xlabel("Classes")
        ax_output.set_ylabel("Output")

    plt.tight_layout()
    save_dir = Path(f"./figures/{framework}/{fusion}/expert_logit/")
    save_dir.mkdir(exist_ok=True, parents=True)
    now = datetime.now()
    plt.savefig(
        str(save_dir)
        + f"/{args.data}_epochs_{args.train_epochs}_{args.modality}_seed_{args.seed}-{now.strftime('%Y-%m-%d_%H:%M:%S')}.pdf",
        dpi=300,
    )

    return


def visualize_expert_logits_simple_weight(
    expert_outputs, routing_weights, output, args, framework=None, fusion=None
):
    """
    Visualizes the logits output by multiple experts for each sample in the minibatch.

    Args:
        expert_outputs (list of torch.Tensor): List of logits from experts,
                                               where each tensor has shape (batch_size, num_classes).
    """
    num_experts = len(expert_outputs)
    batch_size, num_classes = expert_outputs[0].shape
    num_samples_to_plot = min(4, batch_size)  # Plot only the first 8 samples
    num_subplots_per_sample = (
        num_classes + 3
    )  # Subplots: logits for each class, routing weights, final output

    # Convert tensors to NumPy for visualization
    expert_outputs_np = [expert.cpu().detach().numpy() for expert in expert_outputs]
    routing_weights_np = routing_weights.cpu().detach().numpy()
    output_np = output.cpu().detach().numpy()

    # Create figure: Each row corresponds to one sample
    fig, axes = plt.subplots(
        num_samples_to_plot,
        num_subplots_per_sample,
        figsize=(4 * num_subplots_per_sample, 3 * num_samples_to_plot),
        squeeze=False,
    )

    for idx, sample_idx in enumerate(
        random.sample(list(range(batch_size)), num_samples_to_plot)
    ):
        # Plot expert logits for each class
        for class_idx in range(num_classes):
            ax_logits = axes[idx, class_idx]
            logits_for_class = [
                expert_outputs_np[expert_idx][sample_idx, class_idx]
                for expert_idx in range(num_experts)
            ]
            for expert_idx, logit in enumerate(logits_for_class):
                ax_logits.bar(
                    [expert_idx],
                    [logit],
                    color=COLORS[expert_idx],
                    alpha=0.7,
                    label=f"Expert {expert_idx + 1}",
                )
            ax_logits.set_title(
                f"Sample {sample_idx + 1}, Class {class_idx + 1} - Logits"
            )
            ax_logits.set_xlabel("Experts")
            ax_logits.set_ylabel("Logits")
            ax_logits.set_xticks(range(num_experts))
            ax_logits.set_xticklabels([f"Exp {i + 1}" for i in range(num_experts)])
            if sample_idx == 0 and class_idx == 0:  # Add legend only once
                ax_logits.legend()

        # Plot routing weights
        ax_weights = axes[idx, num_classes]
        weights = routing_weights_np  # Shape: (num_experts,)
        ax_weights.bar(range(num_experts), weights, color=COLORS, alpha=0.7)
        ax_weights.set_title(f"Sample {sample_idx + 1} - Routing Weights")
        ax_weights.set_xlabel("Experts")
        ax_weights.set_ylabel("Weights")
        ax_weights.set_xticks(range(num_experts))
        ax_weights.set_xticklabels([f"Exp {i + 1}" for i in range(num_experts)])

        # Plot final output with weighted contributions
        ax_output = axes[idx, num_classes + 1]
        for class_idx in range(num_classes):
            # Compute weighted contribution of each expert
            contributions = [
                routing_weights_np[expert_idx]
                * expert_outputs_np[expert_idx][sample_idx, class_idx]
                for expert_idx in range(num_experts)
            ]
            for expert_idx, contribution in enumerate(contributions):
                ax_output.bar(
                    [class_idx],
                    [contribution],
                    bottom=np.sum(contributions[:expert_idx]),
                    color=COLORS[expert_idx],
                    alpha=0.7,
                )
        ax_output.set_title(f"Sample {sample_idx + 1} - Final Output Contributions")
        ax_output.set_xlabel("Classes")
        ax_output.set_ylabel("Weighted Contributions")

        # Plot final output
        ax_output = axes[idx, num_classes + 2]
        final_output = output_np[sample_idx]  # Shape: (num_classes,)
        ax_output.bar(range(num_classes), final_output, color=COLORS, alpha=0.7)
        ax_output.set_title(f"Sample {sample_idx + 1} - Final Output")
        ax_output.set_xlabel("Classes")
        ax_output.set_ylabel("Output")

    plt.tight_layout()
    save_dir = Path(f"./figures/{framework}/{fusion}/expert_logit/")
    save_dir.mkdir(exist_ok=True, parents=True)
    now = datetime.now()
    plt.savefig(
        str(save_dir)
        + f"/{args.data}_epochs_{args.train_epochs}_{args.modality}_seed_{args.seed}-{now.strftime('%Y-%m-%d_%H:%M:%S')}.pdf",
        dpi=300,
    )

    return


def visualize_sample_weights(
    all_routing_weights, args, framework="imoe", fusion="moepp"
):
    """
    Create a violin plot to visualize weights for each sample.

    Parameters:
    -----------
    all_routing_weights : list of numpy arrays
        List containing weight arrays of shape (4,).
    """
    # Convert list to numpy array
    weights_array = np.array(all_routing_weights)

    # Verify that weights sum to 1 (with small tolerance for floating-point errors)
    weight_sums = weights_array.sum(axis=1)
    print(f"Weight sum statistics: min={weight_sums.min()}, max={weight_sums.max()}")

    # Additional visualization to show distribution of weights
    plt.figure(figsize=(4, 2))

    # Violin plot to show weight distribution
    if len(args.modality) == 2:
        vp = plt.violinplot(
            [
                weights_array[:, 0],  # Uniqueness 1
                weights_array[:, 1],  # Uniqueness 2
                weights_array[:, 2],  # Synergy
                weights_array[:, 3],  # Redundancy
            ],
            showmeans=True,
        )
        # Set individual colors for each violin
        for idx, body in enumerate(vp["bodies"]):
            body.set_facecolor(COLORS[idx])  # Set the face color
            # body.set_edgecolor('black')     # Optional: Set the edge color
            body.set_alpha(0.7)  # Set transparency
        plt.xticks(
            [1, 2, 3, 4], ["Uniqueness 1", "Uniqueness 2", "Synergy", "Redundancy"]
        )
    elif len(args.modality) == 3:
        vp = plt.violinplot(
            [
                weights_array[:, 0],  # Uniqueness 1
                weights_array[:, 1],  # Uniqueness 2
                weights_array[:, 2],  # Uniqueness 3
                weights_array[:, 3],  # Synergy
                weights_array[:, 4],  # Redundancy
            ],
            showmeans=True,
        )
        # Set individual colors for each violin
        for idx, body in enumerate(vp["bodies"]):
            body.set_facecolor(COLORS[idx])  # Set the face color
            # body.set_edgecolor('black')     # Optional: Set the edge color
            body.set_alpha(0.7)  # Set transparency
        plt.xticks(
            [1, 2, 3, 4, 5],
            ["Uniqueness 1", "Uniqueness 2", "Uniqueness 3", "Synergy", "Redundancy"],
        )

    elif len(args.modality) == 4:
        vp = plt.violinplot(
            [
                weights_array[:, 0],  # Uniqueness 1
                weights_array[:, 1],  # Uniqueness 2
                weights_array[:, 2],  # Uniqueness 3
                weights_array[:, 3],  # Uniqueness 4
                weights_array[:, 4],  # Synergy
                weights_array[:, 5],  # Redundancy
            ],
            showmeans=True,
        )
        # Set individual colors for each violin
        for idx, body in enumerate(vp["bodies"]):
            body.set_facecolor(COLORS[idx])  # Set the face color
            # body.set_edgecolor('black')     # Optional: Set the edge color
            body.set_alpha(0.7)  # Set transparency
        plt.xticks(
            [1, 2, 3, 4, 5, 6],
            [
                "Uniqueness 1",
                "Uniqueness 2",
                "Uniqueness 3",
                "Uniqueness 4",
                "Synergy",
                "Redundancy",
            ],
        )
    # plt.title(
    #     f"Distribution of Weights Across All Samples ({args.data}-{framework}-{fusion})"
    # )
    plt.ylabel("Weight Value")
    plt.tight_layout()

    # Save the plot
    save_dir = Path(f"./figures/{framework}/{fusion}/interaction_weight/")
    save_dir.mkdir(exist_ok=True, parents=True)
    now = datetime.now()
    plt.savefig(
        str(save_dir)
        + f"/{args.data}_epochs_{args.train_epochs}_{args.modality}_seed_{args.seed}-{now.strftime('%Y-%m-%d_%H:%M:%S')}.pdf",
        dpi=300,
    )

    return


def visualize_expert_logits_distribution(
    all_expert_outputs, args, framework="imoe", fusion="moepp"
):
    """
    Plot the distribution of logits for each interaction expert and each class.

    Args:
        all_expert_outputs (list of numpy arrays): List of logits from experts,
                                                   where each tensor has shape (batch_size, num_classes).
    """
    all_expert_outputs = [
        np.array(all_expert_output) for all_expert_output in all_expert_outputs
    ]
    num_classes = all_expert_outputs[0].shape[1]
    num_experts = len(all_expert_outputs)

    # Prepare a figure with num_classes rows and num_experts columns
    fig, axes = plt.subplots(
        num_classes,
        num_experts,
        figsize=(5 * num_experts, 4 * num_classes),
        squeeze=False,
    )

    # Iterate over classes and experts
    for class_idx in range(num_classes):
        for expert_idx in range(num_experts):
            # Extract logits for the current class from the current expert
            logits = all_expert_outputs[expert_idx][:, class_idx]

            # Plot histogram
            ax = axes[class_idx, expert_idx]
            ax.hist(
                logits,
                bins=20,
                color=COLORS[expert_idx],
                alpha=0.7,
                edgecolor="#ffffff",
            )
            ax.set_title(f"Expert {expert_idx + 1}, Class {class_idx + 1}")
            ax.set_xlabel("Logit Value")
            ax.set_ylabel("Frequency")

    plt.tight_layout()

    save_dir = Path(f"./figures/{framework}/{fusion}/expert_logit_stat/")
    save_dir.mkdir(exist_ok=True, parents=True)
    now = datetime.now()
    plt.savefig(
        str(save_dir)
        + f"/{args.data}_epochs_{args.train_epochs}_{args.modality}_seed_{args.seed}-{now.strftime('%Y-%m-%d_%H:%M:%S')}.pdf",
        dpi=300,
    )

    return


def visualize_sample_weights_synergy_redundancy_only(
    all_routing_weights, args, framework="imoe", fusion="moepp"
):
    """
    Create a stacked bar plot to visualize weights for each sample for ablation.

    Parameters:
    -----------
    all_routing_weights : list of numpy arrays
        List containing weight arrays of shape (4,)
    """
    # Convert list to numpy array
    weights_array = np.array(all_routing_weights)

    # Verify that weights sum to 1 (with small tolerance for floating-point errors)
    weight_sums = weights_array.sum(axis=1)
    print(f"Weight sum statistics: min={weight_sums.min()}, max={weight_sums.max()}")

    # Additional visualization to show distribution of weights
    plt.figure(figsize=(10, 6))

    plt.violinplot(
        [
            weights_array[:, 0],  # Synergy
            weights_array[:, 1],  # Redundancy
        ]
    )
    plt.xticks([1, 2], ["Synergy", "Redundancy"])

    # plt.title(
    #     f"Distribution of Weights Across All Samples ({args.data}-{framework}-{fusion})"
    # )
    plt.ylabel("Weight Value")
    # Adjust layout
    plt.tight_layout()

    save_dir = Path(f"./figures/{framework}/{fusion}/interaction_weight/")
    save_dir.mkdir(exist_ok=True, parents=True)
    now = datetime.now()
    plt.savefig(
        str(save_dir)
        + f"/{args.data}_epochs_{args.train_epochs}_{args.modality}_seed_{args.seed}-{now.strftime('%Y-%m-%d_%H:%M:%S')}.pdf",
        dpi=300,
    )

    return


def plot_loss_curves_MI_shared(
    args, plotting_total_losses=[], plotting_interaction_losses=[], is_latent=False
):

    Path(f"./figures/loss_curves/{args.data}").mkdir(exist_ok=True, parents=True)
    epochs = range(1, args.train_epochs + 1)

    plt.figure(figsize=(10, 6))  # Adjust figure size as needed

    for name, losses in plotting_total_losses.items():
        plt.plot(epochs, losses, label=name)

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    # plt.title("Training Total Loss Curves")
    plt.legend()
    plt.grid(True)
    if is_latent:
        Path(f"./figures/loss_curves/{args.data}/interaction_shared/total_loss").mkdir(
            exist_ok=True, parents=True
        )
        plt.savefig(
            f"./figures/loss_curves/{args.data}/interaction_shared/total_loss/{args.modality}_SP_{args.fusion_sparse}_GT_{args.gate}_MR_{args.weighting_method}_epoch_{args.train_epochs}.pdf",
            dpi=300,
        )
    else:
        Path(f"./figures/loss_curves/{args.data}/interaction_shared/total_loss").mkdir(
            exist_ok=True, parents=True
        )
        plt.savefig(
            f"./figures/loss_curves/{args.data}/interaction_shared/total_loss/{args.modality}_SP_{args.fusion_sparse}_GT_{args.gate}_MR_{args.weighting_method}_epoch_{args.train_epochs}.pdf",
            dpi=300,
        )

    plt.figure(figsize=(10, 6))  # Adjust figure size as needed

    for name, losses in plotting_interaction_losses.items():
        plt.plot(epochs, losses, label=name)

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    # plt.title("Training Interaction Loss Curves")
    plt.legend()
    plt.grid(True)
    if is_latent:
        Path(
            f"./figures/loss_curves/{args.data}/interaction_shared/interaction_loss"
        ).mkdir(exist_ok=True, parents=True)
        plt.savefig(
            f"./figures/loss_curves/{args.data}/interaction_shared/interaction_loss/{args.modality}_SP_{args.fusion_sparse}_GT_{args.gate}_MR_{args.weighting_method}_epoch_{args.train_epochs}.pdf",
            dpi=300,
        )
    else:
        Path(
            f"./figures/loss_curves/{args.data}/interaction_shared/interaction_loss"
        ).mkdir(exist_ok=True, parents=True)
        plt.savefig(
            f"./figures/loss_curves/{args.data}/interaction_shared/interaction_loss/{args.modality}_SP_{args.fusion_sparse}_GT_{args.gate}_MR_{args.weighting_method}_epoch_{args.train_epochs}.pdf",
            dpi=300,
        )

    plt.figure(figsize=(10, 6))  # Adjust figure size as needed

    return


def plot_loss_curves_MI_shared_baselines(
    args, plotting_total_losses=[], plotting_interaction_losses=[], is_latent=False
):

    Path(f"./figures/loss_curves/{args.data}/{args.model}").mkdir(
        exist_ok=True, parents=True
    )
    epochs = range(1, args.train_epochs + 1)

    plt.figure(figsize=(10, 6))  # Adjust figure size as needed

    for name, losses in plotting_total_losses.items():
        plt.plot(epochs, losses, label=name)

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    # plt.title("Training Total Loss Curves")
    plt.legend()
    plt.grid(True)
    if is_latent:
        Path(
            f"./figures/loss_curves/{args.data}/{args.model}/interaction_shared/total_loss"
        ).mkdir(exist_ok=True, parents=True)
        plt.savefig(
            f"./figures/loss_curves/{args.data}/{args.model}/interaction_shared/total_loss/{args.modality}_SP_{args.fusion_sparse}_GT_{args.gate}_MR_{args.weighting_method}_epoch_{args.train_epochs}.pdf",
            dpi=300,
        )
    else:
        Path(
            f"./figures/loss_curves/{args.data}/{args.model}/interaction_shared/total_loss"
        ).mkdir(exist_ok=True, parents=True)
        plt.savefig(
            f"./figures/loss_curves/{args.data}/{args.model}/interaction_shared/total_loss/{args.modality}_MR_{args.weighting_method}_epoch_{args.train_epochs}.pdf",
            dpi=300,
        )

    plt.figure(figsize=(10, 6))  # Adjust figure size as needed

    for name, losses in plotting_interaction_losses.items():
        plt.plot(epochs, losses, label=name)

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    # plt.title("Training Interaction Loss Curves")
    plt.legend()
    plt.grid(True)
    if is_latent:
        Path(
            f"./figures/loss_curves/{args.data}/{args.model}/interaction_shared/interaction_loss"
        ).mkdir(exist_ok=True, parents=True)
        plt.savefig(
            f"./figures/loss_curves/{args.data}/{args.model}/interaction_shared/interaction_loss/{args.modality}_MR_{args.weighting_method}_epoch_{args.train_epochs}.pdf",
            dpi=300,
        )
    else:
        Path(
            f"./figures/loss_curves/{args.data}/{args.model}/interaction_shared/interaction_loss"
        ).mkdir(exist_ok=True, parents=True)
        plt.savefig(
            f"./figures/loss_curves/{args.data}/{args.model}/interaction_shared/interaction_loss/{args.modality}_MR_{args.weighting_method}_epoch_{args.train_epochs}.pdf",
            dpi=300,
        )

    plt.figure(figsize=(10, 6))  # Adjust figure size as needed

    return
