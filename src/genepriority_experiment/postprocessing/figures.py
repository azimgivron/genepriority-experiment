# pylint: disable=R0913
"""
Figures module
==============

This module provides post-processing functions to generate and save visualizations 
of evaluation metrics such as ROC curves and BEDROC boxplots. These visualizations 
help in comparing model performance across different configurations, splits, and metrics.
"""
from typing import List, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from genepriority import Evaluation

# Okabe-Ito color palette
COLORS = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#F0E442"]
# Line styles for multiple models
LINESTYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 1))]


def plot_bedroc_boxplots(
    bedroc: np.ndarray,
    model_names: List[str],
    output_file: str,
    figsize: Tuple[int, int],
    sharey: bool,
):
    """
    Plots boxplots of BEDROC scores for multiple alpha values and latent dimensions
    without plotting outliers, with a shared y-axis and a single legend on the side.

    Args:
        bedroc (np.ndarray): BEDROC scores array of shape (alphas, diseases, models).
        model_names (List[str]): Names of the models being compared.
        output_file (str): File path where the BEDROC boxplot figure will be saved.
        figsize (Tuple[int, int]): Figure size in inches (width, height).
        sharey (bool): Whether to share the y axis.
    """
    if bedroc.shape[-1] > len(COLORS):
        raise ValueError("Not enough colors.")

    n_alphas = len(Evaluation.alphas)
    fig, axs = plt.subplots(1, n_alphas, figsize=figsize, sharey=sharey)
    if n_alphas == 1:
        axs = [axs]

    for i, alpha in enumerate(Evaluation.alphas):
        box = sns.boxplot(
            data=bedroc[i],
            ax=axs[i],
            palette=COLORS[: bedroc.shape[-1]],
            showfliers=False,
        )
        axs[i].set_xticks(range(len(model_names)))
        axs[i].set_xticklabels(["" for _ in model_names])
        axs[i].yaxis.set_tick_params(labelsize=14)
        axs[i].set_title(
            f"$\\alpha={float(alpha):.1f}$\nTop {Evaluation.alpha_map[alpha]}",
            fontsize=16,
            weight="bold",
        )
        axs[i].grid(axis="y", alpha=0.3)
        if box.legend_ is not None:
            box.legend_.remove()

    axs[0].set_ylabel("BEDROC", fontsize=16)
    fig.subplots_adjust(bottom=0.15)
    handles = [mpatches.Patch(color=c, label=m) for c, m in zip(COLORS, model_names)]
    fig.legend(
        handles,
        model_names,
        loc="lower center",
        bbox_to_anchor=(0.5, 0),
        ncol=4,
        fontsize=16,
    )
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()


def plot_auc_boxplots(
    auc: np.ndarray,
    model_names: List[str],
    output_file: str,
    figsize: Tuple[int, int],
):
    """
    Plots boxplots of AUC scores for latent dimensions
    without plotting outliers and a single legend on the side.

    Args:
        auc (np.ndarray): A 2D array containing the AUC for
            each model and disease. Shape: (diseases, models).
        model_names (List[str]): Names of the models being compared.
        output_file (str): File path where the AUC boxplot figure will be saved.
        figsize (Tuple[int, int]): Figure size in inches (width, height).
    """
    if auc.shape[-1] > len(COLORS):
        raise ValueError("Not enough colors.")

    fig, axis = plt.subplots(1, 1, figsize=figsize)
    _ = sns.boxplot(
        data=auc,
        ax=axis,
        palette=COLORS[: auc.shape[-1]],
        showfliers=False,
    )
    axis.set_ylabel("AUROC", fontsize=16)
    axis.set_xticks(range(len(model_names)))
    axis.set_xticklabels(["" for _ in model_names])
    axis.yaxis.set_tick_params(labelsize=14)
    axis.grid(axis="y", alpha=0.3)
    fig.subplots_adjust(bottom=0.15)
    handles = [mpatches.Patch(color=c, label=m) for c, m in zip(COLORS, model_names)]
    fig.legend(
        handles,
        model_names,
        loc="lower center",
        bbox_to_anchor=(0.5, 0),
        ncol=2,
        fontsize=16,
    )
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()


def plot_avg_precision_boxplots(
    avg_pr: np.ndarray,
    model_names: List[str],
    output_file: str,
    figsize: Tuple[int, int],
):
    """
    Plots boxplots of Average Precision scores for latent dimensions
    without plotting outliers and a single legend on the side.

    Args:
        avg_pr (np.ndarray): A 2D array containing the Average Precision for
            each model and disease. Shape: (diseases, models).
        model_names (List[str]): Names of the models being compared.
        output_file (str): File path where the Average Precision boxplot figure will be saved.
        figsize (Tuple[int, int]): Figure size in inches (width, height).
    """
    if avg_pr.shape[-1] > len(COLORS):
        raise ValueError("Not enough colors.")

    fig, axis = plt.subplots(1, 1, figsize=figsize)
    _ = sns.boxplot(
        data=avg_pr,
        ax=axis,
        palette=COLORS[: avg_pr.shape[-1]],
        showfliers=False,
    )
    axis.set_xticks(range(len(model_names)))
    axis.set_xticklabels(["" for _ in model_names])
    axis.yaxis.set_tick_params(labelsize=14)
    axis.grid(axis="y", alpha=0.3)
    axis.set_ylabel("AUPRC", fontsize=16)
    fig.subplots_adjust(bottom=0.15)
    handles = [mpatches.Patch(color=c, label=m) for c, m in zip(COLORS, model_names)]
    fig.legend(
        handles,
        model_names,
        loc="lower center",
        bbox_to_anchor=(0.5, 0),
        ncol=2,
    )
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()


def plot_roc_curves(
    roc: List[np.ndarray],
    model_names: List[str],
    output_file: str,
    figsize: Tuple[int, int],
):
    """
    Plots ROC curves for multiple models with distinct linestyles.

    Args:
        roc (List[np.ndarray]): FPR-TPR values, shape (2, n_points), for each model.
        model_names (List[str]): Names of the models being compared.
        output_file (str): File path where the ROC curve figure will be saved.
        figsize (Tuple[int, int]): Figure size in inches (width, height).
    """
    if len(model_names) > len(COLORS):
        raise ValueError("Not enough colors for the number of models.")

    _, ax = plt.subplots(1, 1, figsize=figsize)
    for i, name in enumerate(model_names):
        color = COLORS[i]
        ls = LINESTYLES[i % len(LINESTYLES)]
        ax.plot(
            roc[i][0], roc[i][1], linestyle=ls, color=color, linewidth=2, label=name
        )
    ax.set_xlabel("False Positive Rate", fontsize=14)
    ax.set_ylabel("True Positive Rate", fontsize=14)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=16)
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()


def plot_pr_curves(
    pr: List[np.ndarray],
    model_names: List[str],
    output_file: str,
    figsize: Tuple[int, int],
):
    """
    Plots Precision-Recall curves for multiple models with distinct linestyles.

    Args:
        pr (List[np.ndarray]): Precision-Recall values, shape (2, n_points), for each model.
        model_names (List[str]): Names of the models being compared.
        output_file (str): File path where the PR curve figure will be saved.
        figsize (Tuple[int, int]): Figure size in inches (width, height).
    """
    if len(model_names) > len(COLORS):
        raise ValueError("Not enough colors for the number of models.")

    _, ax = plt.subplots(1, 1, figsize=figsize)
    for i, name in enumerate(model_names):
        color = COLORS[i]
        ls = LINESTYLES[i % len(LINESTYLES)]
        ax.plot(pr[i][1], pr[i][0], linestyle=ls, color=color, linewidth=2, label=name)
    ax.set_xlabel("Recall", fontsize=14)
    ax.set_ylabel("Precision", fontsize=14)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=16)
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()
