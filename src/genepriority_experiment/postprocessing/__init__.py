"""Postprocessing Module"""
from .dataframes import generate_auc_loss_table, generate_bedroc_table
from .figures import (
    plot_auc_boxplots,
    plot_avg_precision_boxplots,
    plot_bedroc_boxplots,
)
from .model_evaluation_collection import ModelEvaluationCollection


def plot_roc_curves():
    return


__all__ = [
    "generate_auc_loss_table",
    "generate_bedroc_table",
    "plot_auc_boxplots",
    "plot_bedroc_boxplots",
    "plot_avg_precision_boxplots",
    "ModelEvaluationCollection",
]
