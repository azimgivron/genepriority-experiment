"""
Dataframes module
=================

Post-processing producing dataframes for summarizing evaluation metrics and results.
"""
from typing import Dict, List

import numpy as np
import pandas as pd


def generate_auc_loss_table(
    auc_loss: np.ndarray,
    model_names: List[str],
    avg_auc_loss_name: str = "Averaged 1-AUC error",
    std_auc_loss_name: str = "Std 1-AUC error",
) -> pd.DataFrame:
    """
    Generates a table summarizing AUC loss averages and standard
    deviations for each model.

    Args:
        auc_loss (np.ndarray): A 2D array containing the AUC loss for
            each model and fold. Shape: (fold, models).
        model_names (List[str]): Names of the models corresponding to the AUC losses.
        avg_auc_loss_name (str, optional): Column name for averaged 1-AUC error.
            Defaults to "Averaged 1-AUC error".
        std_auc_loss_name (str, optional): Column name for standard deviation of 1-AUC error.
            Defaults to "Std 1-AUC error".

    Returns:
        pd.DataFrame: A dataframe summarizing AUC loss metrics.
    """
    auc_loss = np.hstack((np.mean(auc_loss, axis=0), np.std(auc_loss, axis=0)))
    auc_loss = auc_loss.reshape((2, -1)).T
    dataframe = pd.DataFrame(
        auc_loss, columns=[avg_auc_loss_name, std_auc_loss_name], index=model_names
    ).map(lambda x: f"{x:.2e}")
    return dataframe


def generate_bedroc_table(
    bedroc: np.ndarray,
    model_names: List[str],
    alpha_map: Dict[float, str],
    avg_bedroc_score_name: str = "Averaged BEDROC score",
    std_bedroc_score_name: str = "Std BEDROC score",
) -> pd.DataFrame:
    """
    Generates a table summarizing the averaged BEDROC scores and their
    standard deviations for each model across specified alpha values.

    Args:
        bedroc (np.ndarray): A 3D array containing BEDROC scores for each model,
            each fold and different alpha values.
            Shape: (alphas, fold, models).
        model_names (List[str]): Names of the models corresponding to the BEDROC scores.
        alpha_map (Dict[float, str]): A mapping of alpha values (e.g., 0.2, 0.5) to
            descriptive strings used for table column naming.
        avg_bedroc_score_name (str, optional): Column name prefix for averaged BEDROC scores.
            Defaults to "Averaged BEDROC score".
        std_bedroc_score_name (str, optional): Column name prefix for standard deviation
            of BEDROC scores. Defaults to "Std BEDROC score".

    Returns:
        pd.DataFrame: A DataFrame summarizing the averaged BEDROC scores and their
        standard deviations for each model across different alpha values. Columns
        are dynamically generated based on the alpha values.
    """
    if len(alpha_map) != bedroc.shape[0]:
        raise ValueError

    mean = bedroc.mean(axis=1).T  # shape = (model, alphas)
    std = bedroc.std(axis=1).T  # shape = (model, alphas)
    bedroc = np.hstack((mean, std))  # shape = (model, 2*alphas)

    column_names = np.array(
        [f"{avg_bedroc_score_name} (top {alpha})" for alpha in alpha_map.values()]
        + [f"{std_bedroc_score_name} (top {alpha})" for alpha in alpha_map.values()]
    )

    dataframe = pd.DataFrame(bedroc, columns=column_names, index=model_names).map(
        lambda x: f"{x:.2e}"
    )

    return dataframe
