"""
ModelEvaluationCollection module
===============================

This module provides the `ModelEvaluationCollection` class, which encapsulates the
evaluation results of multiple models. It offers utility methods to compute and
aggregate evaluation metrics such as AUC loss and BEDROC scores across different models.
This class facilitates comparison and analysis of model performance in a structured
and efficient manner.

"""
from typing import Dict, Iterator, List, Tuple

import numpy as np
from genepriority import Evaluation


class ModelEvaluationCollection:
    """
    A collection of evaluation results for multiple models, with methods to compute
    and access aggregated metrics such as AUC loss and BEDROC scores.

    Attributes:
        model_results (Dict[str, Evaluation]): A dictionary where keys are model names
            and values are Evaluation objects containing metrics for the respective model.
    """

    def __init__(self, model_results: Dict[str, Evaluation]):
        """
        Initializes the ModelEvaluationCollection.

        Args:
            model_results (Dict[str, Evaluation]): A dictionary mapping model names
                to their respective Evaluation objects.
        """
        for key, val in model_results.items():
            if not isinstance(key, str):
                raise TypeError(
                    f"Invalid dictionary key: Expected `str`, but got {type(key)} for key "
                    f"`{key}`. Ensure all keys in `model_results` are strings."
                )
            if not isinstance(val, Evaluation):
                raise TypeError(
                    f"Invalid dictionary value for key `{key}`: Expected `Evaluation`, "
                    f"but got {type(val)}. Ensure all values in `model_results` are instances "
                    "of the `Evaluation` class."
                )
        self.model_results = model_results

    @property
    def model_names(self) -> List[str]:
        """
        Retrieves the names of the models in the collection.

        Returns:
            List[str]: A list of model names.
        """
        return list(self.model_results.keys())

    @property
    def evaluations(self) -> List[Evaluation]:
        """
        Retrieves the list of Evaluation objects in the collection.

        Returns:
            List[Evaluation]: A list of Evaluation objects corresponding to the models.
        """
        return list(self.model_results.values())

    def __len__(self) -> int:
        """Compute the number of evaluation objects.

        Returns:
            int: The number of evaluations.
        """
        return len(self.model_names)

    def items(self) -> Iterator[Tuple[str, Evaluation]]:
        """
        Provides an iterator over the model names and their corresponding Evaluation objects.

        Returns:
            Iterator[Tuple[str, Evaluation]]: An iterator of (model_name, Evaluation) pairs.
        """
        return self.model_results.items()

    def __iter__(self) -> Iterator[Evaluation]:
        """
        Allows the `ModelEvaluationCollection` object to be iterable over its Evaluation objects.

        Returns:
            Iterator[Evaluation]: An iterator over the Evaluation objects.
        """
        return iter(self.evaluations)

    def compute_auc(self) -> np.ndarray:
        """
        Calculates the AUC for each model and fold.

        Returns:
            np.ndarray: A 2D array containing the AUC for
                each model and for each fold. Shape: (fold, models).
        """
        auc = np.array([eval_res.compute_avg_auc() for eval_res in self.evaluations])
        return auc.T

    def compute_avg_precision(self) -> np.ndarray:
        """
        Calculates the Avgerage Precision for each model and fold.

        Returns:
            np.ndarray: A 2D array containing the Average Precision for
                each model and for each fold. Shape: (fold, models).
        """
        avg_pr = np.array(
            [eval_res.compute_avg_precision() for eval_res in self.evaluations]
        )
        return avg_pr.T

    def compute_bedroc_scores(self) -> np.ndarray:
        """
        Calculates the BEDROC scores for several alpah values per fold and
        for each model.

        Returns:
            np.ndarray: A 3D array containing the BEDROC scores for each fold,
            across different alpha values, for each model. Shape: (alphas, fold, models).
        """
        bedroc = np.array(
            [eval_res.compute_bedroc_scores() for eval_res in self.evaluations]
        )  # shape = (models, alphas, fold)
        # reorder to (alphas, fold, models)
        bedroc = bedroc.transpose(1, 2, 0)
        return bedroc

    def compute_roc(self) -> List[np.ndarray]:
        """
        Calculates the ROC curve for each model.

        Returns:
            List[np.ndarray]: 2D arrays containing the ROC for
                each model. Shape: (2, n_thresholds).
        """
        roc = [eval_res.compute_avg_roc_curve() for eval_res in self.evaluations]
        return roc

    def compute_pr(self) -> List[np.ndarray]:
        """
        Calculates the PR curve for each model.

        Returns:
            List[np.ndarray]: 2D arrays containing the PR for
                each model. Shape: (2, n_thresholds).
        """
        pr = [eval_res.compute_avg_pr_curve() for eval_res in self.evaluations]
        return pr
