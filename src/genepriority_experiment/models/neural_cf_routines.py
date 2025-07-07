from typing import Any, Dict

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


class GeneDiseaseDataset(Dataset):
    """
    PyTorch Dataset for gene–disease association pairs.

    Each sample consists of:
        - gene index (int)
        - disease index (int)
        - gene side features (FloatTensor)
        - disease side features (FloatTensor)
        - label (FloatTensor)

    Attributes:
        idx_pairs (np.ndarray): Array of shape (N, 2) containing (gene_idx, disease_idx) pairs.
        labels (np.ndarray): Array of shape (N,) with association labels for each pair.
        gene_feats (torch.FloatTensor): Tensor of shape (G, F) holding gene features.
        disease_feats (torch.FloatTensor): Tensor of shape (D, F') holding disease features.
    """

    def __init__(
        self,
        idx_pairs: np.ndarray,
        M: np.ndarray,
        gene_feats: np.ndarray,
        disease_feats: np.ndarray,
    ) -> None:
        """
        Initialize the dataset.

        Args:
            idx_pairs (np.ndarray): Array of shape (N, 2) with (gene_idx, disease_idx) pairs.
            M (np.ndarray): Full GxD association matrix from which labels are extracted.
            gene_feats (np.ndarray): Array of shape (G, F) with gene side-information.
            disease_feats (np.ndarray): Array of shape (D, F') with disease side-information.
        Returns:
            None
        """
        self.idx_pairs: np.ndarray = idx_pairs
        self.labels: np.ndarray = M[idx_pairs[:, 0], idx_pairs[:, 1]].astype(np.float32)
        self.gene_feats: torch.FloatTensor = torch.FloatTensor(gene_feats)
        self.disease_feats: torch.FloatTensor = torch.FloatTensor(disease_feats)

    def __len__(self) -> int:
        """
        Get dataset size.

        Returns:
            int: Number of (gene, disease) pairs in the dataset.
        """
        return len(self.idx_pairs)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        """
        Retrieve a single sample.

        Args:
            i (int): Index of the sample (0 <= i < __len__()).

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing:
                'gene' (torch.LongTensor): Scalar gene index.
                'disease' (torch.LongTensor): Scalar disease index.
                'g_feat' (torch.FloatTensor): Gene feature vector of shape (F,).
                'd_feat' (torch.FloatTensor): Disease feature vector of shape (F',).
                'label' (torch.FloatTensor): Association label scalar.
        """
        g, d = self.idx_pairs[i]
        return {
            "gene": torch.tensor(g, dtype=torch.long),
            "disease": torch.tensor(d, dtype=torch.long),
            "g_feat": self.gene_feats[g],
            "d_feat": self.disease_feats[d],
            "label": torch.tensor(self.labels[i], dtype=torch.float32),
        }


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """
    Run one epoch of training.

    Args:
        model: NeuralCF model.
        loader: DataLoader for training set.
        criterion: Loss function (e.g., nn.MSELoss()).
        optimizer: Optimizer (e.g., Adam).
        device: Computation device.

    Returns:
        Average training loss over the epoch.
    """
    model.train()
    total_loss = 0.0
    for batch in loader:
        optimizer.zero_grad()
        g = batch["gene"].to(device)
        d = batch["disease"].to(device)
        gf = batch["g_feat"].to(device)
        df = batch["d_feat"].to(device)
        y = batch["label"].to(device)

        pred = model(g, d, gf, df)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()

        batch_loss = loss.item()
        total_loss += batch_loss * g.size(0)
    return total_loss / len(loader.dataset)


def validate_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """
    Run one epoch of validation.

    Args:
        model: NeuralCF model.
        loader: DataLoader for validation set.
        criterion: Loss function.
        device: Computation device.

    Returns:
        Average validation loss over the epoch.
    """
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            g = batch["gene"].to(device)
            d = batch["disease"].to(device)
            gf = batch["g_feat"].to(device)
            df = batch["d_feat"].to(device)
            y = batch["label"].to(device)

            pred = model(g, d, gf, df)
            loss = criterion(pred, y)

            batch_loss = loss.item()
            total_loss += batch_loss * g.size(0)
    return total_loss / len(loader.dataset)


def predict_full_matrix(
    model: nn.Module,
    G: int,
    D: int,
    gene_feats: np.ndarray,
    disease_feats: np.ndarray,
    device: torch.device,
    batch_size: int = 1024,
) -> np.ndarray:
    """
    Predict the full gene–disease association matrix.

    Args:
        model: Trained NeuralCF model.
        G: Number of genes.
        D: Number of diseases.
        gene_feats: Array (G, F) of gene features.
        disease_feats: Array (D, F') of disease features.
        device: Computation device.
        batch_size: Batch size for scanning genes.

    Returns:
        preds: NumPy array of shape (G, D) with predicted scores.
    """
    model.eval()
    preds = np.zeros((G, D), dtype=np.float32)
    with torch.no_grad():
        for g_start in range(0, G, batch_size):
            g_end = min(G, g_start + batch_size)
            g_idx = torch.arange(g_start, g_end, device=device)
            gf = torch.tensor(
                gene_feats[g_start:g_end], dtype=torch.float32, device=device
            )
            d_idx = (
                torch.arange(D, device=device).unsqueeze(0).repeat(g_end - g_start, 1)
            )
            gf_full = gf.unsqueeze(1).repeat(1, D, 1).view(-1, gf.shape[1])
            df_full = torch.tensor(disease_feats, dtype=torch.float32, device=device)
            df_full = (
                df_full.unsqueeze(0)
                .repeat(g_end - g_start, 1, 1)
                .view(-1, disease_feats.shape[1])
            )
            g_idx_full = g_idx.unsqueeze(1).repeat(1, D).view(-1)
            out = model(g_idx_full, d_idx.view(-1), gf_full, df_full)
            preds[g_start:g_end] = out.view(g_end - g_start, D).cpu().numpy()
    return preds
