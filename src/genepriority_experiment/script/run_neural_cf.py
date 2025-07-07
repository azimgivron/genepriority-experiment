import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
import logging
from genepriority import Results, Evaluation
from genepriority.utils import serialize
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
from tqdm import tqdm
from typing import Dict

from genepriority_experiment.models.neural_cf import NeuralCF
from genepriority_experiment.models.neural_cf_routines import (
    GeneDiseaseDataset,
    predict_full_matrix,
    train_epoch,
    validate_epoch,
)


def run_fold(fold: int, args: argparse.Namespace, data: Dict[str, np.ndarray]) -> None:
    """
    Train and evaluate NeuralCF for a single fold.

    Args:
        fold: Fold index (0-based).
        args: Parsed command-line arguments.
    """
    # Device & TensorBoard
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fold_log_dir: Path = args.log_dir / f"fold{fold+1}-NeuralCF"
    if fold_log_dir.exists():
        for file in fold_log_dir.iterdir():
            file.unlink()
    fold_log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=fold_log_dir)

    # Load data once
    gene_disease = data["gene_disease"]
    # assume masks shape = (5, G, D)
    mask_train_all = data["mask_train"]
    mask_val_all = data["mask_val"]
    mask_test_all = data["mask_test"]
    gene_feats = data["gene_feats"]
    disease_feats = data["disease_feats"]

    # Select masks for this fold
    mask_train = mask_train_all[fold]
    mask_val = mask_val_all[fold]
    mask_test = mask_test_all[fold]

    # Create index arrays
    train_idx = np.vstack(np.where(mask_train)).T
    val_idx = np.vstack(np.where(mask_val)).T

    # Feature scaling
    scaler_g = StandardScaler().fit(gene_feats)
    gene_feats_scaled = scaler_g.transform(gene_feats)
    scaler_d = StandardScaler().fit(disease_feats)
    disease_feats_scaled = scaler_d.transform(disease_feats)

    # DataLoaders
    train_ds = GeneDiseaseDataset(
        train_idx, gene_disease, gene_feats_scaled, disease_feats_scaled
    )
    val_ds = GeneDiseaseDataset(
        val_idx, gene_disease, gene_feats_scaled, disease_feats_scaled
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    # Instantiate model per fold
    num_genes, num_diseases = gene_disease.shape
    model = NeuralCF(
        num_genes=num_genes,
        num_diseases=num_diseases,
        embedding_dim=args.embedding_dim,
        gene_feat_dim=gene_feats.shape[1],
        disease_feat_dim=disease_feats.shape[1],
        hidden_dims=args.hidden_dims,
        dropout=args.dropout
    ).to(device)
    
    # Summary
    batch = next(iter(train_loader))
    g = batch["gene"].to(device)
    d = batch["disease"].to(device)
    gf = batch["g_feat"].to(device)
    df = batch["d_feat"].to(device)
    model_summary = summary(
        model,
        input_data=(g, d, gf, df),
        col_names=("output_size", "num_params", "trainable")
    )
    writer.add_text("hyperparameters", str(model_summary))
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    criterion = nn.MSELoss()

    if args.load_model is not None:
        checkpoint = torch.load(args.load_model, map_location=device)
        model.load_state_dict(checkpoint)
        logging.info(f"Loaded model weights from {args.load_model}")

    # Training loop
    best_val = float('inf')
    no_improve = 0
    patience = args.patience
    best_state: Dict[str, torch.Tensor] = {}
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate_epoch(model, val_loader, criterion, device)
        writer.add_scalar("training_loss", np.sqrt(train_loss), epoch)
        writer.add_scalar("testing_loss", np.sqrt(val_loss), epoch)
        
        # Early-stopping logic
        if val_loss < best_val:
            best_val = val_loss
            no_improve = 0
            # Save a copy of the current best weights
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1

        if no_improve >= patience:
            logging.info(
                f"No improvement for {patience} epochs (stopping at epoch {epoch})."
            )
            break

    writer.close()
    if best_state:
        model.load_state_dict(best_state)
        logging.info("Model weights rolled back to best epoch (val_loss = %.4f).", best_val)

    if args.save_model is not None:
        base_path = args.save_model
        numbered_path = base_path.parent / f"{fold}:{base_path.name}"
        torch.save(model.state_dict(), numbered_path)
        logging.info(f"Saved trained model weights to {numbered_path}")

    # Full-matrix prediction
    y_pred = predict_full_matrix(
        model,
        num_genes,
        num_diseases,
        gene_feats_scaled,
        disease_feats_scaled,
        device,
        batch_size=args.batch_size,
    )
    return Results(gene_disease, y_pred, mask_test)


def main(args: argparse.Namespace) -> None:
    """
    Run cross-validation over folds.

    Args:
        args: Parsed command-line arguments.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    data = np.load(args.data_path)
    n_folds = data["mask_train"].shape[0]
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for fold in tqdm(range(n_folds), desc="Folds", unit="fold"):
        results.append(run_fold(fold, args, data))
    evaluation = Evaluation(results)
    serialize(evaluation, output_dir / "results.pickle")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="NCF gene–disease prioritization with cross-validation"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Base output directory for results.",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("/home/TheGreatestCoder/code/experiments/data/data.npz"),
        help="npz file with arrays gene_disease, mask_train, mask_val, mask_test, gene_feats, disease_feats (default: %(default)s).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="The batch size (default: %(default)s).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="The learning rate (default: %(default)s).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1000,
        help="The number of epochs (default: %(default)s).",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=100,
        help="The patience for early stopping (default: %(default)s).",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=.3,
        help="The dropout probability (default: %(default)s).",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("/home/TheGreatestCoder/code/logs"),
        help="Base TensorBoard log directory (default: %(default)s).",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=128,
        help="Embedding dimension (default: %(default)s).",
    )
    parser.add_argument(
        "--hidden-dims",
        type=int,
        nargs="+",
        default=[128, 64, 32],
        help="List of MLP hidden layer sizes (default: %(default)s).",
    )
    parser.add_argument(
        "--weight-decay", type=float, default=.01,
        help="Weight decay (L2 penalty) for the optimizer (default: %(default)s)."
    )
    parser.add_argument(
        "--load-model",
        type=Path,
        default=None,
        help="Path to a .pt/.pth checkpoint to load before training.",
    )
    parser.add_argument(
        "--save-model",
        type=Path,
        default=None,
        help="Path where to save the trained model state_dict.",
    )

    args = parser.parse_args()
    main(args)
