"""
TGAT (Temporal Graph Attention) for IBM Credit Card Fraud Detection
Final corrected version:
 - Uses your TGAT.py API (time_enc_dim, n_heads, dropout)
 - Calls build_batch_sequences_from_subgraphs(..., max_neighbors_per_hop=k, L=L)
 - Unpacks 5 return vals and ignores last (_)
 - Handles zero-neighbor batches in training, validation, and evaluation
 - Avoids duplicate optimizer parameters
"""

import os
import random
import time
from typing import List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

# Import components from your TGAT.py
from TGAT import TimeEncoder, NeighborFinder, TGATModel, build_batch_sequences_from_subgraphs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TGATFraudClassifier(nn.Module):
    """
    TGAT-based fraud classifier. Wraps TGAT embedding model and an MLP classifier.
    """
    def __init__(self, tgat_model: TGATModel, embed_dim: int):
        super().__init__()
        self.tgat = tgat_model
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2)
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.classifier(z)


# -------------------------
# Data loader / conversion
# -------------------------
def load_ibm_temporal_data(csv_path: str):
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df):,} transactions")

    # Normalize Amount
    if 'Amount' in df.columns:
        df['Amount'] = df['Amount'].replace(r'[\$,]', '', regex=True).astype(float)

    # Map fraud
    if 'Is Fraud?' in df.columns:
        df['Fraud_Label'] = df['Is Fraud?'].map({'No': 0, 'Yes': 1})
    else:
        raise ValueError("CSV must contain 'Is Fraud?' column")

    df['transaction_id'] = np.arange(len(df)).astype(int)

    # Construct datetimes (use Time if available)
    if 'Time' in df.columns:
        def make_datetime(row):
            try:
                date_str = f"{int(row['Year']):04d}-{int(row['Month']):02d}-{int(row['Day']):02d} {row['Time']}"
                return pd.to_datetime(date_str, format='%Y-%m-%d %H:%M', errors='coerce')
            except Exception:
                return pd.NaT
        df['datetime'] = df.apply(make_datetime, axis=1)
    else:
        df['datetime'] = pd.to_datetime(df[['Year', 'Month', 'Day']].astype(int).astype(str).agg('-'.join, axis=1),
                                        format='%Y-%m-%d', errors='coerce')

    n_before = len(df)
    df = df.dropna(subset=['datetime']).reset_index(drop=True)
    if len(df) < n_before:
        print(f"Dropped {n_before - len(df)} rows due to invalid timestamps")

    # Normalize timestamps to [0,1]
    df['timestamp'] = (df['datetime'] - df['datetime'].min()).dt.total_seconds().astype(float)
    max_ts = df['timestamp'].max()
    if max_ts <= 0:
        raise ValueError("Timestamp max <= 0, check your dates")
    df['timestamp'] = df['timestamp'] / max_ts

    # Sort and reassign ids
    df = df.sort_values('timestamp').reset_index(drop=True)
    df['transaction_id'] = np.arange(len(df)).astype(int)
    num_nodes = len(df)

    # Build temporal edges (User and Card)
    events: List[Tuple[int, int, float, Optional[torch.Tensor]]] = []
    print("Building temporal edges (User and Card)...")
    for col in ['User', 'Card']:
        if col not in df.columns:
            continue
        for _, group in df.groupby(col):
            if len(group) <= 1:
                continue
            group = group.sort_values('timestamp')
            idxs = group['transaction_id'].values
            tss = group['timestamp'].values
            amounts = group['Amount'].values if 'Amount' in group.columns else np.zeros(len(group))
            mccs = group['MCC'].values if 'MCC' in group.columns else np.zeros(len(group))
            for i in range(len(idxs) - 1):
                src = int(idxs[i])
                dst = int(idxs[i + 1])
                ts = float(tss[i + 1])
                edge_feat = torch.tensor([
                    float(amounts[i + 1] - amounts[i]),
                    float(tss[i + 1] - tss[i]),
                    float(mccs[i + 1]) if len(mccs) > 0 else 0.0
                ], dtype=torch.float)
                events.append((src, dst, ts, edge_feat))

    events.sort(key=lambda x: x[2])
    print(f"Created {len(events):,} temporal edges")

    labels = df['Fraud_Label'].astype(int).values
    fraud_count = int(labels.sum())
    print(f"Fraud transactions: {fraud_count:,} ({fraud_count/len(labels)*100:.2f}%)")

    node_timestamp = df.set_index('transaction_id')['timestamp'].to_dict()
    return events, labels, num_nodes, node_timestamp, df


# -------------------------
# Helper to fix zero-neighbor batches
# -------------------------
def ensure_nonzero_neighbors(neigh_feats: torch.Tensor, neigh_time_enc: torch.Tensor, key_padding_mask: torch.Tensor,
                             node_features: nn.Embedding, time_encoder: TimeEncoder):
    """
    If neigh_feats has second dim == 0 (Ltot==0), replace with shape (B,1,feat) and mark it padded.
    Returns possibly-modified (neigh_feats, neigh_time_enc, key_padding_mask).
    """
    if neigh_feats.shape[1] != 0:
        return neigh_feats, neigh_time_enc, key_padding_mask

    B = neigh_feats.shape[0]
    feat_dim = node_features.weight.shape[1]
    # time_enc dim: neigh_time_enc expected shape (B, L, 2*time_encoder.time_dim)
    time_enc_dim = 2 * time_encoder.time_dim if hasattr(time_encoder, 'time_dim') else (neigh_time_enc.shape[2] if neigh_time_enc.dim() >= 3 else 2 * time_encoder.time_dim)
    neigh_feats = torch.zeros(B, 1, feat_dim)
    neigh_time_enc = torch.zeros(B, 1, time_enc_dim)
    key_padding_mask = torch.ones(B, 1, dtype=torch.bool)
    return neigh_feats, neigh_time_enc, key_padding_mask


# -------------------------
# Evaluation function (test)
# -------------------------
def evaluate_tgat(classifier: TGATFraudClassifier, tgat_model: TGATModel, time_encoder: TimeEncoder,
                  node_features: nn.Embedding, neigh_finder: NeighborFinder, node_timestamp: dict,
                  labels: np.ndarray, test_mask: np.ndarray,
                  batch_size: int = 256, k: int = 10, L: int = 2):
    classifier.eval()
    tgat_model.eval()
    time_encoder.eval()

    test_nodes = np.where(test_mask)[0].tolist()
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for i in range(0, len(test_nodes), batch_size):
            batch_nodes = test_nodes[i:i + batch_size]
            batch_times = [node_timestamp[n] for n in batch_nodes]
            subgraphs = neigh_finder.get_recursive_subgraph(batch_nodes, batch_times, L, k)

            tgt_feats, neigh_feats, neigh_time_enc, key_padding_mask, _ = build_batch_sequences_from_subgraphs(
                subgraphs, node_features.weight.cpu(), time_encoder, max_neighbors_per_hop=k, L=L
            )

            # fix zero-neighbor batches
            neigh_feats, neigh_time_enc, key_padding_mask = ensure_nonzero_neighbors(
                neigh_feats, neigh_time_enc, key_padding_mask, node_features, time_encoder
            )

            tgt_feats = tgt_feats.to(device)
            neigh_feats = neigh_feats.to(device)
            neigh_time_enc = neigh_time_enc.to(device)
            key_padding_mask = key_padding_mask.to(device)

            z = tgat_model(tgt_feats, neigh_feats, neigh_time_enc, key_padding_mask)  # (B, embed_dim)
            logits = classifier(z)
            probs = F.softmax(logits, dim=1)
            preds = logits.argmax(dim=1).cpu().numpy()
            probs_fraud = probs[:, 1].cpu().numpy()

            all_preds.extend(preds.tolist())
            all_probs.extend(probs_fraud.tolist())

    test_nodes_list = test_nodes
    test_labels = [labels[n] for n in test_nodes_list]
    acc = accuracy_score(test_labels, all_preds)
    precision = precision_score(test_labels, all_preds, zero_division=0)
    recall = recall_score(test_labels, all_preds, zero_division=0)
    f1 = f1_score(test_labels, all_preds, zero_division=0)
    try:
        roc_auc = roc_auc_score(test_labels, all_probs) if len(np.unique(test_labels)) > 1 else 0.0
    except ValueError:
        roc_auc = 0.0

    print(f"\nTGAT TEST SET EVALUATION")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}\n")

    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc
    }


# -------------------------
# Training
# -------------------------
def train_tgat_fraud(events: List[Tuple[int, int, float, Optional[torch.Tensor]]],
                     labels: np.ndarray,
                     node_timestamp: dict,
                     num_nodes: int,
                     train_mask: np.ndarray,
                     val_mask: np.ndarray,
                     epochs: int = 30,
                     batch_size: int = 256,
                     lr: float = 1e-3,
                     node_feat_dim: int = 64,
                     time_dim: int = 16,
                     embed_dim: int = 64,
                     k: int = 10,
                     L: int = 2):
    # Node features
    node_features = nn.Embedding(num_nodes, node_feat_dim).to(device)
    nn.init.xavier_uniform_(node_features.weight)

    # TGAT modules
    time_encoder = TimeEncoder(time_dim=time_dim).to(device)
    tgat_model = TGATModel(node_feat_dim=node_feat_dim, time_enc_dim=time_dim, embed_dim=embed_dim,
                           n_layers=2, n_heads=2, dropout=0.1).to(device)

    # classifier contains TGAT as submodule
    classifier = TGATFraudClassifier(tgat_model, embed_dim=embed_dim).to(device)

    # NeighborFinder
    neigh_finder = NeighborFinder(num_nodes=num_nodes, max_neighbors=2000)
    neigh_finder.num_nodes = num_nodes

    # Prepopulate neighborfinder by streaming events
    print("Processing temporal events to populate temporal adjacency (NeighborFinder)...")
    for i, (src, dst, ts, ef) in enumerate(events):
        if i % 10000 == 0:
            print(f"  Processed {i:,}/{len(events):,} events")
        neigh_finder.insert_edge(src, dst, ts, ef)
    print("NeighborFinder populated.")

    # Optimizer: classifier.parameters() already includes tgat_model
    params = list(classifier.parameters()) + list(node_features.parameters()) + list(time_encoder.parameters())
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()

    train_nodes = np.where(train_mask)[0].tolist()
    val_nodes = np.where(val_mask)[0].tolist()

    best_val_acc = 0.0
    best_state = None
    start_time = time.time()

    print("Starting supervised training...")
    for epoch in range(1, epochs + 1):
        classifier.train(); tgat_model.train(); time_encoder.train()
        random.shuffle(train_nodes)
        epoch_loss = 0.0
        steps = 0

        for i in range(0, len(train_nodes), batch_size):
            batch_nodes = train_nodes[i:i + batch_size]
            batch_labels = torch.tensor([labels[n] for n in batch_nodes], dtype=torch.long, device=device)

            batch_times = [node_timestamp[n] for n in batch_nodes]
            subgraphs = neigh_finder.get_recursive_subgraph(batch_nodes, batch_times, L, k)

            tgt_feats, neigh_feats, neigh_time_enc, key_padding_mask, _ = build_batch_sequences_from_subgraphs(
                subgraphs, node_features.weight.cpu(), time_encoder, max_neighbors_per_hop=k, L=L
            )

            # fix zero-neighbor batches
            neigh_feats, neigh_time_enc, key_padding_mask = ensure_nonzero_neighbors(
                neigh_feats, neigh_time_enc, key_padding_mask, node_features, time_encoder
            )

            tgt_feats = tgt_feats.to(device)
            neigh_feats = neigh_feats.to(device)
            neigh_time_enc = neigh_time_enc.to(device)
            key_padding_mask = key_padding_mask.to(device)

            # Compute embedding and classify
            z = tgat_model(tgt_feats, neigh_feats, neigh_time_enc, key_padding_mask)  # (B, embed_dim)

            optimizer.zero_grad()
            logits = classifier(z)
            loss = criterion(logits, batch_labels)

            if torch.isnan(loss):
                print(f"Warning: NaN loss at epoch {epoch}, batch starting {i}. Skipping batch.")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            steps += 1

        avg_loss = epoch_loss / max(1, steps)
        print(f"Epoch {epoch}/{epochs} | Avg Loss: {avg_loss:.4f}")

        # Validation
        classifier.eval(); tgat_model.eval(); time_encoder.eval()
        val_preds = []; val_probs = []
        for i in range(0, len(val_nodes), batch_size):
            batch_nodes = val_nodes[i:i + batch_size]
            batch_times = [node_timestamp[n] for n in batch_nodes]
            subgraphs = neigh_finder.get_recursive_subgraph(batch_nodes, batch_times, L, k)

            tgt_feats, neigh_feats, neigh_time_enc, key_padding_mask, _ = build_batch_sequences_from_subgraphs(
                subgraphs, node_features.weight.cpu(), time_encoder, max_neighbors_per_hop=k, L=L
            )

            # fix zero-neighbor batches in validation
            neigh_feats, neigh_time_enc, key_padding_mask = ensure_nonzero_neighbors(
                neigh_feats, neigh_time_enc, key_padding_mask, node_features, time_encoder
            )

            tgt_feats = tgt_feats.to(device)
            neigh_feats = neigh_feats.to(device)
            neigh_time_enc = neigh_time_enc.to(device)
            key_padding_mask = key_padding_mask.to(device)

            with torch.no_grad():
                z = tgat_model(tgt_feats, neigh_feats, neigh_time_enc, key_padding_mask)
                logits = classifier(z)
                probs = F.softmax(logits, dim=1)
                preds = logits.argmax(dim=1).cpu().numpy()
                probs_fraud = probs[:, 1].cpu().numpy()
            val_preds.extend(preds.tolist()); val_probs.extend(probs_fraud.tolist())

        val_labels_list = [labels[n] for n in val_nodes]
        val_pred_labels = val_preds
        val_acc = accuracy_score(val_labels_list, val_pred_labels)
        val_prec = precision_score(val_labels_list, val_pred_labels, zero_division=0)
        val_rec = recall_score(val_labels_list, val_pred_labels, zero_division=0)
        val_f1 = f1_score(val_labels_list, val_pred_labels, zero_division=0)
        try:
            val_auc = roc_auc_score(val_labels_list, val_probs) if len(np.unique(val_labels_list)) > 1 else 0.0
        except ValueError:
            val_auc = 0.0

        print(f"  Val - Acc: {val_acc:.4f} | Prec: {val_prec:.4f} | Rec: {val_rec:.4f} | F1: {val_f1:.4f} | AUC: {val_auc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {
                'tgat_model': tgat_model.state_dict(),
                'classifier': classifier.state_dict(),
                'node_features': node_features.state_dict(),
                'time_encoder': time_encoder.state_dict()
            }
            print("  ✓ New best validation accuracy. Saving checkpoint in memory.")

    total_time = time.time() - start_time
    print(f"\nTraining finished in {total_time:.2f}s. Best val acc: {best_val_acc:.4f}")

    # Restore best
    if best_state is not None:
        tgat_model.load_state_dict(best_state['tgat_model'])
        classifier.load_state_dict(best_state['classifier'])
        node_features.load_state_dict(best_state['node_features'])
        time_encoder.load_state_dict(best_state['time_encoder'])
        print("Loaded best model from validation.")

    os.makedirs('saved_models', exist_ok=True)
    save_path = 'saved_models/tgat_fraud_best.pt'
    torch.save({
        'tgat_model_state': tgat_model.state_dict(),
        'classifier_state': classifier.state_dict(),
        'node_features_state': node_features.state_dict(),
        'time_encoder_state': time_encoder.state_dict(),
        'config': {
            'num_nodes': num_nodes,
            'node_feat_dim': node_feat_dim,
            'time_dim': time_dim,
            'embed_dim': embed_dim,
            'k': k,
            'L': L
        }
    }, save_path)
    print(f"Saved TGAT model and artifacts to {save_path}")

    return classifier, tgat_model, node_features, neigh_finder


# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    random.seed(0); torch.manual_seed(0); np.random.seed(0)

    csv_path = "data/IBM_dataset/ibm_balanced_10to1.csv"
    events, labels, num_nodes, node_timestamp, df = load_ibm_temporal_data(csv_path)

    # Splits
    all_idx = np.arange(num_nodes)
    train_idx, test_val_idx = train_test_split(all_idx, test_size=0.3, random_state=42, stratify=labels)
    val_idx, test_idx = train_test_split(test_val_idx, test_size=0.5, random_state=42, stratify=labels[test_val_idx])

    train_mask = np.zeros(num_nodes, dtype=bool)
    val_mask = np.zeros(num_nodes, dtype=bool)
    test_mask = np.zeros(num_nodes, dtype=bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    print(f"\nDataset split:")
    print(f"  Train: {train_mask.sum():,} nodes ({train_mask.sum()/num_nodes*100:.1f}%)")
    print(f"  Val:   {val_mask.sum():,} nodes ({val_mask.sum()/num_nodes*100:.1f}%)")
    print(f"  Test:  {test_mask.sum():,} nodes ({test_mask.sum()/num_nodes*100:.1f}%)")

    print(f"\nFraud distribution:")
    print(f"  Train: {labels[train_mask].sum():,} fraud / {train_mask.sum():,} total ({labels[train_mask].mean()*100:.2f}%)")
    print(f"  Val:   {labels[val_mask].sum():,} fraud / {val_mask.sum():,} total ({labels[val_mask].mean()*100:.2f}%)")
    print(f"  Test:  {labels[test_mask].sum():,} fraud / {test_mask.sum():,} total ({labels[test_mask].mean()*100:.2f}%)")

    classifier, tgat_model, node_features, neigh_finder = train_tgat_fraud(
        events=events,
        labels=labels,
        node_timestamp=node_timestamp,
        num_nodes=num_nodes,
        train_mask=train_mask,
        val_mask=val_mask,
        epochs=5,
        batch_size=256,
        lr=1e-3,
        node_feat_dim=64,
        time_dim=16,
        embed_dim=64,
        k=10,
        L=2
    )

    results = evaluate_tgat(classifier, tgat_model, TimeEncoder(time_dim=16).to(device),
                            node_features, neigh_finder, node_timestamp,
                            labels=labels, test_mask=test_mask,
                            batch_size=256, k=10, L=2)

    print("TGAT training + evaluation finished.")
