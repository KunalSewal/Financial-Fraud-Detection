"""
TGN + TGAT Simple Ensemble for Fraud Detection
===============================================
Strategy: Load pre-trained models and combine their predictions.

NO TRAINING - just loads saved models and ensembles them!

Ensemble Methods:
1. Averaging: Average the probabilities from both models
2. Weighted: Weighted average based on individual model performance
3. Voting: Majority vote on predicted class
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from typing import List, Tuple, Optional
import os

# Import model components
from TGAT import TGATModel, TimeEncoder as TGATTimeEncoder, NeighborFinder as TGATNeighborFinder
from tgn import TGNModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


# =====================================================
# Individual Model Wrappers (same as original files)
# =====================================================

class TGATFraudClassifier(nn.Module):
    """TGAT-based fraud classifier"""
    def __init__(self, tgat_model: TGATModel, embed_dim: int, node_feat_dim: int, device='cpu'):
        super().__init__()
        self.tgat = tgat_model
        self.device = device
        self.node_feat_dim = node_feat_dim
        self.time_dim = 50
        self.time_encoder = TGATTimeEncoder(time_dim=self.time_dim, learnable=True).to(device)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2)
        )
    
    def get_node_embeddings_simple(self, node_ids: List[int], node_features: torch.Tensor, 
                                   neigh_finder: TGATNeighborFinder, current_time: float, k: int = 10):
        batch_size = len(node_ids)
        device = self.device
        target_feats = node_features[node_ids]
        max_neighbors = k
        neigh_feats_list = []
        neigh_time_enc_list = []
        
        for node_id in node_ids:
            neighs = neigh_finder.get_prev_neighbors(node_id, current_time, k)
            
            if len(neighs) == 0:
                neigh_feats = torch.zeros(max_neighbors, self.node_feat_dim, device=device)
                neigh_time_enc = torch.zeros(max_neighbors, 2 * self.time_dim, device=device)
            else:
                neigh_ids = [n[0] for n in neighs[:max_neighbors]]
                neigh_times = [n[1] for n in neighs[:max_neighbors]]
                
                while len(neigh_ids) < max_neighbors:
                    neigh_ids.append(neigh_ids[-1] if neigh_ids else node_id)
                    neigh_times.append(neigh_times[-1] if neigh_times else current_time)
                
                neigh_ids = neigh_ids[:max_neighbors]
                neigh_times = neigh_times[:max_neighbors]
                
                neigh_feats = node_features[neigh_ids].to(device)
                time_deltas = torch.tensor([current_time - t for t in neigh_times], device=device)
                neigh_time_enc = self.time_encoder(time_deltas)
            
            neigh_feats_list.append(neigh_feats)
            neigh_time_enc_list.append(neigh_time_enc)
        
        neigh_feats_padded = torch.stack(neigh_feats_list, dim=0)
        neigh_time_enc_padded = torch.stack(neigh_time_enc_list, dim=0)
        key_padding_mask = torch.zeros(batch_size, max_neighbors, dtype=torch.bool, device=device)
        
        embeddings = self.tgat(target_feats, neigh_feats_padded, neigh_time_enc_padded, key_padding_mask)
        return embeddings
    
    def forward(self, node_ids: List[int], node_features: torch.Tensor, 
                neigh_finder: TGATNeighborFinder, current_time: float = 1.0):
        z = self.get_node_embeddings_simple(node_ids, node_features, neigh_finder, current_time)
        return self.classifier(z)


class TGNFraudClassifier(nn.Module):
    """TGN-based fraud classifier"""
    def __init__(self, tgn_model: TGNModel, embed_dim: int):
        super().__init__()
        self.tgn = tgn_model
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2)
        )
    
    def forward(self, node_ids: List[int]):
        z = self.tgn.get_node_embeddings(node_ids)
        return self.classifier(z)


# =====================================================
# Data Loading (for testing)
# =====================================================

def load_ibm_temporal_data(csv_path: str):
    """Load IBM dataset and convert to temporal event format."""
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df):,} transactions")
    
    if 'Amount' in df.columns:
        df['Amount'] = df['Amount'].replace(r'[\$,]', '', regex=True).astype(float)
    
    if 'Is Fraud?' in df.columns:
        df['Fraud_Label'] = df['Is Fraud?'].map({'No': 0, 'Yes': 1})
    
    df['transaction_id'] = np.arange(len(df))
    
    df['timestamp'] = pd.to_datetime(
        df[['Year', 'Month', 'Day']].astype(str).agg('-'.join, axis=1),
        format='%Y-%m-%d',
        errors='coerce'
    )
    df['timestamp'] = (df['timestamp'] - df['timestamp'].min()).dt.total_seconds()
    df['timestamp'] = df['timestamp'] / df['timestamp'].max()
    
    df = df.sort_values('timestamp').reset_index(drop=True)
    df['transaction_id'] = np.arange(len(df))
    
    # Create node features for TGAT
    node_features = torch.zeros(len(df), 3)
    amount_mean = df['Amount'].mean()
    amount_std = df['Amount'].std()
    node_features[:, 0] = torch.tensor((df['Amount'] - amount_mean) / amount_std, dtype=torch.float)
    node_features[:, 1] = torch.tensor(df['MCC'] / df['MCC'].max(), dtype=torch.float)
    node_features[:, 2] = torch.tensor(df['timestamp'], dtype=torch.float)
    
    # Build events
    events = []
    print("Building temporal edges...")
    for col in ['User', 'Card']:
        if col not in df.columns:
            continue
        for _, group in df.groupby(col):
            indices = group['transaction_id'].values
            timestamps = group['timestamp'].values
            amounts = group['Amount'].values
            mccs = group['MCC'].values
            
            for i in range(len(indices) - 1):
                src = int(indices[i])
                dst = int(indices[i + 1])
                ts = float(timestamps[i + 1])
                edge_feat = torch.tensor([
                    amounts[i + 1] - amounts[i],
                    timestamps[i + 1] - timestamps[i],
                    float(mccs[i + 1])
                ], dtype=torch.float)
                events.append((src, dst, ts, edge_feat))
    
    events.sort(key=lambda x: x[2])
    print(f"Created {len(events):,} temporal edges")
    
    labels = df['Fraud_Label'].values
    print(f"Fraud transactions: {labels.sum():,} ({labels.mean()*100:.2f}%)")
    
    return events, node_features, labels, len(df)


# =====================================================
# Load Saved Models
# =====================================================

def load_tgn_model(model_path: str, num_nodes: int, device='cpu'):
    """Load pre-trained TGN model from saved checkpoint."""
    print(f"\nLoading TGN model from {model_path}...")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"TGN model not found at {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['tgn_config']
    
    # Initialize TGN
    tgn = TGNModel(
        num_nodes=config['num_nodes'],
        mem_dim=config['mem_dim'],
        msg_dim=config['msg_dim'],
        edge_dim=config['edge_dim'],
        time_dim=config['time_dim'],
        embed_dim=config['embed_dim'],
        k=config['k'],
        device=device
    )
    
    model = TGNFraudClassifier(tgn, embed_dim=config['embed_dim'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"  ✓ TGN model loaded successfully")
    return model


def load_tgat_model(model_path: str, events: List[Tuple], num_nodes: int, 
                    node_feat_dim: int, device='cpu'):
    """
    Load pre-trained TGAT model from saved checkpoint.
    """
    print(f"\nLoading TGAT model from {model_path}...")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"TGAT model not found at {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['tgat_config']
    
    # Initialize TGAT
    tgat = TGATModel(
        node_feat_dim=node_feat_dim,
        time_enc_dim=config.get('time_dim', 50),
        embed_dim=config.get('embed_dim', 100),
        n_layers=config.get('n_layers', 2),
        n_heads=config.get('n_heads', 2),
        dropout=config.get('dropout', 0.1)
    )
    
    model = TGATFraudClassifier(tgat, embed_dim=config.get('embed_dim', 100), 
                                node_feat_dim=node_feat_dim, device=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Build neighbor finder from events
    print("  Building TGAT neighbor finder from events...")
    neigh_finder = TGATNeighborFinder(num_nodes=num_nodes, max_neighbors=500)
    for i, (src, dst, ts, edge_feat) in enumerate(events):
        if i % 10000 == 0 and i > 0:
            print(f"    Processed {i:,}/{len(events):,} edges")
        neigh_finder.insert_edge(src, dst, ts, edge_feat)
    
    print(f"  ✓ TGAT model loaded successfully")
    
    return model, neigh_finder


def initialize_tgn_memory(tgn_model: TGNFraudClassifier, events: List[Tuple], device='cpu'):
    """
    Populate TGN memory from events (needed if TGN model doesn't save memory state).
    """
    print("\nInitializing TGN memory from events...")
    tgn_model.eval()
    
    with torch.no_grad():
        for i, (src, dst, ts, edge_feat) in enumerate(events):
            if i % 10000 == 0 and i > 0:
                print(f"  Processed {i:,}/{len(events):,} events")
            
            tgn_model.tgn.neigh_finder.insert_edge(src, dst, ts, edge_feat)
            
            src_t = torch.tensor([src], device=device)
            dst_t = torch.tensor([dst], device=device)
            
            src_mem = tgn_model.tgn.memory.get_memory(src_t)
            dst_mem = tgn_model.tgn.memory.get_memory(dst_t)
            
            last_src_time = tgn_model.tgn.memory.last_update[src]
            dt_src = ts - last_src_time.item()
            dt_enc = tgn_model.tgn.time_encoder(torch.tensor([dt_src], device=device))
            
            if edge_feat is not None:
                edge_feat_t = edge_feat.unsqueeze(0).to(device)
            else:
                edge_feat_t = None
            
            raw_msg, _ = tgn_model.tgn.compute_raw_messages(src_mem, dst_mem, edge_feat_t, dt_enc)
            tgn_model.tgn.memory.update_with_messages([src, dst], 
                                                      torch.cat([raw_msg, raw_msg]), 
                                                      torch.tensor([ts, ts], device=device))
    
    print("  ✓ TGN memory initialized")


# =====================================================
# Ensemble Evaluation
# =====================================================

def ensemble_predict(tgn_model: TGNFraudClassifier, 
                    tgat_model: TGATFraudClassifier,
                    node_features: torch.Tensor,
                    tgat_neigh_finder: TGATNeighborFinder,
                    node_ids: List[int],
                    method: str = 'average',
                    tgn_weight: float = 0.5,
                    tgat_weight: float = 0.5,
                    batch_size: int = 8192,
                    device='cpu'):
    """
    Get ensemble predictions.
    
    Methods:
    - 'average': Simple average of probabilities
    - 'weighted': Weighted average (use tgn_weight, tgat_weight)
    - 'voting': Majority vote
    """
    all_preds = []
    all_probs = []
    
    tgn_model.eval()
    tgat_model.eval()
    
    print(f"\nEnsemble method: {method}")
    if method == 'weighted':
        print(f"  Weights: TGN={tgn_weight:.3f}, TGAT={tgat_weight:.3f}")
    
    with torch.no_grad():
        for i in range(0, len(node_ids), batch_size):
            if i % (batch_size * 5) == 0:
                print(f"  Processing {i:,}/{len(node_ids):,} nodes...")
            
            batch_nodes = node_ids[i:i + batch_size]
            
            # Get TGN predictions
            tgn_logits = tgn_model(batch_nodes)
            tgn_probs = F.softmax(tgn_logits, dim=1).cpu().numpy()
            
            # Get TGAT predictions
            tgat_logits = tgat_model(batch_nodes, node_features, 
                                     tgat_neigh_finder, current_time=1.0)
            tgat_probs = F.softmax(tgat_logits, dim=1).cpu().numpy()
            
            # Combine based on method
            if method == 'average':
                combined_probs = (tgn_probs + tgat_probs) / 2
            
            elif method == 'weighted':
                combined_probs = (tgn_weight * tgn_probs + tgat_weight * tgat_probs)
            
            elif method == 'voting':
                tgn_preds = np.argmax(tgn_probs, axis=1)
                tgat_preds = np.argmax(tgat_probs, axis=1)
                combined_preds = []
                combined_probs_list = []
                
                for j in range(len(tgn_preds)):
                    if tgn_preds[j] == tgat_preds[j]:
                        # Both agree
                        combined_preds.append(tgn_preds[j])
                        combined_probs_list.append(
                            (tgn_probs[j, 1] + tgat_probs[j, 1]) / 2
                        )
                    else:
                        # Disagree - use higher confidence
                        if tgn_probs[j, tgn_preds[j]] > tgat_probs[j, tgat_preds[j]]:
                            combined_preds.append(tgn_preds[j])
                            combined_probs_list.append(tgn_probs[j, 1])
                        else:
                            combined_preds.append(tgat_preds[j])
                            combined_probs_list.append(tgat_probs[j, 1])
                
                combined_preds = np.array(combined_preds)
                all_preds.extend(combined_preds.tolist())
                all_probs.extend(combined_probs_list)
                continue
            
            # For average and weighted methods
            preds = np.argmax(combined_probs, axis=1)
            probs_fraud = combined_probs[:, 1]
            
            all_preds.extend(preds.tolist())
            all_probs.extend(probs_fraud.tolist())
    
    return np.array(all_preds), np.array(all_probs)


def evaluate_ensemble(tgn_model: TGNFraudClassifier,
                     tgat_model: TGATFraudClassifier,
                     node_features: torch.Tensor,
                     tgat_neigh_finder: TGATNeighborFinder,
                     labels: np.ndarray,
                     test_mask: np.ndarray,
                     method: str = 'average',
                     tgn_weight: float = 0.5,
                     tgat_weight: float = 0.5,
                     batch_size: int = 8192,
                     device='cpu'):
    """Evaluate ensemble on test set."""
    print(f"\n{'='*60}")
    print(f"ENSEMBLE EVALUATION")
    print(f"{'='*60}")
    
    test_nodes = np.where(test_mask)[0].tolist()
    
    preds, probs = ensemble_predict(
        tgn_model, tgat_model, node_features, tgat_neigh_finder,
        test_nodes, method, tgn_weight, tgat_weight, batch_size, device
    )
    
    test_labels = labels[test_nodes]
    
    # Compute metrics
    acc = accuracy_score(test_labels, preds)
    precision = precision_score(test_labels, preds, zero_division=0)
    recall = recall_score(test_labels, preds, zero_division=0)
    f1 = f1_score(test_labels, preds, zero_division=0)
    roc_auc = roc_auc_score(test_labels, probs) if len(np.unique(test_labels)) > 1 else 0.0
    
    print(f"\n{'='*60}")
    print(f"ENSEMBLE TEST RESULTS ({method.upper()})")
    print(f"{'='*60}")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print(f"{'='*60}")
    
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc
    }


# =====================================================
# Main
# =====================================================

if __name__ == "__main__":
    # Load data
    csv_path = "data/ibm/ibm_fraud_29k_nonfraud_60k.csv"
    events, node_features, labels, num_nodes = load_ibm_temporal_data(csv_path)
    
    # Create splits
    all_indices = np.arange(num_nodes)
    train_idx, test_val_idx = train_test_split(
        all_indices, test_size=0.3, random_state=42, stratify=labels
    )
    val_idx, test_idx = train_test_split(
        test_val_idx, test_size=0.5, random_state=42, stratify=labels[test_val_idx]
    )
    
    train_mask = np.zeros(num_nodes, dtype=bool)
    val_mask = np.zeros(num_nodes, dtype=bool)
    test_mask = np.zeros(num_nodes, dtype=bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True
    
    print(f"\n{'='*60}")
    print(f"DATASET INFO")
    print(f"{'='*60}")
    print(f"Total nodes: {num_nodes:,}")
    print(f"Test set: {test_mask.sum():,} ({test_mask.sum()/num_nodes*100:.1f}%)")
    print(f"Test fraud: {labels[test_mask].mean()*100:.2f}%")
    
    # Load TGN model
    tgn_model_path = "saved_models/tgn_fraud_best.pt"
    tgn_model = load_tgn_model(tgn_model_path, num_nodes, device)
    
    # Initialize TGN memory (needed if not saved in checkpoint)
    initialize_tgn_memory(tgn_model, events, device)
    
    # Load/Initialize TGAT model
    # NOTE: TGAT_fraud.py doesn't save model - you need to train it first
    # or modify TGAT_fraud.py to save the model
    tgat_model, tgat_neigh_finder = load_tgat_model(events, num_nodes, 
                                                     node_features.shape[1], device)
    
    print(f"\n{'='*60}")
    print(f"⚠  WARNING: TGAT model is UNTRAINED!")
    print(f"   Run TGAT_fraud.py first and modify it to save the model.")
    print(f"   For demonstration, proceeding with untrained TGAT...")
    print(f"{'='*60}")
    
    # Evaluate ensemble with different methods
    methods = ['average', 'weighted', 'voting']
    
    for method in methods:
        if method == 'weighted':
            # Set weights based on individual model performance
            # For demo: assume TGN performs better (0.6 vs 0.4)
            # In practice, get these from validation set
            evaluate_ensemble(
                tgn_model, tgat_model, node_features, tgat_neigh_finder,
                labels, test_mask, method=method,
                tgn_weight=0.6, tgat_weight=0.4,
                batch_size=8192, device=device
            )
        else:
            evaluate_ensemble(
                tgn_model, tgat_model, node_features, tgat_neigh_finder,
                labels, test_mask, method=method,
                batch_size=8192, device=device
            )
    
    print("\n✓ Ensemble evaluation complete!")
    print("\nTo use trained TGAT:")
    print("1. Add model saving to TGAT_fraud.py (like TGN does)")
    print("2. Train TGAT: python TGAT_fraud.py")
    print("3. Modify load_tgat_model() to load the saved checkpoint")