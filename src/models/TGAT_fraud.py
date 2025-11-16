"""
TGAT (Temporal Graph Attention) for IBM Credit Card Fraud Detection
Adapts TGAT model for fraud classification on temporal transaction data.

Key differences from TGN:
- No memory module (stateless)
- Uses multi-layer temporal attention
- Recursive L-hop temporal subgraph sampling
- Pure attention-based embeddings
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
import time
from typing import List, Tuple, Optional

# Import TGAT components
from TGAT import (
    TGATModel, TimeEncoder, NeighborFinder
)


class TGATFraudClassifier(nn.Module):
    """
    TGAT-based fraud classifier for transaction nodes.
    Uses TGAT temporal attention embeddings + MLP classifier.
    Simplified version that uses node features directly.
    """
    def __init__(self, tgat_model: TGATModel, embed_dim: int, node_feat_dim: int, device='cpu'):
        super().__init__()
        self.tgat = tgat_model
        self.device = device
        self.node_feat_dim = node_feat_dim
        self.time_dim = 50  # Must match what we pass to TGATModel
        self.time_encoder = TimeEncoder(time_dim=self.time_dim, learnable=True).to(device)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2)
        )
    
    def get_node_embeddings_simple(self, node_ids: List[int], node_features: torch.Tensor, 
                                   neigh_finder: NeighborFinder, current_time: float, k: int = 10):
        """
        Simplified embedding: just use node features + single-hop neighbors.
        """
        batch_size = len(node_ids)
        device = self.device
        
        # Get target node features
        target_feats = node_features[node_ids]  # (B, feat_dim)
        
        # Get k nearest neighbors for each node
        max_neighbors = k
        neigh_feats_list = []
        neigh_time_enc_list = []
        
        for node_id in node_ids:
            neighs = neigh_finder.get_prev_neighbors(node_id, current_time, k)
            
            if len(neighs) == 0:
                # No neighbors - use padding
                neigh_feats = torch.zeros(max_neighbors, self.node_feat_dim, device=device)
                neigh_time_enc = torch.zeros(max_neighbors, 2 * self.time_dim, device=device)  # Time encoder outputs 2*time_dim
            else:
                # Get neighbor features
                neigh_ids = [n[0] for n in neighs[:max_neighbors]]
                neigh_times = [n[1] for n in neighs[:max_neighbors]]
                
                # Pad if needed
                while len(neigh_ids) < max_neighbors:
                    neigh_ids.append(neigh_ids[-1] if neigh_ids else node_id)
                    neigh_times.append(neigh_times[-1] if neigh_times else current_time)
                
                neigh_ids = neigh_ids[:max_neighbors]
                neigh_times = neigh_times[:max_neighbors]
                
                neigh_feats = node_features[neigh_ids].to(device)  # (K, feat_dim)
                
                # Time encoding - TimeEncoder outputs 2*time_dim already
                time_deltas = torch.tensor([current_time - t for t in neigh_times], device=device)
                neigh_time_enc = self.time_encoder(time_deltas)  # (K, 2*time_dim)
            
            neigh_feats_list.append(neigh_feats)
            neigh_time_enc_list.append(neigh_time_enc)
        
        # Stack into batches
        neigh_feats_padded = torch.stack(neigh_feats_list, dim=0)  # (B, K, feat_dim)
        neigh_time_enc_padded = torch.stack(neigh_time_enc_list, dim=0)  # (B, K, 2*time_dim)
        
        # Create padding mask (all False since we padded manually)
        key_padding_mask = torch.zeros(batch_size, max_neighbors, dtype=torch.bool, device=device)
        
        # Apply TGAT
        embeddings = self.tgat(target_feats, neigh_feats_padded, neigh_time_enc_padded, key_padding_mask)
        return embeddings  # (B, embed_dim)
    
    def forward(self, node_ids: List[int], node_features: torch.Tensor, 
                neigh_finder: NeighborFinder, current_time: float = 1.0):
        """
        Get fraud predictions for transaction nodes.
        Returns: logits (B, 2)
        """
        z = self.get_node_embeddings_simple(node_ids, node_features, neigh_finder, current_time)
        return self.classifier(z)


def load_ibm_temporal_data(csv_path: str):
    """
    Load IBM dataset and convert to temporal event format.
    Returns: events list, node features, labels, num_nodes
    """
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df):,} transactions")
    
    # Convert Amount to float
    if 'Amount' in df.columns:
        df['Amount'] = df['Amount'].replace(r'[\$,]', '', regex=True).astype(float)
    
    # Map 'Is Fraud?' to binary label
    if 'Is Fraud?' in df.columns:
        df['Fraud_Label'] = df['Is Fraud?'].map({'No': 0, 'Yes': 1})
    
    # Create unique transaction IDs (node IDs)
    df['transaction_id'] = np.arange(len(df))
    
    # Create timestamp from Year, Month, Day columns
    df['timestamp'] = pd.to_datetime(
        df[['Year', 'Month', 'Day']].astype(str).agg('-'.join, axis=1),
        format='%Y-%m-%d',
        errors='coerce'
    )
    df['timestamp'] = (df['timestamp'] - df['timestamp'].min()).dt.total_seconds()
    df['timestamp'] = df['timestamp'] / df['timestamp'].max()  # normalize to [0,1]
    
    # Sort by timestamp
    df = df.sort_values('timestamp').reset_index(drop=True)
    df['transaction_id'] = np.arange(len(df))  # reassign after sort
    
    # Create node features (simple: amount, MCC, timestamp)
    node_features = torch.zeros(len(df), 3)
    amount_mean = df['Amount'].mean()
    amount_std = df['Amount'].std()
    node_features[:, 0] = torch.tensor((df['Amount'] - amount_mean) / amount_std, dtype=torch.float)
    node_features[:, 1] = torch.tensor(df['MCC'] / df['MCC'].max(), dtype=torch.float)
    node_features[:, 2] = torch.tensor(df['timestamp'], dtype=torch.float)
    
    # Build events: connect transactions by same User or Card
    events = []
    
    print("Building temporal edges...")
    # Group by User and create edges
    for user_id, group in df.groupby('User'):
        indices = group['transaction_id'].values
        timestamps = group['timestamp'].values
        amounts = group['Amount'].values
        mccs = group['MCC'].values
        
        for i in range(len(indices) - 1):
            src = int(indices[i])
            dst = int(indices[i + 1])
            ts = float(timestamps[i + 1])
            # Edge features: amount diff, time diff, MCC
            edge_feat = torch.tensor([
                amounts[i + 1] - amounts[i],
                timestamps[i + 1] - timestamps[i],
                float(mccs[i + 1])
            ], dtype=torch.float)
            events.append((src, dst, ts, edge_feat))
    
    # Group by Card and create edges
    for card_id, group in df.groupby('Card'):
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
    
    # Sort events by timestamp
    events.sort(key=lambda x: x[2])
    print(f"Created {len(events):,} temporal edges")
    
    # Extract labels
    labels = df['Fraud_Label'].values
    fraud_count = labels.sum()
    print(f"Fraud transactions: {fraud_count:,} ({fraud_count/len(labels)*100:.2f}%)")
    
    return events, node_features, labels, len(df)


def train_tgat_fraud(model: TGATFraudClassifier, events: List[Tuple], node_features: torch.Tensor,
                     labels: np.ndarray, train_mask: np.ndarray, val_mask: np.ndarray,
                     epochs: int = 5, batch_size: int = 256, lr: float = 0.001, device='cpu'):
    """
    Train TGAT model for fraud classification.
    """
    model = model.to(device)
    node_features = node_features.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()
    
    print(f"\nTraining on device: {device}")
    print(f"Using standard CrossEntropyLoss for balanced dataset")
    
    best_val_acc = 0
    best_model_state = None
    
    print(f"\nStarting training for {epochs} epochs...")
    start_time = time.time()
    
    # Build neighbor finder from events
    print("Building temporal neighbor structure...")
    neigh_finder = NeighborFinder(num_nodes=len(labels), max_neighbors=500)
    for src, dst, ts, edge_feat in events:
        neigh_finder.insert_edge(src, dst, ts, edge_feat)
    print("Neighbor structure built.")
    
    # Get train and val node indices
    train_nodes = np.where(train_mask)[0].tolist()
    val_nodes = np.where(val_mask)[0].tolist()
    
    for epoch in range(epochs):
        model.train()
        
        # Shuffle training nodes
        np.random.shuffle(train_nodes)
        
        epoch_loss = 0
        num_batches = 0
        
        # Mini-batch training
        for i in range(0, len(train_nodes), batch_size):
            batch_nodes = train_nodes[i:i + batch_size]
            batch_labels = torch.tensor([labels[n] for n in batch_nodes], 
                                       dtype=torch.long, device=device)
            
            optimizer.zero_grad()
            logits = model(batch_nodes, node_features, neigh_finder, current_time=1.0)
            loss = criterion(logits, batch_labels)
            
            if torch.isnan(loss):
                print(f"Warning: NaN loss at epoch {epoch+1}, batch {num_batches}, skipping")
                continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            # Progress update every 100 batches
            if num_batches % 1 == 0:
                print(f"  Epoch {epoch+1}/{epochs} | Batch {num_batches}/{len(train_nodes)//batch_size} | Loss: {loss.item():.4f}")
        
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
        
        # Validation every epoch (since we only have 5)
        if (epoch + 1) % 1 == 0:
            print(f"\nEpoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")
            print(f"  Validating...")
            model.eval()
            with torch.no_grad():
                # Evaluate on train
                train_preds = []
                train_probs = []
                for i in range(0, len(train_nodes), batch_size):
                    batch_nodes = train_nodes[i:i + batch_size]
                    logits = model(batch_nodes, node_features, neigh_finder, current_time=1.0)
                    probs = F.softmax(logits, dim=1)
                    preds = logits.argmax(dim=1).cpu().numpy()
                    probs_fraud = probs[:, 1].cpu().numpy()
                    train_preds.extend(preds)
                    train_probs.extend(probs_fraud)
                train_labels_subset = [labels[n] for n in train_nodes]
                train_acc = accuracy_score(train_labels_subset, train_preds)
                train_precision = precision_score(train_labels_subset, train_preds, zero_division=0)
                train_recall = recall_score(train_labels_subset, train_preds, zero_division=0)
                train_f1 = f1_score(train_labels_subset, train_preds, zero_division=0)
                
                # Evaluate on val
                val_preds = []
                val_probs = []
                for i in range(0, len(val_nodes), batch_size):
                    batch_nodes = val_nodes[i:i + batch_size]
                    logits = model(batch_nodes, node_features, neigh_finder, current_time=1.0)
                    probs = F.softmax(logits, dim=1)
                    preds = logits.argmax(dim=1).cpu().numpy()
                    probs_fraud = probs[:, 1].cpu().numpy()
                    val_preds.extend(preds)
                    val_probs.extend(probs_fraud)
                val_labels_subset = [labels[n] for n in val_nodes]
                val_acc = accuracy_score(val_labels_subset, val_preds)
                val_precision = precision_score(val_labels_subset, val_preds, zero_division=0)
                val_recall = recall_score(val_labels_subset, val_preds, zero_division=0)
                val_f1 = f1_score(val_labels_subset, val_preds, zero_division=0)
                val_roc_auc = roc_auc_score(val_labels_subset, val_probs) if len(np.unique(val_labels_subset)) > 1 else 0.0
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_model_state = model.state_dict().copy()
                    print(f"    ✓ New best validation accuracy!")
                
                print(f"    Train - Acc: {train_acc:.4f} | Prec: {train_precision:.4f} | Rec: {train_recall:.4f} | F1: {train_f1:.4f}")
                print(f"    Val   - Acc: {val_acc:.4f} | Prec: {val_precision:.4f} | Rec: {val_recall:.4f} | F1: {val_f1:.4f} | AUC: {val_roc_auc:.4f}\n")
    
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f}s")
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print("Loaded best model from validation")

    # Save model to disk (like TGN_fraud.py)
    import os
    os.makedirs('saved_models', exist_ok=True)
    model_path = 'saved_models/tgat_fraud_best.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'tgat_config': {
            'embed_dim': getattr(model.tgat, 'embed_dim', None),
            'time_dim': getattr(model.tgat, 'time_dim', None),
            'n_layers': getattr(model.tgat, 'n_layers', None),
            'n_heads': getattr(model.tgat, 'n_heads', None),
            'dropout': getattr(model.tgat, 'dropout', None),
            'node_feat_dim': getattr(model.tgat, 'node_feat_dim', None)
        }
    }, model_path)
    print(f"Model saved to {model_path}")

    return model


def evaluate_tgat_fraud(model: TGATFraudClassifier, node_features: torch.Tensor,
                        neigh_finder: NeighborFinder, labels: np.ndarray, 
                        test_mask: np.ndarray, batch_size: int = 256, device='cpu'):
    """
    Evaluate TGAT fraud classifier on test set.
    """
    print("\nEvaluating on test set...")
    model.eval()
    node_features = node_features.to(device)
    test_nodes = np.where(test_mask)[0].tolist()
    
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for i in range(0, len(test_nodes), batch_size):
            if i % (batch_size * 10) == 0:
                print(f"  Processed {i:,}/{len(test_nodes):,} test samples...")
            
            batch_nodes = test_nodes[i:i + batch_size]
            logits = model(batch_nodes, node_features, neigh_finder, current_time=1.0)
            probs = F.softmax(logits, dim=1)
            preds = logits.argmax(dim=1).cpu().numpy()
            probs_fraud = probs[:, 1].cpu().numpy()
            
            all_preds.extend(preds)
            all_probs.extend(probs_fraud)
    
    test_labels = [labels[n] for n in test_nodes]
    
    # Compute metrics
    acc = accuracy_score(test_labels, all_preds)
    precision = precision_score(test_labels, all_preds, zero_division=0)
    recall = recall_score(test_labels, all_preds, zero_division=0)
    f1 = f1_score(test_labels, all_preds, zero_division=0)
    roc_auc = roc_auc_score(test_labels, all_probs) if len(np.unique(test_labels)) > 1 else 0.0
    
    print(f"\n{'='*60}")
    print(f"TGAT TEST SET EVALUATION RESULTS")
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


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load IBM balanced dataset
    csv_path = "data/ibm/ibm_fraud_29k_nonfraud_60k.csv"
    events, node_features, labels, num_nodes = load_ibm_temporal_data(csv_path)
    
    # Create train/val/test split (same as GNN: 70/15/15)
    all_indices = np.arange(num_nodes)
    train_idx, test_val_idx = train_test_split(
        all_indices,
        test_size=0.3,
        random_state=42,
        stratify=labels
    )
    val_idx, test_idx = train_test_split(
        test_val_idx,
        test_size=0.5,
        random_state=42,
        stratify=labels[test_val_idx]
    )
    
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
    
    # Initialize TGAT model
    print("\nInitializing TGAT model...")
    node_feat_dim = node_features.shape[1]  # 3 features
    tgat = TGATModel(
        node_feat_dim=node_feat_dim,
        time_enc_dim=50,
        embed_dim=100,
        n_layers=2,
        n_heads=2,
        dropout=0.1
    )
    
    # Build neighbor finder for initial structure
    neigh_finder = NeighborFinder(num_nodes=num_nodes, max_neighbors=500)
    for src, dst, ts, edge_feat in events:
        neigh_finder.insert_edge(src, dst, ts, edge_feat)
    
    model = TGATFraudClassifier(tgat, embed_dim=100, node_feat_dim=node_feat_dim, device=device)
    
    # Train model
    model = train_tgat_fraud(
        model, events, node_features, labels, train_mask, val_mask,
        epochs=15, batch_size=786432, lr=0.001, device=device
    )
    
    # Evaluate on test set
    results = evaluate_tgat_fraud(model, node_features, neigh_finder, labels, 
                                  test_mask, batch_size=786432, device=device)
    
    print("\nTGAT training complete!")
    print("Compare these results with GNN and TGN baselines.")
