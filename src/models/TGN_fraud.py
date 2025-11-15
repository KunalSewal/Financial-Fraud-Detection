"""
TGN (Temporal Graph Networks) for IBM Credit Card Fraud Detection
Adapts TGN model for fraud classification on temporal transaction data.

Key adaptations:
- Load IBM balanced dataset (10:1 ratio)
- Convert to temporal event format (chronological)
- Use TGN embeddings + classifier for fraud prediction
- Train/val/test splits matching GNN baseline
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
import time
from collections import defaultdict
from typing import List, Tuple, Optional

# Import TGN components
from tgn import (
    TGNModel, TimeEncoder, MessageFunction, MemoryModule, 
    RawMessageStore, NeighborFinder, TemporalAttentionEmbedding
)


class TGNFraudClassifier(nn.Module):
    """
    TGN-based fraud classifier for transaction nodes.
    Uses TGN temporal embeddings + MLP classifier.
    """
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
        """
        Get fraud predictions for transaction nodes.
        Returns: logits (B, 2)
        """
        z = self.tgn.get_node_embeddings(node_ids)  # (B, embed_dim)
        return self.classifier(z)


def load_ibm_temporal_data(csv_path: str):
    """
    Load IBM dataset and convert to temporal event format.
    Returns: events list, node features, labels, node_id_map
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
    # Normalize to [0, 1] range for numerical stability
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
    
    # Build events: connect transactions by same User or Card
    # Each edge represents a temporal connection between consecutive transactions
    events = []
    edge_features = []
    
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
                amounts[i + 1] - amounts[i],  # amount change
                timestamps[i + 1] - timestamps[i],  # time delta
                float(mccs[i + 1])  # merchant category
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
    
    return events, labels, len(df)


def train_tgn_fraud(model: TGNFraudClassifier, events: List[Tuple], labels: np.ndarray,
                    train_mask: np.ndarray, val_mask: np.ndarray,
                    epochs: int = 5, batch_size: int = 256, lr: float = 0.001, device='cpu'):
    """
    Train TGN model for fraud classification.
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()
    
    print(f"\nTraining on device: {device}")
    print(f"Using standard CrossEntropyLoss for balanced dataset")
    
    best_val_acc = 0
    best_model_state = None
    
    print(f"\nStarting training for {epochs} epochs...")
    start_time = time.time()
    
    # Process events to update TGN memory and neighbor structures
    print("Processing temporal events to build TGN memory...")
    with torch.no_grad():
        for i, (src, dst, ts, edge_feat) in enumerate(events):
            if i % 10000 == 0:
                print(f"  Processed {i:,}/{len(events):,} events")
            
            # Update neighbor finder
            model.tgn.neigh_finder.insert_edge(src, dst, ts, edge_feat)
            
            # Compute and store messages
            src_t = torch.tensor([src], device=device)
            dst_t = torch.tensor([dst], device=device)
            ts_t = torch.tensor([ts], device=device)
            
            src_mem = model.tgn.memory.get_memory(src_t)
            dst_mem = model.tgn.memory.get_memory(dst_t)
            
            # Time encoding
            last_src_time = model.tgn.memory.last_update[src]
            dt_src = ts - last_src_time.item()
            dt_enc = model.tgn.time_encoder(torch.tensor([dt_src], device=device))
            
            # Compute raw messages
            if edge_feat is not None:
                edge_feat_t = edge_feat.unsqueeze(0).to(device)
            else:
                edge_feat_t = None
            
            raw_msg, _ = model.tgn.compute_raw_messages(src_mem, dst_mem, edge_feat_t, dt_enc)
            
            # Update memory immediately (simplified compared to full TGN batching)
            model.tgn.memory.update_with_messages([src, dst], 
                                                  torch.cat([raw_msg, raw_msg]), 
                                                  torch.tensor([ts, ts], device=device))
    
    print("Temporal event processing complete. Starting supervised training...\n")
    
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
            logits = model(batch_nodes)
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
            if num_batches % 100 == 0:
                print(f"  Epoch {epoch+1}/{epochs} | Batch {num_batches}/{len(train_nodes)//batch_size} | Loss: {loss.item():.4f}")
        
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
        
        # Validation every 5 epochs
        if (epoch + 1) % 5 == 0:
            print(f"\n  Validating epoch {epoch+1}...")
            model.eval()
            with torch.no_grad():
                # Evaluate on train
                train_preds = []
                train_probs = []
                print(f"    Evaluating training set...")
                for i in range(0, len(train_nodes), batch_size):
                    batch_nodes = train_nodes[i:i + batch_size]
                    logits = model(batch_nodes)
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
                print(f"    Evaluating validation set...")
                for i in range(0, len(val_nodes), batch_size):
                    batch_nodes = val_nodes[i:i + batch_size]
                    logits = model(batch_nodes)
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
                
                print(f"  Epoch {epoch+1:3d} | Loss: {avg_loss:.4f}")
                print(f"    Train - Acc: {train_acc:.4f} | Prec: {train_precision:.4f} | Rec: {train_recall:.4f} | F1: {train_f1:.4f}")
                print(f"    Val   - Acc: {val_acc:.4f} | Prec: {val_precision:.4f} | Rec: {val_recall:.4f} | F1: {val_f1:.4f} | AUC: {val_roc_auc:.4f}\n")
    
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f}s")
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print("Loaded best model from validation")
    
    # Save model to disk
    import os
    os.makedirs('saved_models', exist_ok=True)
    model_path = 'saved_models/tgn_fraud_best.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'tgn_config': {
            'num_nodes': model.tgn.num_nodes,
            'mem_dim': model.tgn.mem_dim,
            'msg_dim': model.tgn.msg_dim,
            'edge_dim': model.tgn.edge_dim,
            'time_dim': model.tgn.time_dim,
            'embed_dim': model.tgn.embed_dim,
            'k': model.tgn.k
        }
    }, model_path)
    print(f"Model saved to {model_path}")
    
    return model


def evaluate_tgn_fraud(model: TGNFraudClassifier, labels: np.ndarray, 
                       test_mask: np.ndarray, batch_size: int = 256, device='cpu'):
    """
    Evaluate TGN fraud classifier on test set.
    """
    print("\nEvaluating on test set...")
    model.eval()
    test_nodes = np.where(test_mask)[0].tolist()
    
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for i in range(0, len(test_nodes), batch_size):
            if i % (batch_size * 50) == 0:
                print(f"  Processed {i:,}/{len(test_nodes):,} test samples...")
            
            batch_nodes = test_nodes[i:i + batch_size]
            logits = model(batch_nodes)
            probs = F.softmax(logits, dim=1)
            preds = logits.argmax(dim=1).cpu().numpy()
            probs_fraud = probs[:, 1].cpu().numpy()
            
            all_preds.extend(preds)
            all_probs.extend(probs_fraud)
    
    print(f"  Completed evaluation on {len(test_nodes):,} test samples")
    
    test_labels = [labels[n] for n in test_nodes]
    
    # Compute metrics
    acc = accuracy_score(test_labels, all_preds)
    precision = precision_score(test_labels, all_preds, zero_division=0)
    recall = recall_score(test_labels, all_preds, zero_division=0)
    f1 = f1_score(test_labels, all_preds, zero_division=0)
    roc_auc = roc_auc_score(test_labels, all_probs) if len(np.unique(test_labels)) > 1 else 0.0
    
    print(f"\n{'='*60}")
    print(f"TGN TEST SET EVALUATION RESULTS")
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
    csv_path = "data/IBM_dataset/ibm_balanced_10to1.csv"
    events, labels, num_nodes = load_ibm_temporal_data(csv_path)
    
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
    
    # Initialize TGN model
    print("\nInitializing TGN model...")
    tgn = TGNModel(
        num_nodes=num_nodes,
        mem_dim=100,
        msg_dim=100,
        edge_dim=3,  # amount_diff, time_diff, MCC
        time_dim=50,
        embed_dim=100,
        k=10,  # use 10 most recent neighbors
        device=device
    )
    
    model = TGNFraudClassifier(tgn, embed_dim=100)
    
    # Train model
    model = train_tgn_fraud(
        model, events, labels, train_mask, val_mask,
        epochs=5, batch_size=256, lr=0.001, device=device
    )
    
    # Evaluate on test set
    results = evaluate_tgn_fraud(model, labels, test_mask, batch_size=256, device=device)
    
    print("\nTGN training complete!")
    print("Compare these results with the basic GNN baseline.")
