"""
Basic Graph Neural Network (GNN) for transaction classification on IBM credit card dataset.
This script defines a simple 2-layer GCN model for baseline comparison with HMSTA.

Graph Construction:
- Nodes: Transactions
- Edges: User-to-User connections (transactions by same user/card)
- Features: Amount, MCC, temporal features, user statistics
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
import time

# Removed FocalLoss - using standard CrossEntropyLoss for balanced dataset

class BasicGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout
        
        # Initialize weights
        self._reset_parameters()
    
    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, edge_index):
        # Replace NaN/Inf with 0
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x

def load_preprocessed_graph(csv_path):
    """
    Load preprocessed IBM dataset and construct transaction graph.
    
    Args:
        csv_path: Path to preprocessed CSV with features
        
    Returns:
        PyTorch Geometric Data object
    """
    print(f"Loading preprocessed data from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df):,} transactions")
    
    # Build graph edges: connect transactions by same User or Card
    print("Building graph edges...")
    edge_index = [[], []]

    # Use correct column names for grouping
    user_col = 'User_ID' if 'User_ID' in df.columns else 'User'
    card_col = 'Card_ID' if 'Card_ID' in df.columns else 'Card'

    # Group by user
    user_groups = df.groupby(user_col).groups
    for user_id, indices in user_groups.items():
        indices_list = list(indices)
        for i in range(len(indices_list) - 1):
            edge_index[0].append(indices_list[i])
            edge_index[1].append(indices_list[i + 1])
            edge_index[0].append(indices_list[i + 1])
            edge_index[1].append(indices_list[i])

    # Group by card
    card_groups = df.groupby(card_col).groups
    for card_id, indices in card_groups.items():
        indices_list = list(indices)
        for i in range(len(indices_list) - 1):
            edge_index[0].append(indices_list[i])
            edge_index[1].append(indices_list[i + 1])
            edge_index[0].append(indices_list[i + 1])
            edge_index[1].append(indices_list[i])
    
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    print(f"Created {edge_index.shape[1]:,} edges")
    
    # --- Feature selection and label extraction for IBM dataset ---
    # Convert Amount to float (remove $ and commas)
    if 'Amount' in df.columns:
        df['Amount'] = df['Amount'].replace(r'[\$,]', '', regex=True).astype(float)

    # Map 'Is Fraud?' to binary label
    if 'Is Fraud?' in df.columns:
        df['Fraud_Label'] = df['Is Fraud?'].map({'No': 0, 'Yes': 1})
    elif 'Fraud_Label' not in df.columns:
        raise ValueError("No fraud label column found!")

    # Select only numeric columns for features (exclude label and obvious IDs)
    exclude_cols = {'Fraud_Label', 'Is Fraud?', 'User', 'Card', 'Time', 'Merchant Name', 'Merchant City', 'Merchant State', 'Use Chip', 'Errors?'}
    feature_cols = [col for col in df.columns if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])]
    x = torch.tensor(df[feature_cols].fillna(0).values, dtype=torch.float)
    print(f"Node features shape: {x.shape}")

    # Labels
    y = torch.tensor(df['Fraud_Label'].values, dtype=torch.long)
    fraud_count = (y == 1).sum().item()
    print(f"Fraud transactions: {fraud_count:,} ({fraud_count/len(y)*100:.2f}%)")

    return Data(x=x, edge_index=edge_index, y=y)

def train_gnn(data, train_mask, val_mask, epochs=100, lr=0.01, hidden_dim=64):
    """
    Train GNN model with train/val split.
    Uses standard CrossEntropyLoss for balanced dataset.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nTraining on device: {device}")
    
    model = BasicGNN(data.num_node_features, hidden_dim, 2).to(device)
    data = data.to(device)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    
    # Use standard CrossEntropyLoss for balanced dataset (10:1 ratio)
    criterion = nn.CrossEntropyLoss()
    print(f"Using standard CrossEntropyLoss for balanced dataset (10:1 ratio)")
    
    best_val_acc = 0
    best_model_state = None
    
    print(f"\nStarting training for {epochs} epochs...")
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        
        # Standard training without oversampling
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        
        # Loss computed on training set only
        loss = criterion(out[train_mask], data.y[train_mask])
        
        # Check for NaN
        if torch.isnan(loss):
            print(f"Warning: NaN loss at epoch {epoch+1}, skipping")
            continue
        
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Validation
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                logits = model(data.x, data.edge_index)
                
                # Train metrics
                train_preds = logits[train_mask].argmax(dim=1).cpu()
                train_labels = data.y[train_mask].cpu()
                train_acc = accuracy_score(train_labels, train_preds)
                
                # Val metrics
                val_preds = logits[val_mask].argmax(dim=1).cpu()
                val_labels = data.y[val_mask].cpu()
                val_acc = accuracy_score(val_labels, val_preds)
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_model_state = model.state_dict().copy()
                
                print(f"Epoch {epoch+1:3d} | Loss: {loss.item():.4f} | "
                      f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
    
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f}s")
    
    # Load best model
    model.load_state_dict(best_model_state)
    return model

def evaluate_gnn(model, data, test_mask):
    """
    Comprehensive evaluation of GNN model.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        probs = F.softmax(logits, dim=1)
        preds = logits[test_mask].argmax(dim=1).cpu().numpy()
        labels = data.y[test_mask].cpu().numpy()
        probs_fraud = probs[test_mask, 1].cpu().numpy()
        
        # Compute all metrics
        acc = accuracy_score(labels, preds)
        precision = precision_score(labels, preds, zero_division=0)
        recall = recall_score(labels, preds, zero_division=0)
        f1 = f1_score(labels, preds, zero_division=0)
        roc_auc = roc_auc_score(labels, probs_fraud) if len(np.unique(labels)) > 1 else 0.0
        
        print(f"\n{'='*60}")
        print(f"TEST SET EVALUATION RESULTS")
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
    # Path to balanced dataset
    csv_path = "data/ibm/ibm_fraud_29k_nonfraud_60k.csv"

    # Load graph
    data = load_preprocessed_graph(csv_path)
    num_nodes = data.num_nodes

    # Stratified split: 70% train, 15% val, 15% test
    all_indices = np.arange(num_nodes)
    train_idx, test_val_idx = train_test_split(
        all_indices,
        test_size=0.3,
        random_state=42,
        stratify=data.y.numpy()
    )
    val_idx, test_idx = train_test_split(
        test_val_idx,
        test_size=0.5,
        random_state=42,
        stratify=data.y[test_val_idx].numpy()
    )

    # Create masks
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    print(f"\nDataset split:")
    print(f"  Train: {train_mask.sum():,} nodes ({train_mask.sum()/num_nodes*100:.1f}%)")
    print(f"  Val:   {val_mask.sum():,} nodes ({val_mask.sum()/num_nodes*100:.1f}%)")
    print(f"  Test:  {test_mask.sum():,} nodes ({test_mask.sum()/num_nodes*100:.1f}%)")

    # Show fraud distribution in each set
    print(f"\nFraud distribution:")
    print(f"  Train: {data.y[train_mask].sum().item():,} fraud / {train_mask.sum().item():,} total ({data.y[train_mask].float().mean()*100:.2f}%)")
    print(f"  Val:   {data.y[val_mask].sum().item():,} fraud / {val_mask.sum().item():,} total ({data.y[val_mask].float().mean()*100:.2f}%)")
    print(f"  Test:  {data.y[test_mask].sum().item():,} fraud / {test_mask.sum().item():,} total ({data.y[test_mask].float().mean()*100:.2f}%)")
    
    # Train model
    model = train_gnn(data, train_mask, val_mask, epochs=25, lr=0.01, hidden_dim=64)
    
    # Evaluate on test set
    results = evaluate_gnn(model, data, test_mask)
    
    print("\nBaseline GNN training complete!")
    print(f"Use these results to compare with HMSTA model performance.")
