"""
Simple baseline to test if training loop works.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score
import pandas as pd

from src.data.temporal_graph_builder import load_and_build_temporal_graph

class SimpleGNN(nn.Module):
    """Simple 2-layer MLP baseline to test training."""
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 2)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def train_simple():
    print("Training simple MLP baseline...")
    
    # Load data (same as HMSTA)
    graph_dict = load_and_build_temporal_graph('ethereum', 'data/transaction_dataset.csv')
    num_nodes = graph_dict['num_nodes']
    node_to_id = graph_dict['node_to_id']
    
    # Load features
    df = pd.read_csv('data/transaction_dataset.csv')
    node_feature_dim = 166
    node_features = torch.zeros((num_nodes, node_feature_dim), dtype=torch.float32)
    node_labels = torch.zeros(num_nodes, dtype=torch.long)
    
    for addr in df['Address'].unique():
        if addr in node_to_id:
            node_id = node_to_id[addr]
            addr_data = df[df['Address'] == addr]
            node_labels[node_id] = int(addr_data['FLAG'].iloc[-1])
            
            features = []
            for col in df.columns:
                if col not in ['Index', 'Address', 'FLAG'] and pd.api.types.is_numeric_dtype(df[col]):
                    val = addr_data[col].mean()
                    if pd.isna(val) or np.isinf(val):
                        val = 0.0
                    features.append(val)
            
            features = features[:node_feature_dim]
            if len(features) < node_feature_dim:
                features.extend([0.0] * (node_feature_dim - len(features)))
            node_features[node_id] = torch.tensor(features, dtype=torch.float32)
    
    # Normalize
    node_features = torch.nan_to_num(node_features, nan=0.0, posinf=1.0, neginf=-1.0)
    feat_min = node_features.min(dim=0, keepdim=True)[0]
    feat_max = node_features.max(dim=0, keepdim=True)[0]
    feat_range = feat_max - feat_min
    feat_range[feat_range == 0] = 1.0
    node_features = (node_features - feat_min) / feat_range
    node_features = torch.nan_to_num(node_features, nan=0.0, posinf=1.0, neginf=-1.0)
    
    # Splits
    num_nodes = node_features.size(0)
    indices = torch.randperm(num_nodes)
    train_size = int(0.6 * num_nodes)
    val_size = int(0.2 * num_nodes)
    
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    train_mask[indices[:train_size]] = True
    val_mask[indices[train_size:train_size + val_size]] = True
    test_mask[indices[train_size + val_size:]] = True
    
    print(f"Train: {train_mask.sum()}, Val: {val_mask.sum()}, Test: {test_mask.sum()}")
    print(f"Features shape: {node_features.shape}")
    print(f"Labels: Fraud={node_labels.sum()}, Normal={(node_labels==0).sum()}")
    
    # Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleGNN(node_feature_dim, hidden_dim=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Class weights
    fraud_count = node_labels[train_mask].sum().item()
    normal_count = (node_labels[train_mask] == 0).sum().item()
    total = fraud_count + normal_count
    weight_fraud = total / (2.0 * fraud_count)
    weight_normal = total / (2.0 * normal_count)
    class_weights = torch.tensor([weight_normal, weight_fraud], device=device)
    
    print(f"\\nClass weights: Normal={weight_normal:.2f}, Fraud={weight_fraud:.2f}")
    
    # Train
    node_features = node_features.to(device)
    node_labels = node_labels.to(device)
    
    best_val_auc = 0
    for epoch in range(50):
        model.train()
        optimizer.zero_grad()
        
        logits = model(node_features)
        loss = F.cross_entropy(logits[train_mask], node_labels[train_mask], weight=class_weights)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Evaluate
        if epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                logits = model(node_features)
                probs = F.softmax(logits, dim=1)
                
                train_probs = probs[train_mask, 1].cpu().numpy()
                train_labels = node_labels[train_mask].cpu().numpy()
                train_auc = roc_auc_score(train_labels, train_probs)
                
                val_probs = probs[val_mask, 1].cpu().numpy()
                val_labels = node_labels[val_mask].cpu().numpy()
                val_auc = roc_auc_score(val_labels, val_probs)
                
                preds = logits[val_mask].argmax(dim=1).cpu().numpy()
                val_f1 = f1_score(val_labels, preds)
                
                print(f"Epoch {epoch:2d} | Loss: {loss.item():.4f} | Train AUC: {train_auc:.4f} | Val AUC: {val_auc:.4f} | Val F1: {val_f1:.4f}")
                
                if val_auc > best_val_auc:
                    best_val_auc = val_auc
    
    # Final test
    model.eval()
    with torch.no_grad():
        logits = model(node_features)
        probs = F.softmax(logits, dim=1)
        
        test_probs = probs[test_mask, 1].cpu().numpy()
        test_labels = node_labels[test_mask].cpu().numpy()
        test_auc = roc_auc_score(test_labels, test_probs)
        
        preds = logits[test_mask].argmax(dim=1).cpu().numpy()
        test_f1 = f1_score(test_labels, preds)
        
        print(f"\\n✅ Final Test | AUC: {test_auc:.4f} | F1: {test_f1:.4f}")

if __name__ == '__main__':
    train_simple()
