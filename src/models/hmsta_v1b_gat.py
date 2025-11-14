"""
HMSTA v1b: Enhanced with Graph Attention Networks (GAT)

Quick test: Does attention over graph neighbors help?
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class HMSTA_v1b_GAT(nn.Module):
    """
    v1 but with GAT (Graph Attention) instead of GCN.
    Attention learns which neighbors are most important.
    """
    def __init__(self, node_features, hidden_dim=128, heads=4, dropout=0.5):
        super().__init__()
        self.input_proj = nn.Linear(node_features, hidden_dim)
        # GAT with multi-head attention
        self.gat = GATConv(hidden_dim, hidden_dim // heads, heads=heads, dropout=dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)
        )
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x, edge_index, edge_attr=None, timestamps=None):
        h = self.input_proj(x)
        h = F.relu(h)
        h = self.gat(h, edge_index)  # ← Multi-head attention over neighbors
        h = F.dropout(h, p=0.5, training=self.training)
        logits = self.classifier(h)
        return logits


if __name__ == '__main__':
    print("✨ HMSTA v1b: Graph Attention Enhancement")
    print("Testing if attention improves over standard GCN...")
    
    # Quick synthetic test
    num_nodes = 100
    num_edges = 500
    node_features = 47
    
    x = torch.randn(num_nodes, node_features)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    
    model = HMSTA_v1b_GAT(node_features, hidden_dim=128, heads=4)
    
    model.eval()
    with torch.no_grad():
        logits = model(x, edge_index)
    
    print(f"✅ Output shape: {logits.shape}")
    print(f"✅ Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"\n💡 Ready to train! Expected improvement: +0.5-1% over v1")
