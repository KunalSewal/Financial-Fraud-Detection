"""
Multi-Path Temporal Graph Neural Network (MPTGNN).

Based on:
Saldaña-Ulloa et al., "A Temporal Graph Network Algorithm for Detecting 
Fraudulent Transactions on Online Payment Platforms"
Algorithms, 17(12), 552, 2024

Key innovations:
- Multiple propagation paths (short-term and long-term)
- Temporal attention mechanism
- Path-wise aggregation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List


class TemporalAttention(nn.Module):
    """
    Temporal attention mechanism for weighting different time windows.
    """
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, temporal_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            temporal_features: [batch, num_paths, hidden_dim]
            
        Returns:
            Weighted features [batch, hidden_dim]
        """
        # Compute attention scores
        scores = self.attention(temporal_features)  # [batch, num_paths, 1]
        weights = F.softmax(scores, dim=1)
        
        # Weighted sum
        output = (temporal_features * weights).sum(dim=1)
        
        return output


class MultiPathConvolution(nn.Module):
    """
    Multi-path graph convolution that processes different temporal ranges.
    
    Paths:
    - Short-term: Recent interactions (high weight on recent edges)
    - Medium-term: Intermediate history
    - Long-term: Full historical context
    """
    
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_paths: int = 3,
        edge_dim: int = 4
    ):
        super().__init__()
        self.num_paths = num_paths
        
        # Separate convolution for each path
        self.path_convs = nn.ModuleList([
            nn.Linear(in_dim + edge_dim, out_dim)
            for _ in range(num_paths)
        ])
        
        # Temporal decay parameters (learnable)
        self.temporal_decay = nn.Parameter(torch.ones(num_paths))
        
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_time: torch.Tensor,
        current_time: float
    ) -> torch.Tensor:
        """
        Args:
            x: Node features [num_nodes, in_dim]
            edge_index: [2, num_edges]
            edge_attr: [num_edges, edge_dim]
            edge_time: [num_edges] timestamps
            current_time: Current timestamp
            
        Returns:
            Multi-path features [num_nodes, num_paths, out_dim]
        """
        num_nodes = x.size(0)
        out_dim = self.path_convs[0].out_features
        
        # Initialize output
        path_outputs = []
        
        # Compute time deltas
        time_delta = current_time - edge_time
        
        for path_idx in range(self.num_paths):
            # Temporal weighting (exponential decay)
            # Different paths have different decay rates
            decay_rate = self.temporal_decay[path_idx]
            temporal_weight = torch.exp(-decay_rate * time_delta)
            
            # Source and destination nodes
            src = edge_index[0]
            dst = edge_index[1]
            
            # Get source features and edge features
            src_features = x[src]
            combined = torch.cat([src_features, edge_attr], dim=-1)
            
            # Transform
            messages = self.path_convs[path_idx](combined)
            
            # Weight by temporal importance
            weighted_messages = messages * temporal_weight.unsqueeze(-1)
            
            # Aggregate to destination nodes
            aggregated = torch.zeros(num_nodes, out_dim, device=x.device)
            aggregated.scatter_add_(
                0,
                dst.unsqueeze(-1).expand_as(weighted_messages),
                weighted_messages
            )
            
            path_outputs.append(aggregated)
        
        # Stack paths: [num_nodes, num_paths, out_dim]
        return torch.stack(path_outputs, dim=1)


class MPTGNN(nn.Module):
    """
    Multi-Path Temporal Graph Neural Network.
    
    Architecture:
    1. Multi-path temporal convolutions (short/medium/long-term)
    2. Temporal attention (weight different paths)
    3. Path aggregation
    4. Classification layer
    """
    
    def __init__(
        self,
        node_dim: int,
        edge_dim: int = 4,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_paths: int = 3,
        num_classes: int = 2,
        dropout: float = 0.1
    ):
        """
        Args:
            node_dim: Node feature dimension
            edge_dim: Edge feature dimension
            hidden_dim: Hidden dimension
            num_layers: Number of MP-TGN layers
            num_paths: Number of temporal paths (default: 3)
            num_classes: Number of output classes
            dropout: Dropout probability
        """
        super().__init__()
        
        self.node_dim = node_dim
        self.num_paths = num_paths
        self.num_layers = num_layers
        
        # Input projection
        self.input_proj = nn.Linear(node_dim, hidden_dim)
        
        # Multi-path convolution layers
        self.mp_convs = nn.ModuleList([
            MultiPathConvolution(hidden_dim, hidden_dim, num_paths, edge_dim)
            for _ in range(num_layers)
        ])
        
        # Temporal attention layers
        self.temporal_attentions = nn.ModuleList([
            TemporalAttention(hidden_dim)
            for _ in range(num_layers)
        ])
        
        # Batch normalization
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim)
            for _ in range(num_layers)
        ])
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Output classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        edge_time: Optional[torch.Tensor] = None,
        current_time: Optional[float] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Node features [num_nodes, node_dim]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge features [num_edges, edge_dim]
            edge_time: Edge timestamps [num_edges]
            current_time: Current time (for temporal weighting)
            
        Returns:
            Logits [num_nodes, num_classes]
        """
        num_nodes = x.size(0)
        
        # Default edge features and times if not provided
        if edge_attr is None:
            edge_attr = torch.ones(edge_index.size(1), 1, device=x.device)
        
        if edge_time is None:
            edge_time = torch.zeros(edge_index.size(1), device=x.device)
        
        if current_time is None:
            current_time = edge_time.max().item() if len(edge_time) > 0 else 0.0
        
        # Input projection
        h = self.input_proj(x)
        h = F.relu(h)
        h = self.dropout(h)
        
        # Multi-path temporal convolutions
        for i in range(self.num_layers):
            # Multi-path convolution
            path_features = self.mp_convs[i](
                h, edge_index, edge_attr, edge_time, current_time
            )  # [num_nodes, num_paths, hidden_dim]
            
            # Temporal attention aggregation
            h_new = self.temporal_attentions[i](path_features)
            
            # Residual connection
            h = h + h_new
            
            # Batch norm
            h = self.batch_norms[i](h)
            
            # Activation and dropout
            h = F.relu(h)
            h = self.dropout(h)
        
        # Classification
        logits = self.classifier(h)
        
        return logits
    
    def get_path_weights(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        edge_time: Optional[torch.Tensor] = None,
        current_time: Optional[float] = None
    ) -> List[torch.Tensor]:
        """
        Get attention weights for each temporal path (for visualization).
        
        Returns:
            List of attention weights per layer [num_nodes, num_paths]
        """
        num_nodes = x.size(0)
        
        if edge_attr is None:
            edge_attr = torch.ones(edge_index.size(1), 1, device=x.device)
        if edge_time is None:
            edge_time = torch.zeros(edge_index.size(1), device=x.device)
        if current_time is None:
            current_time = edge_time.max().item() if len(edge_time) > 0 else 0.0
        
        h = self.input_proj(x)
        h = F.relu(h)
        
        attention_weights = []
        
        for i in range(self.num_layers):
            path_features = self.mp_convs[i](
                h, edge_index, edge_attr, edge_time, current_time
            )
            
            # Get attention scores
            scores = self.temporal_attentions[i].attention(path_features)
            weights = F.softmax(scores, dim=1).squeeze(-1)
            attention_weights.append(weights)
            
            # Continue forward pass
            h_new = (path_features * weights.unsqueeze(-1)).sum(dim=1)
            h = h + h_new
            h = self.batch_norms[i](h)
            h = F.relu(h)
        
        return attention_weights
