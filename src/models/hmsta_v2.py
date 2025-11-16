"""
HMSTA v2: Clean Incremental Implementation for Ablation Study

This file contains 6 progressive versions of HMSTA architecture:
- v0: Baseline MLP (no graph, no temporal)
- v1: + Graph Convolution (uses edge_index)
- v2: + Temporal Encoding (uses timestamps)
- v3: + Temporal Memory (GRU-based memory per node)
- v4: + Multi-Path Reasoning (short/long-term aggregation)
- v5: + Anomaly Attention (fraud-specific attention)

Each version builds incrementally on the previous one for proper ablation study.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
try:
    from torch_geometric.utils import scatter
except ImportError:
    # Fallback if torch-scatter not available
    from torch_scatter import scatter


# =============================================================================
# Version 0: Baseline MLP (No Graph, No Temporal)
# =============================================================================

class HMSTA_v0_Baseline(nn.Module):
    """
    Baseline MLP for comparison.
    Only uses node features, ignores graph structure and temporal information.
    """
    def __init__(self, node_features, hidden_dim=128, dropout=0.5):
        super().__init__()
        self.input_proj = nn.Linear(node_features, hidden_dim)
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
    
    def forward(self, x, edge_index=None, edge_attr=None, timestamps=None):
        """
        Args:
            x: Node features [num_nodes, node_features]
            edge_index: Not used in baseline
            edge_attr: Not used in baseline
            timestamps: Not used in baseline
        
        Returns:
            logits: [num_nodes, 2] class predictions
        """
        h = self.input_proj(x)
        h = F.relu(h)
        h = F.dropout(h, p=0.5, training=self.training)
        logits = self.classifier(h)
        return logits


# =============================================================================
# Version 1: + Graph Convolution
# =============================================================================

class HMSTA_v1_Graph(nn.Module):
    """
    Baseline + Graph Convolution.
    Uses graph structure via GCN layer.
    """
    def __init__(self, node_features, hidden_dim=128, dropout=0.5):
        super().__init__()
        self.input_proj = nn.Linear(node_features, hidden_dim)
        self.gcn = GCNConv(hidden_dim, hidden_dim)
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
        """
        Args:
            x: Node features [num_nodes, node_features]
            edge_index: Graph edges [2, num_edges] ← USED!
            edge_attr: Not used yet
            timestamps: Not used yet
        
        Returns:
            logits: [num_nodes, 2] class predictions
        """
        h = self.input_proj(x)
        h = F.relu(h)
        h = self.gcn(h, edge_index)  # ← Graph convolution applied
        h = F.dropout(h, p=0.5, training=self.training)
        logits = self.classifier(h)
        return logits


# =============================================================================
# Version 2: + Temporal Encoding
# =============================================================================

class HMSTA_v2_Temporal(nn.Module):
    """
    v1 + Temporal Encoding.
    Uses graph structure + adds per-node temporal context.
    """
    def __init__(self, node_features, hidden_dim=128, dropout=0.5):
        super().__init__()
        self.input_proj = nn.Linear(node_features, hidden_dim)
        self.time_encoder = nn.Linear(1, hidden_dim)
        self.gcn = GCNConv(hidden_dim, hidden_dim)
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
        """
        Args:
            x: Node features [num_nodes, node_features]
            edge_index: Graph edges [2, num_edges] ← USED
            edge_attr: Not used yet
            timestamps: Edge timestamps [num_edges] OR node timestamps [num_nodes] ← USED!
        
        Returns:
            logits: [num_nodes, 2] class predictions
        """
        h = self.input_proj(x)
        
        # Add temporal encoding per node
        if timestamps is not None:
            num_nodes = x.size(0)
            
            # Check if timestamps are edge-level or node-level
            if timestamps.size(0) == edge_index.size(1):
                # Edge timestamps: scatter to nodes (max timestamp of incoming edges)
                node_times = scatter(timestamps, edge_index[1], dim=0, dim_size=num_nodes, reduce='max')
            else:
                # Node timestamps: use directly
                node_times = timestamps
            
            # Normalize to [0, 1] range
            if node_times.max() > node_times.min():
                node_times = (node_times - node_times.min()) / (node_times.max() - node_times.min() + 1e-8)
            time_emb = self.time_encoder(node_times.unsqueeze(-1))
            h = h + time_emb  # ← Add temporal context
        
        h = F.relu(h)
        h = self.gcn(h, edge_index)
        h = F.dropout(h, p=0.5, training=self.training)
        logits = self.classifier(h)
        return logits


# =============================================================================
# Version 3: + Temporal Memory
# =============================================================================

class TemporalMemoryModule(nn.Module):
    """Simple GRU-based memory for temporal patterns (TGN-lite)"""
    def __init__(self, memory_dim):
        super().__init__()
        self.memory_dim = memory_dim
        self.gru = nn.GRUCell(memory_dim, memory_dim)
        self.memory = None
    
    def reset(self, num_nodes, device):
        """Initialize memory for all nodes"""
        self.memory = torch.zeros(num_nodes, self.memory_dim, device=device)
    
    def update(self, node_ids, messages):
        """
        Update memory for specific nodes
        Args:
            node_ids: [num_updates] node indices to update
            messages: [num_updates, memory_dim] new information
        """
        if self.memory is None:
            raise RuntimeError("Memory not initialized. Call reset() first.")
        
        old_memory = self.memory[node_ids]
        new_memory = self.gru(messages, old_memory)
        # Use clone to avoid inplace modification
        self.memory = self.memory.clone()
        self.memory[node_ids] = new_memory
    
    def get(self, node_ids=None):
        """Get memory for specific nodes or all nodes"""
        if self.memory is None:
            raise RuntimeError("Memory not initialized. Call reset() first.")
        if node_ids is None:
            return self.memory
        return self.memory[node_ids]


class HMSTA_v3_Memory(nn.Module):
    """
    v2 + Temporal Memory.
    Uses graph + temporal encoding + GRU-based memory per node.
    """
    def __init__(self, node_features, hidden_dim=128, dropout=0.5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_proj = nn.Linear(node_features, hidden_dim)
        self.time_encoder = nn.Linear(1, hidden_dim)
        self.memory_module = TemporalMemoryModule(hidden_dim)
        self.gcn = GCNConv(hidden_dim, hidden_dim)
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
    
    def reset_memory(self, num_nodes, device):
        """Initialize memory before training/inference"""
        self.memory_module.reset(num_nodes, device)
    
    def forward(self, x, edge_index, edge_attr=None, timestamps=None):
        """
        Args:
            x: Node features [num_nodes, node_features]
            edge_index: Graph edges [2, num_edges] ← USED
            edge_attr: Not used yet
            timestamps: Edge timestamps [num_edges] ← USED
        
        Returns:
            logits: [num_nodes, 2] class predictions
        """
        num_nodes = x.size(0)
        device = x.device
        
        # Initialize memory if needed
        if self.memory_module.memory is None:
            self.reset_memory(num_nodes, device)
        
        h = self.input_proj(x)
        
        # Add temporal encoding
        if timestamps is not None:
            # Check if timestamps are edge-level or node-level
            if timestamps.size(0) == edge_index.size(1):
                # Edge timestamps: scatter to nodes
                node_times = scatter(timestamps, edge_index[1], dim=0, dim_size=num_nodes, reduce='max')
            else:
                # Node timestamps: use directly
                node_times = timestamps
            
            if node_times.max() > node_times.min():
                node_times = (node_times - node_times.min()) / (node_times.max() - node_times.min() + 1e-8)
            time_emb = self.time_encoder(node_times.unsqueeze(-1))
            h = h + time_emb
        
        h = F.relu(h)
        
        # Retrieve and update memory
        memory = self.memory_module.get()
        h = h + memory  # ← Incorporate temporal memory
        
        # Update memory with new information (for nodes involved in edges)
        # Detach to prevent backprop through memory update
        unique_nodes = torch.unique(edge_index.flatten())
        self.memory_module.update(unique_nodes, h[unique_nodes].detach())
        
        h = self.gcn(h, edge_index)
        h = F.dropout(h, p=0.5, training=self.training)
        logits = self.classifier(h)
        return logits


# =============================================================================
# Version 4: + Multi-Path Reasoning
# =============================================================================

class MultiPathAggregator(nn.Module):
    """Aggregates multiple temporal paths with learned weights"""
    def __init__(self, hidden_dim, num_paths=2):
        super().__init__()
        self.num_paths = num_paths
        self.attention = nn.Linear(hidden_dim, num_paths)
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, path_embeddings):
        """
        Args:
            path_embeddings: List of [num_nodes, hidden_dim] tensors
        
        Returns:
            aggregated: [num_nodes, hidden_dim]
        """
        # Stack paths: [num_nodes, num_paths, hidden_dim]
        stacked = torch.stack(path_embeddings, dim=1)
        
        # Compute attention weights for each path
        # Average over hidden_dim to get per-path scores
        path_scores = stacked.mean(dim=-1)  # [num_nodes, num_paths]
        weights = F.softmax(path_scores, dim=-1).unsqueeze(-1)  # [num_nodes, num_paths, 1]
        
        # Weighted aggregation
        aggregated = (stacked * weights).sum(dim=1)  # [num_nodes, hidden_dim]
        aggregated = self.layer_norm(aggregated)
        
        return aggregated


class HMSTA_v4_MultiPath(nn.Module):
    """
    v3 + Multi-Path Reasoning.
    Uses graph + temporal + memory + multi-scale temporal paths.
    """
    def __init__(self, node_features, hidden_dim=128, dropout=0.5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_proj = nn.Linear(node_features, hidden_dim)
        self.time_encoder = nn.Linear(1, hidden_dim)
        self.memory_module = TemporalMemoryModule(hidden_dim)
        
        # Two paths: recent (short-term) and historical (long-term)
        self.gcn_recent = GCNConv(hidden_dim, hidden_dim)
        self.gcn_history = GCNConv(hidden_dim, hidden_dim)
        self.path_aggregator = MultiPathAggregator(hidden_dim, num_paths=2)
        
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
    
    def reset_memory(self, num_nodes, device):
        """Initialize memory before training/inference"""
        self.memory_module.reset(num_nodes, device)
    
    def forward(self, x, edge_index, edge_attr=None, timestamps=None):
        """
        Args:
            x: Node features [num_nodes, node_features]
            edge_index: Graph edges [2, num_edges] ← USED
            edge_attr: Not used yet
            timestamps: Edge timestamps [num_edges] ← USED for filtering
        
        Returns:
            logits: [num_nodes, 2] class predictions
        """
        num_nodes = x.size(0)
        device = x.device
        
        # Initialize memory if needed
        if self.memory_module.memory is None:
            self.reset_memory(num_nodes, device)
        
        h = self.input_proj(x)
        
        # Add temporal encoding
        if timestamps is not None:
            # Check if timestamps are edge-level or node-level
            if timestamps.size(0) == edge_index.size(1):
                # Edge timestamps: scatter to nodes
                node_times = scatter(timestamps, edge_index[1], dim=0, dim_size=num_nodes, reduce='max')
            else:
                # Node timestamps: use directly
                node_times = timestamps
            
            if node_times.max() > node_times.min():
                node_times = (node_times - node_times.min()) / (node_times.max() - node_times.min() + 1e-8)
            time_emb = self.time_encoder(node_times.unsqueeze(-1))
            h = h + time_emb
        
        h = F.relu(h)
        
        # Retrieve and update memory
        memory = self.memory_module.get()
        h = h + memory
        unique_nodes = torch.unique(edge_index.flatten())
        self.memory_module.update(unique_nodes, h[unique_nodes].detach())
        
        # Multi-path reasoning: split edges by recency
        if timestamps is not None:
            # Recent path: top 50% most recent edges
            median_time = timestamps.median()
            recent_mask = timestamps >= median_time
            recent_edges = edge_index[:, recent_mask]
            history_edges = edge_index[:, ~recent_mask]
        else:
            # If no timestamps, use all edges for both paths
            recent_edges = edge_index
            history_edges = edge_index
        
        # Process each path separately
        h_recent = self.gcn_recent(h, recent_edges) if recent_edges.size(1) > 0 else h
        h_history = self.gcn_history(h, history_edges) if history_edges.size(1) > 0 else h
        
        # Aggregate paths
        h = self.path_aggregator([h_recent, h_history])  # ← Multi-scale aggregation
        
        h = F.dropout(h, p=0.5, training=self.training)
        logits = self.classifier(h)
        return logits


# =============================================================================
# Version 5: + Anomaly Attention (Full HMSTA)
# =============================================================================

class AnomalyAttentionModule(nn.Module):
    """Fraud-specific attention mechanism"""
    def __init__(self, hidden_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.attention = nn.MultiheadAttention(
            hidden_dim, 
            num_heads, 
            dropout=dropout,
            batch_first=False
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Learnable anomaly query vectors
        self.anomaly_queries = nn.Parameter(torch.randn(num_heads, hidden_dim))
    
    def forward(self, x):
        """
        Args:
            x: Node embeddings [num_nodes, hidden_dim]
        
        Returns:
            attended: [num_nodes, hidden_dim] with anomaly-focused attention
        """
        # Prepare for attention: [seq_len, batch, hidden_dim]
        x_t = x.unsqueeze(1).transpose(0, 1)  # [num_nodes, 1, hidden_dim] -> [1, num_nodes, hidden_dim]
        
        # Use anomaly queries
        queries = self.anomaly_queries.mean(dim=0, keepdim=True).unsqueeze(1)  # [1, 1, hidden_dim]
        queries = queries.expand(1, x.size(0), -1)  # [1, num_nodes, hidden_dim]
        
        # Self-attention with anomaly bias
        attended, _ = self.attention(queries, x_t, x_t)  # [1, num_nodes, hidden_dim]
        attended = attended.squeeze(0)  # [num_nodes, hidden_dim]
        
        # Residual connection and layer norm
        out = self.layer_norm(x + attended)
        
        return out


class HMSTA_v5_Full(nn.Module):
    """
    Full HMSTA: v4 + Anomaly Attention.
    Complete architecture with all components:
    - Graph convolution
    - Temporal encoding
    - Temporal memory
    - Multi-path reasoning
    - Fraud-specific attention
    """
    def __init__(self, node_features, hidden_dim=128, dropout=0.5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_proj = nn.Linear(node_features, hidden_dim)
        self.time_encoder = nn.Linear(1, hidden_dim)
        self.memory_module = TemporalMemoryModule(hidden_dim)
        
        # Multi-path convolutions
        self.gcn_recent = GCNConv(hidden_dim, hidden_dim)
        self.gcn_history = GCNConv(hidden_dim, hidden_dim)
        self.path_aggregator = MultiPathAggregator(hidden_dim, num_paths=2)
        
        # Anomaly-aware attention
        self.anomaly_attention = AnomalyAttentionModule(hidden_dim, num_heads=4)
        
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
    
    def reset_memory(self, num_nodes, device):
        """Initialize memory before training/inference"""
        self.memory_module.reset(num_nodes, device)
    
    def forward(self, x, edge_index, edge_attr=None, timestamps=None):
        """
        Complete forward pass using all HMSTA components.
        
        Args:
            x: Node features [num_nodes, node_features]
            edge_index: Graph edges [2, num_edges] ← USED
            edge_attr: Edge features (optional)
            timestamps: Edge timestamps [num_edges] ← USED
        
        Returns:
            logits: [num_nodes, 2] class predictions
        """
        num_nodes = x.size(0)
        device = x.device
        
        # Initialize memory if needed
        if self.memory_module.memory is None:
            self.reset_memory(num_nodes, device)
        
        # 1. Input projection
        h = self.input_proj(x)
        
        # 2. Temporal encoding
        if timestamps is not None:
            # Check if timestamps are edge-level or node-level
            if timestamps.size(0) == edge_index.size(1):
                # Edge timestamps: scatter to nodes
                node_times = scatter(timestamps, edge_index[1], dim=0, dim_size=num_nodes, reduce='max')
            else:
                # Node timestamps: use directly
                node_times = timestamps
            
            if node_times.max() > node_times.min():
                node_times = (node_times - node_times.min()) / (node_times.max() - node_times.min() + 1e-8)
            time_emb = self.time_encoder(node_times.unsqueeze(-1))
            h = h + time_emb
        
        h = F.relu(h)
        
        # 3. Temporal memory
        memory = self.memory_module.get()
        h = h + memory
        unique_nodes = torch.unique(edge_index.flatten())
        self.memory_module.update(unique_nodes, h[unique_nodes])
        
        # 4. Multi-path reasoning
        if timestamps is not None:
            median_time = timestamps.median()
            recent_mask = timestamps >= median_time
            recent_edges = edge_index[:, recent_mask]
            history_edges = edge_index[:, ~recent_mask]
        else:
            recent_edges = edge_index
            history_edges = edge_index
        
        h_recent = self.gcn_recent(h, recent_edges) if recent_edges.size(1) > 0 else h
        h_history = self.gcn_history(h, history_edges) if history_edges.size(1) > 0 else h
        h = self.path_aggregator([h_recent, h_history])
        
        # 5. Anomaly-aware attention
        h = self.anomaly_attention(h)  # ← Fraud-specific attention applied
        
        h = F.dropout(h, p=0.5, training=self.training)
        logits = self.classifier(h)
        return logits


# =============================================================================
# Factory Function
# =============================================================================

def create_hmsta_model(version, node_features, hidden_dim=128, dropout=0.5):
    """
    Factory function to create HMSTA models for ablation study.
    
    Args:
        version: 0-5 indicating which version to create
        node_features: Number of input features
        hidden_dim: Hidden layer dimension
        dropout: Dropout rate
    
    Returns:
        model: HMSTA model of specified version
    """
    models = {
        0: HMSTA_v0_Baseline,
        1: HMSTA_v1_Graph,
        2: HMSTA_v2_Temporal,
        3: HMSTA_v3_Memory,
        4: HMSTA_v4_MultiPath,
        5: HMSTA_v5_Full
    }
    
    if version not in models:
        raise ValueError(f"Invalid version {version}. Must be 0-5.")
    
    return models[version](node_features, hidden_dim, dropout)


# Version descriptions for reporting
VERSION_DESCRIPTIONS = {
    0: "Baseline MLP (no graph, no temporal)",
    1: "+ Graph Convolution",
    2: "+ Temporal Encoding",
    3: "+ Temporal Memory (GRU)",
    4: "+ Multi-Path Reasoning",
    5: "+ Anomaly Attention (Full HMSTA)"
}
