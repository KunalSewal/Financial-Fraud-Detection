"""
Model implementations for financial fraud detection.
Includes baseline models (MLP, GraphSAGE) and temporal models (TGN, TGAT).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv, TransformerConv
from torch_geometric.nn import global_mean_pool, global_max_pool


class MLPClassifier(nn.Module):
    """
    Simple Multi-Layer Perceptron baseline.
    Uses only node features, ignores graph structure.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = [128, 64],
        num_classes: int = 2,
        dropout: float = 0.3
    ):
        """
        Initialize MLP classifier.
        
        Args:
            input_dim: Input feature dimension
            hidden_dims: List of hidden layer dimensions
            num_classes: Number of output classes
            dropout: Dropout probability
        """
        super(MLPClassifier, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes
        self.dropout = dropout
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Node features [num_nodes, input_dim]
            
        Returns:
            predictions [num_nodes, num_classes]
        """
        return self.network(x)


class GraphSAGE(nn.Module):
    """
    GraphSAGE model for node classification.
    Static GNN baseline that aggregates neighborhood information.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.3,
        aggregator: str = "mean"
    ):
        """
        Initialize GraphSAGE model.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            num_layers: Number of SAGE layers
            num_classes: Number of output classes
            dropout: Dropout probability
            aggregator: Aggregation function ("mean", "max", "lstm")
        """
        super(GraphSAGE, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.dropout = dropout
        
        # Build SAGE layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # Input layer
        self.convs.append(SAGEConv(input_dim, hidden_dim))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Output layer
        self.output = nn.Linear(hidden_dim, num_classes)
        
        self.dropout_layer = nn.Dropout(dropout)
    
    def forward(self, x, edge_index):
        """
        Forward pass.
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]
            
        Returns:
            predictions [num_nodes, num_classes]
        """
        # Apply SAGE layers
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = self.dropout_layer(x)
        
        # Output layer
        x = self.output(x)
        
        return x


class TemporalGNN(nn.Module):
    """
    Simple temporal GNN with temporal attention.
    Processes sequences of graph snapshots.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.3
    ):
        """
        Initialize Temporal GNN.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            num_layers: Number of GNN layers
            num_classes: Number of output classes
            dropout: Dropout probability
        """
        super(TemporalGNN, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        
        # Spatial layers (GNN)
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.convs.append(SAGEConv(input_dim, hidden_dim))
            else:
                self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        
        # Temporal layer (LSTM)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )
        
        # Output layer
        self.output = nn.Linear(hidden_dim, num_classes)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, edge_index, temporal_mask=None):
        """
        Forward pass.
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]
            temporal_mask: Optional temporal ordering mask
            
        Returns:
            predictions [num_nodes, num_classes]
        """
        # Spatial aggregation
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = self.dropout(x)
        
        # Temporal aggregation (if temporal mask provided)
        if temporal_mask is not None:
            # Reshape for LSTM: [batch, seq_len, features]
            x = x.unsqueeze(0)
            x, _ = self.lstm(x)
            x = x.squeeze(0)
        
        # Output
        x = self.output(x)
        
        return x


class TGN(nn.Module):
    """
    Temporal Graph Network (TGN) with memory module.
    Based on: Rossi et al., "Temporal Graph Networks for Deep Learning on Dynamic Graphs" (ICML 2020)
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        memory_dim: int = 128,
        time_dim: int = 32,
        num_classes: int = 2,
        dropout: float = 0.3
    ):
        """
        Initialize TGN model.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            memory_dim: Memory dimension
            time_dim: Time encoding dimension
            num_classes: Number of output classes
            dropout: Dropout probability
        """
        super(TGN, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.memory_dim = memory_dim
        self.time_dim = time_dim
        self.num_classes = num_classes
        
        # Memory initialization
        self.memory = None
        self.last_update = None
        
        # Time encoding
        self.time_encoder = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.ReLU(),
            nn.Linear(time_dim, time_dim)
        )
        
        # Message function
        self.message_fn = nn.Sequential(
            nn.Linear(input_dim + memory_dim + time_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, memory_dim)
        )
        
        # Memory updater (GRU)
        self.memory_updater = nn.GRUCell(memory_dim, memory_dim)
        
        # Embedding layer
        self.embedding = nn.Sequential(
            nn.Linear(input_dim + memory_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Output layer
        self.output = nn.Linear(hidden_dim, num_classes)
    
    def init_memory(self, num_nodes, device):
        """Initialize memory for all nodes."""
        self.memory = torch.zeros(num_nodes, self.memory_dim).to(device)
        self.last_update = torch.zeros(num_nodes).to(device)
    
    def compute_temporal_embedding(self, nodes, current_time):
        """
        Compute temporal embeddings for nodes.
        
        Args:
            nodes: Node indices
            current_time: Current timestamp
            
        Returns:
            Temporal embeddings
        """
        if self.memory is None:
            device = nodes.device if torch.is_tensor(nodes) else 'cpu'
            self.init_memory(nodes.max().item() + 1 if torch.is_tensor(nodes) else len(nodes), device)
        
        # Time delta encoding
        time_delta = current_time - self.last_update[nodes]
        time_encoding = self.time_encoder(time_delta.unsqueeze(-1))
        
        return time_encoding
    
    def forward(self, x, edge_index, edge_time=None):
        """
        Forward pass.
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]
            edge_time: Edge timestamps [num_edges]
            
        Returns:
            predictions [num_nodes, num_classes]
        """
        num_nodes = x.size(0)
        device = x.device
        
        # Initialize memory if needed
        if self.memory is None:
            self.init_memory(num_nodes, device)
        
        # Compute embeddings with memory
        h = torch.cat([x, self.memory], dim=-1)
        h = self.embedding(h)
        
        # Output
        out = self.output(h)
        
        return out
    
    def update_memory(self, nodes, messages):
        """Update memory for specific nodes."""
        if self.memory is None:
            return
        
        self.memory[nodes] = self.memory_updater(
            messages,
            self.memory[nodes]
        )


class TGAT(nn.Module):
    """
    Temporal Graph Attention Network (TGAT).
    Based on: Xu et al., "Inductive Representation Learning on Temporal Graphs" (ICLR 2020)
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        num_classes: int = 2,
        dropout: float = 0.3
    ):
        """
        Initialize TGAT model.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            num_layers: Number of attention layers
            num_heads: Number of attention heads
            num_classes: Number of output classes
            dropout: Dropout probability
        """
        super(TGAT, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_classes = num_classes
        
        # Temporal attention layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # Input layer
        self.convs.append(
            GATConv(input_dim, hidden_dim // num_heads, heads=num_heads)
        )
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            self.convs.append(
                GATConv(hidden_dim, hidden_dim // num_heads, heads=num_heads)
            )
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Output layer
        self.output = nn.Linear(hidden_dim, num_classes)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, edge_index, edge_time=None):
        """
        Forward pass.
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]
            edge_time: Edge timestamps [num_edges] (optional)
            
        Returns:
            predictions [num_nodes, num_classes]
        """
        # Apply temporal attention layers
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = self.dropout(x)
        
        # Output
        x = self.output(x)
        
        return x


def get_model(model_name: str, config: dict) -> nn.Module:
    """
    Factory function to create models by name.
    
    Args:
        model_name: Name of the model
        config: Configuration dictionary
        
    Returns:
        Initialized model
    """
    models = {
        "mlp": MLPClassifier,
        "graphsage": GraphSAGE,
        "temporal_gnn": TemporalGNN,
        "tgn": TGN,
        "tgat": TGAT
    }
    
    if model_name.lower() not in models:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(models.keys())}")
    
    return models[model_name.lower()](**config)

