"""
Temporal Graph Network (TGN) Implementation.

Based on:
Rossi et al., "Temporal Graph Networks for Deep Learning on Dynamic Graphs"
ICML 2020 Workshop on Graph Representation Learning

This is a FULL implementation with:
- Memory module (stores historical node states)
- Time encoding (continuous-time Fourier features)
- Message function (aggregates neighbor interactions)
- Memory updater (GRU-based state evolution)
- Graph attention (attentive aggregation)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class TimeEncoder(nn.Module):
    """
    Time encoding using Fourier features (Bochner's theorem).
    
    Encodes continuous timestamps into learnable periodic features.
    """
    
    def __init__(self, time_dim: int):
        """
        Args:
            time_dim: Dimension of time encoding output
        """
        super().__init__()
        self.time_dim = time_dim
        
        # Learnable frequency and phase parameters
        self.w = nn.Linear(1, time_dim)
        self.b = nn.Linear(1, time_dim)
        
    def forward(self, timestamps: torch.Tensor) -> torch.Tensor:
        """
        Encode timestamps into periodic features.
        
        Args:
            timestamps: [batch_size] or [batch_size, 1]
            
        Returns:
            Time embeddings [batch_size, time_dim]
        """
        if timestamps.dim() == 1:
            timestamps = timestamps.unsqueeze(-1)
        
        # Fourier time features: cos(w*t + b)
        time_encoding = torch.cos(self.w(timestamps) + self.b(timestamps))
        
        return time_encoding


class MemoryModule(nn.Module):
    """
    Memory module that stores the last state of each node.
    
    Updated after each interaction via GRU-based memory updater.
    """
    
    def __init__(self, num_nodes: int, memory_dim: int):
        """
        Args:
            num_nodes: Maximum number of nodes in graph
            memory_dim: Dimension of memory vectors
        """
        super().__init__()
        self.num_nodes = num_nodes
        self.memory_dim = memory_dim
        
        # Initialize memory with zeros
        self.register_buffer('memory', torch.zeros(num_nodes, memory_dim))
        self.register_buffer('last_update', torch.zeros(num_nodes))
        
        # Memory is updated via GRU
        self.memory_updater = nn.GRUCell(memory_dim, memory_dim)
        
    def get_memory(self, node_ids: torch.Tensor) -> torch.Tensor:
        """
        Retrieve memory for specific nodes.
        
        Args:
            node_ids: [batch_size] node indices
            
        Returns:
            Memory vectors [batch_size, memory_dim]
        """
        return self.memory[node_ids]
    
    def get_last_update(self, node_ids: torch.Tensor) -> torch.Tensor:
        """
        Get last update time for nodes.
        
        Args:
            node_ids: [batch_size] node indices
            
        Returns:
            Last update times [batch_size]
        """
        return self.last_update[node_ids]
    
    def update_memory(
        self,
        node_ids: torch.Tensor,
        messages: torch.Tensor,
        timestamps: torch.Tensor
    ):
        """
        Update memory for specific nodes using GRU.
        
        Args:
            node_ids: [batch_size] node indices
            messages: [batch_size, memory_dim] aggregated messages
            timestamps: [batch_size] current timestamps
        """
        # Get current memory
        current_memory = self.memory[node_ids]
        
        # Update via GRU
        updated_memory = self.memory_updater(messages, current_memory)
        
        # Write back to memory
        self.memory[node_ids] = updated_memory
        self.last_update[node_ids] = timestamps
        
    def reset_memory(self):
        """Reset all memory to zeros."""
        self.memory.zero_()
        self.last_update.zero_()
    
    def detach_memory(self):
        """Detach memory from computation graph (for backprop truncation)."""
        self.memory = self.memory.detach()


class MessageFunction(nn.Module):
    """
    Message function that aggregates information from neighbors.
    
    For each edge (u, v) at time t, compute message based on:
    - Source node features
    - Destination node features
    - Edge features
    - Time encoding
    - Node memories
    """
    
    def __init__(
        self,
        node_dim: int,
        memory_dim: int,
        edge_dim: int,
        time_dim: int,
        message_dim: int
    ):
        """
        Args:
            node_dim: Node feature dimension
            memory_dim: Memory dimension
            edge_dim: Edge feature dimension
            time_dim: Time encoding dimension
            message_dim: Output message dimension
        """
        super().__init__()
        
        # Total input dimension
        input_dim = 2 * (node_dim + memory_dim) + edge_dim + time_dim
        
        # Message MLP
        self.message_mlp = nn.Sequential(
            nn.Linear(input_dim, message_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(message_dim * 2, message_dim),
            nn.ReLU()
        )
        
    def forward(
        self,
        src_features: torch.Tensor,
        dst_features: torch.Tensor,
        src_memory: torch.Tensor,
        dst_memory: torch.Tensor,
        edge_features: torch.Tensor,
        time_encoding: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute messages for edges.
        
        Args:
            src_features: [num_edges, node_dim]
            dst_features: [num_edges, node_dim]
            src_memory: [num_edges, memory_dim]
            dst_memory: [num_edges, memory_dim]
            edge_features: [num_edges, edge_dim]
            time_encoding: [num_edges, time_dim]
            
        Returns:
            Messages [num_edges, message_dim]
        """
        # Concatenate all features
        message_input = torch.cat([
            src_features,
            dst_features,
            src_memory,
            dst_memory,
            edge_features,
            time_encoding
        ], dim=-1)
        
        # Pass through MLP
        messages = self.message_mlp(message_input)
        
        return messages


class MessageAggregator(nn.Module):
    """
    Aggregate messages for each node.
    
    Multiple messages can arrive at a node; we need to aggregate them.
    Options: mean, max, attention-based
    """
    
    def __init__(self, message_dim: int, aggregation: str = 'mean'):
        """
        Args:
            message_dim: Message dimension
            aggregation: 'mean', 'max', or 'attention'
        """
        super().__init__()
        self.aggregation = aggregation
        
        if aggregation == 'attention':
            self.attention = nn.Linear(message_dim, 1)
    
    def forward(
        self,
        messages: torch.Tensor,
        node_ids: torch.Tensor,
        num_nodes: int
    ) -> torch.Tensor:
        """
        Aggregate messages per node.
        
        Args:
            messages: [num_messages, message_dim]
            node_ids: [num_messages] destination node IDs
            num_nodes: Total number of nodes
            
        Returns:
            Aggregated messages [num_nodes, message_dim]
        """
        message_dim = messages.size(-1)
        aggregated = torch.zeros(num_nodes, message_dim, device=messages.device)
        
        if self.aggregation == 'mean':
            # Mean aggregation with scatter_add
            aggregated.scatter_add_(0, node_ids.unsqueeze(-1).expand_as(messages), messages)
            
            # Count messages per node
            counts = torch.zeros(num_nodes, device=messages.device)
            counts.scatter_add_(0, node_ids, torch.ones_like(node_ids, dtype=torch.float))
            counts = counts.clamp(min=1).unsqueeze(-1)
            
            aggregated = aggregated / counts
            
        elif self.aggregation == 'max':
            # Max aggregation
            aggregated, _ = scatter_max(messages, node_ids, dim=0, dim_size=num_nodes)
            
        elif self.aggregation == 'attention':
            # Attention-based aggregation
            attention_scores = self.attention(messages).squeeze(-1)
            attention_weights = torch.zeros(num_nodes, device=messages.device)
            attention_weights.scatter_add_(0, node_ids, attention_scores.exp())
            attention_weights = attention_weights.clamp(min=1e-8)
            
            # Weighted messages
            weighted_messages = messages * (attention_scores.exp() / attention_weights[node_ids]).unsqueeze(-1)
            aggregated.scatter_add_(0, node_ids.unsqueeze(-1).expand_as(weighted_messages), weighted_messages)
        
        return aggregated


class TGN(nn.Module):
    """
    Full Temporal Graph Network implementation.
    
    Components:
    1. Memory module (stores node states)
    2. Time encoder (encodes timestamps)
    3. Message function (computes edge messages)
    4. Message aggregator (aggregates to nodes)
    5. Memory updater (GRU-based update)
    6. Embedding layer (final node embeddings)
    7. Classifier (fraud prediction)
    """
    
    def __init__(
        self,
        num_nodes: int,
        node_dim: int,
        edge_dim: int = 4,
        memory_dim: int = 128,
        time_dim: int = 32,
        message_dim: int = None,  # If None, defaults to memory_dim
        embedding_dim: int = 128,
        num_classes: int = 2,
        aggregation: str = 'mean',
        dropout: float = 0.1
    ):
        """
        Initialize TGN model.
        
        Args:
            num_nodes: Maximum number of nodes
            node_dim: Node feature dimension
            edge_dim: Edge feature dimension
            memory_dim: Memory dimension
            time_dim: Time encoding dimension
            message_dim: Message dimension (must equal memory_dim for GRU)
            embedding_dim: Final embedding dimension
            num_classes: Number of output classes
            aggregation: Message aggregation method
            dropout: Dropout probability
        """
        super().__init__()
        
        self.num_nodes = num_nodes
        self.node_dim = node_dim
        self.memory_dim = memory_dim
        
        # Message dim must equal memory dim for GRU
        if message_dim is None:
            message_dim = memory_dim
        elif message_dim != memory_dim:
            import warnings
            warnings.warn(f"message_dim ({message_dim}) should equal memory_dim ({memory_dim}) for GRU. "
                         f"Setting message_dim = memory_dim.")
            message_dim = memory_dim
        
        # 1. Memory module
        self.memory = MemoryModule(num_nodes, memory_dim)
        
        # 2. Time encoder
        self.time_encoder = TimeEncoder(time_dim)
        
        # 3. Message function
        self.message_fn = MessageFunction(
            node_dim=node_dim,
            memory_dim=memory_dim,
            edge_dim=edge_dim,
            time_dim=time_dim,
            message_dim=message_dim
        )
        
        # 4. Message aggregator
        self.message_aggregator = MessageAggregator(message_dim, aggregation)
        
        # 5. Embedding layer (combines features and memory)
        self.embedding_layer = nn.Sequential(
            nn.Linear(node_dim + memory_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU()
        )
        
        # 6. Classifier
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim // 2, num_classes)
        )
        
    def compute_messages(
        self,
        edge_index: torch.Tensor,
        node_features: torch.Tensor,
        edge_features: torch.Tensor,
        edge_times: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute messages for all edges.
        
        Args:
            edge_index: [2, num_edges] edge connectivity
            node_features: [num_nodes, node_dim] node features
            edge_features: [num_edges, edge_dim] edge features
            edge_times: [num_edges] edge timestamps
            
        Returns:
            (messages, destination_nodes)
        """
        src_nodes = edge_index[0]
        dst_nodes = edge_index[1]
        
        # Get node features
        src_features = node_features[src_nodes]
        dst_features = node_features[dst_nodes]
        
        # Get node memories
        src_memory = self.memory.get_memory(src_nodes)
        dst_memory = self.memory.get_memory(dst_nodes)
        
        # Get last update times
        src_last_update = self.memory.get_last_update(src_nodes)
        dst_last_update = self.memory.get_last_update(dst_nodes)
        
        # Compute time deltas
        src_time_delta = edge_times - src_last_update
        dst_time_delta = edge_times - dst_last_update
        
        # Encode time (use average time delta)
        time_delta = (src_time_delta + dst_time_delta) / 2
        time_encoding = self.time_encoder(time_delta)
        
        # Compute messages
        messages = self.message_fn(
            src_features, dst_features,
            src_memory, dst_memory,
            edge_features,
            time_encoding
        )
        
        return messages, dst_nodes
    
    def update_memory_with_messages(
        self,
        node_ids: torch.Tensor,
        messages: torch.Tensor,
        timestamps: torch.Tensor
    ):
        """
        Update node memory with aggregated messages.
        """
        # Aggregate messages per node
        aggregated_messages = self.message_aggregator(
            messages, node_ids, self.num_nodes
        )
        
        # Get unique nodes that received messages
        unique_nodes = torch.unique(node_ids)
        
        # Update memory for these nodes
        self.memory.update_memory(
            unique_nodes,
            aggregated_messages[unique_nodes],
            timestamps[0].expand(len(unique_nodes))  # Use first timestamp
        )
    
    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
        edge_features: Optional[torch.Tensor] = None,
        edge_times: Optional[torch.Tensor] = None,
        node_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for node classification.
        
        Args:
            node_features: [num_nodes, node_dim]
            edge_index: [2, num_edges] (optional, for memory update)
            edge_features: [num_edges, edge_dim] (optional)
            edge_times: [num_edges] (optional)
            node_ids: [batch_size] nodes to classify (if None, classify all)
            
        Returns:
            Logits [num_nodes, num_classes] or [batch_size, num_classes]
        """
        num_nodes = node_features.size(0)
        
        # Update memory if edges provided
        if edge_index is not None and edge_times is not None:
            if edge_features is None:
                edge_features = torch.ones(edge_index.size(1), 1, device=edge_index.device)
            
            # Compute and aggregate messages
            messages, dst_nodes = self.compute_messages(
                edge_index, node_features, edge_features, edge_times
            )
            
            # Update memory
            self.update_memory_with_messages(dst_nodes, messages, edge_times)
        
        # Get nodes to classify
        if node_ids is None:
            node_ids = torch.arange(num_nodes, device=node_features.device)
        
        # Get current memory
        node_memory = self.memory.get_memory(node_ids)
        
        # Combine features and memory
        node_embedding = torch.cat([
            node_features[node_ids],
            node_memory
        ], dim=-1)
        
        # Embed
        embeddings = self.embedding_layer(node_embedding)
        
        # Classify
        logits = self.classifier(embeddings)
        
        return logits
    
    def reset(self):
        """Reset memory (e.g., between epochs)."""
        self.memory.reset_memory()
    
    def detach_memory(self):
        """Detach memory from computation graph."""
        self.memory.detach_memory()
