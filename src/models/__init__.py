"""
Models module - Temporal Graph Neural Networks.

This module contains:
- TGN: Temporal Graph Network (ICML 2020)
- MPTGNN: Multi-Path Temporal GNN (Algorithms 2024)
- HMSTA: Hybrid Multi-Scale Temporal Attention (NOVEL - Ours!)
"""

from .tgn import TGN, TimeEncoder, MemoryModule, MessageFunction, MessageAggregator
from .mptgnn import MPTGNN, TemporalAttention, MultiPathConvolution
from .hmsta import (
    HMSTA, 
    AnomalyAttention, 
    TemporalExplainer, 
    PathAggregator,
    create_hmsta_model
)

__all__ = [
    'TGN',
    'TimeEncoder',
    'MemoryModule',
    'MessageFunction',
    'MessageAggregator',
    'MPTGNN',
    'TemporalAttention',
    'MultiPathConvolution',
    'HMSTA',
    'AnomalyAttention',
    'TemporalExplainer',
    'PathAggregator',
    'create_hmsta_model'
]
