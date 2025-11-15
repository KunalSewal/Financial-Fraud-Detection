"""
TGN + TGAT Hybrid (Option A)
Hybrid model: use TGN's per-node memory as node features for TGAT attention.
Produces a fraud classifier using TGAT embeddings (computed from TGN memory).

Place this file alongside: tgn.py and TGAT.py
Run similarly to your TGN_fraud.py / TGAT_fraud.py scripts.
"""

import os
import time
import random
from typing import List, Tuple, Optional, Any
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

# Import components from your existing files
# tgn.py should export: TGNModel, TimeEncoder, MessageFunction, MemoryModule, RawMessageStore, NeighborFinder, TemporalAttentionEmbedding
# TGAT.py should export: TimeEncoder (or TimeEncoder equivalent), NeighborFinder (if needed), TGATModel
from tgn import TGNModel  # we will use TGNModel and its internal neighbor finder & memory
from TGAT import TimeEncoder as TGATTimeEncoder, TGATModel, NeighborFinder as TGATNeighborFinder  # alias TGAT components

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Hybrid run device:", device)


# -------------------------
# Hybrid model wrapper
# -------------------------
class HybridFraudClassifier(nn.Module):
    """
    Wrapper that computes TGAT embeddings using TGN memory vectors as node features,
    then applies a classifier MLP on top of TGAT output.
    """
    def __init__(self, tgn_model: TGNModel, tgat_model: TGATModel, embed_dim: int):
        super().__init__()
        self.tgn = tgn_model  # contains memory, neighbor finder, time encoder (if any)
        self.tgat = tgat_model
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2)
        )

    def forward_from_embeddings(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: (B, embed_dim) TGAT embeddings
        returns logits (B,2)
        """
        return self.classifier(z)


# -------------------------
# Data loader / conversion (same approach as your other scripts)
# -------------------------
def load_ibm_temporal_data(csv_path: str):
    """
    Loads the IBM CSV and converts it to chronological events:
    Returns: events list, labels array, num_nodes, node_timestamp dict, dataframe
    """
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df):,} rows")

    # Clean Amount
    if 'Amount' in df.columns:
        df['Amount'] = df['Amount'].replace(r'[\$,]', '', regex=True).astype(float)

    # Map fraud column
    if 'Is Fraud?' in df.columns:
        df['Fraud_Label'] = df['Is Fraud?'].map({'No': 0, 'Yes': 1})
    else:
        raise ValueError("CSV must contain 'Is Fraud?' column")

    # Transaction id
    df['transaction_id'] = np.arange(len(df)).astype(int)

    # Build datetime from Year, Month, Day and optional Time
    if 'Time' in df.columns:
        def make_datetime(row):
            try:
                return pd.to_datetime(f"{int(row['Year']):04d}-{int(row['Month']):02d}-{int(row['Day']):02d} {row['Time']}",
                                      format='%Y-%m-%d %H:%M', errors='coerce')
            except Exception:
                return pd.NaT
        df['datetime'] = df.apply(make_datetime, axis=1)
    else:
        df['datetime'] = pd.to_datetime(df[['Year', 'Month', 'Day']].astype(int).astype(str).agg('-'.join, axis=1),
                                        format='%Y-%m-%d', errors='coerce')

    # drop invalid datetimes
    n_before = len(df)
    df = df.dropna(subset=['datetime']).reset_index(drop=True)
    if len(df) < n_before:
        print(f"Dropped {n_before - len(df)} rows due to invalid timestamps")

    # normalized timestamp [0,1]
    df['timestamp'] = (df['datetime'] - df['datetime'].min()).dt.total_seconds().astype(float)
    max_ts = df['timestamp'].max()
    if max_ts <= 0:
        raise ValueError("Timestamps invalid (max <= 0)")
    df['timestamp'] = df['timestamp'] / max_ts

    # sort chronologically
    df = df.sort_values('timestamp').reset_index(drop=True)
    df['transaction_id'] = np.arange(len(df)).astype(int)
    num_nodes = len(df)

    # Build events by User and Card
    events: List[Tuple[int, int, float, Optional[torch.Tensor]]] = []
    print("Building temporal edges (User and Card grouping)...")
    for col in ['User', 'Card']:
        if col not in df.columns:
            continue
        grouped = df.groupby(col)
        for key, group in grouped:
            if len(group) <= 1:
                continue
            group = group.sort_values('timestamp')
            idxs = group['transaction_id'].values
            tss = group['timestamp'].values
            amounts = group['Amount'].values if 'Amount' in group.columns else np.zeros(len(group))
            mccs = group['MCC'].values if 'MCC' in group.columns else np.zeros(len(group))
            for i in range(len(idxs) - 1):
                src = int(idxs[i])
                dst = int(idxs[i + 1])
                ts = float(tss[i + 1])
                edge_feat = torch.tensor([
                    float(amounts[i + 1] - amounts[i]),
                    float(tss[i + 1] - tss[i]),
                    float(mccs[i + 1]) if len(mccs) > 0 else 0.0
                ], dtype=torch.float)
                events.append((src, dst, ts, edge_feat))

    events.sort(key=lambda x: x[2])
    print(f"Created {len(events):,} temporal edges")

    labels = df['Fraud_Label'].astype(int).values
    fraud_count = int(labels.sum())
    print(f"Fraud transactions: {fraud_count:,} ({fraud_count/len(labels)*100:.2f}%)")

    node_timestamp = df.set_index('transaction_id')['timestamp'].to_dict()
    return events, labels, num_nodes, node_timestamp, df


# -------------------------
# Helper: recursive subgraph (compatible with TGAT builder)
# -------------------------
def get_recursive_subgraph_from_neighborfinder(neigh_finder, seed_nodes: List[int], seed_times: List[float], L: int, k: int):
    """
    Build L-hop temporal subgraph per seed using the neighborfinder adj lists.
    Returns list of dicts like {'nodes_per_hop': [[seed], hop1_nodes, hop2_nodes, ...],
                              'times_per_hop': [[seed_time], hop1_times, ...]}
    This mirrors TGAT_fraud's get_recursive_subgraph.
    """
    batch = []
    for b, seed in enumerate(seed_nodes):
        t_cut = seed_times[b]
        nodes_per_hop = [[seed]]
        times_per_hop = [[t_cut]]
        frontier = [seed]
        for layer in range(L):
            next_frontier = []
            next_times = []
            seen = set()
            for node in frontier:
                # read adjacency deque for node in neigh_finder
                if node < 0 or node >= neigh_finder.num_nodes:
                    continue
                dq = neigh_finder.adj[node]  # deque of (nbr, ts, ef)
                # iterate reversed (most recent first)
                for nbr, ts, ef in reversed(dq):
                    if ts < t_cut and nbr not in seen:
                        next_frontier.append(nbr)
                        next_times.append(ts)
                        seen.add(nbr)
                        if len(next_frontier) >= k:
                            break
                if len(next_frontier) >= k:
                    break
            nodes_per_hop.append(next_frontier)
            times_per_hop.append(next_times)
            frontier = next_frontier
        batch.append({'nodes_per_hop': nodes_per_hop, 'times_per_hop': times_per_hop})
    return batch


# -------------------------
# Build batch sequences (use TGN memory as node features)
# -------------------------
def build_batch_sequences_from_subgraphs_using_memory(subgraphs: List[dict],
                                                     memory_tensor: torch.Tensor,
                                                     tgat_time_encoder: TGATTimeEncoder,
                                                     max_neighbors_total: int,
                                                     L: int):
    """
    Given subgraphs and TGN memory tensor (num_nodes x mem_dim), build
    - target_feats (B, mem_dim)
    - neigh_feats_padded (B, Ltot, mem_dim)
    - neigh_time_enc_padded (B, Ltot, 2*time_dim)
    - key_padding_mask (B, Ltot)  True for pads
    """
    B = len(subgraphs)
    mem_dim = memory_tensor.size(1)
    neighbor_lists = []
    neighbor_times = []
    target_ids = []
    target_times = []

    for sg in subgraphs:
        root = sg['nodes_per_hop'][0][0]
        root_time = sg['times_per_hop'][0][0]
        target_ids.append(root)
        target_times.append(root_time)
        neighs = []
        times = []
        for hop in range(1, L + 1):
            nodes_h = sg['nodes_per_hop'][hop]
            times_h = sg['times_per_hop'][hop]
            for nid, ts in zip(nodes_h, times_h):
                neighs.append(nid)
                times.append(ts)
        seen = set()
        uniq_neighs = []
        uniq_times = []
        for nid, ts in zip(neighs, times):
            if nid in seen:
                continue
            seen.add(nid)
            uniq_neighs.append(nid)
            uniq_times.append(ts)
        uniq_neighs = uniq_neighs[:max_neighbors_total]
        uniq_times = uniq_times[:max_neighbors_total]
        neighbor_lists.append(uniq_neighs)
        neighbor_times.append(uniq_times)

    Ltot = max((len(lst) for lst in neighbor_lists), default=0)
    
    # If no neighbors at all, return empty tensors with single dummy neighbor to avoid attention errors
    if Ltot == 0:
        Ltot = 1  # Use at least 1 position to avoid 0-length sequences
    
    target_feats = torch.zeros(B, mem_dim)
    neigh_feats = torch.zeros(B, Ltot, mem_dim)
    neigh_time_enc = torch.zeros(B, Ltot, 2 * tgat_time_encoder.time_dim)
    key_padding_mask = torch.ones(B, Ltot, dtype=torch.bool)

    for i in range(B):
        tid = target_ids[i]
        t = target_times[i]
        target_feats[i] = memory_tensor[tid].cpu()
        for j in range(Ltot):
            if j < len(neighbor_lists[i]):
                nid = neighbor_lists[i][j]
                ts = neighbor_times[i][j]
                neigh_feats[i, j] = memory_tensor[nid].cpu()
                dt = float(t - ts)
                neigh_time_enc[i, j] = tgat_time_encoder(torch.tensor(dt).float().to(tgat_time_encoder.omega.device)).detach().cpu()
                key_padding_mask[i, j] = False

    return target_feats, neigh_feats, neigh_time_enc, key_padding_mask


# -------------------------
# Evaluate helper
# -------------------------
def evaluate_hybrid(classifier: HybridFraudClassifier, tgat_model: TGATModel, tgat_time_encoder: TGATTimeEncoder,
                    memory_tensor: torch.Tensor, neigh_finder, node_timestamp: dict,
                    labels: np.ndarray, test_mask: np.ndarray, batch_size: int = 256,
                    k: int = 10, L: int = 2):
    classifier.eval(); tgat_model.eval(); tgat_time_encoder.eval()
    test_nodes = np.where(test_mask)[0].tolist()
    all_preds = []; all_probs = []
    with torch.no_grad():
        for i in range(0, len(test_nodes), batch_size):
            batch_nodes = test_nodes[i:i + batch_size]
            times = [1.0] * len(batch_nodes)  # Use future time to get all neighbors
            subgraphs = get_recursive_subgraph_from_neighborfinder(neigh_finder, batch_nodes, times, L, k)
            tgt_feats, neigh_feats, neigh_time_enc, key_padding_mask = build_batch_sequences_from_subgraphs_using_memory(
                subgraphs, memory_tensor, tgat_time_encoder, max_neighbors_total=k * L, L=L
            )
            tgt_feats = tgt_feats.to(device); neigh_feats = neigh_feats.to(device); neigh_time_enc = neigh_time_enc.to(device); key_padding_mask = key_padding_mask.to(device)
            z = tgat_model(tgt_feats, neigh_feats, neigh_time_enc, key_padding_mask)  # (B, embed_dim)
            logits = classifier.forward_from_embeddings(z)
            probs = F.softmax(logits, dim=1)
            preds = logits.argmax(dim=1).cpu().numpy()
            probs_fraud = probs[:, 1].cpu().numpy()
            all_preds.extend(preds.tolist()); all_probs.extend(probs_fraud.tolist())

    test_nodes_list = test_nodes
    test_labels = [labels[n] for n in test_nodes_list]
    acc = accuracy_score(test_labels, all_preds)
    precision = precision_score(test_labels, all_preds, zero_division=0)
    recall = recall_score(test_labels, all_preds, zero_division=0)
    f1 = f1_score(test_labels, all_preds, zero_division=0)
    try:
        roc_auc = roc_auc_score(test_labels, all_probs) if len(np.unique(test_labels)) > 1 else 0.0
    except Exception:
        roc_auc = 0.0

    print(f"\nHYBRID TEST EVAL")
    print(f"Accuracy: {acc:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f} | AUC: {roc_auc:.4f}\n")
    return {'accuracy': acc, 'precision': precision, 'recall': recall, 'f1': f1, 'roc_auc': roc_auc}


# -------------------------
# Training procedure
# -------------------------
def train_hybrid(events: List[Tuple[int, int, float, Optional[torch.Tensor]]],
                 labels: np.ndarray,
                 node_timestamp: dict,
                 num_nodes: int,
                 train_mask: np.ndarray,
                 val_mask: np.ndarray,
                 epochs: int = 30,
                 batch_size: int = 256,
                 lr: float = 1e-3,
                 mem_dim: int = 100,
                 msg_dim: int = 100,
                 edge_dim: int = 3,
                 tgat_time_dim: int = 16,
                 embed_dim: int = 64,
                 k: int = 10,
                 L: int = 2,
                 device_local: torch.device = device):
    """
    Build TGN + TGAT hybrid and train classifier.
    - We instantiate TGNModel and TGATModel. TGAT node feature dim is set to mem_dim (TGN memory dim).
    - First we stream events to populate neighborfinder and update TGN memory (like TGN_fraud).
    - Then supervised training: for each batch of nodes, we build recursive subgraphs,
      use TGN.memory (as node features) to build TGAT inputs, compute TGAT embeddings, and train classifier.
    """
    # Instantiate TGN (for memory + neighbor structure)
    print("Initializing TGN model (memory + raw message store)...")
    tgn_model = TGNModel(num_nodes=num_nodes, mem_dim=mem_dim, msg_dim=msg_dim, edge_dim=edge_dim, time_dim=tgat_time_dim, embed_dim=embed_dim, k=k, device=device_local)
    
    # Create separate TGAT NeighborFinder with larger capacity
    print("Initializing TGAT neighbor finder (separate from TGN)...")
    tgat_neigh_finder = TGATNeighborFinder(num_nodes=num_nodes, max_neighbors=500)
    
    # TGAT model: node_feat_dim must equal mem_dim (we use memory as node features)
    print("Initializing TGAT model (will use TGN memory as node features)...")
    tgat_time_encoder = TGATTimeEncoder(time_dim=tgat_time_dim).to(device_local)
    tgat_model = TGATModel(node_feat_dim=mem_dim, time_enc_dim=tgat_time_dim, embed_dim=embed_dim, n_layers=2).to(device_local)

    classifier = HybridFraudClassifier(tgn_model, tgat_model, embed_dim=embed_dim).to(device_local)

    # Populate neighborfinder and update TGN memory by streaming events
    print("Streaming events to populate neighborfinder & update TGN memory...")
    for i, (src, dst, ts, ef) in enumerate(events):
        if i % 10000 == 0:
            print(f"  Processed {i:,}/{len(events):,} events")
        # insert in BOTH neighborfinders (TGN's for memory updates, TGAT's for subgraph sampling)
        tgn_model.neigh_finder.insert_edge(src, dst, ts, ef)
        tgat_neigh_finder.insert_edge(src, dst, ts, ef)
        # compute raw messages and update memory (simple immediate updates, similar to earlier script)
        src_t = torch.tensor([src], device=device_local)
        dst_t = torch.tensor([dst], device=device_local)
        src_mem = tgn_model.memory.get_memory(src_t)
        dst_mem = tgn_model.memory.get_memory(dst_t)
        # compute dt for src/dst relative to their last update
        last_src_t = tgn_model.memory.last_update[src].item()
        dt_src = ts - last_src_t
        dt_enc = tgn_model.time_encoder(torch.tensor([dt_src], device=device_local))
        if ef is not None:
            ef_t = ef.unsqueeze(0).to(device_local)
        else:
            ef_t = None
        raw_msg, _ = tgn_model.compute_raw_messages(src_mem, dst_mem, ef_t, dt_enc)
        # update both memories with the raw message (agg simplification)
        try:
            tgn_model.memory.update_with_messages([int(src), int(dst)], torch.cat([raw_msg, raw_msg], dim=0), torch.tensor([ts, ts], device=device_local))
        except Exception:
            # fallback: update individually
            tgn_model.memory.update_with_messages([int(src)], raw_msg.detach(), torch.tensor([ts], device=device_local))
            tgn_model.memory.update_with_messages([int(dst)], raw_msg.detach(), torch.tensor([ts], device=device_local))

    print("Streaming complete. TGN memory populated.")
    
    # Debug: Check neighbor structure
    sample_nodes = [0, 100, 1000, 5000]
    for n in sample_nodes:
        if n < num_nodes:
            tgn_neighbors = len(tgn_model.neigh_finder.adj[n])
            tgat_neighbors = len(tgat_neigh_finder.adj[n])
            print(f"  Node {n}: TGN={tgn_neighbors} neighbors, TGAT={tgat_neighbors} neighbors")

    # Prepare optimizer: train tgat_model, classifier, and optionally fine-tune TGN memory/time encoder
    params = list(tgat_model.parameters()) + list(classifier.parameters()) + list(tgn_model.time_encoder.parameters())
    # Optionally include TGN memory updater params if you want to fine-tune memory updater (not necessary)
    # params += list(tgn_model.memory.parameters())  # GRUCell is part of memory module
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()

    train_nodes = np.where(train_mask)[0].tolist()
    val_nodes = np.where(val_mask)[0].tolist()

    best_val_acc = 0.0
    best_state = None
    start_time = time.time()

    print("Starting supervised training of classifier on TGAT embeddings (fed with TGN memory)...")
    for epoch in range(1, epochs + 1):
        classifier.train(); tgat_model.train(); tgat_time_encoder.train()
        random.shuffle(train_nodes)
        epoch_loss = 0.0
        steps = 0

        for i in range(0, len(train_nodes), batch_size):
            batch_nodes = train_nodes[i:i + batch_size]
            batch_labels = torch.tensor([labels[n] for n in batch_nodes], dtype=torch.long, device=device_local)
            # Use max timestamp (1.0) instead of node's own timestamp to get all available neighbors
            batch_times = [1.0] * len(batch_nodes)  # Use future time to get all neighbors
            # build recursive subgraphs using the TGAT neighborfinder (not TGN's)
            subgraphs = get_recursive_subgraph_from_neighborfinder(tgat_neigh_finder, batch_nodes, batch_times, L, k)
            
            # Debug first batch
            if i == 0 and steps == 0:
                total_neighbors = sum(len(sg['nodes_per_hop'][1]) for sg in subgraphs)
                print(f"  First batch: {len(batch_nodes)} nodes, total 1-hop neighbors: {total_neighbors}")
            
            # collect memory tensor (num_nodes x mem_dim)
            memory_tensor = tgn_model.memory.memory  # buffer on device; we will use cpu view for building
            tgt_feats, neigh_feats, neigh_time_enc, key_padding_mask = build_batch_sequences_from_subgraphs_using_memory(
                subgraphs, memory_tensor, tgat_time_encoder, max_neighbors_total=k * L, L=L
            )
            tgt_feats = tgt_feats.to(device_local); neigh_feats = neigh_feats.to(device_local); neigh_time_enc = neigh_time_enc.to(device_local); key_padding_mask = key_padding_mask.to(device_local)

            # compute TGAT embeddings
            z = tgat_model(tgt_feats, neigh_feats, neigh_time_enc, key_padding_mask)  # (B, embed_dim)

            optimizer.zero_grad()
            logits = classifier.forward_from_embeddings(z)
            loss = criterion(logits, batch_labels)
            if torch.isnan(loss):
                print(f"Warning: NaN loss at epoch {epoch}, batch start {i}. Skipping batch.")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            steps += 1

        avg_loss = epoch_loss / max(1, steps)
        print(f"Epoch {epoch}/{epochs} | Avg Loss: {avg_loss:.4f}")

        # Validation
        if epoch % 1 == 0:
            classifier.eval(); tgat_model.eval(); tgat_time_encoder.eval()
            val_preds = []; val_probs = []
            for i in range(0, len(val_nodes), batch_size):
                batch_nodes = val_nodes[i:i + batch_size]
                batch_times = [1.0] * len(batch_nodes)  # Use future time to get all neighbors
                subgraphs = get_recursive_subgraph_from_neighborfinder(tgat_neigh_finder, batch_nodes, batch_times, L, k)
                memory_tensor = tgn_model.memory.memory
                tgt_feats, neigh_feats, neigh_time_enc, key_padding_mask = build_batch_sequences_from_subgraphs_using_memory(
                    subgraphs, memory_tensor, tgat_time_encoder, max_neighbors_total=k * L, L=L
                )
                tgt_feats = tgt_feats.to(device_local); neigh_feats = neigh_feats.to(device_local); neigh_time_enc = neigh_time_enc.to(device_local); key_padding_mask = key_padding_mask.to(device_local)
                with torch.no_grad():
                    z = tgat_model(tgt_feats, neigh_feats, neigh_time_enc, key_padding_mask)
                    logits = classifier.forward_from_embeddings(z)
                    probs = F.softmax(logits, dim=1)
                    preds = logits.argmax(dim=1).cpu().numpy()
                    probs_f = probs[:, 1].cpu().numpy()
                val_preds.extend(preds.tolist()); val_probs.extend(probs_f.tolist())

            val_labels_list = [labels[n] for n in val_nodes]
            # Because we computed predictions for all val nodes in batches we should compare total (make sure sizes match)
            # For simplicity we'll compute metrics by picking first len(val_preds) elements
            if len(val_preds) == 0:
                val_acc = val_prec = val_rec = val_f1 = val_auc = 0.0
            else:
                # If val_preds length < val_nodes, sample corresponding val labels (this usually won't happen)
                val_pred_labels = val_preds[:len(val_nodes)]
                val_probs_used = val_probs[:len(val_nodes)]
                try:
                    val_acc = accuracy_score(val_labels_list, val_pred_labels)
                    val_prec = precision_score(val_labels_list, val_pred_labels, zero_division=0)
                    val_rec = recall_score(val_labels_list, val_pred_labels, zero_division=0)
                    val_f1 = f1_score(val_labels_list, val_pred_labels, zero_division=0)
                    val_auc = roc_auc_score(val_labels_list, val_probs_used) if len(np.unique(val_labels_list)) > 1 else 0.0
                except Exception:
                    val_acc = val_prec = val_rec = val_f1 = val_auc = 0.0

            print(f"  Val - Acc: {val_acc:.4f} | Prec: {val_prec:.4f} | Rec: {val_rec:.4f} | F1: {val_f1:.4f} | AUC: {val_auc:.4f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {
                    'tgn_state': tgn_model.state_dict(),
                    'tgat_state': tgat_model.state_dict(),
                    'classifier_state': classifier.state_dict(),
                    'time_encoder_state': tgat_time_encoder.state_dict(),
                    'memory_state': tgn_model.memory.memory.detach().cpu()
                }
                print("  ✓ New best validation accuracy; checkpoint saved in memory.")

    total_time = time.time() - start_time
    print(f"\nTraining finished in {total_time:.2f}s. Best val acc: {best_val_acc:.4f}")

    # restore best
    if best_state is not None:
        tgn_model.load_state_dict(best_state['tgn_state'])
        tgat_model.load_state_dict(best_state['tgat_state'])
        classifier.load_state_dict(best_state['classifier_state'])
        tgat_time_encoder.load_state_dict(best_state['time_encoder_state'])
        # memory restored if needed (optional)
        # tgn_model.memory.memory = best_state['memory_state'].to(tgn_model.memory.memory.device)
        print("Loaded best model from validation checkpoint.")

    # Save artifact
    os.makedirs('saved_models', exist_ok=True)
    save_path = 'saved_models/tgn_tgat_hybrid_best.pt'
    torch.save({
        'tgn_state': tgn_model.state_dict(),
        'tgat_state': tgat_model.state_dict(),
        'classifier_state': classifier.state_dict(),
        'time_encoder_state': tgat_time_encoder.state_dict(),
        'config': {
            'num_nodes': num_nodes,
            'mem_dim': mem_dim,
            'msg_dim': msg_dim,
            'edge_dim': edge_dim,
            'tgat_time_dim': tgat_time_dim,
            'embed_dim': embed_dim,
            'k': k,
            'L': L
        }
    }, save_path)
    print(f"Saved hybrid artifact to {save_path}")

    return classifier, tgat_model, tgn_model.memory.memory, tgat_neigh_finder


# -------------------------
# Main: run hybrid training (mirrors TGN_fraud and TGAT_fraud)
# -------------------------
if __name__ == "__main__":
    random.seed(0); torch.manual_seed(0); np.random.seed(0)

    csv_path = "data/ibm/ibm_fraud_29k_nonfraud_60k.csv"
    events, labels, num_nodes, node_timestamp, df = load_ibm_temporal_data(csv_path)

    # train/val/test split (70/15/15) stratified by label
    all_idx = np.arange(num_nodes)
    train_idx, test_val_idx = train_test_split(all_idx, test_size=0.3, random_state=42, stratify=labels)
    val_idx, test_idx = train_test_split(test_val_idx, test_size=0.5, random_state=42, stratify=labels[test_val_idx])

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

    # Train hybrid
    classifier, tgat_model, memory_tensor, neigh_finder = train_hybrid(
        events=events,
        labels=labels,
        node_timestamp=node_timestamp,
        num_nodes=num_nodes,
        train_mask=train_mask,
        val_mask=val_mask,
        epochs=5,
        batch_size=256,
        lr=1e-3,
        mem_dim=100,
        msg_dim=100,
        edge_dim=3,
        tgat_time_dim=16,
        embed_dim=64,
        k=10,
        L=2
    )

    # Evaluate on test
    results = evaluate_hybrid(classifier, tgat_model, TGATTimeEncoder(time_dim=16).to(device),
                              memory_tensor=classifier.tgn.memory.memory, neigh_finder=neigh_finder, node_timestamp=node_timestamp,
                              labels=labels, test_mask=test_mask, batch_size=256, k=10, L=2)

    print("Hybrid training + evaluation finished.")
