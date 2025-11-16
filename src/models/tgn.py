"""
Temporal Graph Networks (TGN) - PyTorch implementation (TGN-attn variant)
Implements:
 - per-node memory (GRU updater)
 - raw message store (Algorithm 1 behavior)
 - temporal attention embedding (neighbors + time encoding)
 - neighbor sampling (most recent k)
 - train loop with negative sampling for link prediction

Assumptions:
 - Input data: chronological list/array of events: (src, dst, timestamp, edge_feat_tensor)
 - Nodes are indexed [0..num_nodes-1]
 - Edge features optional; use zeros if not available

Author: ChatGPT (GPT-5 Thinking mini)
"""

import math
import random
from collections import defaultdict, deque
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# --------------------------
# Utilities / Data handling
# --------------------------

class TemporalEdgeDataset(Dataset):
    """
    Simple dataset wrapper. Expects `events` to be a list of tuples:
    (src, dst, timestamp, edge_feat_tensor)
    Events must be sorted by timestamp ascending.
    """
    def __init__(self, events: List[Tuple[int,int,float,Optional[torch.Tensor]]]):
        self.events = events

    def __len__(self):
        return len(self.events)

    def __getitem__(self, idx):
        return self.events[idx]

def collate_events(batch):
    """
    Collate a list of events into batched tensors.
    batch: list of (src, dst, ts, edge_feat) ; edge_feat may be None
    Returns: dict of tensors
    """
    srcs = torch.tensor([b[0] for b in batch], dtype=torch.long)
    dsts = torch.tensor([b[1] for b in batch], dtype=torch.long)
    ts   = torch.tensor([b[2] for b in batch], dtype=torch.float)
    # edge features: assume tensors of same dim or None
    if batch[0][3] is None:
        edge_feats = None
    else:
        edge_feats = torch.stack([b[3] for b in batch], dim=0)  # (B, edge_dim)
    return {'src': srcs, 'dst': dsts, 'ts': ts, 'edge_feats': edge_feats}

# --------------------------
# Core Model Components
# --------------------------

class TimeEncoder(nn.Module):
    """
    Time encoding similar to Time2Vec / sinusoidal encodings used in TGAT/TGN.
    Input: delta time scalar (batch,)
    Output: (batch, time_dim)
    """
    def __init__(self, time_dim: int):
        super().__init__()
        self.time_dim = time_dim
        # Frequencies and bias parameters learnable
        self.w = nn.Parameter(torch.randn(time_dim))
        self.b = nn.Parameter(torch.randn(time_dim))

    def forward(self, dt: torch.Tensor):
        # dt: (batch,) or (batch,1)
        # Expand dt to (batch, time_dim) via outer product
        dt = dt.unsqueeze(-1)  # (batch,1)
        x = dt * self.w + self.b  # broadcast -> (batch, time_dim)
        return torch.sin(x)  # (batch, time_dim)

class MessageFunction(nn.Module):
    """
    Produces raw message vector for an event (src,dst).
    Input: src_mem, dst_mem, edge_feat, delta_time_enc
    Output: message vector
    """
    def __init__(self, mem_dim, edge_dim, time_dim, msg_dim):
        super().__init__()
        input_dim = mem_dim * 2 + (edge_dim if edge_dim else 0) + time_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, msg_dim),
            nn.ReLU(),
            nn.Linear(msg_dim, msg_dim)
        )

    def forward(self, s_src, s_dst, edge_feat, dt_enc):
        parts = [s_src, s_dst]
        if edge_feat is not None:
            parts.append(edge_feat)
        parts.append(dt_enc)
        x = torch.cat(parts, dim=-1)
        return self.mlp(x)

class MemoryModule(nn.Module):
    """
    Memory storage and updater (GRUCell style).
    memory: (num_nodes, mem_dim)
    last_update_ts: vector of last timestamp per node
    """
    def __init__(self, num_nodes: int, mem_dim: int, msg_dim: int, device='cpu'):
        super().__init__()
        self.num_nodes = num_nodes
        self.mem_dim = mem_dim
        # memory stored as buffer tensor (not a parameter)
        self.register_buffer('memory', torch.zeros(num_nodes, mem_dim, device=device))
        self.register_buffer('last_update', torch.zeros(num_nodes, device=device))
        self.updater = nn.GRUCell(msg_dim, mem_dim)

    def get_memory(self, node_ids: torch.LongTensor):
        return self.memory[node_ids]

    def set_memory(self, node_ids: torch.LongTensor, new_mem: torch.Tensor):
        self.memory[node_ids] = new_mem.detach()  # detach to avoid accidental graph through storage

    def update_with_messages(self, node_ids: List[int], agg_messages: torch.Tensor, timestamps: torch.Tensor):
        """
        node_ids: list of node indices (length M)
        agg_messages: (M, msg_dim)
        timestamps: (M,) new timestamps for those nodes
        """
        if len(node_ids) == 0:
            return
        node_ids_t = torch.tensor(node_ids, dtype=torch.long, device=self.memory.device)
        cur_mem = self.memory[node_ids_t]                  # (M, mem_dim)
        new_mem = self.updater(agg_messages.to(cur_mem.dtype), cur_mem)
        # write back (no grads propagate through memory stores in this simplified update)
        self.memory[node_ids_t] = new_mem.detach()
        self.last_update[node_ids_t] = timestamps.detach()

class RawMessageStore:
    """
    Stores the last raw message per node (or aggregated messages).
    For simplicity we store one message per node (the last). In the paper
    they aggregate messages produced in the same batch before updating memory.
    Here we mimic: raw_message[node] = last message (tensor) ; if none -> None
    """
    def __init__(self, num_nodes:int, msg_dim:int, device='cpu'):
        self.num_nodes = num_nodes
        self.msg_dim = msg_dim
        self.device = device
        # Use a dict mapping node -> tensor message (to save memory for sparse)
        self.store = dict()

    def set_messages(self, node_ids: List[int], messages: torch.Tensor):
        # messages: (M, msg_dim)
        for i, nid in enumerate(node_ids):
            self.store[int(nid)] = messages[i].detach().to(self.device)

    def get_messages_for_nodes(self, node_ids: List[int]) -> Tuple[List[int], Optional[torch.Tensor]]:
        """
        Return filtered node_ids that have stored messages and a tensor (K, msg_dim)
        """
        found = []
        msgs = []
        for nid in node_ids:
            if int(nid) in self.store:
                found.append(int(nid))
                msgs.append(self.store[int(nid)])
        if not found:
            return [], None
        return found, torch.stack(msgs, dim=0).to(self.device)

    def clear_nodes(self, node_ids: List[int]):
        for nid in node_ids:
            self.store.pop(int(nid), None)

# --------------------------
# Neighbor finder (most recent)
# --------------------------
class NeighborFinder:
    """
    Maintains adjacency lists with timestamps and edge features.
    For each node we keep a deque of (neighbor_id, ts, edge_feat_tensor) sorted by time (most recent at right).
    """
    def __init__(self, num_nodes:int, max_neighbors:int=100):
        self.num_nodes = num_nodes
        self.max_neighbors = max_neighbors
        self.adj = [deque() for _ in range(num_nodes)]  # list of deques

    def insert_edge(self, src:int, dst:int, ts: float, edge_feat: Optional[torch.Tensor]):
        # insert both directions (for undirected)
        self._insert_one(src, dst, ts, edge_feat)
        self._insert_one(dst, src, ts, edge_feat)

    def _insert_one(self, node:int, nbr:int, ts: float, edge_feat: Optional[torch.Tensor]):
        dq = self.adj[node]
        dq.append((nbr, ts, edge_feat))
        # keep deque length bounded
        if len(dq) > self.max_neighbors:
            dq.popleft()

    def get_most_recent_neighbors(self, node_ids: List[int], k:int):
        """
        For each node in node_ids, return up to k most recent neighbors as lists.
        Output shapes:
          neigh_ids: list of lists of length k (padded with -1)
          neigh_timestamps: list of lists of length k (padded with 0)
          neigh_edge_feats: list of lists of tensors or None
        """
        neigh_ids = []
        neigh_ts = []
        neigh_edge_feats = []
        for nid in node_ids:
            dq = self.adj[nid]
            # take from right (most recent)
            recent = list(dq)[-k:] if len(dq) > 0 else []
            # pad to k with (-1,0,None)
            pad = k - len(recent)
            recent_padded = [(-1, 0.0, None)] * pad + recent  # oldest-first so final order oldest->newest
            ids = [x[0] for x in recent_padded]
            ts  = [x[1] for x in recent_padded]
            efs = [x[2] for x in recent_padded]
            neigh_ids.append(ids)
            neigh_ts.append(ts)
            neigh_edge_feats.append(efs)
        return neigh_ids, neigh_ts, neigh_edge_feats

# --------------------------
# Temporal Attention Embedder (TGN-attn)
# --------------------------
class TemporalAttentionEmbedding(nn.Module):
    """
    For each target node, attend over its k most recent neighbors. Query is target node memory.
    Keys/Values are neighbor memories + time encoding + edge features.
    We implement a multi-head attention per node: sequence length = k.
    """
    def __init__(self, mem_dim:int, edge_dim:int, time_dim:int, out_dim:int, num_heads:int=2):
        super().__init__()
        self.mem_dim = mem_dim
        self.edge_dim = edge_dim if edge_dim else 0
        self.time_dim = time_dim
        self.input_dim = mem_dim + self.edge_dim + time_dim
        self.num_heads = num_heads
        self.out_dim = out_dim
        # a small linear to project neighbor (mem + edge + time) into key/value space
        self.key_lin = nn.Linear(self.input_dim, out_dim)
        self.value_lin = nn.Linear(self.input_dim, out_dim)
        # project query (node_mem) into same dim
        self.query_lin = nn.Linear(mem_dim, out_dim)
        self.attention = nn.MultiheadAttention(embed_dim=out_dim, num_heads=num_heads, batch_first=True)
        # final combiner
        self.combine = nn.Linear(mem_dim + out_dim, out_dim)

    def forward(self, node_mems: torch.Tensor, neigh_mems: torch.Tensor,
                neigh_edge_feats: Optional[torch.Tensor], neigh_dt_enc: torch.Tensor):
        """
        node_mems: (B, mem_dim)
        neigh_mems: (B, k, mem_dim)
        neigh_edge_feats: (B, k, edge_dim) or None
        neigh_dt_enc: (B, k, time_dim)
        Returns: z (B, out_dim)
        """
        B, k, _ = neigh_dt_enc.shape
        # build neighbor input
        if self.edge_dim > 0 and neigh_edge_feats is not None:
            neigh_input = torch.cat([neigh_mems, neigh_edge_feats, neigh_dt_enc], dim=-1)  # (B,k,input_dim)
        else:
            neigh_input = torch.cat([neigh_mems, neigh_dt_enc], dim=-1)
        keys = self.key_lin(neigh_input)    # (B,k,out_dim)
        vals = self.value_lin(neigh_input)  # (B,k,out_dim)
        queries = self.query_lin(node_mems).unsqueeze(1)  # (B,1,out_dim)
        # use nn.MultiheadAttention which expects (B, seq_len, embed)
        attn_out, _ = self.attention(queries, keys, vals, need_weights=False)  # (B,1,out_dim)
        attn_out = attn_out.squeeze(1)  # (B, out_dim)
        z = F.relu(self.combine(torch.cat([node_mems, attn_out], dim=-1)))  # (B, out_dim)
        return z

# --------------------------
# Decoder for link prediction
# --------------------------
class LinkPredictor(nn.Module):
    """
    Simple MLP scoring function on concatenated embeddings.
    """
    def __init__(self, in_dim:int, hidden:int=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, z_src, z_dst):
        x = torch.cat([z_src, z_dst], dim=-1)
        return torch.sigmoid(self.net(x)).squeeze(-1)  # (B,)

# --------------------------
# Full TGN Model (orchestrates modules)
# --------------------------
class TGNModel(nn.Module):
    def __init__(self, num_nodes:int, mem_dim:int=172, msg_dim:int=100,
                 edge_dim:int=0, time_dim:int=100, embed_dim:int=100, k: int = 10,
                 device='cpu'):
        super().__init__()
        self.device = device
        self.num_nodes = num_nodes
        self.mem_dim = mem_dim
        self.msg_dim = msg_dim
        self.edge_dim = edge_dim
        self.time_dim = time_dim
        self.embed_dim = embed_dim
        self.k = k

        # modules
        self.time_encoder = TimeEncoder(time_dim).to(device)
        self.msg_fn = MessageFunction(mem_dim, edge_dim, time_dim, msg_dim).to(device)
        self.memory = MemoryModule(num_nodes, mem_dim, msg_dim, device=device).to(device)
        self.raw_store = RawMessageStore(num_nodes, msg_dim, device=device)
        self.neigh_finder = NeighborFinder(num_nodes, max_neighbors=500)  # store a lot
        self.embedder = TemporalAttentionEmbedding(mem_dim, edge_dim, time_dim, embed_dim).to(device)
        self.predictor = LinkPredictor(in_dim=embed_dim*2).to(device)

    def compute_raw_messages(self, src_mems, dst_mems, edge_feats, dt_enc):
        """
        Compute raw messages for both src and dst (directional or symmetric).
        Returns raw_msg_src, raw_msg_dst both shape (B, msg_dim)
        """
        # per paper raw message uses s_src, s_dst, edge, dt
        raw = self.msg_fn(src_mems, dst_mems, edge_feats, dt_enc)  # (B, msg_dim)
        # symmetric for both nodes (paper sometimes computes both directions - we reuse same)
        return raw, raw

    def get_node_embeddings(self, node_ids: List[int]):
        """
        Helper to get embeddings z_i(t) for a set of nodes using current memory & neighbors.
        Returns (B, embed_dim)
        """
        device = self.device
        node_ids_t = torch.tensor(node_ids, dtype=torch.long, device=device)
        node_mems = self.memory.get_memory(node_ids_t)  # (B, mem_dim)
        # fetch most recent neighbors
        neigh_ids, neigh_ts, neigh_edge_feats = self.neigh_finder.get_most_recent_neighbors(node_ids, self.k)
        # build neighbor mems tensor (B,k,mem_dim) and edge_feat tensor (B,k,edge_dim) and dt enc (B,k,time_dim)
        B = len(node_ids)
        k = self.k
        neigh_mems = torch.zeros(B, k, self.mem_dim, device=device)
        neigh_edge_t = None if self.edge_dim == 0 else torch.zeros(B, k, self.edge_dim, device=device)
        neigh_dt_enc = torch.zeros(B, k, self.time_dim, device=device)
        for i in range(B):
            for j in range(k):
                nid = neigh_ids[i][j]
                if nid == -1:
                    # leave zeros (padding)
                    continue
                neigh_mems[i,j] = self.memory.get_memory(torch.tensor(nid, device=device))
                if self.edge_dim > 0 and neigh_edge_feats[i][j] is not None:
                    neigh_edge_t[i,j] = neigh_edge_feats[i][j].to(device)
                dt = torch.tensor(self.memory.last_update[nid].item(), device=device)
                # compute delta time between target node last update and neighbor timestamp:
                # We'll use neighbor timestamp stored in neigh_ts
                neigh_dt_enc[i,j] = self.time_encoder(torch.tensor(neigh_ts[i][j], device=device).unsqueeze(0))
        # compute embeddings with attention
        z = self.embedder(node_mems, neigh_mems, neigh_edge_t, neigh_dt_enc)  # (B, embed_dim)
        return z

# --------------------------
# Training Loop (Algorithm 1 style)
# --------------------------
def train_tgn(model: TGNModel, events: List[Tuple[int,int,float,Optional[torch.Tensor]]],
              num_epochs:int=1, batch_size:int=200, lr:float=1e-3, neg_ratio:int=1, device='cpu'):
    """
    events: chronological list of (src,dst,ts,edge_feat)
    """
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    dataset = TemporalEdgeDataset(events)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_events)

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        for batch_idx, batch in enumerate(loader):
            src = batch['src'].to(device)
            dst = batch['dst'].to(device)
            ts  = batch['ts'].to(device)
            edge_feats = batch['edge_feats'].to(device) if batch['edge_feats'] is not None else None
            B = src.size(0)

            # Step A: BEFORE predicting for this batch, update memory using previously stored raw messages
            # Gather nodes involved in this batch
            nodes_in_batch = set(src.tolist() + dst.tolist())
            # Check raw store for messages for these nodes
            node_list = list(nodes_in_batch)
            found_nodes, found_msgs = model.raw_store.get_messages_for_nodes(node_list)
            if found_nodes and found_msgs is not None:
                # Update memory for those nodes using the stored messages
                # For timestamp we use model.memory.last_update (we don't have a new ts for those old stored messages)
                # But per Algorithm 1: aggregate messages and update memory with their associated timestamps
                timestamps = torch.tensor([model.memory.last_update[n] for n in found_nodes], device=device)
                model.memory.update_with_messages(found_nodes, found_msgs.to(device), timestamps)

                # After using them, clear from raw store
                model.raw_store.clear_nodes(found_nodes)

            # Step B: compute embeddings for src and dst using current memory (which just got updated)
            src_list = src.tolist()
            dst_list = dst.tolist()
            # embeddings
            z_src = model.get_node_embeddings(src_list)   # (B, embed_dim)
            z_dst = model.get_node_embeddings(dst_list)   # (B, embed_dim)

            # Step C: scoring positive pairs
            pos_scores = model.predictor(z_src, z_dst)  # (B,)

            # Step D: negative sampling (for each src sample a random negative dst)
            neg_dst_ids = torch.randint(0, model.num_nodes, (B * neg_ratio,), device=device)
            # get embeddings for neg dst (repeat src embeddings accordingly if neg_ratio>1)
            z_neg_dst = model.get_node_embeddings(neg_dst_ids.tolist())  # (B*neg_ratio, embed_dim)
            # expand z_src to match neg samples
            z_src_rep = z_src.repeat_interleave(neg_ratio, dim=0)  # (B*neg_ratio, embed_dim)
            neg_scores = model.predictor(z_src_rep, z_neg_dst)  # (B*neg_ratio,)

            # Loss: binary cross entropy; pos labeled 1, neg labeled 0
            pos_loss = F.binary_cross_entropy(pos_scores, torch.ones_like(pos_scores))
            neg_loss = F.binary_cross_entropy(neg_scores, torch.zeros_like(neg_scores))
            loss = pos_loss + neg_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Step E: compute raw messages for current batch (use memory before writing new memory)
            # Need dt encoding: delta between event ts and last_update per node
            last_src_times = torch.tensor([model.memory.last_update[n] for n in src.tolist()], device=device)
            dt_src = ts - last_src_times
            dt_src_enc = model.time_encoder(dt_src)

            last_dst_times = torch.tensor([model.memory.last_update[n] for n in dst.tolist()], device=device)
            dt_dst = ts - last_dst_times
            dt_dst_enc = model.time_encoder(dt_dst)

            # current memory for nodes (we used memory above; get again)
            cur_src_mem = model.memory.get_memory(src)
            cur_dst_mem = model.memory.get_memory(dst)
            raw_src_msgs, raw_dst_msgs = model.compute_raw_messages(cur_src_mem, cur_dst_mem,
                                                                    edge_feats, dt_src_enc)
            # Store raw messages for future batches (for both src & dst)
            # We'll store one message per node (latest). If multiple events for same node in batch,
            # we aggregate by taking the latest (here the batch is chronological so last wins).
            # Build mapping node->message for all nodes in the batch:
            per_node_msgs = defaultdict(list)
            for i in range(B):
                per_node_msgs[int(src[i].item())].append(raw_src_msgs[i].detach().cpu())  # store on cpu to save GPU memory
                per_node_msgs[int(dst[i].item())].append(raw_dst_msgs[i].detach().cpu())
            # For each node take mean of messages in this batch (aggregation)
            node_ids_to_store = []
            node_msgs_to_store = []
            for nid, msgs in per_node_msgs.items():
                agg = torch.stack(msgs, dim=0).mean(dim=0).to(device)
                node_ids_to_store.append(nid)
                node_msgs_to_store.append(agg)
            if node_ids_to_store:
                model.raw_store.set_messages(node_ids_to_store, torch.stack(node_msgs_to_store, dim=0).to(device))

            # Finally: update neighbor lists with edges from this batch (so future embeddings see them)
            for i in range(B):
                s = int(src[i].item()); d = int(dst[i].item()); t = float(ts[i].item())
                ef = edge_feats[i] if edge_feats is not None else None
                model.neigh_finder.insert_edge(s, d, t, ef)

            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1} Batch {batch_idx} Loss {loss.item():.4f}")

    print("Training complete.")
