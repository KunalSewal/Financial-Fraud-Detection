"""Quick script to fix all scatter operations in hmsta_v2.py"""

# Read the file
with open('src/models/hmsta_v2.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Define the old pattern (without dim_size)
old_pattern = '''        # Add temporal encoding
        if timestamps is not None:
            node_times = scatter(timestamps, edge_index[1], dim=0, reduce='max')
            if node_times.max() > node_times.min():
                node_times = (node_times - node_times.min()) / (node_times.max() - node_times.min() + 1e-8)
            time_emb = self.time_encoder(node_times.unsqueeze(-1))
            h = h + time_emb'''

# Define the new pattern (with dim_size and flexibility)
new_pattern = '''        # Add temporal encoding
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
            h = h + time_emb'''

# Replace all occurrences
content_fixed = content.replace(old_pattern, new_pattern)

# Count replacements
count = content.count(old_pattern)
print(f"Found {count} occurrences to fix")

# Write back
with open('src/models/hmsta_v2.py', 'w', encoding='utf-8') as f:
    f.write(content_fixed)

print(f"✅ Fixed all scatter operations in hmsta_v2.py")
print(f"   Replaced {count} occurrences")
