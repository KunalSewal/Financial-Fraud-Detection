"""
Model Comparison Script - Validate HMSTA Novelty

Compare:
1. MLP (baseline)
2. GraphSAGE (baseline)
3. TGN (component)
4. MPTGNN (component)
5. HMSTA (our novel architecture)

This demonstrates that HMSTA > individual components = novelty!
"""

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import time

from src.models.hmsta import create_hmsta_model
from src.models.tgn import TGN
from src.models.mptgnn import MPTGNN
from src.data.temporal_graph_builder import load_and_build_temporal_graph


def evaluate_model(model, data, mask, device):
    """Evaluate a model and return metrics."""
    model.eval()
    with torch.no_grad():
        x = data.x.to(device)
        edge_index = data.edge_index.to(device)
        edge_attr = data.edge_attr.to(device) if data.edge_attr is not None else None
        timestamps = data.timestamps.to(device) if hasattr(data, 'timestamps') else torch.zeros(edge_index.size(1)).to(device)
        
        # Forward pass (handle different model outputs)
        if isinstance(model, (TGN, MPTGNN)):
            logits = model(x, edge_index, edge_attr, timestamps)
        else:  # HMSTA
            output = model(x, edge_index, edge_attr, timestamps)
            logits = output['logits']
        
        # Get predictions
        probs = F.softmax(logits[mask], dim=-1)
        preds = logits[mask].argmax(dim=-1)
        labels = data.y[mask].cpu().numpy()
        
        # Compute metrics
        probs_cpu = probs[:, 1].cpu().numpy()
        preds_cpu = preds.cpu().numpy()
        
        metrics = {
            'accuracy': (preds_cpu == labels).mean(),
            'roc_auc': roc_auc_score(labels, probs_cpu),
            'f1': f1_score(labels, preds_cpu),
            'precision': precision_score(labels, preds_cpu, zero_division=0),
            'recall': recall_score(labels, preds_cpu, zero_division=0)
        }
        
        return metrics


def load_baseline_results():
    """Load pre-trained baseline results."""
    baselines = {
        'MLP': {
            'roc_auc': 0.9399,
            'f1': 0.8650,
            'precision': 0.8523,
            'recall': 0.8780
        },
        'GraphSAGE': {
            'roc_auc': 0.9131,
            'f1': 0.8482,
            'precision': 0.8356,
            'recall': 0.8611
        }
    }
    return baselines


def train_simple_model(model, data, optimizer, device, num_epochs=50):
    """Quick training for comparison models."""
    print(f"   Training for {num_epochs} epochs...")
    
    best_val_auc = 0
    best_model_state = None
    
    for epoch in range(1, num_epochs + 1):
        # Train
        model.train()
        optimizer.zero_grad()
        
        x = data.x.to(device)
        edge_index = data.edge_index.to(device)
        edge_attr = data.edge_attr.to(device) if data.edge_attr is not None else None
        timestamps = data.timestamps.to(device) if hasattr(data, 'timestamps') else torch.zeros(edge_index.size(1)).to(device)
        
        if isinstance(model, (TGN, MPTGNN)):
            logits = model(x, edge_index, edge_attr, timestamps)
        else:
            output = model(x, edge_index, edge_attr, timestamps)
            logits = output['logits']
        
        loss = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask].to(device))
        loss.backward()
        optimizer.step()
        
        # Validate
        if epoch % 10 == 0:
            val_metrics = evaluate_model(model, data, data.val_mask, device)
            print(f"      Epoch {epoch:2d}: Val AUC = {val_metrics['roc_auc']:.4f}")
            
            if val_metrics['roc_auc'] > best_val_auc:
                best_val_auc = val_metrics['roc_auc']
                best_model_state = model.state_dict().copy()
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model


def compare_models():
    """
    Compare all models to validate HMSTA novelty.
    
    Key Question: Is HMSTA better than its components?
    - If HMSTA > TGN: Multi-path helps
    - If HMSTA > MPTGNN: Temporal memory helps
    - If HMSTA > TGN + MPTGNN individually: Hybrid is better!
    """
    print("=" * 80)
    print("Model Comparison - Validating HMSTA Novelty")
    print("=" * 80)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n📱 Using device: {device}")
    
    # Load data
    print("\n📦 Loading Ethereum dataset...")
    data = load_and_build_temporal_graph('data/transaction_dataset.csv', source='ethereum')
    print(f"   • Nodes: {data.num_nodes:,}")
    print(f"   • Edges: {data.num_edges:,}")
    
    results = {}
    
    # ==================== Baseline Results ====================
    print("\n" + "=" * 80)
    print("1. Loading Baseline Results")
    print("=" * 80)
    
    baselines = load_baseline_results()
    results.update(baselines)
    
    for name, metrics in baselines.items():
        print(f"\n{name}:")
        print(f"   • ROC-AUC: {metrics['roc_auc']:.4f}")
        print(f"   • F1 Score: {metrics['f1']:.4f}")
    
    # ==================== TGN (Component 1) ====================
    print("\n" + "=" * 80)
    print("2. Training TGN (Component 1 - Temporal Memory)")
    print("=" * 80)
    
    print("\n🏗️  Creating TGN model...")
    tgn = TGN(
        node_features=data.num_node_features,
        edge_features=data.edge_attr.size(1),
        hidden_dim=128,
        num_nodes=data.num_nodes
    ).to(device)
    
    optimizer_tgn = torch.optim.Adam(tgn.parameters(), lr=0.001)
    tgn = train_simple_model(tgn, data, optimizer_tgn, device, num_epochs=50)
    
    print("\n📊 Evaluating TGN...")
    tgn_metrics = evaluate_model(tgn, data, data.test_mask, device)
    results['TGN'] = tgn_metrics
    
    print(f"   • ROC-AUC: {tgn_metrics['roc_auc']:.4f}")
    print(f"   • F1 Score: {tgn_metrics['f1']:.4f}")
    
    # ==================== MPTGNN (Component 2) ====================
    print("\n" + "=" * 80)
    print("3. Training MPTGNN (Component 2 - Multi-Path)")
    print("=" * 80)
    
    print("\n🏗️  Creating MPTGNN model...")
    mptgnn = MPTGNN(
        in_channels=data.num_node_features,
        hidden_channels=128,
        out_channels=2,
        num_paths=3
    ).to(device)
    
    optimizer_mptgnn = torch.optim.Adam(mptgnn.parameters(), lr=0.001)
    mptgnn = train_simple_model(mptgnn, data, optimizer_mptgnn, device, num_epochs=50)
    
    print("\n📊 Evaluating MPTGNN...")
    mptgnn_metrics = evaluate_model(mptgnn, data, data.test_mask, device)
    results['MPTGNN'] = mptgnn_metrics
    
    print(f"   • ROC-AUC: {mptgnn_metrics['roc_auc']:.4f}")
    print(f"   • F1 Score: {mptgnn_metrics['f1']:.4f}")
    
    # ==================== HMSTA (Our Novel Architecture) ====================
    print("\n" + "=" * 80)
    print("4. Training HMSTA (Novel Hybrid Architecture)")
    print("=" * 80)
    
    print("\n Creating HMSTA model...")
    hmsta = create_hmsta_model(
        dataset_name='ethereum',
        node_features=data.num_node_features,
        edge_features=data.edge_attr.size(1),
        hidden_dim=128,
        num_nodes=data.num_nodes
    ).to(device)
    
    optimizer_hmsta = torch.optim.Adam(hmsta.parameters(), lr=0.001)
    
    # Train HMSTA (more epochs since it's more complex)
    print(f"   Training for 100 epochs...")
    best_val_auc = 0
    best_model_state = None
    
    for epoch in range(1, 101):
        # Train
        hmsta.train()
        optimizer_hmsta.zero_grad()
        
        x = data.x.to(device)
        edge_index = data.edge_index.to(device)
        edge_attr = data.edge_attr.to(device)
        timestamps = data.timestamps.to(device) if hasattr(data, 'timestamps') else torch.zeros(edge_index.size(1)).to(device)
        
        output = hmsta(x, edge_index, edge_attr, timestamps)
        loss = F.cross_entropy(output['logits'][data.train_mask], data.y[data.train_mask].to(device))
        loss.backward()
        optimizer_hmsta.step()
        
        # Validate
        if epoch % 10 == 0:
            val_metrics = evaluate_model(hmsta, data, data.val_mask, device)
            print(f"      Epoch {epoch:3d}: Val AUC = {val_metrics['roc_auc']:.4f}")
            
            if val_metrics['roc_auc'] > best_val_auc:
                best_val_auc = val_metrics['roc_auc']
                best_model_state = hmsta.state_dict().copy()
    
    # Restore best model
    if best_model_state is not None:
        hmsta.load_state_dict(best_model_state)
    
    print("\n📊 Evaluating HMSTA...")
    hmsta_metrics = evaluate_model(hmsta, data, data.test_mask, device)
    results['HMSTA'] = hmsta_metrics
    
    print(f"   • ROC-AUC: {hmsta_metrics['roc_auc']:.4f}")
    print(f"   • F1 Score: {hmsta_metrics['f1']:.4f}")
    
    # ==================== Summary & Analysis ====================
    print("\n" + "=" * 80)
    print("📊 FINAL COMPARISON - VALIDATING NOVELTY")
    print("=" * 80)
    
    print(f"\n{'Model':<20} {'ROC-AUC':<12} {'F1 Score':<12} {'Type':<25}")
    print("-" * 70)
    
    model_order = ['MLP', 'GraphSAGE', 'TGN', 'MPTGNN', 'HMSTA']
    model_types = {
        'MLP': 'Baseline (Static)',
        'GraphSAGE': 'Baseline (Static GNN)',
        'TGN': 'Component (Temporal)',
        'MPTGNN': 'Component (Multi-Path)',
        'HMSTA': '🌟 NOVEL (Hybrid) 🌟'
    }
    
    for name in model_order:
        if name in results:
            metrics = results[name]
            print(f"{name:<20} {metrics['roc_auc']:<12.4f} {metrics['f1']:<12.4f} {model_types[name]:<25}")
    
    # Compute improvements
    print("\n" + "=" * 80)
    print("📈 IMPROVEMENT ANALYSIS (vs Components)")
    print("=" * 80)
    
    if 'TGN' in results and 'MPTGNN' in results:
        tgn_auc = results['TGN']['roc_auc']
        mptgnn_auc = results['MPTGNN']['roc_auc']
        hmsta_auc = results['HMSTA']['roc_auc']
        
        improvement_over_tgn = (hmsta_auc - tgn_auc) / tgn_auc * 100
        improvement_over_mptgnn = (hmsta_auc - mptgnn_auc) / mptgnn_auc * 100
        improvement_over_best_baseline = (hmsta_auc - 0.9399) / 0.9399 * 100
        
        print(f"\nHMSTA vs Individual Components:")
        print(f"   • vs TGN:      {improvement_over_tgn:+.2f}% {'✅ Better!' if improvement_over_tgn > 0 else '❌ Worse'}")
        print(f"   • vs MPTGNN:   {improvement_over_mptgnn:+.2f}% {'✅ Better!' if improvement_over_mptgnn > 0 else '❌ Worse'}")
        print(f"   • vs MLP:      {improvement_over_best_baseline:+.2f}% {'✅ Better!' if improvement_over_best_baseline > 0 else '❌ Worse'}")
        
        # Novelty validation
        print("\n" + "=" * 80)
        print("🎯 NOVELTY VALIDATION")
        print("=" * 80)
        
        if improvement_over_tgn > 0 and improvement_over_mptgnn > 0:
            print("\n✅ NOVELTY CONFIRMED!")
            print("   HMSTA outperforms BOTH individual components.")
            print("   This validates that hybrid architecture provides synergistic benefits.")
            print("\n   📝 For Presentation:")
            print("      'Our novel HMSTA architecture combines TGN and MPTGNN,")
            print(f"      achieving {hmsta_auc:.4f} ROC-AUC, which is:")
            print(f"      • {improvement_over_tgn:.1f}% better than TGN alone")
            print(f"      • {improvement_over_mptgnn:.1f}% better than MPTGNN alone")
            print(f"      • {improvement_over_best_baseline:.1f}% better than best baseline (MLP)'")
        elif improvement_over_tgn > 0 or improvement_over_mptgnn > 0:
            print("\n⚠️  PARTIAL NOVELTY")
            print("   HMSTA outperforms one component but not the other.")
            print("   Consider tuning hyperparameters or training longer.")
        else:
            print("\n❌ NOVELTY NOT VALIDATED")
            print("   HMSTA does not outperform individual components.")
            print("   Possible issues:")
            print("      • Need more training epochs")
            print("      • Hyperparameter tuning required")
            print("      • Architecture needs adjustment")
    
    # Save results
    print("\n" + "=" * 80)
    print("💾 Saving Results")
    print("=" * 80)
    
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    
    # Save JSON
    json_path = results_dir / 'model_comparison.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Results saved to {json_path}")
    
    # Create comparison plot
    create_comparison_plot(results, model_order, results_dir)
    
    return results


def create_comparison_plot(results, model_order, output_dir):
    """Create visualization of model comparison."""
    print("\n📊 Creating comparison visualization...")
    
    # Prepare data
    models = []
    aucs = []
    f1s = []
    
    for name in model_order:
        if name in results:
            models.append(name)
            aucs.append(results[name]['roc_auc'])
            f1s.append(results[name]['f1'])
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # ROC-AUC comparison
    colors = ['#3498db' if m != 'HMSTA' else '#e74c3c' for m in models]
    ax1.barh(models, aucs, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax1.set_xlabel('ROC-AUC Score', fontsize=12, fontweight='bold')
    ax1.set_title('Model Comparison: ROC-AUC', fontsize=14, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3, linestyle='--')
    ax1.set_xlim(0.85, max(aucs) * 1.02)
    
    # Add value labels
    for i, (model, auc) in enumerate(zip(models, aucs)):
        ax1.text(auc + 0.002, i, f'{auc:.4f}', va='center', fontweight='bold')
    
    # F1 Score comparison
    ax2.barh(models, f1s, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax2.set_xlabel('F1 Score', fontsize=12, fontweight='bold')
    ax2.set_title('Model Comparison: F1 Score', fontsize=14, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3, linestyle='--')
    ax2.set_xlim(0.80, max(f1s) * 1.02)
    
    # Add value labels
    for i, (model, f1) in enumerate(zip(models, f1s)):
        ax2.text(f1 + 0.002, i, f'{f1:.4f}', va='center', fontweight='bold')
    
    plt.tight_layout()
    
    # Save
    plot_path = output_dir / 'model_comparison.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✅ Plot saved to {plot_path}")
    
    plt.close()


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("HMSTA Novelty Validation")
    print("=" * 80)
    print("\nThis script will:")
    print("  1. Load baseline results (MLP, GraphSAGE)")
    print("  2. Train TGN (component 1)")
    print("  3. Train MPTGNN (component 2)")
    print("  4. Train HMSTA (novel hybrid)")
    print("  5. Compare all models")
    print("  6. Validate novelty claim\n")
    
    input("Press Enter to start comparison...")
    
    results = compare_models()
    
    print("\n" + "=" * 80)
    print("✅ COMPARISON COMPLETE!")
    print("=" * 80)
    print("\nCheck results/ directory for:")
    print("  • model_comparison.json (detailed metrics)")
    print("  • model_comparison.png (visualization)")
    print("\nUse these results in your presentation to validate novelty!")
    print()
