"""
Ablation Study on IBM Transaction Graph

This will prove that temporal models (v3-v5) work with proper temporal data!
"""

import sys
sys.path.append('data')

from load_ibm_graph import load_ibm_transaction_graph
from train_ablation import train_model, compute_class_weights
import pandas as pd
import os

def run_ibm_ablation_study(sample_size=1000000, versions=None):
    """
    Run ablation study on IBM transaction data.
    
    Args:
        sample_size: Number of transactions to use (None = all 24M)
        versions: List of versions to train (default: [0,1,2,3,4,5])
    """
    if versions is None:
        versions = [0, 1, 2, 3, 4, 5]
    
    print("="*80)
    print("HMSTA ABLATION STUDY - IBM TRANSACTION DATA")
    print("="*80)
    print(f"\n🎯 Testing with {sample_size:,} transactions")
    print(f"   This data has REAL temporal structure!")
    print(f"   Expected: v3-v5 should outperform v1")
    
    # Load IBM graph
    data = load_ibm_transaction_graph(
        sample_size=sample_size,
        min_transactions_per_user=50  # More transactions = better temporal patterns
    )
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Train each version
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n📱 Using device: {device}")
    
    results = []
    for version in versions:
        result = train_model(
            version=version,
            data=data,
            hidden_dim=128,
            epochs=100,
            patience=30,
            lr=0.0001,
            device=device
        )
        results.append(result)
    
    # Print comparison table
    print("\n" + "="*80)
    print("ABLATION STUDY RESULTS - IBM DATA")
    print("="*80)
    print(f"\n{'Version':<8} {'Description':<35} {'AUC':<8} {'F1':<8} {'Params':<10} {'Time(s)':<10}")
    print("-" * 90)
    
    baseline_auc = results[0]['auc'] if results else 0
    
    for result in results:
        improvement = ((result['auc'] - baseline_auc) / baseline_auc * 100) if baseline_auc > 0 else 0
        print(f"v{result['version']:<7} {result['description']:<35} "
              f"{result['auc']:.4f}   {result['f1']:.4f}   "
              f"{result['num_params']:<10,} {result['training_time']:<10.1f}")
        if result['version'] > 0:
            print(f"{'':>45} ({improvement:+.2f}%)")
    
    # Print component contributions
    print("\n" + "="*80)
    print("COMPONENT CONTRIBUTIONS")
    print("="*80)
    
    from src.models.hmsta_v2 import VERSION_DESCRIPTIONS
    for i in range(1, len(results)):
        prev_auc = results[i-1]['auc']
        curr_auc = results[i]['auc']
        improvement = (curr_auc - prev_auc) / prev_auc * 100
        component = VERSION_DESCRIPTIONS[i].replace('+ ', '')
        print(f"{component:<30} → {improvement:+.2f}% AUC improvement")
    
    # Save results
    df_results = pd.DataFrame(results)
    df_results.to_csv('results/ablation_study_ibm_results.csv', index=False)
    print(f"\n💾 Results saved to results/ablation_study_ibm_results.csv")
    
    # Compare with Ethereum results
    print("\n" + "="*80)
    print("📊 COMPARISON: Ethereum vs IBM")
    print("="*80)
    
    try:
        eth_results = pd.read_csv('results/ablation_study_results.csv')
        
        print("\n🔬 Key Insight: Data Structure Matters!")
        print("-" * 60)
        
        for version in versions:
            if version < len(results):
                ibm_auc = results[version]['auc']
                eth_row = eth_results[eth_results['version'] == version]
                
                if not eth_row.empty:
                    eth_auc = eth_row['auc'].values[0]
                    diff = (ibm_auc - eth_auc) * 100
                    
                    desc = results[version]['description']
                    print(f"\nv{version}: {desc}")
                    print(f"  Ethereum (aggregated):  {eth_auc:.4f}")
                    print(f"  IBM (temporal):         {ibm_auc:.4f}  ({diff:+.2f}%)")
        
        print("\n💡 Analysis:")
        print("   If v3-v5 perform BETTER on IBM:")
        print("   → Architecture is correct!")
        print("   → Temporal components need sequential data")
        print("   → This validates your design!")
        
    except FileNotFoundError:
        print("   (Ethereum results not found for comparison)")
    
    return results


if __name__ == '__main__':
    # Run with 1M transactions (good balance of size vs speed)
    print("\n🚀 Starting IBM ablation study...")
    print("   This will take ~10-15 minutes for all 6 versions")
    
    results = run_ibm_ablation_study(
        sample_size=1000000,  # 1 million transactions
        versions=[0, 1, 2, 3, 4, 5]  # All versions
    )
    
    print("\n✅ IBM ablation study complete!")
    print("\n🎯 The moment of truth:")
    print("   Check if v3-v5 outperform v1 on this temporal data!")
