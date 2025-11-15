"""
Ablation study on IBM TRANSACTION-LEVEL graph.

This uses individual transactions as nodes (not aggregated users).
This is what temporal GNNs are designed for!
"""
import torch
import pandas as pd
from data.load_ibm_transaction_level import load_ibm_transaction_level_graph
from train_ablation import train_model

def run_ibm_transaction_ablation(
    sample_size=50000,  # 50K transactions (manageable size)
    versions=[0, 1, 2, 3, 4, 5],
    epochs=100,
    patience=30
):
    print("\n🚀 Starting IBM TRANSACTION-LEVEL ablation study...")
    print(f"   This uses INDIVIDUAL transactions, not aggregated users!")
    print(f"   Expected: v3-v5 should now work better with sequential data")
    
    print("\n" + "="*80)
    print("HMSTA ABLATION STUDY - IBM TRANSACTION-LEVEL DATA")
    print("="*80)
    
    print(f"\n🎯 Testing with {sample_size:,} individual transactions")
    print(f"   Each transaction is a node (not user aggregation)")
    print(f"   Edges connect transactions chronologically")
    
    print("="*80)
    
    # Load transaction-level graph
    data = load_ibm_transaction_level_graph(
        sample_size=sample_size,
        max_time_gap_days=7
    )
    
    # Store results
    results = []
    
    # Train each version
    for version in versions:
        print(f"\n{'='*80}")
        
        result = train_model(
            version=version,
            data=data,
            hidden_dim=128,
            epochs=epochs,
            patience=patience,
            lr=0.0001
        )
        
        results.append(result)
    
    # Create results DataFrame
    df_results = pd.DataFrame(results)
    
    # Display results
    print(f"\n{'='*80}")
    print("ABLATION STUDY RESULTS - IBM TRANSACTION-LEVEL DATA")
    print("="*80)
    print()
    print(df_results.to_string(index=False))
    
    # Calculate component contributions
    print(f"\n{'='*80}")
    print("COMPONENT CONTRIBUTIONS")
    print("="*80)
    
    baseline_auc = df_results.iloc[0]['auc']
    for i in range(1, len(df_results)):
        improvement = (df_results.iloc[i]['auc'] - df_results.iloc[i-1]['auc']) / df_results.iloc[i-1]['auc'] * 100
        print(f"{df_results.iloc[i]['description']:<40} → {improvement:+.2f}% AUC improvement")
    
    # Save results
    output_file = 'results/ablation_study_ibm_txn_results.csv'
    df_results.to_csv(output_file, index=False)
    print(f"\n💾 Results saved to {output_file}")
    
    # Compare with user-level results
    print(f"\n{'='*80}")
    print("📊 COMPARISON: User-Level vs Transaction-Level")
    print("="*80)
    print("\n🔬 Testing the hypothesis:")
    print("   User-level: Features are pre-aggregated → Simple models win")
    print("   Transaction-level: Raw data → Temporal models should win!")
    print("-"*60)
    
    try:
        user_results = pd.read_csv('results/ablation_study_ibm_results.csv')
        
        for i in range(len(df_results)):
            version = df_results.iloc[i]['version']
            user_row = user_results[user_results['version'] == f'v{version}']
            
            if len(user_row) > 0:
                user_auc = user_row.iloc[0]['test_auc']
                txn_auc = df_results.iloc[i]['auc']
                diff = txn_auc - user_auc
                
                print(f"\nv{version}: {df_results.iloc[i]['description']}")
                print(f"  User-level:        {user_auc:.4f}")
                print(f"  Transaction-level: {txn_auc:.4f}  ({diff:+.4f})")
    except FileNotFoundError:
        print("   (User-level results not found for comparison)")
    
    print(f"\n{'='*80}")
    print("💡 KEY INSIGHTS:")
    print("="*80)
    
    # Check if v3-v5 improved over v1
    v1_auc = df_results[df_results['version'] == 'v1']['auc'].values[0]
    
    for version in [3, 4, 5]:
        version_str = f'v{version}'
        if version_str in df_results['version'].values:
            v_auc = df_results[df_results['version'] == version_str]['auc'].values[0]
            if v_auc > v1_auc:
                improvement = (v_auc - v1_auc) / v1_auc * 100
                print(f"   ✅ {version_str} OUTPERFORMS v1 by {improvement:.2f}%!")
                print(f"      → Temporal components work with sequential data!")
            else:
                degradation = (v1_auc - v_auc) / v1_auc * 100
                print(f"   ⚠️  {version_str} underperforms v1 by {degradation:.2f}%")
    
    print(f"\n✅ IBM transaction-level ablation study complete!")
    print("\n🎯 The moment of truth:")
    print("   If v3-v5 > v1 → Architecture validated!")
    print("   If v1 still wins → Need to investigate further")
    
    return df_results


if __name__ == '__main__':
    results = run_ibm_transaction_ablation(
        sample_size=50000,  # 50K transactions
        versions=[0, 1, 2, 3, 4, 5]
    )
