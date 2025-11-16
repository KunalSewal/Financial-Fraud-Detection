"""
Quick Training Launcher for Phase 1
====================================

Simple script to start training TGN or MPTGNN on Ethereum.

Usage:
    python train.py tgn              # Train TGN with defaults
    python train.py mptgnn           # Train MPTGNN with defaults
    python train.py both             # Train both models
    python train.py tgn --wandb      # Train with W&B online
    python train.py tgn --offline    # Train with W&B offline
    python train.py tgn --no-wandb   # Train without W&B
"""

import argparse
import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run a command and stream output."""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"{'='*60}\n")
    
    try:
        process = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        for line in process.stdout:
            print(line, end='')
        
        process.wait()
        
        if process.returncode != 0:
            print(f"\n❌ Error: Command failed with exit code {process.returncode}")
            return False
        
        print(f"\n✅ Completed: {description}\n")
        return True
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Training interrupted by user")
        process.terminate()
        return False
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Quick training launcher for Phase 1 models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python train.py tgn                    # Train TGN with defaults
    python train.py mptgnn                 # Train MPTGNN with defaults
    python train.py both                   # Train both models
    python train.py tgn --wandb            # Train TGN with W&B online
    python train.py tgn --offline          # Train TGN with W&B offline
    python train.py tgn --no-wandb         # Train TGN without W&B
    python train.py tgn --memory-dim 256   # Custom hyperparameters
        """
    )
    
    parser.add_argument(
        'model',
        choices=['tgn', 'mptgnn', 'both'],
        help='Which model to train'
    )
    
    # W&B options
    wandb_group = parser.add_mutually_exclusive_group()
    wandb_group.add_argument('--wandb', action='store_true', help='Use W&B online mode')
    wandb_group.add_argument('--offline', action='store_true', help='Use W&B offline mode')
    wandb_group.add_argument('--no-wandb', action='store_true', help='Disable W&B')
    
    # Hyperparameters for TGN
    parser.add_argument('--memory-dim', type=int, help='Memory dimension (TGN)')
    parser.add_argument('--time-dim', type=int, help='Time encoding dimension (TGN)')
    parser.add_argument('--embedding-dim', type=int, help='Embedding dimension (TGN)')
    
    # Hyperparameters for MPTGNN
    parser.add_argument('--hidden-dim', type=int, help='Hidden dimension (MPTGNN)')
    parser.add_argument('--num-layers', type=int, help='Number of layers (MPTGNN)')
    
    # Common hyperparameters
    parser.add_argument('--dropout', type=float, help='Dropout rate')
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    parser.add_argument('--patience', type=int, help='Early stopping patience')
    parser.add_argument('--seed', type=int, help='Random seed')
    parser.add_argument('--run-name', type=str, help='W&B run name')
    
    args = parser.parse_args()
    
    # Verify training scripts exist
    tgn_script = Path('train_tgn_ethereum.py')
    mptgnn_script = Path('train_mptgnn_ethereum.py')
    
    if not tgn_script.exists() or not mptgnn_script.exists():
        print("❌ Error: Training scripts not found!")
        print("Expected files:")
        print(f"  - {tgn_script}")
        print(f"  - {mptgnn_script}")
        sys.exit(1)
    
    # Build command arguments
    cmd_args = []
    
    # W&B configuration
    if args.no_wandb:
        cmd_args.append('--no-wandb')
    elif args.offline:
        cmd_args.append('--wandb-offline')
    elif args.wandb:
        pass  # Default is online mode
    
    # Hyperparameters
    if args.memory_dim:
        cmd_args.extend(['--memory-dim', str(args.memory_dim)])
    if args.time_dim:
        cmd_args.extend(['--time-dim', str(args.time_dim)])
    if args.embedding_dim:
        cmd_args.extend(['--embedding-dim', str(args.embedding_dim)])
    if args.hidden_dim:
        cmd_args.extend(['--hidden-dim', str(args.hidden_dim)])
    if args.num_layers:
        cmd_args.extend(['--num-layers', str(args.num_layers)])
    if args.dropout:
        cmd_args.extend(['--dropout', str(args.dropout)])
    if args.lr:
        cmd_args.extend(['--lr', str(args.lr)])
    if args.epochs:
        cmd_args.extend(['--epochs', str(args.epochs)])
    if args.patience:
        cmd_args.extend(['--patience', str(args.patience)])
    if args.seed:
        cmd_args.extend(['--seed', str(args.seed)])
    if args.run_name:
        cmd_args.extend(['--run-name', args.run_name])
    
    cmd_args_str = ' '.join(cmd_args)
    
    # Print configuration
    print("\n" + "="*60)
    print("🎯 Phase 1 Training Launcher")
    print("="*60)
    print(f"Model: {args.model.upper()}")
    
    if args.no_wandb:
        print("W&B: Disabled")
    elif args.offline:
        print("W&B: Offline mode")
    elif args.wandb:
        print("W&B: Online mode")
    else:
        print("W&B: Default (offline)")
    
    if cmd_args:
        print(f"Arguments: {cmd_args_str}")
    
    print("="*60)
    
    # Execute training
    success = True
    
    if args.model == 'tgn' or args.model == 'both':
        cmd = f"python train_tgn_ethereum.py {cmd_args_str}"
        success = run_command(cmd, "Training TGN on Ethereum")
        
        if not success and args.model == 'both':
            print("\n⚠️ TGN training failed, skipping MPTGNN")
            sys.exit(1)
    
    if args.model == 'mptgnn' or args.model == 'both':
        cmd = f"python train_mptgnn_ethereum.py {cmd_args_str}"
        success = run_command(cmd, "Training MPTGNN on Ethereum")
    
    # Summary
    print("\n" + "="*60)
    if success:
        print("✅ Training completed successfully!")
        print("="*60)
        print("\nNext steps:")
        print("  1. Check results in checkpoints/ directory")
        if not args.no_wandb:
            print("  2. View W&B dashboard for detailed metrics")
        print("  3. Compare with baselines:")
        print("     - MLP: 93.99% ROC-AUC")
        print("     - GraphSAGE: 91.31% ROC-AUC")
        print("\nFor more info: python train.py --help")
    else:
        print("❌ Training failed")
        print("="*60)
        sys.exit(1)

if __name__ == '__main__':
    main()
