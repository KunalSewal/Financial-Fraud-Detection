"""
Experiment runner with Weights & Biases integration.

Handles:
- Experiment configuration
- Model training with logging
- Hyperparameter sweeps
- Result tracking
"""

import os
import torch
import wandb
from pathlib import Path
from typing import Dict, Optional, List
import yaml
from datetime import datetime


class ExperimentRunner:
    """
    Unified experiment runner with W&B tracking.
    """
    
    def __init__(
        self,
        project_name: str = "financial-fraud-tgnn",
        entity: Optional[str] = None,
        offline: bool = False
    ):
        """
        Args:
            project_name: W&B project name
            entity: W&B entity (username/team)
            offline: If True, run in offline mode
        """
        self.project_name = project_name
        self.entity = entity
        self.offline = offline
        
        # Determine if W&B is available
        self.use_wandb = not offline
        
    def init_run(
        self,
        config: Dict,
        name: Optional[str] = None,
        tags: Optional[List[str]] = None,
        notes: Optional[str] = None
    ):
        """
        Initialize a new experiment run.
        
        Args:
            config: Experiment configuration dict
            name: Run name (auto-generated if None)
            tags: List of tags for organization
            notes: Description of experiment
        """
        if name is None:
            # Auto-generate name
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = config.get('model', 'unknown')
            dataset_name = config.get('dataset', 'unknown')
            name = f"{model_name}_{dataset_name}_{timestamp}"
        
        if self.use_wandb:
            wandb.init(
                project=self.project_name,
                entity=self.entity,
                name=name,
                config=config,
                tags=tags,
                notes=notes,
                reinit=True
            )
            
            # Store config as artifact
            config_artifact = wandb.Artifact(
                name=f"config_{wandb.run.id}",
                type="config"
            )
            
            # Save config to temp file
            config_path = f"temp_config_{wandb.run.id}.yaml"
            with open(config_path, 'w') as f:
                yaml.dump(config, f)
            
            config_artifact.add_file(config_path)
            wandb.log_artifact(config_artifact)
            
            # Clean up temp file
            os.remove(config_path)
        
        return name
    
    def log_metrics(
        self,
        metrics: Dict,
        step: Optional[int] = None,
        commit: bool = True
    ):
        """
        Log metrics to W&B.
        
        Args:
            metrics: Dictionary of metric_name: value
            step: Training step/epoch
            commit: Whether to commit immediately
        """
        if self.use_wandb:
            wandb.log(metrics, step=step, commit=commit)
    
    def log_model(
        self,
        model_path: str,
        name: str = "model",
        metadata: Optional[Dict] = None
    ):
        """
        Log model checkpoint to W&B.
        
        Args:
            model_path: Path to model checkpoint
            name: Artifact name
            metadata: Additional metadata
        """
        if self.use_wandb:
            artifact = wandb.Artifact(
                name=name,
                type="model",
                metadata=metadata or {}
            )
            artifact.add_file(model_path)
            wandb.log_artifact(artifact)
    
    def log_dataset(
        self,
        dataset_path: str,
        name: str = "dataset",
        metadata: Optional[Dict] = None
    ):
        """
        Log dataset to W&B.
        
        Args:
            dataset_path: Path to dataset
            name: Artifact name
            metadata: Additional metadata
        """
        if self.use_wandb:
            artifact = wandb.Artifact(
                name=name,
                type="dataset",
                metadata=metadata or {}
            )
            
            if os.path.isdir(dataset_path):
                artifact.add_dir(dataset_path)
            else:
                artifact.add_file(dataset_path)
            
            wandb.log_artifact(artifact)
    
    def log_figure(
        self,
        figure,
        name: str,
        step: Optional[int] = None
    ):
        """
        Log matplotlib/plotly figure to W&B.
        
        Args:
            figure: Matplotlib or Plotly figure
            name: Figure name
            step: Training step
        """
        if self.use_wandb:
            wandb.log({name: figure}, step=step)
    
    def log_table(
        self,
        data: List[List],
        columns: List[str],
        name: str = "results_table"
    ):
        """
        Log data table to W&B.
        
        Args:
            data: List of rows
            columns: Column names
            name: Table name
        """
        if self.use_wandb:
            table = wandb.Table(data=data, columns=columns)
            wandb.log({name: table})
    
    def finish(self):
        """Finish the current run."""
        if self.use_wandb:
            wandb.finish()
    
    def watch_model(
        self,
        model: torch.nn.Module,
        log_freq: int = 100,
        log_graph: bool = False
    ):
        """
        Watch model parameters during training.
        
        Args:
            model: PyTorch model
            log_freq: Logging frequency
            log_graph: Whether to log computation graph
        """
        if self.use_wandb:
            wandb.watch(model, log="all", log_freq=log_freq, log_graph=log_graph)


def create_sweep_config(
    method: str = 'bayes',
    metric_name: str = 'val_f1',
    metric_goal: str = 'maximize'
) -> Dict:
    """
    Create hyperparameter sweep configuration.
    
    Args:
        method: 'grid', 'random', or 'bayes'
        metric_name: Metric to optimize
        metric_goal: 'minimize' or 'maximize'
        
    Returns:
        Sweep configuration dictionary
    """
    sweep_config = {
        'method': method,
        'metric': {
            'name': metric_name,
            'goal': metric_goal
        },
        'parameters': {
            'learning_rate': {
                'distribution': 'log_uniform_values',
                'min': 1e-5,
                'max': 1e-2
            },
            'hidden_dim': {
                'values': [64, 128, 256, 512]
            },
            'memory_dim': {
                'values': [64, 128, 256]
            },
            'num_layers': {
                'values': [2, 3, 4]
            },
            'dropout': {
                'distribution': 'uniform',
                'min': 0.0,
                'max': 0.5
            },
            'batch_size': {
                'values': [32, 64, 128, 256]
            }
        }
    }
    
    return sweep_config


def run_sweep(
    sweep_config: Dict,
    train_function,
    count: int = 20,
    project_name: str = "financial-fraud-tgnn"
):
    """
    Run hyperparameter sweep.
    
    Args:
        sweep_config: Sweep configuration from create_sweep_config()
        train_function: Training function to call
        count: Number of runs
        project_name: W&B project name
    """
    sweep_id = wandb.sweep(sweep_config, project=project_name)
    wandb.agent(sweep_id, function=train_function, count=count)


# Example usage functions

def log_training_progress(
    runner: ExperimentRunner,
    epoch: int,
    train_loss: float,
    train_metrics: Dict,
    val_loss: float,
    val_metrics: Dict
):
    """
    Helper to log training progress.
    
    Args:
        runner: ExperimentRunner instance
        epoch: Current epoch
        train_loss: Training loss
        train_metrics: Training metrics dict
        val_loss: Validation loss
        val_metrics: Validation metrics dict
    """
    metrics = {
        'epoch': epoch,
        'train/loss': train_loss,
        'val/loss': val_loss
    }
    
    # Add training metrics
    for key, value in train_metrics.items():
        metrics[f'train/{key}'] = value
    
    # Add validation metrics
    for key, value in val_metrics.items():
        metrics[f'val/{key}'] = value
    
    runner.log_metrics(metrics, step=epoch)


def log_final_results(
    runner: ExperimentRunner,
    test_metrics: Dict,
    model_path: str,
    confusion_matrix=None,
    roc_curve=None
):
    """
    Log final test results.
    
    Args:
        runner: ExperimentRunner instance
        test_metrics: Test metrics dict
        model_path: Path to best model
        confusion_matrix: Matplotlib figure (optional)
        roc_curve: Matplotlib figure (optional)
    """
    # Log test metrics
    test_log = {f'test/{k}': v for k, v in test_metrics.items()}
    runner.log_metrics(test_log)
    
    # Log model
    runner.log_model(
        model_path,
        name="best_model",
        metadata=test_metrics
    )
    
    # Log figures
    if confusion_matrix is not None:
        runner.log_figure(confusion_matrix, "confusion_matrix")
    
    if roc_curve is not None:
        runner.log_figure(roc_curve, "roc_curve")
    
    # Create results table
    results_data = [[k, v] for k, v in test_metrics.items()]
    runner.log_table(
        results_data,
        columns=["Metric", "Value"],
        name="test_results"
    )
