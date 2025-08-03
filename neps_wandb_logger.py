"""
Alternative WandB logger that's more NEPS-friendly.

This version avoids step ordering issues by letting WandB handle steps automatically
or by using timestamps instead of manual step numbers.
"""

import time
from pathlib import Path
import uuid
import wandb
import yaml
import matplotlib.pyplot as plt
import numpy as np


class NepsWandbLogger:
    """
    NEPS-friendly WandB logger that avoids step ordering issues.
    
    This logger uses one of two strategies:
    1. Let WandB auto-manage steps (no explicit step parameter)
    2. Use timestamp-based steps for guaranteed monotonic ordering
    """

    def __init__(
        self,
        log_dir: str | Path,
        project_name: str = "nlp-classifier-automl",
        run_name: str = None,
        config: dict = None,
        use_auto_step: bool = True,
    ):
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        self.log_dir = log_dir
        self.training_time = time.time()
        self.epoch_time = time.time()
        self.use_auto_step = use_auto_step
        self.start_time = time.time()

        # Create unique run name to avoid conflicts in parallel NEPS execution
        if run_name:
            unique_run_name = f"{run_name}_{uuid.uuid4().hex[:8]}"
        else:
            unique_run_name = f"neps_run_{uuid.uuid4().hex[:8]}"

        # Initialize wandb with settings optimized for NEPS
        try:
            wandb.init(
                project=project_name,
                name=unique_run_name,
                config=config,
                dir=str(log_dir),
                reinit=True,
                settings=wandb.Settings(
                    start_method="thread",
                    _disable_stats=True,  # Disable system stats to reduce overhead
                    _disable_meta=True,   # Disable metadata collection
                )
            )
            self.wandb_available = True
        except Exception as e:
            print(f"Warning: WandB initialization failed: {e}")
            print("Continuing without WandB logging...")
            self.wandb_available = False

    def _get_step(self, original_step=None):
        """Get step number based on strategy."""
        if not self.use_auto_step and original_step is not None:
            # Use timestamp-based step to ensure monotonic ordering
            return int((time.time() - self.start_time) * 1000)  # Milliseconds since start
        return None  # Let WandB auto-manage steps

    def add_data_distribution(self, train_dfs, val_dfs, test_dfs):
        """Log data distribution information."""
        if not self.wandb_available:
            return

        try:
            for task, df in train_dfs.items():          
                counts = df["label"].value_counts().sort_index()
                wandb.log({
                    f"{task}/class_hist": wandb.plot.bar(
                        wandb.Table(data=[[c, n] for c, n in counts.items()],
                                  columns=["class", "count"]),
                        "class", "count",
                        title=f"{task} class distribution"
                    )
                }, step=self._get_step())
            
            sizes = {t: len(df) for t, df in train_dfs.items()}
            wandb.log({
                "tasks/train_size": wandb.plot.bar(
                    wandb.Table(data=[[t, n] for t, n in sizes.items()],
                              columns=["task", "train_samples"]),
                    "task", "train_samples",
                    title="Train samples per task"
                )
            }, step=self._get_step())

            final_sizes = {
                "train": sizes,
                "val": {t: len(val_dfs[t]) for t in val_dfs},
                "test": {t: len(test_dfs[t]) for t in test_dfs},
            }

            split_sizes_table = wandb.Table(
                columns=["split", "task", "n"],
                data=[(s, t, n) for s, d in final_sizes.items() for t, n in d.items()]
            )
            wandb.log({"split_sizes": split_sizes_table}, step=self._get_step())
            
        except Exception as e:
            print(f"Warning: Failed to log data distribution: {e}")

    def step(self, total_loss: float, task, step):
        """Log training step metrics."""
        if not self.wandb_available:
            return

        try:
            wandb.log({
                f"{task}/mean_train_loss": total_loss
            }, step=self._get_step(step))
        except Exception as e:
            print(f"Warning: Failed to log training step: {e}")

    def log_evaluation(self, epoch, task, val_accuracy, val_auc):
        """Log validation metrics."""
        if not self.wandb_available:
            return

        try:
            wandb.log({
                f"{task}/val_acc": val_accuracy, 
                f"{task}/val_auc": val_auc,
                f"{task}/epoch": epoch
            }, step=self._get_step(epoch))
        except Exception as e:
            print(f"Warning: Failed to log evaluation: {e}")

    def epoch_info(self, epoch, mean_val_accuracy, mean_val_auc, mean_train_loss):
        """Log epoch summary metrics."""
        if not self.wandb_available:
            return

        try:
            wandb.log({
                "mean_val_accuracy": mean_val_accuracy, 
                "mean_val_roc_auc": mean_val_auc, 
                "mean_train_loss": mean_train_loss,
                "epoch": epoch
            }, step=self._get_step(epoch))
        except Exception as e:
            print(f"Warning: Failed to log epoch info: {e}")

    def close(self):
        """Close the WandB run."""
        if not self.wandb_available:
            return

        try:
            # Mark run as finished in run_info.yaml if it exists
            run_info_file = self.log_dir / "run_info.yaml"
            if run_info_file.exists():
                try:
                    with run_info_file.open("r") as file:
                        run_info = yaml.safe_load(file)
                    run_info["finished"] = True
                    with run_info_file.open("w") as file:
                        yaml.dump(run_info, file)
                except Exception as e:
                    print(f"Warning: Failed to update run_info.yaml: {e}")

            wandb.finish()
        except Exception as e:
            print(f"Warning: Failed to close WandB: {e}")


def get_logger(
    log_dir: str | Path,
    use_auto_step: bool = True,
    **kwargs,
):
    """
    Creates and returns a NEPS-friendly WandB logger.
    
    Args:
        log_dir: Directory for logging
        use_auto_step: If True, let WandB auto-manage steps. If False, use timestamp-based steps.
        **kwargs: Additional arguments for the logger
    """
    return NepsWandbLogger(log_dir=log_dir, use_auto_step=use_auto_step, **kwargs)
