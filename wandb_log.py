import time
from pathlib import Path

import torch
import wandb
import yaml
import matplotlib.pyplot as plt

import numpy as np

class WandbLogger:
    """
    Handles logging of training metrics to Weights & Biases.

    This class logs training metrics to wandb and saves run information to a YAML file.
    It ensures proper tracking of training progress and allows exporting logs for further analysis.

    Args:
        training_directory (str | Path): Path to the directory where run information will be stored.
        project_name (str, optional): Name of the wandb project. Defaults to "hpo4tabpfn".
        run_name (str, optional): Name of the wandb run. Defaults to None.
        config (dict, optional): Configuration dictionary to log to wandb. Defaults to None.
    """

    def __init__(
        self,
        log_dir: str | Path,
        project_name: str = "nlp-classifier-automl",
        run_name: str = None,
        config: dict = None,
    ):
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        self.log_dir = log_dir
        self.training_time = time.time()
        self.epoch_time = time.time()
        self._step_counter = 0  # Add monotonic step counter for NEPS
        self._step_counter = 0  # Internal step counter to ensure monotonic steps

        # Initialize wandb with unique run ID to avoid conflicts in NEPS
        import uuid
        unique_run_id = f"{run_name}_{uuid.uuid4().hex[:8]}" if run_name else None
        
        # Initialize wandb
        wandb.init(
            project=project_name,
            name=unique_run_id,
            config=config,
            dir=str(log_dir),
            reinit=True,
            resume=False,
        )

    def add_data_distribution(self, train_dfs, val_dfs, test_dfs):
        # Use internal step counter for data distribution logging
        for task, df in train_dfs.items():          
            counts = df["label"].value_counts().sort_index()
            wandb.log({f"{task}/class_hist": wandb.plot.bar(
                wandb.Table(data=[[c, n] for c, n in counts.items()],
                            columns=["class", "count"]),
                "class", "count",
                title=f"{task} class distribution")}, step=self._get_next_step())
        
        sizes = {t: len(df) for t, df in train_dfs.items()}
        wandb.log({"tasks/train_size": wandb.plot.bar(
            wandb.Table(data=[[t, n] for t, n in sizes.items()],
                        columns=["task", "train_samples"]),
            "task", "train_samples",
            title="Train samples per task")}, step=self._get_next_step())

        final_sizes = {
            "train": sizes,
            "val":   {t: len(val_dfs[t])  for t in val_dfs},
            "test":  {t: len(test_dfs[t]) for t in test_dfs},
        }

        split_sizes_table = wandb.Table(
            columns=["split", "task", "n"],
            data=[(s, t, n) for s, d in final_sizes.items() for t, n in d.items()]
        )
        wandb.log({"split_sizes": split_sizes_table}, step=self._get_next_step())


    def _get_next_step(self):
        """Get the next monotonic step number."""
        self._step_counter += 1
        return self._step_counter

    def step(self, total_loss: float, task, step):
        """
        Logs training metrics for the current epoch and updates run information.

        Args:
            total_loss (float): The total training loss for the current epoch.
            epoch (int): The current epoch number.
            optimizer (torch.optim.Optimizer): The optimizer used during training.
        """
        print('hereeee')
        # Use internal step counter instead of provided step to ensure monotonic ordering
        monotonic_step = self._get_next_step()
        wandb.log({f"{task}/mean_train_loss": total_loss}, step=monotonic_step)

    def log_evaluation(self, epoch, task, val_accuracy, val_auc):
        # Use internal step counter instead of epoch to ensure monotonic ordering
        monotonic_step = self._get_next_step()
        wandb.log({
            f"{task}/val_acc": val_accuracy, 
            f"{task}/val_auc": val_auc,
            f"{task}/epoch": epoch  # Log original epoch as a metric instead
        }, step=monotonic_step)

    def epoch_info(self, epoch, mean_val_accuracy, mean_val_auc, mean_train_loss):
        # Use internal step counter instead of epoch to ensure monotonic ordering
        monotonic_step = self._get_next_step()
        wandb.log({
            "mean_val_accuracy": mean_val_accuracy, 
            "mean_val_roc_auc": mean_val_auc, 
            "mean_train_loss": mean_train_loss,
            "epoch": epoch  # Log original epoch as a metric instead
        }, step=monotonic_step)

    def close(self):
        """
        Marks the training run as finished and closes the wandb run.
        """
        wandb.finish()


def get_logger(
    log_dir: str | Path,
    **kwargs,
):
    """
    Creates and returns the appropriate logger based on the logger_name parameter.
    Args:
        logger_name (str): The name of the logger to create. Currently supports "wandb".
        **kwargs: Additional keyword arguments to pass to the logger constructor.
    """
    return WandbLogger(log_dir=log_dir, **kwargs)
