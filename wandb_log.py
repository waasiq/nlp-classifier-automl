import time
from pathlib import Path

import torch
import wandb
import yaml


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

        # Initialize wandb
        wandb.init(
            project=project_name,
            name=run_name,
            config=config,
            dir=str(log_dir),
        )

    def step(self, total_loss: float, epoch: int, optimizer: torch.optim.Optimizer):
        """
        Logs training metrics for the current epoch and updates run information.

        Args:
            total_loss (float): The total training loss for the current epoch.
            epoch (int): The current epoch number.
            optimizer (torch.optim.Optimizer): The optimizer used during training.
        """
        # run info yaml
        with (self.training_directory / "run_info.yaml").open("w") as file:
            yaml.dump(
                {
                    "epoch": epoch,
                    "training_loss": total_loss,
                    "training_time": time.time() - self.training_time,
                    "epoch_time": time.time() - self.epoch_time,
                    "finished": False,
                },
                file,
            )

        # wandb logging
        log_dict = {
            "Loss/train": total_loss,
            "epoch": epoch,
            "training_time": time.time() - self.training_time,
            "epoch_time": time.time() - self.epoch_time,
        }

        # Log learning rates for each parameter group
        for i, param_group in enumerate(optimizer.param_groups):
            log_dict[f"Learning Rate/group{i}"] = param_group["lr"]

        wandb.log(log_dict, step=epoch)

    def log_evaluation(self, evaluation: dict, step: int = 0):
        """
        Logs evaluation metrics to wandb.

        This method logs various evaluation metrics including:
        - Validation loss (val_loss)
        - Mean real data accuracy (mean_real_data_accuracy)
        - Training loss (train_loss)
        - Individual dataset accuracies and normalized accuracies for iris, wine, and breast_cancer datasets

        Args:
            evaluation (dict): Dictionary containing evaluation metrics from evaluate_model function.
                Expected keys: val_loss, mean_real_data_accuracy, train_loss, and dataset-specific metrics.
            step (int, optional): The current epoch or step. Defaults to 0.
        """
        # Log validation loss
        if "val_loss" in evaluation:
            wandb.log({"val_loss": evaluation["val_loss"]}, step=step)

        # Log mean real data accuracy
        if "mean_real_data_accuracy" in evaluation:
            wandb.log(
                {"mean_real_data_accuracy": evaluation["mean_real_data_accuracy"]},
                step=step,
            )

        # Log individual dataset metrics
        for dataset_name, metrics in evaluation.items():
            if isinstance(metrics, dict) and "accuracy" in metrics:
                wandb.log(
                    {
                        f"{dataset_name}/accuracy": metrics["accuracy"],
                        f"{dataset_name}/norm_accuracy": metrics["norm_accuracy"],
                    },
                    step=step,
                )

        # Log training loss if available
        if "train_loss" in evaluation:
            wandb.log({"train_loss": evaluation["train_loss"]}, step=step)

        # Log objective value for hyperparameter optimization
        if "val_loss" in evaluation:
            wandb.log({"objective_to_minimize": evaluation["val_loss"]}, step=step)

    def close(self):
        """
        Marks the training run as finished and closes the wandb run.
        """
        # Mark run as finished in run_info.yaml
        with (self.training_directory / "run_info.yaml").open("r") as file:
            run_info = yaml.safe_load(file)
        run_info["finished"] = True
        with (self.training_directory / "run_info.yaml").open("w") as file:
            yaml.dump(run_info, file)

        wandb.finish()


def get_logger(
    log_dir: str | Path,
    **kwargs,
):
    """
    Creates and returns the appropriate logger based on the logger_name parameter.
    Args:
        logger_name (str): The name of the logger to create. Currently supports "wandb".
        training_directory (str | Path): The directory where the logger will save its data.
        **kwargs: Additional keyword arguments to pass to the logger constructor.
    """
    return WandbLogger(log_dir=log_dir, **kwargs)
