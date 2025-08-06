import optuna
import torch
import pandas as pd
import random
from itertools import chain

# --- Imports from your project ---
# Make sure your project structure allows these imports
from automl.core import TextAutoML
from automl.models import BertMultiTaskClassifier
from transformers import DistilBertModel

# --- 1. Configuration ---
N_TRIALS = 30
EPOCHS = 3
DATA_SUBSET_FRACTION = 0.20  # Use 20% of data for each trial
BASELINE_ACCURACY = 0.55 

# Define the tasks and their properties
TASKS = {
    "dbpedia": {"num_classes": 14, "num_samples": 560000},
    "ag_news": {"num_classes": 4, "num_samples": 120000},
    "amazon": {"num_classes": 5, "num_samples": 3600000},
    "imdb": {"num_classes": 2, "num_samples": 25000},
    "yelp": {"num_classes": 5, "num_samples": 560000},
}

# --- 2. Mock Data Generation ---
# This function simulates your dataframes. Replace with your actual data loading.
def get_mock_dataframes():
    train_dfs = {}
    val_dfs = {}
    for task_name, config in TASKS.items():
        # Create smaller mock dataframes for quick testing
        n_mock_samples = 1000
        texts = [f"This is sample text for {task_name} item #{i}" for i in range(n_mock_samples)]
        labels = [random.randint(0, config["num_classes"] - 1) for _ in range(n_mock_samples)]
        df = pd.DataFrame({"text": texts, "label": labels})
        train_dfs[task_name] = df
        val_dfs[task_name] = df # Using same data for validation for simplicity
    return train_dfs, val_dfs

# --- 3. Optuna Objective Function ---
def objective(trial: optuna.Trial):
    # We instantiate AutoML class to use its utility functions
    automl_instance = TextAutoML(epochs=EPOCHS)
    
    # --- Suggest Hyperparameters ---
    params = {
        "lr_backbone": trial.suggest_categorical("lr_backbone", [5e-5, 4e-5, 3e-5, 2e-5]),
        "lr_head": trial.suggest_float("lr_head", 1e-4, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32]),
        "num_bert_layers_to_unfreeze": trial.suggest_int("num_bert_layers_to_unfreeze", 1, 2),
        "common_dropout": trial.suggest_float("common_dropout", 0.1, 0.5),
        "head_dropout": trial.suggest_float("head_dropout", 0.1, 0.5),
    }

    # --- Prepare DataLoaders with Subsampling ---
    train_dfs, val_dfs = get_mock_dataframes()
    train_loaders, val_loaders = automl_instance.prepare_dataloaders(
        train_dfs, val_dfs, params["batch_size"], DATA_SUBSET_FRACTION
    )

    # --- Create Model and Optimizer ---
    bert_backbone = DistilBertModel.from_pretrained('distilbert-base-uncased')
    model = BertMultiTaskClassifier(
        bert_backbone=bert_backbone,
        task_num_classes={name: conf["num_classes"] for name, conf in TASKS.items()},
        num_bert_layers_to_unfreeze=params["num_bert_layers_to_unfreeze"],
        common_dropout_prob=params["common_dropout"],
        head_dropout_prob=params["head_dropout"],
    ).to(automl_instance.device)

    head_params = chain(model.shared_common_backbone.parameters(), model.classification_heads.parameters())
    optimizer = torch.optim.AdamW([
        {"params": model.bert.parameters(), "lr": params["lr_backbone"]},
        {"params": head_params, "lr": params["lr_head"]}
    ])

    # --- Train and Evaluate using the modified _train_loop ---
    avg_val_accuracy = automl_instance._train_loop(
        model, optimizer, train_loaders, val_loaders, trial, BASELINE_ACCURACY
    )

    return avg_val_accuracy

# --- 4. Main Execution Block ---
if __name__ == "__main__":
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2)
    study = optuna.create_study(direction="maximize", pruner=pruner)
    
    print(f"Starting HPO for {N_TRIALS} trials...")
    study.optimize(objective, n_trials=N_TRIALS)
    
    print("\n--- HPO Finished ---")
    print(f"Best trial avg accuracy: {study.best_value:.4f}")
    print("Best hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    # --- Visualization ---
    if optuna.visualization.is_available():
        print("\nGenerating visualizations... Close the plots to exit.")
        fig1 = optuna.visualization.plot_optimization_history(study)
        fig1.show()
        
        fig2 = optuna.visualization.plot_param_importances(study)
        fig2.show()