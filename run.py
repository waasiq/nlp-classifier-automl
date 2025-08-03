"""A STARTER KIT SCRIPT for SS25 AutoML Exam --- Modality III: Text

You are not expected to follow this script or be constrained to it.

For a test run:
1) Download datasets (see, README) at chosen path
2) Run the script: 
```
python run.py \
    --data-path <path-to-downloaded-data> \
    --dataset amazon \
    --epochs 1
```

"""
from __future__ import annotations
import sys
import math
import logging
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report
import yaml
import pandas as pd
from wandb_log import get_logger
from automl.core import TextAutoML, custom_multiclass_roc_auc
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
from automl.datasets import (
    AGNewsDataset,
    AmazonReviewsDataset,
    DBpediaDataset,
    IMDBDataset,
    YelpDataset,
)
from hydra import compose, initialize
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)

FINAL_TEST_DATASET= "yelp"

# function load the dataset and create dfs
def load_dataset(dataset: str, data_path: Path, val_size: float, seed: int, is_mtl: bool = False):
    match dataset:
        case "ag_news":
            dataset_class = AGNewsDataset
        case "imdb":
            dataset_class = IMDBDataset
        case "amazon":
            dataset_class = AmazonReviewsDataset
        case "dbpedia":
            dataset_class = DBpediaDataset
        case "yelp":
            dataset_class = YelpDataset
        case _:
            raise ValueError(f"Invalid dataset: {dataset}")
    
    dataset_classes = {dataset: dataset_class}

    if is_mtl:
        dataset_classes = {
            "ag_news": AGNewsDataset,
            "imdb": IMDBDataset,
            "amazon": AmazonReviewsDataset,
            "dbpedia": DBpediaDataset,
            "yelp": YelpDataset,
        }
        dataset = "mtl"

    
    data_path = Path(data_path) if isinstance(data_path, str) else data_path
    data_infos = {dataset: dataset_class(data_path).create_dataloaders(val_size=val_size, random_state=seed) for dataset, dataset_class in dataset_classes.items()}
    train_dfs = {dataset: data_info['train_df'] for dataset, data_info in data_infos.items()}
    val_dfs = {dataset: data_info.get('val_df', None) for dataset, data_info in data_infos.items()}
    test_dfs = {dataset: data_info['test_df'] for dataset, data_info in data_infos.items()}
    num_classes = {dataset: data_info['num_classes'] for dataset, data_info in data_infos.items()}
    return dataset_classes, train_dfs, val_dfs, test_dfs, num_classes

def main_loop(
        train_dfs: dict[str, pd.DataFrame],
        val_dfs: dict[str, pd.DataFrame],
        test_dfs: dict[str, pd.DataFrame],
        num_classes: dict[str, int],
        dataset_classes: dict[str, str],
        data_fraction: float,
        output_path: Path,
        seed: int,
        approach: str,
        vocab_size: int = 10000,
        token_length: int = 128,
        epochs: int = 5,
        batch_size: int = 32,
        lr: float = 0.0001,
        weight_decay: float = 0.01,
        ffnn_hidden: int = 128,
        lstm_emb_dim: int = 128,
        lstm_hidden_dim: int = 128,
        fraction_layers_to_finetune: float = 1.0,
        load_path: Path = None,
        pipeline_directory: Path | None = None,
    ) -> None:
    #create run_name with random 6 characters
    run_name = f"{''.join(dataset_classes.keys())}_config_{pipeline_directory}_{np.random.randint(100000, 999999)}"
    plotter = get_logger(log_dir=pipeline_directory, run_name=run_name)

    logger.info("Fitting Text AutoML")

    np.random.seed(seed)

    n_class_samples = round(sum(len(v) for v in train_dfs.values()) * data_fraction / len(train_dfs))
    for dataset, train_df in train_dfs.items():
        _subsample = np.random.choice(
            list(range(len(train_df))),
            size=n_class_samples,
            replace=len(train_df) < n_class_samples,
        )
        train_dfs[dataset] = train_df.iloc[_subsample]

    logger.info(
        [f"Train size: {len(train_dfs[dataset])}, Validation size: {len(val_dfs[dataset])}, Test size: {len(test_dfs[dataset])}" for dataset in dataset_classes.keys()]
    )
    # plotter.add_data_distribution(train_dfs, val_dfs, test_dfs)
    logger.info(f"Number of classes: {num_classes}")

    # Initialize the TextAutoML instance with the best parameters
    automl = TextAutoML(
        seed=seed,
        approach=approach,
        vocab_size=vocab_size,
        token_length=token_length,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        ffnn_hidden=ffnn_hidden,
        lstm_emb_dim=lstm_emb_dim,
        lstm_hidden_dim=lstm_hidden_dim,
        fraction_layers_to_finetune=fraction_layers_to_finetune,
        plotter=plotter
    )

    # Fit the AutoML model on the training and validation datasets
    user_result = automl.fit(
        train_dfs,
        val_dfs,
        num_classes=num_classes,
        load_path=load_path,
        save_path=output_path,
    )
    logger.info("Training complete")

    # Predict on the test set
    val_err = user_result.get("objective_to_minimize", math.inf)
    test_roc_auc_accumulated = 0.0
    test_n = 0
    for task, test_df in test_dfs.items():
        if task == FINAL_TEST_DATASET:
            continue
        (test_preds, test_labels), test_probs = automl.predict(test_df, task)
        n_classes = num_classes[task]
        y_true_binarized = label_binarize(test_labels, classes=np.arange(n_classes))
        
        if n_classes == 2:
            auc =  roc_auc_score(test_labels, test_probs[:, 1])
        else:
        # Multiclass classification - use one-vs-rest approach
            auc = custom_multiclass_roc_auc(y_true_binarized, test_probs)
        test_roc_auc_accumulated += auc
        test_n += 1
        user_result["info_dict"][f"{task}_roc_auc_score"] = float(auc)
        output_path = Path(output_path)
        # Write the predictions of X_test to disk
        logger.info("Writing predictions to disk")
        task_output_path = output_path / task
        task_output_path.mkdir(parents=True, exist_ok=True)
        with (task_output_path / "score.yaml").open("w") as f:
            yaml.safe_dump({"val_err": float(val_err)}, f)
        logger.info(f"Saved validataion score at {task_output_path / 'score.yaml'}")
        with (task_output_path / "test_preds.npy").open("wb") as f:
            np.save(f, test_preds)
        logger.info(f"Saved tet prediction at {task_output_path / 'test_preds.npy'}")

    # Average the ROC AUC scores across all tasks
    average_roc_auc = test_roc_auc_accumulated / test_n
    user_result["info_dict"]["test_mean_roc_auc"] = float(average_roc_auc)
    # In case of running on the final exam data, also add the predictions.npy
    # to the correct location for auto evaluation.
    if FINAL_TEST_DATASET in dataset_classes.keys(): 
        test_output_path = output_path / "predictions.npy"
        test_output_path.parent.mkdir(parents=True, exist_ok=True)
        with test_output_path.open("wb") as f:
            np.save(f, test_preds)

    # Check if test_labels has missing data
    if not np.isnan(test_labels).any():
        acc = accuracy_score(test_labels, test_preds)
        logger.info(f"Accuracy on test set: {acc}")
        with (output_path / "score.yaml").open("a+") as f:
            yaml.safe_dump({"test_err": float(1-acc)}, f)
        
        # Log detailed classification report for better insight
        logger.info("Classification Report:")
        logger.info(f"\n{classification_report(test_labels, test_preds)}")
    else:
        # This is the setting for the exam dataset, you will not have access to the labels
        logger.info(f"No test labels available for dataset '{dataset_classes.keys()}'")
    plotter.close()
    return user_result

if __name__ == "__main__":
    overrides = sys.argv[1:]
    with initialize(config_path="./configs", version_base="1.3"):
        cfg = compose(config_name="train", overrides=overrides)
    args = OmegaConf.to_object(cfg)
    out_dir = Path(args["output_path"])
    out_dir.mkdir(parents=True, exist_ok=True) 
    logging.basicConfig(level=logging.INFO, filename=out_dir / "run.log", 
                        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
    # load datasets:
    dataset_classes, train_dfs, val_dfs, test_dfs, num_classes = load_dataset(
        dataset=args["dataset"],
        data_path=Path(args["data_path"]).absolute(),
        seed=args["seed"],
        val_size= args["val_size"],
        is_mtl=args["is_mtl"]
    )
    main_loop(
        train_dfs=train_dfs,
        val_dfs=val_dfs,
        test_dfs=test_dfs,
        num_classes=num_classes,
        dataset_classes=dataset_classes,
        data_fraction=args["data_fraction"],
        output_path=out_dir.absolute(),
        seed=args["seed"],
        approach=args["approach"],
        vocab_size=args["model_config"]["vocab_size"],
        token_length=args["token_length"],
        epochs=args["epochs"],
        batch_size=args["batch_size"],
        lr=args["lr"],
        weight_decay=args["weight_decay"],
        ffnn_hidden=args["ffnn_hidden_layer_dim"],
        lstm_emb_dim=args["model_config"]["lstm_emb_dim"],
        lstm_hidden_dim=args["model_config"]["lstm_hidden_dim"],
        load_path=Path(args["load_path"]) if args["load_path"] else None,
    )
