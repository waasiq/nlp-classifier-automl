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

import argparse
import logging
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report
import yaml
import sys
from wandb_log import get_logger
import neps
from automl.core import TextAutoML
from automl.datasets import (
    AGNewsDataset,
    AmazonReviewsDataset,
    DBpediaDataset,
    IMDBDataset,
    YelpDataset,
)
from run import main_loop, load_dataset
from hydra import compose, initialize
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)


def neps_training_wrapper(args, dataset_classes, train_dfs, val_dfs, test_dfs, num_classes, out_dir):
    def evaluate_pipeline(pipeline_directory, lr, epochs, batch_size):
        return main_loop(
            train_dfs=train_dfs,
            val_dfs=val_dfs,
            test_dfs=test_dfs,
            num_classes=num_classes,
            dataset_classes=dataset_classes,
            pipeline_directory=pipeline_directory,
            data_fraction=args["data_fraction"],
            output_path=out_dir.absolute(),
            seed=args["seed"],
            approach=args["approach"],
            vocab_size=args["model_config"]["vocab_size"],
            token_length=args["token_length"],
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            weight_decay=args["weight_decay"],
            ffnn_hidden=args["ffnn_hidden_layer_dim"],
            lstm_emb_dim=args["model_config"]["lstm_emb_dim"],
            lstm_hidden_dim=args["model_config"]["lstm_hidden_dim"],
            load_path=Path(args["load_path"]) if args["load_path"] else None,
            model_name=args["model_name"],
        )
    return evaluate_pipeline

def get_args():
    overrides = sys.argv[1:]
    with initialize(config_path="./configs", version_base="1.3"):
        cfg = compose(config_name="train", overrides=overrides)
    cfg = OmegaConf.to_object(cfg)
    assert "root_directory" in cfg["neps"], (
        "neps config must contain 'root_directory' key to store results."
        " Please specify it with `++neps.root_directory=results/<run_name>`."
    )
    return cfg

if __name__ == "__main__":
    conf = get_args()
    out_dir = Path(conf["output_path"])
    out_dir.mkdir(parents=True, exist_ok=True) 
    logging.basicConfig(level=logging.INFO, filename=out_dir / "run.log", 
                        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
    nep_configs = conf.pop("neps")

    dataset_classes, train_dfs, val_dfs, test_dfs, num_classes = load_dataset(
        dataset=conf["dataset"],
        data_path=Path(conf["data_path"]).absolute(),
        seed=conf["seed"],
        val_size= conf["val_size"],
        is_mtl=conf["is_mtl"]
    )

    neps.run(
        evaluate_pipeline= neps_training_wrapper(conf, dataset_classes, train_dfs, val_dfs, test_dfs, num_classes, out_dir),
        **nep_configs
    )
