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
import neps
from automl.core import TextAutoML
from automl.datasets import (
    AGNewsDataset,
    AmazonReviewsDataset,
    DBpediaDataset,
    IMDBDataset,
    YelpDataset,
)
from run import main_loop, parse_arguments
from hydra import compose, initialize
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)


def neps_training_wrapper(args):
    def evaluate_pipeline(lr, data_fraction):
        return main_loop(
            dataset=args["dataset"],
            output_path=Path(args["output_path"]).absolute(),
            data_path=Path(args["data_path"]).absolute(),
            seed=args["seed"],
            approach=args["approach"],
            vocab_size=args["model_config"]["vocab_size"],
            token_length=args["token_length"],
            epochs=args["epochs"],
            batch_size=args["batch_size"],
            lr=lr,
            weight_decay=args["weight_decay"],
            ffnn_hidden=args["ffnn_hidden_layer_dim"],
            lstm_emb_dim=args["model_config"]["lstm_emb_dim"],
            lstm_hidden_dim=args["model_config"]["lstm_hidden_dim"],
            data_fraction=data_fraction,
            load_path=Path(args["load_path"]) if args["load_path"] else None,
            is_mtl=args["is_mtl"]
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
    # parser = argparse.ArgumentParser()
    # args = parse_arguments(parser)
    conf = get_args()
    out_dir = Path(conf["output_path"])
    out_dir.mkdir(parents=True, exist_ok=True) 
    logging.basicConfig(level=logging.INFO, filename=out_dir / "run.log", 
                        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
    nep_configs = conf.pop("neps")
    neps.run(
        evaluate_pipeline= neps_training_wrapper(conf),
        **nep_configs
    )
