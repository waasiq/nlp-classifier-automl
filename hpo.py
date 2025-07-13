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
import neps
from automl.core import TextAutoML
from automl.datasets import (
    AGNewsDataset,
    AmazonReviewsDataset,
    DBpediaDataset,
    IMDBDataset,
)
from run import main_loop
logger = logging.getLogger(__name__)

FINAL_TEST_DATASET=...  # TBA later


def evaluate_pipeline(batch_size, lr, weight_decay, lstm_emb_dim, lstm_hidden_dim):
    return main_loop(
        dataset=args.dataset,
        output_path=Path(args.output_path).absolute(),
        data_path=Path(args.data_path).absolute(),
        seed=args.seed,
        approach=args.approach,
        vocab_size=args.vocab_size,
        token_length=args.token_length,
        epochs=args.epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        ffnn_hidden=args.ffnn_hidden_layer_dim,
        lstm_emb_dim=lstm_emb_dim,
        lstm_hidden_dim=lstm_hidden_dim,
        data_fraction=args.data_fraction,
        load_path=Path(args.load_path) if args.load_path else None,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="The name of the dataset to run on.",
        choices=["ag_news", "imdb", "amazon", "dbpedia",]
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help=(
            "The path to save the predictions to."
            " By default this will just save to the cwd as `./results`."
        )
    )
    parser.add_argument(
        "--load-path",
        type=Path,
        default=None,
        help="The path to resume checkpoint from."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help=(
            "The path to laod the data from."
            " By default this will look up cwd for `./.data/`."
        )
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "Random seed for reproducibility if you are using any randomness,"
            " i.e. torch, numpy, pandas, sklearn, etc."
        )
    )
    parser.add_argument(
        "--approach",
        type=str,
        default="transformer",
        choices=["tfidf", "lstm", "transformer"],
        help=(
            "The approach to use for the AutoML system. "
            "Options are 'tfidf', 'lstm', or 'transformer'."
        )
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=1000,
        help="The size of the vocabulary to use for the text dataset."
    )
    parser.add_argument(
        "--token-length",
        type=int,
        default=128,
        help="The maximum length of tokens to use for the text dataset."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="The number of epochs to train the model for."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="The batch size to use for training and evaluation."
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help="The learning rate to use for the optimizer."
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="The weight decay to use for the optimizer."
    )

    parser.add_argument(
        "--lstm-emb-dim",
        type=int,
        default=64,
        help="The embedding dimension to use for the LSTM model."
    )

    parser.add_argument(
        "--lstm-hidden-dim",
        type=int,
        default=64,
        help="The hidden size to use for the LSTM model."
    )

    parser.add_argument(
        "--ffnn-hidden-layer-dim",
        type=int,
        default=64,
        help="The hidden size to use for the model."
    )

    parser.add_argument(
        "--data-fraction",
        type=float,
        default=1,
        help="Subsampling of training set, in fraction (0, 1]."
    )
    args = parser.parse_args()

    logger.info(f"Running text dataset {args.dataset}\n{args}")

    if args.output_path is None:
        args.output_path =  (
            Path.cwd().absolute() / 
            "results" / 
            f"dataset={args.dataset}" / 
            f"seed={args.seed}"
        )
    if args.data_path is None:
        args.data_path = Path.cwd().absolute() / ".data"

    args.output_path = Path(args.output_path).absolute()
    args.output_path.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO, filename=args.output_path / "run.log", 
                        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
    neps.run(
        evaluate_pipeline= evaluate_pipeline,
        pipeline_space={
            # "vocab_size": neps.Integer(lower=1000, upper=100000),
            # "token_length": neps.Categorical(choices=[128, 256, 512]),
            # "epochs": neps.Integer(lower=1, upper=10),
            "batch_size": neps.Categorical(choices=[16, 32, 64]),
            "lr": neps.Float(lower=1e-5, upper=1e-2, log=True),
            "weight_decay": neps.Categorical(choices=[0.0, 0.1]),
            # "ffnn_hidden": neps.Integer(lower=32, upper=512),
            "lstm_emb_dim": neps.Integer(lower=32, upper=512),
            "lstm_hidden_dim": neps.Integer(lower=32, upper=512),
        },
        max_evaluations_total=100,
        optimizer="bayesian_optimization"
    )