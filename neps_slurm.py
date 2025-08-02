import os, subprocess, time, logging
from pathlib import Path
import neps
import argparse
import logging

logger = logging.getLogger(__name__)
# ---- 1. the Slurm-aware objective ------------------------------------------

def evaluate_pipeline_wrapper(args):
    def evaluate_pipeline_slurm(
        pipeline_directory: Path,          # NePS gives this automatically:contentReference[oaicite:0]{index=0}
        batch_size, lr, weight_decay,
        lstm_emb_dim, lstm_hidden_dim,
    ):
        # ---- create a one-off bash script ---------------------------------------
        bash = f"""#!/bin/bash
    #SBATCH --time=0-02:00
    #SBATCH --partition=dllabdlc_gpu-rtx2080
    #SBATCH --gres=gpu:1
    #SBATCH --cpus-per-task=4
    #SBATCH --mem=2G
    #SBATCH --job-name=neps_trial
    #SBATCH --output={pipeline_directory}/%j.out
    #SBATCH --error={pipeline_directory}/%j.err

    source ~/.bashrc
    conda activate nlp

    python - << 'PY'
from pathlib import Path
from run import main_loop
import json, os

val = main_loop(
    dataset       = '{args.dataset}',
    output_path   = Path('{args.output_path}').absolute(),
    data_path     = Path('{args.data_path}').absolute(),
    seed          = {args.seed},
    approach      = '{args.approach}',
    vocab_size    = {args.vocab_size},
    token_length  = {args.token_length},
    epochs        = {args.epochs},
    batch_size    = {batch_size},
    lr            = {lr},
    weight_decay  = {weight_decay},
    ffnn_hidden   = {args.ffnn_hidden_layer_dim},
    lstm_emb_dim  = {lstm_emb_dim},
    lstm_hidden_dim={lstm_hidden_dim},
    data_fraction = {args.data_fraction},
    load_path     = '{str(args.load_path) if args.load_path else ""}' or None,
)
(Path('{pipeline_directory}') / 'val_error.txt').write_text(str(val))
PY
    """

        script_file = pipeline_directory / "run.sh"
        script_file.write_text(bash)

        # ---- submit and get the Slurm job-ID ------------------------------------
        job_id = subprocess.check_output(["sbatch", script_file], text=True) \
                            .strip().split()[-1]
        logging.info("Submitted Slurm job %s", job_id)

        # ---- wait until the file appears ----------------------------------------
        val_path = pipeline_directory / "val_error.txt"
        while not val_path.exists():
            time.sleep(30)

        return float(val_path.read_text())
    return evaluate_pipeline_slurm

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
        choices=["lstm", "transformer"],
        help=(
            "The approach to use for the AutoML system. "
            "Options are 'lstm', or 'transformer'."
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

    # logger.info(f"Running text dataset {args.dataset}\n{args}")

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
        evaluate_pipeline=evaluate_pipeline_wrapper(args),
        pipeline_space={
            "batch_size":      neps.Categorical([16, 32, 64]),
            "lr":              neps.Float(1e-5, 1e-2, log=True),
            "weight_decay":    neps.Categorical([0.0, 0.1]),
            "lstm_emb_dim":    neps.Integer(32, 512),
            "lstm_hidden_dim": neps.Integer(32, 512),
        },
        max_evaluations_total=100,
        optimizer="bayesian_optimization",
    )
