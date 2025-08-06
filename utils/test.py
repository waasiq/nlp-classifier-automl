import sys
import pandas as pd
from pathlib import Path
from omegaconf import OmegaConf
from hydra import initialize, compose
from run import load_dataset
from hpo import neps_training_wrapper

configs_df = pd.read_csv("top5.csv")  # contains columns: lr, batch_size

results_file = Path("full_training_results.csv")

with initialize(config_path="./configs", version_base="1.3"):
    cfg = compose(config_name="train")
args = OmegaConf.to_object(cfg)

out_dir = Path(args["output_path"])
out_dir.mkdir(parents=True, exist_ok=True) 

dataset_classes, train_dfs, val_dfs, test_dfs, num_classes = load_dataset(
    dataset=args["dataset"],
    data_path=Path(args["data_path"]).absolute(),
    seed=args["seed"],
    val_size=args["val_size"],
    is_mtl=args["ismtl"]
)
for , row in configs_df.iterrows():
    lr = row["lr"]
    batch_size = row["batch_size"]

    # Run training
    func = neps_training_wrapper(
        args, dataset_classes, train_dfs, val_dfs, test_dfs, num_classes, out_dir
    )

    result_dict = func("final", lr=lr, batch_size=batch_size)
    row_data = {
        "lr": lr,
        "batch_size": batch_size,
        **result_dict
    }

    df = pd.DataFrame([row_data])

    # Append to CSV (create if not exists)
    df.to_csv(results_file, mode="a", header=not results_file.exists(), index=False)

    print(f"Saved results for lr={lr}, batch_size={batch_size}")
