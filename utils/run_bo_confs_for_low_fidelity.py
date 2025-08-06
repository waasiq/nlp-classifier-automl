from pathlib import Path
import yaml
import copy
from hydra import compose, initialize
from omegaconf import OmegaConf
from run import load_dataset
import sys
from hpo import neps_training_wrapper
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import csv

def extract_config_suffix(path: Path) -> int:
    if path.name.startswith("config_"):
        return path.name[7:]
    return -1

class CorrelationEvaluation:
    def __init__(self, result_dir: str):
        self.current_confs = []
        self.next_confs = []
        self.get_configs(result_dir=result_dir)
    
    def get_configs(self, result_dir: str) -> list[dict]:
        result_dir = Path(result_dir)
        configs_path = result_dir / "configs"
        if not configs_path.is_dir():
            raise FileNotFoundError(f"Expected 'configs' inside {result_dir}")
        result = {}
        for cfg_dir in sorted(configs_path.glob("config_*"), key=extract_config_suffix):
            config_id = extract_config_suffix(cfg_dir)
            cfg_path = cfg_dir / "config.yaml"
            report_path = cfg_dir / "report.yaml"

            with cfg_path.open("r", encoding="utf-8") as fh:
                config_data = yaml.safe_load(fh)

            if not report_path.exists():
                print(f"skipping config {config_id}, beacause of not existing report")
                continue
            with report_path.open("r", encoding="utf-8") as fh:
                report = yaml.safe_load(fh)
                if report["reported_as"] != "success":
                    print(f"skipping config {config_id}, beacause of not successful report")
                    continue
            result = {
                "config_id": config_id,
                "fidelity_id": '1',
                "loss": report.get("objective_to_minimize"),
                "test_roc_auc": report["extra"]["test_mean_roc_auc"],
                "conf": config_data,
            }
            self.current_confs.append(result)
        print(self.current_confs)
        
    def generate_lower_fidelity(self):
        overrides = sys.argv[1:]
        with initialize(config_path="./configs", version_base="1.3"):
            cfg = compose(config_name="train", overrides=overrides)
        args = OmegaConf.to_object(cfg)
        out_dir = Path(args["output_path"])
        out_dir.mkdir(parents=True, exist_ok=True) 
        # load datasets:
        dataset_classes, train_dfs, val_dfs, test_dfs, num_classes = load_dataset(
            dataset=args["dataset"],
            data_path=Path(args["data_path"]).absolute(),
            seed=args["seed"],
            val_size= args["val_size"],
            is_mtl=args["is_mtl"]
        )   
        func = neps_training_wrapper(args, dataset_classes, train_dfs, val_dfs, test_dfs, num_classes, out_dir)
    
        for conf in self.current_confs:
            if int(conf["config_id"]) < 6:
                continue
            print(f"evaluate lower fidelity for config {conf["config_id"]}")
            lr = conf["conf"]["lr"]
            bs = conf["conf"]["batch_size"]
            res = func("bo-manual", lr, epochs=2, batch_size=bs)
            result = {
                "config_id": conf["config_id"],
                "fidelity_id": '0',
                "loss": res["objective_to_minimize"],
                "test_roc_auc": res["info_dict"]["test_mean_roc_auc"],
                "conf": {"lr": lr, "batch_size": bs, "epochs": 2},
            }
            print(result)
            self.next_confs.append(result)
            with open("low_fid_res1.csv", mode="a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(result.values())

    def calculate_spearman(self):
        current_dict = {c["config_id"]: c["loss"] for c in self.current_confs}
        next_dict = {n["config_id"]: n["loss"] for n in self.next_confs}
        
        common_ids = set(current_dict.keys()) & set(next_dict.keys())
        
        if len(common_ids) < 3:
            print("Not enough paired configs to calculate correlation.")
            return
        
        losses_current = [current_dict[cid] for cid in common_ids]
        losses_next = [next_dict[cid] for cid in common_ids]
        
        corr, p_val = spearmanr(losses_current, losses_next)
        print(f"Spearman Correlation: {corr:.4f}, P-value: {p_val:.4f}")

        # --- Plot ---
        plt.figure(figsize=(8, 6))
        plt.scatter(losses_next, losses_current, alpha=0.7, edgecolors='k')
        plt.xlabel("Loss at Low Fidelity (e.g., epochs=2)")
        plt.ylabel("Loss at High Fidelity (e.g., epochs=full)")
        plt.title(f"Spearman Correlation = {corr:.2f}, p = {p_val:.2f}")
        
        lims = [min(min(losses_next), min(losses_current)), max(max(losses_next), max(losses_current))]
        plt.plot(lims, lims, 'r--', label='y = x')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig("fidelity_spearman_plot.png")
        plt.show()

evaluation = CorrelationEvaluation(result_dir="neps_results/random_search2")
evaluation.generate_lower_fidelity()
evaluation.calculate_spearman()
