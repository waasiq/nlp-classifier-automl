from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict

import yaml
import numpy as np
import matplotlib.pyplot as plt
import heapq
import csv
from itertools import cycle
from matplotlib.axes import Axes
import wandb


api = wandb.Api()


def extract_config_suffix(path: Path) -> int:
    if path.name.startswith("config_"):
        return path.name[7:]
    return -1

_marker_cycle = cycle([".", ">", ","])

def get_plot_marker():
    return next(_marker_cycle)

class HPOVisualization:
    """Utility to parse TabPFN HPO results and extract necessary data."""

    def __init__(self, result_dir: str | Path, method: str = "lstm"):
        self.result_dir = Path(result_dir)
        self.method = method
        self.val_losses: Dict[int, float] = {}
        self.reports: Dict[int, dict] = {}
        self.runs: Dict[int, dict] = {}
        self.fidelities = set()
        self._load_reports()

    def _load_reports(self) -> None:
        configs_path = self.result_dir / "configs"
        if not configs_path.is_dir():
            raise FileNotFoundError(f"Expected 'configs' inside {self.result_dir}")

        for cfg_dir in sorted(configs_path.glob("config_*"), key=extract_config_suffix):
            config_id = extract_config_suffix(cfg_dir)
            cfg_path = cfg_dir / "config.yaml"
            report_path = cfg_dir / "report.yaml"
            
            
            if not report_path.is_file():
                print(f"skipping config {config_id}, beacause of not existing report")
                continue

            with report_path.open("r", encoding="utf-8") as fh:
                report = yaml.safe_load(fh)
                if report["reported_as"] != "success":
                    print(f"skipping config {config_id}, beacause of not successful report")
                    continue
            wandb_dir = cfg_dir / "wandb"
            for run in wandb_dir.glob("run-*"):
                run_name = str(run).split("-")[-1]
                self.runs[config_id] = run_name
                break

            with cfg_path.open("r", encoding="utf-8") as fh:
                config = yaml.safe_load(fh)

            fidelity_id = 0
            trial_id = config_id
            if "_" in config_id:
                trial_id, fidelity_id = config_id.split('_')
            

            self.val_losses[config_id] = float(report["objective_to_minimize"])
            results = {
                "config_name": config_id, 
                "objective_to_minimize": report["objective_to_minimize"], 
                "evaluation_duration": report["evaluation_duration"],
                "trial_id": trial_id,
                "fidelity_id": fidelity_id}
            
            results.update(report["extra"])
            results.update(config)
            self.reports[config_id] = results
            
            self.fidelities.add(results["epochs"])

        if not self.val_losses:
            raise RuntimeError("No valid report.yaml files found.")
    
    def get_plot_for_given_config(self):
        fig, ax = plt.subplots(figsize=(6, 4))
        for run_name in self.runs.values():
            run = api.run(f"nastaran78/nlp-classifier-automl/{run_name}")
            history = run.history()

            val_loss_curve = history["mean_val_roc_auc"]  # or your metric name
            ax.plot(range(1, len(val_loss_curve) + 1), val_loss_curve, marker=',', lw=2)
            ax.set_title("Learning Curve")
            ax.set_xlabel("Epochs/Budget")
            ax.set_ylabel("Validation ROC AUC")
            ax.grid(True, ls=":", alpha=0.6)
        # for fidelity in self.fidelities:
        #     ax.plot([float(fidelity), int(fidelity)], [0,1], marker='.', lw=2)

        ax.legend()
        fig.tight_layout()
        Path("plots").mkdir(exist_ok=True)
        fig.savefig(f"plots/{self.method}-all-learning-curves.png", dpi=150)
        print(f"File: plots/{self.method}-all-learning-curves.png ............Saved")


    def get_top_k_configs(self, k: int, save: Path) -> None:
        """ Get top-k configs based on the objective to minimize and save the detail in csv"""
        configs_distinct_dict = {}
        for report in self.reports.values():
            key = (report["lr"], report["batch_size"])
            if key in configs_distinct_dict and report["epochs"] < configs_distinct_dict[key]["epochs"]:
                continue
            else:
                configs_distinct_dict[key] = report
                
        configs_distinct = list(configs_distinct_dict.values())
        headers = configs_distinct[0].keys()
        rows = sorted(configs_distinct, key=lambda x: x["objective_to_minimize"], reverse=False)[:k]
        with open(save, 'w+') as f:
            writer = csv.DictWriter(f, fieldnames=list(headers))
            writer.writeheader()
            writer.writerows(rows)

    def plt_learning_curve(self, ax: Axes | None = None) -> Axes:
        steps = np.arange(1, len(self.val_losses) + 1)
        val_losses = np.array(list(self.val_losses.values()))

        ax = ax or plt.gca()
        ax.plot(steps, val_losses, marker='o', lw=2, label=f"{self.method} val loss")
        ax.set_title("Validation loss vs Config index")
        ax.set_xlabel("Config index")
        ax.set_ylabel("Validation loss")
        ax.grid(True, ls=":", alpha=0.6)
        return ax

    def plot_incumbent(self, ax: Axes | None = None) -> Axes:
        incumbent = np.minimum.accumulate(list(self.val_losses.values()))
        steps = np.arange(1, len(incumbent) + 1)

        ax = ax or plt.gca()
        ax.plot(steps, incumbent, marker=get_plot_marker(), lw=2, label=f"{self.method} incumbent")
        ax.set_title("Incumbent over configs")
        ax.set_xlabel("Config index")
        ax.set_ylabel("Best val loss so far")
        ax.grid(True, ls=":", alpha=0.6)
        return ax

    def plot_incumbent_over_epoch(self, ax: Axes | None = None) -> Axes:
        incumbent = np.minimum.accumulate(list(self.val_losses.values()))
        costs = [self.reports.get(conf)["epochs"] for conf in self.val_losses.keys()]
        wall_times = np.cumsum(costs)
        ax = ax or plt.gca()
        ax.plot(wall_times, incumbent, marker='o', lw=2, label=f"{self.method} incumbent")
        ax.set_title("Incumbent over epochs")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Best val loss so far")
        ax.grid(True, ls=":", alpha=0.6)
        return ax

    def plot_pareto(self, save: Path | None = None) -> None:
        """Scatter loss vs cost, highlight the non‑dominated front."""
        costs = [self.reports.get(conf)["epochs"] for conf in self.val_losses.keys()]
        c_arr = np.array(costs)
        l_arr = np.array(self.val_losses)

        idx = np.argsort(c_arr)
        c_sorted, l_sorted = c_arr[idx], l_arr[idx]
        pareto_mask = np.ones_like(c_sorted, dtype=bool)
        best_loss_so_far = math.inf
        for i, loss in enumerate(l_sorted):
            if loss < best_loss_so_far:
                best_loss_so_far = loss
            else:
                pareto_mask[i] = False

        plt.figure(figsize=(6, 4))
        plt.scatter(c_arr, l_arr, alpha=0.75, label="configs", color="tab:blue")
        plt.scatter(
            c_sorted[pareto_mask],
            l_sorted[pareto_mask],
            color="crimson",
            label="Pareto front",
            zorder=5,
        )
        plt.title(f"Pareto plot – {self.method}")
        plt.xlabel("Training cost (s)")
        plt.ylabel("Validation loss")
        plt.grid(True, ls=":", alpha=0.6)
        plt.legend()
        plt.tight_layout()

        if save:
            save.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save, dpi=150)
            print(f"[saved] {save}")
        else:
            plt.show()
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    # parser.add_argument("--result-dir", type=str, required=True)
    # parser.add_argument("--model", required=True, choices=["lstm", "bert"])
    # args = parser.parse_args()

    asha_viz = HPOVisualization(Path("neps_results/asha2"), 'ASHA')
    # bo_viz = HPOVisualization('neps_results/bayesian_optimization2', 'BO')
    hb_viz = HPOVisualization('neps_results/hyperband', 'HB')
    # asha_viz.get_plot_for_given_config()
    # # bo_viz.get_plot_for_given_config()
    # hb_viz.get_plot_for_given_config()
    
    asha_viz.get_top_k_configs(k=10, save=Path("plots") / "top10-asha.csv")
    # bo_viz.get_top_k_configs(k=10, save=Path("plots") / "top10-bo.csv")
    hb_viz.get_top_k_configs(k=10, save=Path("plots") / "top10-hb.csv")

    # # Plot both curves on the same figure
    # fig, ax = plt.subplots(figsize=(6, 4))
    # asha_viz.plt_learning_curve(ax=ax)
    # bo_viz.plt_learning_curve(ax=ax)
    # hb_viz.plt_learning_curve(ax=ax)
    # ax.legend()
    # fig.tight_layout()
    # Path("plots").mkdir(exist_ok=True)
    # fig.savefig("plots/learning_curve_plot.png", dpi=150)
    # print("[saved] plots/learning_curve_plot.png")

    # fig, ax = plt.subplots(figsize=(6, 4))
    # asha_viz.plot_incumbent(ax=ax)
    # bo_viz.plot_incumbent(ax=ax)
    # hb_viz.plot_incumbent(ax=ax)
    # ax.legend()
    # fig.tight_layout()
    # Path("plots").mkdir(exist_ok=True)
    # fig.savefig("plots/incumbent_plot.png", dpi=150)
    # print("[saved] plots/incumbent_plot.png")

    # fig, ax = plt.subplots(figsize=(6, 4))
    # asha_viz.plot_incumbent_over_epoch(ax=ax)
    # bo_viz.plot_incumbent_over_epoch(ax=ax)
    # hb_viz.plot_incumbent_over_epoch(ax=ax)
    # ax.legend()
    # fig.tight_layout()
    # Path("plots").mkdir(exist_ok=True)
    # fig.savefig("plots/incumbent_over_time_plot.png", dpi=150)
    # print("[saved] plots/incumbent_over_time_plot.png")


if __name__ == "__main__":
    main()
