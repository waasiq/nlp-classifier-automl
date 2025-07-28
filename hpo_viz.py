from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import List

import yaml
import numpy as np
import matplotlib.pyplot as plt


class HPOVisualization:
    """Utility to parse TabPFN HPO results + produce two plots."""

    def __init__(self, result_dir: str | Path, model_name: str):
        self.result_dir = Path(result_dir)
        self.model_name = model_name

        self.costs: List[float] = []
        self.val_losses: List[float] = []
        self._load_reports()

    def _load_reports(self) -> None:
        """Read *report.yaml* for every config_<idx>/ sub‑folder."""
        configs_path = self.result_dir / "configs"
        if not configs_path.is_dir():
            raise FileNotFoundError(
                f"expected child directory 'configs' inside {self.result_dir}"
            )

        idx = 1
        while True:
            cfg_dir = configs_path / f"config_{idx}_0"
            report_path = cfg_dir / "report.yaml"
            if not (cfg_dir.is_dir() and report_path.is_file()):
                break

            with report_path.open("r", encoding="utf-8") as fh:
                report = yaml.safe_load(fh)

            if report.get("err") is not None:
                print(f"[skip] {report_path}: err != None → {report['err']}")
            else:
                self.costs.append(float(report["evaluation_duration"]))
                self.val_losses.append(float(report["objective_to_minimize"]))
            idx += 1

        if not self.val_losses:
            raise RuntimeError("No valid report.yaml files found.")

    def plot_incumbent(self, save: Path | None = None) -> None:
        """Monotonic incumbent‑over‑configs curve."""
        incumbent = np.minimum.accumulate(self.val_losses)
        steps = np.arange(1, len(incumbent) + 1)

        plt.figure(figsize=(6, 4))
        plt.plot(steps, incumbent, marker="o", lw=2, color="tab:orange")
        plt.title(f"Incumbent over configs – {self.model_name}")
        plt.xlabel("Config index")
        plt.ylabel("Best val loss so far")
        plt.grid(True, ls=":", alpha=0.6)
        plt.tight_layout()

        if save:
            save.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save, dpi=150)
            print(f"[saved] {save}")
        else:
            plt.show()
        plt.close()

    def plot_pareto(self, save: Path | None = None) -> None:
        """Scatter loss vs cost, highlight the non‑dominated front."""
        c_arr = np.array(self.costs)
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
        plt.title(f"Pareto plot – {self.model_name}")
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
    parser = argparse.ArgumentParser(
        description="Plot incumbent & Pareto for TabPFN HPO"
    )
    parser.add_argument(
        "--result-dir",
        type=str,
        required=True,
        help="Directory with HPO results (contains 'configs/' sub‑dir)",
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=[
            "lstm",
            "bert"
        ],
        help="Model architecture used during the sweep",
    )

    args = parser.parse_args()
    viz = HPOVisualization(args.result_dir, args.model)
    out_dir = Path("plots") / f"{args.model}"
    viz.plot_incumbent(out_dir / "incumbent.png")
    viz.plot_pareto(out_dir / "pareto.png")


if __name__ == "__main__":
    main()
