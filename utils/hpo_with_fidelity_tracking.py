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
import pandas as pd
import json
from scipy.stats import spearmanr
from typing import Dict, Any

logger = logging.getLogger(__name__)


class FidelityCorrelationTracker:
    """Tracks and analyzes fidelity correlations during NEPS optimization."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = []
        self.tracking_file = self.output_dir / "fidelity_tracking.json"
        self.h_params = ['bert_lr', 'batch_size', 'num_bert_layers_to_unfreeze']

    def log_evaluation(self, config: Dict[str, Any], loss: float):
        result = {**config, 'loss': loss, 'timestamp': pd.Timestamp.now().isoformat()}
        self.results.append(result)

        with open(self.tracking_file, 'w') as f:
            json.dump(self.results, f, indent=2)

        if len(self.results) % 10 == 0 and len(self.results) >= 10:
            self.analyze_current_correlation()

    def _calculate_paired_correlation(self, df: pd.DataFrame) -> Optional[Tuple[float, float, List[float], List[float]]]:
        """
        Calculates Spearman correlation on paired low and high fidelity runs.

        Returns:
            A tuple of (correlation, p_value, low_losses, high_losses) or None.
        """
        if not all(p in df.columns for p in self.h_params):
            logger.warning("Hyperparameter columns not found in results; cannot compute paired correlation.")
            return None

        low_fidelity_runs = df[df['data_fraction'] <= 0.5]
        high_fidelity_runs = df[df['data_fraction'] >= 0.7]

        # 2. Merge dataframes on the hyperparameter keys to find pairs
        # This finds configurations that have been evaluated at BOTH low and high fidelity.
        paired_df = pd.merge(
            low_fidelity_runs,
            high_fidelity_runs,
            on=self.h_params,
            suffixes=('_low', '_high')
        )

        # 3. Check if we have enough pairs for a meaningful correlation
        if len(paired_df) < 3:
            return None

        # 4. Extract the paired lists of losses
        low_losses = paired_df['loss_low'].tolist()
        high_losses = paired_df['loss_high'].tolist()

        # 5. Calculate correlation on the correctly paired data
        corr, p_val = spearmanr(low_losses, high_losses)
        return corr, p_val, low_losses, high_losses

    
    def analyze_current_correlation(self):
        if len(self.results) < 2:
            return
        df = pd.DataFrame(self.results)
        
        # Use the new, correct correlation logic
        correlation_result = self._calculate_paired_correlation(df)
        logger.info(f"Current results: {len(df)} total, "
                    f"{len(df[df['data_fraction'] <= 0.5])} low, "
                    f"{len(df[df['data_fraction'] >= 0.7])} high fidelity")

        if correlation_result is None:
            logger.info("Not enough paired low/high fidelity runs to analyze correlation yet.")
            return
            
        corr, p_val, low_losses, _ = correlation_result
        num_pairs = len(low_losses)
        
        logger.info(f"🔍 Fidelity Correlation Analysis (based on {num_pairs} paired configurations):")
        logger.info(f"   Spearman correlation: {corr:.4f}")
        logger.info(f"   P-value: {p_val:.4f}")
        
        self.create_correlation_plot(df)

    def _interpret_correlation(self, correlation: float) -> str:
        abs_corr = abs(correlation)
        if abs_corr >= 0.8:
            return "Very strong correlation - low fidelity is an excellent predictor"
        elif abs_corr >= 0.6:
            return "Strong correlation - low fidelity is a good predictor"
        elif abs_corr >= 0.4:
            return "Moderate correlation - low fidelity provides useful information"
        elif abs_corr >= 0.2:
            return "Weak correlation - low fidelity has limited predictive value"
        else:
            return "No correlation - low fidelity is not predictive of high fidelity"
    
    def create_correlation_plot(self, df: pd.DataFrame):
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            plt.style.use('default')
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle('NEPS Fidelity Correlation Analysis', fontsize=14)
            
            # Plot 1: Loss vs Data Fraction
            axes[0, 0].scatter(df['data_fraction'], df['loss'], alpha=0.6, c=df.index, cmap='viridis')
            axes[0, 0].set_xlabel('Data Fraction (Fidelity)')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].set_title('Performance vs Fidelity Level')
            
            # Plot 2: Fidelity distribution
            axes[0, 1].hist(df['data_fraction'], bins=20, alpha=0.7, edgecolor='black')
            axes[0, 1].set_xlabel('Data Fraction')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].set_title('Fidelity Distribution')
            
            # Plot 3: Loss distribution by fidelity level
            df['fidelity_level'] = pd.cut(df['data_fraction'], 
                                        bins=[0, 0.5, 0.7, 1.0], 
                                        labels=['Low', 'Medium', 'High'])
            
            fidelity_groups = [group['loss'].values for name, group in df.groupby('fidelity_level') if len(group) > 0]
            fidelity_labels = [name for name, group in df.groupby('fidelity_level') if len(group) > 0]
            
            if fidelity_groups:
                axes[1, 0].boxplot(fidelity_groups, labels=fidelity_labels)
                axes[1, 0].set_xlabel('Fidelity Level')
                axes[1, 0].set_ylabel('Loss')
                axes[1, 0].set_title('Loss Distribution by Fidelity')
            
            if 'lr' in df.columns:
                axes[1, 1].scatter(df['lr'], df['loss'], c=df['data_fraction'], 
                                 cmap='coolwarm', alpha=0.6)
                axes[1, 1].set_xlabel('Learning Rate')
                axes[1, 1].set_ylabel('Loss')
                axes[1, 1].set_title('LR vs Loss (colored by fidelity)')
                axes[1, 1].set_xscale('log')
                
                im = axes[1, 1].scatter(df['lr'], df['loss'], c=df['data_fraction'], 
                                      cmap='coolwarm', alpha=0.6)
                plt.colorbar(im, ax=axes[1, 1], label='Data Fraction')
            
            plt.tight_layout()
            
            plot_file = self.output_dir / f"correlation_plot_step_{len(df)}.png"
            plt.savefig(plot_file, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"📊 Correlation plot saved to {plot_file}")
            
        except ImportError:
            logger.warning("Matplotlib not available for plotting")
        except Exception as e:
            logger.warning(f"Failed to create plot: {e}")
    
    def generate_final_report(self):
        if not self.results:
            return
        
        df = pd.DataFrame(self.results)
        
            
        report_lines = [
            "=== FIDELITY CORRELATION FINAL REPORT ===",
            f"Total Evaluations: {len(df)}",
        ]

        correlation_result = self._calculate_paired_correlation(df)

        if correlation_result:
            corr, p_val, low_losses, high_losses = correlation_result
            report_lines.extend([
                "\n--- PAIRED CORRELATION ANALYSIS ---",
                f"Number of paired configurations found: {len(low_losses)}",
                f"Spearman Correlation: {corr:.4f}",
                f"P-value: {p_val:.4f}",
                f"Interpretation: {self._interpret_correlation(corr)}",
            ])
        else:
            report_lines.append("\nCould not compute a final correlation score: not enough paired data.")

        report_text = "\n".join(report_lines)

        report_file = self.output_dir / "final_correlation_report.txt"
        with open(report_file, 'w') as f:
            f.write(report_text)
                
        logger.info(f"📄 Final report saved to {report_file}")


def neps_training_wrapper_with_tracking(args, tracker: FidelityCorrelationTracker, dataset_classes, train_dfs, val_dfs, test_dfs, num_classes, out_dir: Path):
    """Enhanced training wrapper that tracks fidelity correlations."""
    def evaluate_pipeline(pipeline_directory, data_fraction, **kwargs):
        # Extract parameters from kwargs or use defaults from config
        lr = kwargs.get('lr', args["lr"])  # fallback to config lr
        batch_size = kwargs.get('batch_size', args["batch_size"])
        bert_lr = kwargs.get('bert_lr', args.get("bert_lr", 0.00005))
        num_bert_layers_to_unfreeze = kwargs.get('num_bert_layers_to_unfreeze', 2)
        
        # Run the evaluation - main_loop returns val_err (float) despite -> None annotation
        result_dict = main_loop(
            train_dfs=train_dfs,
            val_dfs=val_dfs,
            test_dfs=test_dfs,
            num_classes=num_classes,
            dataset_classes=dataset_classes,
            pipeline_directory=pipeline_directory,
            data_fraction=data_fraction,
            output_path=out_dir.absolute(),
            seed=args["seed"],
            approach=args["approach"],
            vocab_size=args["model_config"]["vocab_size"],
            token_length=args["token_length"],
            epochs=args["epochs"],
            batch_size=batch_size,
            lr=lr,
            bert_lr=bert_lr,
            weight_decay=args["weight_decay"],
            ffnn_hidden=args["ffnn_hidden_layer_dim"],
            lstm_emb_dim=args["model_config"]["lstm_emb_dim"],
            lstm_hidden_dim=args["model_config"]["lstm_hidden_dim"],
            num_bert_layers_to_unfreeze=num_bert_layers_to_unfreeze,
            load_path=Path(args["load_path"]) if args["load_path"] else None,
            model_name=args["model_name"],
        )
        
        # Log the evaluation to the fidelity tracker
        config = {
            'lr': lr,
            'data_fraction': data_fraction,
            'batch_size': batch_size,
            'bert_lr': bert_lr,
            'num_bert_layers_to_unfreeze': num_bert_layers_to_unfreeze,
            'pipeline_directory': str(pipeline_directory)
        }
        loss = result_dict['objective_to_minimize']
        tracker.log_evaluation(config, loss)
        return result_dict
    return evaluate_pipeline


def get_args():
    """Get configuration arguments."""
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
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    tracker = FidelityCorrelationTracker(out_dir)
    logger.info("🎯 Fidelity correlation tracking enabled")
    
    nep_configs = conf.pop("neps")
    
    dataset_classes, train_dfs, val_dfs, test_dfs, num_classes = load_dataset(
        dataset=conf["dataset"],
        data_path=Path(conf["data_path"]).absolute(),
        seed=conf["seed"],
        val_size=conf["val_size"],
        is_mtl=conf["is_mtl"],
    )
    
    try:
        neps.run(
            evaluate_pipeline=neps_training_wrapper_with_tracking(conf, tracker, dataset_classes, train_dfs, val_dfs, test_dfs, num_classes, out_dir),
            **nep_configs
        )
    except KeyboardInterrupt:
        logger.info("Optimization interrupted by user")
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        raise
    finally:
        logger.info("🔍 Generating final fidelity correlation report...")
        tracker.generate_final_report()
