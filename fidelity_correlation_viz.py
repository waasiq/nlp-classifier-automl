import argparse
import yaml
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def analyze_fidelity(root_dir: str = "neps_results", fidelity_name: str = "epochs"):
    neps_path = Path(root_dir)
    configs_path = neps_path / "configs"

    if not configs_path.is_dir():
        logging.error(f"Directory not found: {configs_path}")
        return

    all_results = []
    logging.info(f"Scanning for trial results in {configs_path}...")

    for trial_path in sorted(configs_path.glob("config_*_*")):
        try:
            config_id, fidelity_id = trial_path.name.split("_")[1:]
            
            config_file = trial_path / "config.yaml"
            report_file = trial_path / "report.yaml"

            if not (config_file.exists() and report_file.exists()):
                logging.warning(f"Skipping incomplete trial: {trial_path.name}")
                continue
            
            with open(report_file, 'r') as f:
                report_data = yaml.safe_load(f)
            
            if report_data.get("reported_as") != "success":
                logging.warning(f"Skipping non-successful trial: {trial_path.name}")
                continue

            with open(config_file, 'r') as f:
                config_data = yaml.safe_load(f)

            result = {
                "config_id": config_id,
                "fidelity_id": fidelity_id,
                "loss": report_data.get("objective_to_minimize"),
                "test_roc_auc": report_data["extra"]["test_mean_roc_auc"],
                fidelity_name: config_data.get(fidelity_name),
                **config_data 
            }
            all_results.append(result)

        except (ValueError, IndexError, yaml.YAMLError) as e:
            logging.error(f"Could not parse {trial_path.name}: {e}")
            continue
    
    if not all_results:
        logging.error("No successful trials found to analyze.")
        return
        
    df = pd.DataFrame(all_results)
    logging.info(f"Successfully parsed {len(df)} successful trial results.")

    fidelities = sorted(df[fidelity_name].unique())
    if len(fidelities) < 2:
        logging.warning("Need at least two different fidelity levels to perform correlation analysis.")
        return

    low_fidelity_val = fidelities[0]
    high_fidelity_val = fidelities[-1]

    df_low = df[df[fidelity_name] == low_fidelity_val]
    df_high = df[df[fidelity_name] == high_fidelity_val]

    paired_df = pd.merge(
        df_low, 
        df_high, 
        on="config_id", 
        suffixes=('_low', '_high')
    )
    print(paired_df)
    num_pairs = len(paired_df)
    logging.info(f"Found {num_pairs} configurations evaluated at both low ({low_fidelity_val:.2f}) and high ({high_fidelity_val:.2f}) fidelity.")

    if num_pairs < 3:
        logging.warning("Not enough paired configurations to calculate a meaningful correlation.")
        return

    low_losses = paired_df['loss_low']
    high_losses = paired_df['loss_high']

    corr, p_val = spearmanr(low_losses, high_losses)

    print("\n" + "="*50)
    print("      FIDELITY CORRELATION ANALYSIS REPORT")
    print("="*50)
    print(f"Paired Runs Analyzed: {num_pairs}")
    print(f"Low Fidelity (data_fraction): {low_fidelity_val:.3f}")
    print(f"High Fidelity (data_fraction): {high_fidelity_val:.3f}")
    print("-" * 50)
    print(f"Spearman Correlation: {corr:.4f}")
    print(f"P-value: {p_val:.4f}")
    print("-" * 50)

    abs_corr = abs(corr)
    if p_val > 0.05:
        print("Interpretation: The correlation is not statistically significant (p > 0.05).")
    elif abs_corr >= 0.8:
        print("Interpretation: Very strong correlation. Low fidelity is an excellent predictor.")
    elif abs_corr >= 0.6:
        print("Interpretation: Strong correlation. Low fidelity is a good predictor.")
    elif abs_corr >= 0.4:
        print("Interpretation: Moderate correlation. Low fidelity provides useful information.")
    else:
        print("Interpretation: Weak or no correlation. Low fidelity may not be reliable.")
    print("="*50 + "\n")

    plt.figure(figsize=(8, 6))
    plt.scatter(low_losses, high_losses, alpha=0.7, edgecolors='k')
    plt.title('Fidelity Correlation: Low vs. High Budget Performance')
    plt.xlabel(f'Loss at Low Fidelity ({fidelity_name}={low_fidelity_val:.2f})')
    plt.ylabel(f'Loss at High Fidelity ({fidelity_name}={high_fidelity_val:.2f})')
    
    lims = [
        min(plt.xlim()[0], plt.ylim()[0]),
        max(plt.xlim()[1], plt.ylim()[1]),
    ]
    plt.plot(lims, lims, 'r--', alpha=0.75, zorder=0, label='y=x (Perfect Rank)')
    
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    plot_filename = "plots/fidelity_correlation_analysis.png"
    plt.savefig(plot_filename)
    logging.info(f"📊 Correlation plot saved to '{plot_filename}'")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze fidelity correlation from a NePS run.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="neps_results",
        help="Path to the root directory of the NePS run (e.g., 'neps_results')."
    )
    args = parser.parse_args()
    
    analyze_fidelity(root_dir=args.output_dir, fidelity_name="epochs")