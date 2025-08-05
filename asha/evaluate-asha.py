import argparse
import yaml
import pandas as pd
from pathlib import Path
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def load_neps_data(root_dir: str) -> pd.DataFrame:
    """Parses all successful trial data from a neps_results directory."""
    neps_path = Path(root_dir)
    configs_path = neps_path / "configs"
    if not configs_path.is_dir():
        logging.error(f"Directory not found: {configs_path}")
        return pd.DataFrame()

    all_results = []
    # Define the hyperparameters that make a configuration unique
    h_params = ['bert_lr', 'batch_size', 'num_bert_layers_to_unfreeze']
    
    for trial_path in sorted(configs_path.glob("config_*_*")):
        try:
            config_id, fidelity_id = trial_path.name.split("_")[1:]
            report_file = trial_path / "report.yaml"
            if not report_file.exists(): continue
            
            with open(report_file, 'r') as f: report_data = yaml.safe_load(f)
            if report_data.get("reported_as") != "success": continue

            with open(trial_path / "config.yaml", 'r') as f: config_data = yaml.safe_load(f)

            result = {
                "config_id": config_id,
                "fidelity_id": fidelity_id,
                "loss": report_data.get("objective_to_minimize"),
                "data_fraction": config_data.get("data_fraction"),
                **{k: config_data[k] for k in h_params}
            }
            all_results.append(result)
        except Exception:
            continue
    
    return pd.DataFrame(all_results)

def analyze_asha_run(df: pd.DataFrame, h_params: list):
    """Analyzes a DataFrame of HPO results to evaluate ASHA's effectiveness."""
    if df.empty or len(df) < 2:
        logging.error("Not enough data to perform analysis.")
        return

    fidelities = sorted(df['data_fraction'].unique())
    low_fidelity_val = fidelities[0]
    high_fidelity_val = fidelities[-1]

    # --- 1. Duplicate Configuration Analysis ---
    # Group by hyperparameters and count how many times each unique config was sampled
    duplicate_counts = df[df['fidelity_id'] == '0'].groupby(h_params).size().reset_index(name='counts')
    duplicates = duplicate_counts[duplicate_counts['counts'] > 1]

    # --- 2. Pruning vs. Promotion Analysis ---
    df_low = df[df['data_fraction'] == low_fidelity_val]
    df_high = df[df['data_fraction'] == high_fidelity_val]

    promoted_ids = set(df_high['config_id'])
    pruned_ids = set(df_low['config_id']) - promoted_ids

    promoted_runs_low_fi = df_low[df_low['config_id'].isin(promoted_ids)]
    pruned_runs_low_fi = df_low[df_low['config_id'].isin(pruned_ids)]
    
    avg_loss_promoted = promoted_runs_low_fi['loss'].mean()
    avg_loss_pruned = pruned_runs_low_fi['loss'].mean()
    
    # Perform t-test to see if the difference in performance is statistically significant
    ttest_result = ttest_ind(promoted_runs_low_fi['loss'], pruned_runs_low_fi['loss'], nan_policy='omit')
    p_value = ttest_result.pvalue

    # --- 3. Generate Report ---
    print("\n" + "="*60)
    print("      ASHA OPTIMIZER PERFORMANCE ANALYSIS")
    print("="*60)

    print("\n## Verdict:")
    if avg_loss_promoted < avg_loss_pruned and p_value < 0.05:
        print("✅ EFFECTIVE: ASHA made statistically significant, correct pruning decisions.")
    elif avg_loss_promoted < avg_loss_pruned:
        print("⚠️ LIKELY EFFECTIVE: ASHA made correct pruning decisions on average, but the difference was not statistically significant.")
    else:
        print("❌ INEFFECTIVE: ASHA's pruning decisions were not optimal and may have been random.")

    print("\n## Analysis of Pruning Decisions")
    print(f"  - ASHA promoted {len(promoted_ids)} unique configs and pruned {len(pruned_ids)} unique configs.")
    print(f"  - Avg. low-fidelity loss of PROMOTED configs: {avg_loss_promoted:.5f}")
    print(f"  - Avg. low-fidelity loss of PRUNED configs:   {avg_loss_pruned:.5f}")
    print(f"  - Is the difference significant? (p-value): {p_value:.4f}")
    
    if not duplicates.empty:
        print("\n## Analysis of Duplicate Configurations")
        print("The following configurations were sampled more than once by ASHA's random sampler:")
        print(duplicates.to_string(index=False))
    else:
        print("\n## Analysis of Duplicate Configurations")
        print("No duplicate configurations were sampled.")

    print("\n## Resource Efficiency")
    print(f"  - Total low-fidelity trials: {len(df_low)}")
    print(f"  - Total high-fidelity trials: {len(df_high)}")
    print(f"  - Insight: ASHA saved costs by running {len(df_low) - len(df_high)} fewer high-fidelity trials.")
    print("="*60 + "\n")

    # --- 4. Generate Plot ---
    plt.figure(figsize=(8, 6))
    plot_df = pd.concat([
        promoted_runs_low_fi.assign(Decision='Promoted'),
        pruned_runs_low_fi.assign(Decision='Pruned')
    ])
    sns.boxplot(data=plot_df, x='Decision', y='loss', palette=['#66c2a5', '#fc8d62'])
    plt.title("ASHA's Pruning Decisions: Performance at Low Fidelity", fontsize=14)
    plt.xlabel('Decision Made by ASHA', fontsize=12)
    plt.ylabel(f'Loss at Low Fidelity (data_fraction={low_fidelity_val:.2f})', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plot_filename = "asha_pruning_analysis.png"
    plt.savefig(plot_filename)
    logging.info(f"📊 ASHA analysis plot saved to '{plot_filename}'")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the effectiveness of an ASHA run from NePS results.")
    parser.add_argument(
        "--path", type=str, default="neps_results",
        help="Path to the root directory of the NePS run (e.g., 'neps_results')."
    )
    args = parser.parse_args()
    
    hyperparameters = ['bert_lr', 'batch_size', 'num_bert_layers_to_unfreeze']
    results_df = load_neps_data(root_dir=args.path)
    analyze_asha_run(results_df, h_params=hyperparameters)