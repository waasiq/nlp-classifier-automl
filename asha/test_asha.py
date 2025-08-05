import argparse
import yaml
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
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
                **config_data
            }
            all_results.append(result)
        except Exception:
            continue
    
    return pd.DataFrame(all_results)

def analyze_asha_run(df: pd.DataFrame):
    """Analyzes a DataFrame of HPO results to evaluate ASHA's performance."""
    if df.empty or len(df) < 2:
        logging.error("Not enough data to perform analysis.")
        return

    fidelities = sorted(df['data_fraction'].unique())
    low_fidelity_val = fidelities[0]
    high_fidelity_val = fidelities[-1]

    # --- 1. Fidelity Correlation Analysis ---
    df_low = df[df['data_fraction'] == low_fidelity_val]
    df_high = df[df['data_fraction'] == high_fidelity_val]
    paired_df = pd.merge(df_low, df_high, on="config_id", suffixes=('_low', '_high'))
    
    num_pairs = len(paired_df)
    corr, p_val = -1, 1.0 # Default values
    if num_pairs >= 3:
        corr, p_val = spearmanr(paired_df['loss_low'], paired_df['loss_high'])

    # --- 2. Promotion vs. Pruning Analysis ---
    promoted_ids = set(df_high['config_id'])
    all_low_fi_ids = set(df_low['config_id'])
    pruned_ids = all_low_fi_ids - promoted_ids

    promoted_runs_low_fi = df_low[df_low['config_id'].isin(promoted_ids)]
    pruned_runs_low_fi = df_low[df_low['config_id'].isin(pruned_ids)]

    avg_loss_promoted = promoted_runs_low_fi['loss'].mean()
    avg_loss_pruned = pruned_runs_low_fi['loss'].mean()

    # --- 3. Generate Report ---
    print("\n" + "="*60)
    print("      ASHA OPTIMIZER PERFORMANCE ANALYSIS")
    print("="*60)

    # Verdict
    print("\n## Verdict:")
    verdict = "UNCLEAR"
    if p_val < 0.05 and corr > 0.4 and avg_loss_promoted < avg_loss_pruned:
        verdict = "✅ EFFECTIVE: ASHA is working correctly and efficiently."
    elif p_val > 0.05:
        verdict = "⚠️ POTENTIALLY INEFFECTIVE: The fidelity correlation is not statistically significant."
    else:
        verdict = "❌ INEFFECTIVE: Low-fidelity performance is not a good predictor."
    print(verdict)

    print("\n## Evidence 1: Fidelity Correlation")
    print(f"  - Spearman Correlation: {corr:.4f} (p-value: {p_val:.4f})")
    if p_val < 0.05 and corr > 0.4:
        print("  - Insight: Strong correlation. Low-budget performance is a reliable predictor.")
    else:
        print("  - Insight: Weak or insignificant correlation. ASHA might be making suboptimal decisions.")

    print("\n## Evidence 2: Pruning Decisions")
    print(f"  - ASHA promoted {len(promoted_ids)} configs and pruned {len(pruned_ids)} configs.")
    print(f"  - Avg. low-fidelity loss of PROMOTED configs: {avg_loss_promoted:.5f}")
    print(f"  - Avg. low-fidelity loss of PRUNED configs:   {avg_loss_pruned:.5f}")
    if avg_loss_promoted < avg_loss_pruned:
        print("  - Insight: ASHA correctly promoted configurations that were better on average.")
    else:
        print("  - Insight: ASHA's pruning decisions were not optimal.")

    print("\n## Evidence 3: Resource Efficiency")
    print(f"  - Total low-fidelity trials: {len(df_low)}")
    print(f"  - Total high-fidelity trials: {len(df_high)}")
    print(f"  - Insight: ASHA saved costs by running {len(df_low) - len(df_high)} fewer high-fidelity trials.")
    print("="*60 + "\n")

    # --- 4. Generate Plots ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("ASHA Performance Evaluation", fontsize=16)

    # Plot 1: Fidelity Correlation
    if num_pairs > 0:
        axes[0].scatter(paired_df['loss_low'], paired_df['loss_high'], alpha=0.7, edgecolors='k')
        axes[0].set_title('Fidelity Correlation')
        axes[0].set_xlabel(f'Loss at Low Fidelity (data_fraction={low_fidelity_val:.2f})')
        axes[0].set_ylabel(f'Loss at High Fidelity (data_fraction={high_fidelity_val:.2f})')
        lims = [min(axes[0].get_xlim()[0], axes[0].get_ylim()[0]), max(axes[0].get_xlim()[1], axes[0].get_ylim()[1])]
        axes[0].plot(lims, lims, 'r--', alpha=0.75, zorder=0, label='y=x (Perfect Rank)')
        axes[0].legend()
    else:
        axes[0].text(0.5, 0.5, 'Not enough paired runs\nto plot correlation.', ha='center', va='center')
    axes[0].grid(True, linestyle='--', alpha=0.6)

    # Plot 2: Pruning Decisions
    plot_df = pd.concat([
        promoted_runs_low_fi.assign(Decision='Promoted'),
        pruned_runs_low_fi.assign(Decision='Pruned')
    ])
    sns.boxplot(data=plot_df, x='Decision', y='loss', ax=axes[1], palette=['#66c2a5', '#fc8d62'])
    axes[1].set_title('Effectiveness of Pruning Decisions')
    axes[1].set_xlabel('Decision Made by ASHA')
    axes[1].set_ylabel(f'Loss at Low Fidelity (data_fraction={low_fidelity_val:.2f})')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plot_filename = "asha_evaluation_report.png"
    plt.savefig(plot_filename)
    logging.info(f"📊 ASHA evaluation plots saved to '{plot_filename}'")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the effectiveness of an ASHA run from NePS results.")
    parser.add_argument(
        "--path",
        type=str,
        default="neps_results",
        help="Path to the root directory of the NePS run (e.g., 'neps_results')."
    )
    args = parser.parse_args()
    
    results_df = load_neps_data(root_dir=args.path)
    analyze_asha_run(results_df)