"""
Fidelity Correlation Analysis for NEPS Results

This script analyzes the correlation between low and high fidelity evaluations
in NEPS optimization results. It computes Spearman correlation coefficients
to understand how well low fidelity evaluations predict high fidelity performance.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, pearsonr
from pathlib import Path
import yaml
import json
import argparse
from typing import Dict, List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FidelityCorrelationAnalyzer:
    """Analyzer for computing fidelity correlations in NEPS results."""
    
    def __init__(self, neps_results_dir: Path):
        """
        Initialize the analyzer with NEPS results directory.
        
        Args:
            neps_results_dir: Path to NEPS results directory
        """
        self.neps_results_dir = Path(neps_results_dir)
        self.results_df = None
        
    def load_neps_results(self) -> pd.DataFrame:
        """
        Load NEPS results from the results directory.
        
        Returns:
            DataFrame with configuration and performance data
        """
        results = []
        
        # Look for result files in the NEPS directory structure
        for result_file in self.neps_results_dir.rglob("**/result.yaml"):
            try:
                with open(result_file, 'r') as f:
                    result_data = yaml.safe_load(f)
                
                # Extract configuration from config.yaml in the same directory
                config_file = result_file.parent / "config.yaml"
                if config_file.exists():
                    with open(config_file, 'r') as f:
                        config_data = yaml.safe_load(f)
                    
                    # Combine result and config data
                    combined_data = {
                        'trial_id': result_file.parent.name,
                        'loss': result_data.get('loss', result_data.get('result', None)),
                        'data_fraction': config_data.get('data_fraction', None),
                        'lr': config_data.get('lr', None),
                        **config_data
                    }
                    results.append(combined_data)
                    
            except Exception as e:
                logger.warning(f"Failed to load result from {result_file}: {e}")
                continue
        
        if not results:
            # Try alternative NEPS result structure
            logger.info("Trying alternative NEPS result structure...")
            results = self._load_alternative_structure()
        
        self.results_df = pd.DataFrame(results)
        logger.info(f"Loaded {len(self.results_df)} results")
        return self.results_df
    
    def _load_alternative_structure(self) -> List[Dict]:
        """Load results from alternative NEPS directory structures."""
        results = []
        
        # Look for any JSON/YAML files that might contain results
        for result_file in self.neps_results_dir.rglob("*.json"):
            try:
                with open(result_file, 'r') as f:
                    data = json.load(f)
                
                if isinstance(data, dict) and 'config' in data and 'result' in data:
                    config = data['config']
                    result_data = {
                        'trial_id': result_file.stem,
                        'loss': data['result'],
                        'data_fraction': config.get('data_fraction', None),
                        'lr': config.get('lr', None),
                        **config
                    }
                    results.append(result_data)
                    
            except Exception as e:
                continue
                
        return results
    
    def compute_fidelity_correlation(self, 
                                   low_fidelity_threshold: float = 0.3,
                                   high_fidelity_threshold: float = 0.8,
                                   method: str = 'spearman') -> Dict:
        """
        Compute correlation between low and high fidelity evaluations.
        
        Args:
            low_fidelity_threshold: Maximum data_fraction for low fidelity
            high_fidelity_threshold: Minimum data_fraction for high fidelity
            method: Correlation method ('spearman' or 'pearson')
            
        Returns:
            Dictionary with correlation results
        """
        if self.results_df is None:
            self.load_neps_results()
        
        df = self.results_df.copy()
        
        # Filter out rows with missing data_fraction or loss
        df = df.dropna(subset=['data_fraction', 'loss'])
        
        # Separate low and high fidelity results
        low_fidelity = df[df['data_fraction'] <= low_fidelity_threshold]
        high_fidelity = df[df['data_fraction'] >= high_fidelity_threshold]
        
        logger.info(f"Low fidelity samples: {len(low_fidelity)}")
        logger.info(f"High fidelity samples: {len(high_fidelity)}")
        
        if len(low_fidelity) == 0 or len(high_fidelity) == 0:
            logger.warning("Insufficient data for correlation analysis")
            return {}
        
        # Group by configuration (excluding data_fraction) to find matching configs
        config_cols = [col for col in df.columns if col not in ['data_fraction', 'loss', 'trial_id']]
        
        correlation_results = {}
        
        # Method 1: Direct correlation between all low and high fidelity results
        if method == 'spearman':
            corr_func = spearmanr
        else:
            corr_func = pearsonr
        
        # Overall correlation
        low_losses = low_fidelity['loss'].values
        high_losses = high_fidelity['loss'].values
        
        if len(low_losses) > 1 and len(high_losses) > 1:
            # Compute correlation between rankings
            low_ranks = pd.Series(low_losses).rank()
            high_ranks = pd.Series(high_losses).rank()
            
            # For matching configurations
            matched_pairs = self._find_matching_configurations(low_fidelity, high_fidelity, config_cols)
            
            if len(matched_pairs) > 2:
                matched_low = [pair[0] for pair in matched_pairs]
                matched_high = [pair[1] for pair in matched_pairs]
                
                correlation, p_value = corr_func(matched_low, matched_high)
                correlation_results['matched_configs'] = {
                    'correlation': correlation,
                    'p_value': p_value,
                    'n_pairs': len(matched_pairs),
                    'method': method
                }
        
        # Method 2: Correlation within fidelity buckets
        fidelity_buckets = self._create_fidelity_buckets(df)
        correlation_results['fidelity_buckets'] = fidelity_buckets
        
        return correlation_results
    
    def _find_matching_configurations(self, 
                                    low_fidelity: pd.DataFrame, 
                                    high_fidelity: pd.DataFrame,
                                    config_cols: List[str]) -> List[Tuple[float, float]]:
        """Find configurations that appear in both low and high fidelity."""
        matched_pairs = []
        
        for _, low_row in low_fidelity.iterrows():
            # Find matching configurations in high fidelity
            mask = True
            for col in config_cols:
                if col in high_fidelity.columns:
                    mask &= (high_fidelity[col] == low_row[col])
            
            matching_high = high_fidelity[mask]
            if len(matching_high) > 0:
                # Take the best (lowest loss) high fidelity result for this config
                best_high = matching_high.loc[matching_high['loss'].idxmin()]
                matched_pairs.append((low_row['loss'], best_high['loss']))
        
        return matched_pairs
    
    def _create_fidelity_buckets(self, df: pd.DataFrame) -> Dict:
        """Create correlation analysis across different fidelity levels."""
        # Create fidelity buckets
        df['fidelity_bucket'] = pd.cut(df['data_fraction'], 
                                     bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0], 
                                     labels=['very_low', 'low', 'medium', 'high', 'very_high'])
        
        bucket_correlations = {}
        buckets = df['fidelity_bucket'].unique()
        
        for i, bucket1 in enumerate(buckets):
            for bucket2 in buckets[i+1:]:
                if pd.isna(bucket1) or pd.isna(bucket2):
                    continue
                    
                bucket1_data = df[df['fidelity_bucket'] == bucket1]['loss']
                bucket2_data = df[df['fidelity_bucket'] == bucket2]['loss']
                
                if len(bucket1_data) > 2 and len(bucket2_data) > 2:
                    # Sample equal numbers for correlation if datasets are unbalanced
                    min_size = min(len(bucket1_data), len(bucket2_data), 50)
                    sample1 = bucket1_data.sample(min_size, random_state=42)
                    sample2 = bucket2_data.sample(min_size, random_state=42)
                    
                    corr, p_val = spearmanr(sample1, sample2)
                    bucket_correlations[f"{bucket1}_vs_{bucket2}"] = {
                        'correlation': corr,
                        'p_value': p_val,
                        'n_samples': min_size
                    }
        
        return bucket_correlations
    
    def visualize_fidelity_correlation(self, 
                                     save_path: Optional[Path] = None,
                                     figsize: Tuple[int, int] = (15, 10)):
        """
        Create visualizations for fidelity correlation analysis.
        
        Args:
            save_path: Path to save the plots
            figsize: Figure size for the plots
        """
        if self.results_df is None:
            self.load_neps_results()
        
        df = self.results_df.dropna(subset=['data_fraction', 'loss'])
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Fidelity Correlation Analysis', fontsize=16)
        
        # Plot 1: Loss vs Data Fraction
        axes[0, 0].scatter(df['data_fraction'], df['loss'], alpha=0.6)
        axes[0, 0].set_xlabel('Data Fraction (Fidelity)')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Performance vs Fidelity')
        
        # Plot 2: Distribution of losses by fidelity buckets
        df['fidelity_bucket'] = pd.cut(df['data_fraction'], 
                                     bins=[0, 0.3, 0.7, 1.0], 
                                     labels=['Low', 'Medium', 'High'])
        
        df.boxplot(column='loss', by='fidelity_bucket', ax=axes[0, 1])
        axes[0, 1].set_title('Loss Distribution by Fidelity Level')
        axes[0, 1].set_xlabel('Fidelity Level')
        
        # Plot 3: Correlation heatmap if we have learning rate data
        if 'lr' in df.columns:
            pivot_table = df.pivot_table(values='loss', 
                                       index=pd.cut(df['data_fraction'], bins=5), 
                                       columns=pd.cut(df['lr'], bins=3), 
                                       aggfunc='mean')
            sns.heatmap(pivot_table, ax=axes[1, 0], annot=True, fmt='.3f')
            axes[1, 0].set_title('Loss Heatmap: Fidelity vs Learning Rate')
        
        # Plot 4: Ranking correlation
        low_fidelity = df[df['data_fraction'] <= 0.3]
        high_fidelity = df[df['data_fraction'] >= 0.7]
        
        if len(low_fidelity) > 0 and len(high_fidelity) > 0:
            axes[1, 1].scatter(low_fidelity['loss'].rank(), 
                             [i for i in range(len(low_fidelity))], 
                             alpha=0.6, label='Low Fidelity Ranks')
            axes[1, 1].scatter(high_fidelity['loss'].rank(), 
                             [i for i in range(len(high_fidelity))], 
                             alpha=0.6, label='High Fidelity Ranks')
            axes[1, 1].set_xlabel('Loss Rank')
            axes[1, 1].set_ylabel('Configuration Index')
            axes[1, 1].set_title('Ranking Comparison')
            axes[1, 1].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plots saved to {save_path}")
        
        plt.show()
    
    def generate_report(self, output_file: Optional[Path] = None) -> str:
        """
        Generate a comprehensive correlation analysis report.
        
        Args:
            output_file: Path to save the report
            
        Returns:
            Report text
        """
        correlation_results = self.compute_fidelity_correlation()
        
        report_lines = [
            "=== FIDELITY CORRELATION ANALYSIS REPORT ===\n",
            f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"NEPS Results Directory: {self.neps_results_dir}",
            f"Total Configurations Analyzed: {len(self.results_df) if self.results_df is not None else 0}\n"
        ]
        
        if 'matched_configs' in correlation_results:
            matched = correlation_results['matched_configs']
            report_lines.extend([
                "--- MATCHED CONFIGURATIONS ANALYSIS ---",
                f"Correlation Coefficient: {matched['correlation']:.4f}",
                f"P-value: {matched['p_value']:.4f}",
                f"Number of Matched Pairs: {matched['n_pairs']}",
                f"Method: {matched['method']}",
                f"Interpretation: {'Strong' if abs(matched['correlation']) > 0.7 else 'Moderate' if abs(matched['correlation']) > 0.3 else 'Weak'} correlation\n"
            ])
        
        if 'fidelity_buckets' in correlation_results:
            report_lines.append("--- FIDELITY BUCKET CORRELATIONS ---")
            for comparison, result in correlation_results['fidelity_buckets'].items():
                report_lines.extend([
                    f"{comparison}:",
                    f"  Correlation: {result['correlation']:.4f}",
                    f"  P-value: {result['p_value']:.4f}",
                    f"  Sample size: {result['n_samples']}"
                ])
            report_lines.append("")
        
        # Add recommendations
        report_lines.extend([
            "--- RECOMMENDATIONS ---",
            "Based on the correlation analysis:",
        ])
        
        if 'matched_configs' in correlation_results:
            corr = correlation_results['matched_configs']['correlation']
            if abs(corr) > 0.7:
                report_lines.append("• Strong correlation suggests low fidelity is a good predictor")
                report_lines.append("• Consider using more aggressive early stopping")
            elif abs(corr) > 0.3:
                report_lines.append("• Moderate correlation suggests reasonable fidelity trade-off")
                report_lines.append("• Current successive halving strategy appears appropriate")
            else:
                report_lines.append("• Weak correlation suggests low fidelity may not be reliable")
                report_lines.append("• Consider increasing minimum data fraction or using different fidelity")
        
        report_text = "\n".join(report_lines)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_text)
            logger.info(f"Report saved to {output_file}")
        
        return report_text


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Analyze fidelity correlations in NEPS results")
    parser.add_argument("--neps-dir", type=Path, required=True,
                       help="Path to NEPS results directory")
    parser.add_argument("--output-dir", type=Path, default=Path("./fidelity_analysis"),
                       help="Output directory for plots and reports")
    parser.add_argument("--low-threshold", type=float, default=0.3,
                       help="Maximum data fraction for low fidelity")
    parser.add_argument("--high-threshold", type=float, default=0.8,
                       help="Minimum data fraction for high fidelity")
    parser.add_argument("--method", choices=['spearman', 'pearson'], default='spearman',
                       help="Correlation method")
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize analyzer
    analyzer = FidelityCorrelationAnalyzer(args.neps_dir)
    
    try:
        # Load results
        analyzer.load_neps_results()
        
        # Compute correlations
        results = analyzer.compute_fidelity_correlation(
            low_fidelity_threshold=args.low_threshold,
            high_fidelity_threshold=args.high_threshold,
            method=args.method
        )
        
        # Generate visualizations
        plot_path = args.output_dir / "fidelity_correlation_plots.png"
        analyzer.visualize_fidelity_correlation(save_path=plot_path)
        
        # Generate report
        report_path = args.output_dir / "fidelity_correlation_report.txt"
        report = analyzer.generate_report(output_file=report_path)
        
        print(report)
        
        # Save results as JSON
        results_path = args.output_dir / "correlation_results.json"
        with open(results_path, 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return obj
            
            json.dump(results, f, indent=2, default=convert_numpy)
        
        logger.info(f"Analysis complete. Results saved to {args.output_dir}")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise


if __name__ == "__main__":
    main()
