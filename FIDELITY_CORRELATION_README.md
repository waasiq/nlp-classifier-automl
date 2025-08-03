# NEPS Fidelity Correlation Analysis

This repository contains tools for analyzing the correlation between low and high fidelity evaluations in NEPS (Neural Architecture Search) optimization, specifically for your LSTM configuration with data_fraction as the fidelity parameter.

## Overview

Your current NEPS setup uses `data_fraction` (ranging from 0.1 to 1.0) as a fidelity parameter with successive halving optimization. This means NEPS trains models on different fractions of your training data to save computational time, but you need to know if training on less data (low fidelity) actually predicts performance on full data (high fidelity).

## Files Created

### 1. `simple_fidelity_analysis.py` (RECOMMENDED START HERE)
A standalone script that analyzes existing NEPS results to compute Spearman correlations between low and high fidelity evaluations.

**Usage:**
```bash
# Analyze your NEPS results (assuming they're in neps_results_df/)
python simple_fidelity_analysis.py neps_results_df/

# With custom thresholds
python simple_fidelity_analysis.py neps_results_df/ --low-threshold 0.2 --high-threshold 0.8

# Verbose output
python simple_fidelity_analysis.py neps_results_df/ --verbose
```

**What it does:**
- Automatically finds NEPS result files in your directory
- Extracts loss values and data_fraction parameters  
- Computes Spearman correlation between low fidelity (≤0.3 data_fraction) and high fidelity (≥0.7 data_fraction)
- Generates plots and a comprehensive analysis report
- Provides recommendations based on correlation strength

### 2. `hpo_with_fidelity_tracking.py`
Enhanced version of your `hpo.py` that tracks fidelity correlations in real-time during optimization.

**Usage:**
```bash
# Replace your normal hpo.py call with this enhanced version
python hpo_with_fidelity_tracking.py dataset=yelp ++neps.root_directory=results/yelp_with_tracking
```

**What it does:**
- Everything your original `hpo.py` does
- Logs every evaluation with its configuration and loss
- Computes correlation analysis every 10 evaluations
- Shows real-time correlation updates in the console
- Generates final comprehensive report when optimization finishes

### 3. `fidelity_correlation_analysis.py`
Comprehensive analysis class with advanced features for detailed correlation analysis.

### 4. `neps_fidelity_tracker.py`
Modular tracking utilities that can be integrated into existing NEPS workflows.

## Quick Start

1. **If you already have NEPS results:**
   ```bash
   python simple_fidelity_analysis.py neps_results_df/
   ```

2. **For new optimization runs:**
   ```bash
   python hpo_with_fidelity_tracking.py dataset=yelp ++neps.root_directory=results/yelp_correlation_test
   ```

## Understanding the Results

### Correlation Coefficients
- **> 0.7**: Strong correlation - low fidelity is an excellent predictor
- **0.4-0.7**: Moderate correlation - low fidelity provides useful information  
- **0.2-0.4**: Weak correlation - low fidelity has limited predictive value
- **< 0.2**: No correlation - low fidelity is not predictive

### Recommendations Based on Results

**Strong Correlation (>0.6):**
- ✅ Your successive halving strategy is working well
- Consider more aggressive early stopping
- You could reduce max data_fraction to save time

**Moderate Correlation (0.3-0.6):**
- ⚠️ Current strategy is reasonable but could be improved
- Monitor correlation as optimization progresses
- Consider adjusting fidelity thresholds

**Weak Correlation (<0.3):**
- ❌ Low fidelity may not be reliable for your problem
- Consider increasing minimum data_fraction 
- Alternative fidelity measures might be needed
- Standard optimization without multi-fidelity might be better

## Key Insights for Your Setup

Your LSTM configuration with `data_fraction` as fidelity is particularly interesting because:

1. **Text data sensitivity**: Text classification performance might be more sensitive to dataset size than other domains
2. **Vocabulary effects**: Smaller data fractions might not capture the full vocabulary, affecting correlation
3. **Class balance**: Reduced data might change class distributions, impacting low→high fidelity correlation

## Output Files

The analysis generates several output files:

- `fidelity_analysis_report.txt`: Human-readable analysis report with recommendations
- `correlation_results.json`: Raw correlation statistics and metadata
- `extracted_data.csv`: Processed evaluation data for further analysis
- `fidelity_analysis_plots.png`: Visualization of fidelity relationships

## Integration with Your Workflow

### Current NEPS Config (`configs/neps/lstm.yaml`):
```yaml
pipeline_space:
  lr:
    lower: 0.000001
    upper: 0.001
    log: True
  data_fraction:
    lower: 0.1
    upper: 1.0
    is_fidelity: True

optimizer: "successive_halving"
max_evaluations_total: 100
root_directory: neps_results_df/
```

This configuration is perfect for fidelity correlation analysis because:
- ✅ `data_fraction` is marked as fidelity parameter
- ✅ Wide range (0.1-1.0) allows good separation of low/high fidelity
- ✅ Successive halving naturally creates the fidelity data we need

## Troubleshooting

### "No NEPS result files found"
- Check that your results directory exists and contains result files
- NEPS might use different file naming - the script looks for various patterns
- Try running with `--verbose` for more detailed logging

### "No valid evaluation data extracted"
- Ensure your result files contain both 'loss' and 'data_fraction' values
- Check that the file format matches expected NEPS structure

### "Insufficient data for correlation analysis"
- You need at least 3 evaluations each in low and high fidelity ranges
- Consider running more NEPS evaluations or adjusting fidelity thresholds

## Example Analysis Output

```
=== NEPS FIDELITY CORRELATION ANALYSIS REPORT ===
Analysis Date: 2025-08-03 14:30:00

📊 DATA SUMMARY
Total Evaluations: 85
Low Fidelity Samples (≤0.3): 23
High Fidelity Samples (≥0.7): 19

🔍 CORRELATION ANALYSIS
Method: Spearman Correlation
Correlation Coefficient: 0.6847
P-value: 0.0023
Statistical Significance: Yes (α=0.05)

📈 INTERPRETATION
Strong correlation - low fidelity is a good predictor

💡 RECOMMENDATIONS
✅ Strong correlation detected - your fidelity strategy is working well!
• Low fidelity evaluations are reliable predictors of high fidelity performance
• Consider more aggressive early stopping to save computational resources
• Successive halving with your current data_fraction range is well-suited
```

This implementation provides a complete solution for understanding and optimizing your multi-fidelity NEPS setup!
