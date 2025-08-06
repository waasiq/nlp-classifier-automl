# Multi Task Learning (MTL) Classifier for NLP Datasets

## Environment Setup

To install the repository, create a conda environment . 

**Conda Environment**

Can also use `conda`, left to individual preference.

```bash
conda create -n automl-text-env python=3.10
conda activate automl-text-env
```

Then install the repository by running the following command:

```bash
pip install -e .
```

## Datasets

You should place your datasets in the `data` folder. The datasets should be in CSV format with the following columns:
- `text`: The text data for classification.
- `label`: The label for the text data.

Example dataset structure:

```data/
├── yelp  
  ├── test.csv
  ├── train.csv
```

## Running the code

The code using Hydra for configuration management. The configuration for training is located in config/train.yaml.

```bash
python run.py 
```

This will run the training script with the configuration specified in config/train.yaml. You can modify the configuration file to change the training parameters.

We use hpo.py for hyperparameter optimization. The configuration for HPO is located in config/neps.yaml.

```bash
python hpo.py
```

### Files in util folder 

The util folder contains utility functions for the project. The main files are:
- `hpo_importance.py`: Computes the importance of hyperparameters using SHAP.
- `hpo_viz.py`: Generates the visualization for HPOs and also saves top HPO trials in the HPO_viz.
- `scaling_top_trials.py`: Reads top configs from CSV file and runs then runs it for complete data fracton.
- `run_bo_confs_for_low_fidelity.py`: Runs the Bayesian Optimization for low fidelity configurations.
- `hpo_with_fidelity_tracking.py`: File which keeps saving fidelity tracking json was used by one of the team members.
- `analyze_fidelity_correlation.py`: Analyzes the correlation between low and hihg fidelity using Spearman correlation.

### Multi Task Learning

We leverage the `MultiTaskLearning` class to train a model on multiple tasks. Here is an image explaining it:
![Multi Task Learning](https://raw.githubusercontent.com/waasiq/nlp-classifier-automl/refs/heads/main/img/multi_task_learning.png)


### Team Members

1. Nastaran Alipour 
2. Waasiq Masood
