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

## Running the code

1. The code using Hydra for configuration management. The configuration for training is located in config/train.yaml.

#### Running the training script

```bash
python run.py 
```

This will run the training script with the configuration specified in config/train.yaml. You can modify the configuration file to change the training parameters.

We use hpo.py for hyperparameter optimization. The configuration for HPO is located in config/neps.yaml.

#### Running the HPO script

```bash
python hpo.py
```


### Files in util folder 



### Multi Task Learning

We leverage the `MultiTaskLearning` class to train a model on multiple tasks. Here is an image explaining it:
![Multi Task Learning](https://raw.githubusercontent.com/waasiq/nlp-classifier-automl/main/img/multi_task_learning.png)
