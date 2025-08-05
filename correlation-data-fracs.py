# Correlation

import pandas as pd
import numpy as np
import logging
from pathlib import Path
import sys
import subprocess
from hydra import compose, initialize
from omegaconf import OmegaConf
from run import main_loop, load_dataset
import yaml

logger = logging.getLogger(__name__)

ALL_DATASETS = ["ag_news", "imdb", "amazon", "dbpedia", "yelp"]

def read_top5_configs(csv_path: str = "top5.csv"):
    try:
        df = pd.read_csv(csv_path)
        logger.info(f"Successfully loaded {len(df)} configurations from {csv_path}")
        return df
    except Exception as e:
        logger.error(f"Error reading {csv_path}: {e}")
        return None

def run_config_on_all_datasets(config_row, base_args, results_dir):
    logger.info(f"Running config {config_row['config_name']} on mtl")
    
    epochs = int(config_row['epochs'])
    batch_size = int(config_row['batch_size'])
    lr = float(config_row['lr'])

    run_output_dir = results_dir / f"config_{config_row['config_name']}" / "mtl"
    run_output_dir.mkdir(parents=True, exist_ok=True)
    
    run_args = base_args.copy()
    run_args.update({
        'dataset': 'ag_news',  
        'approach': 'lstm', 
        'epochs': epochs,
        'batch_size': batch_size,
        'lr': lr,
        'output_path': str(run_output_dir),
        'data_fraction': 0.7,
        'is_mtl': True  
    })
    
    try:
        dataset_classes, train_dfs, val_dfs, test_dfs, num_classes = load_dataset(
            dataset=run_args["dataset"],  
            data_path=Path(run_args["data_path"]).absolute(),
            seed=run_args["seed"],
            val_size=run_args["val_size"],
            is_mtl=run_args["is_mtl"]  
        )
        
        # Run the main loop with LSTM approach on ALL datasets
        result = main_loop(
            train_dfs=train_dfs,
            val_dfs=val_dfs,
            test_dfs=test_dfs,
            num_classes=num_classes,
            dataset_classes=dataset_classes,
            data_fraction=run_args["data_fraction"],
            output_path=Path(run_args["output_path"]).absolute(),
            seed=run_args["seed"],
            approach=run_args["approach"], 
            vocab_size=run_args["model_config"]["vocab_size"],
            token_length=run_args["token_length"],
            epochs=run_args["epochs"],
            batch_size=run_args["batch_size"],
            lr=run_args["lr"],
            weight_decay=run_args["weight_decay"],
            ffnn_hidden=run_args["ffnn_hidden_layer_dim"],
            lstm_emb_dim=run_args["model_config"]["lstm_emb_dim"],
            lstm_hidden_dim=run_args["model_config"]["lstm_hidden_dim"],
            load_path=Path(run_args["load_path"]) if run_args["load_path"] else None,
            model_name=run_args["model_name"],
            bert_lr=run_args["bert_lr"],
            num_bert_layers_to_unfreeze=run_args["num_bert_layers_to_unfreeze"]
        )
        
        logger.info(f"Successfully completed config {config_row['config_name']} on datasets")
        return result
        
    except Exception as e:
        logger.error(f"Error running config {config_row['config_name']} on datasets: {e}")
        return None
        
        logger.info(f"Successfully completed config {config_row['config_name']} on {dataset_name}")
        return result
        
    except Exception as e:
        logger.error(f"Error running config {config_row['config_name']} on {dataset_name}: {e}")
        return None

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    top5_df = read_top5_configs()
    if top5_df is None:
        logger.error("Failed to load configurations. Exiting.")
        return
    
    with initialize(config_path="./configs", version_base="1.3"):
        cfg = compose(config_name="train")
    base_args = OmegaConf.to_object(cfg)
    
    results_dir = Path("correlation_results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = results_dir / "correlation_run.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))
    logging.getLogger().addHandler(file_handler)
    
    logger.info(f"Starting correlation analysis with {len(top5_df)} configurations on ALL datasets combined (MTL)")
    
    all_results = []
    
    for idx, config_row in top5_df.iterrows():
        config_results = {
            'config_name': config_row['config_name'],
            'original_objective': config_row['objective_to_minimize'],
            'epochs': config_row['epochs'],
            'batch_size': config_row['batch_size'],
            'lr': config_row['lr'],
            'all_datasets_result': {}
        }
        
        logger.info(f"=" * 60)
        logger.info(f"Running Configuration {config_row['config_name']} on ALL DATASETS (MTL)")
        logger.info(f"=" * 60)
        
        result = run_config_on_all_datasets(config_row, base_args, results_dir)
        
        if result is not None:
            config_results['all_datasets_result'] = {
                'objective_to_minimize': result.get('objective_to_minimize', float('inf')),
                'info_dict': result.get('info_dict', {})
            }
        else:
            config_results['all_datasets_result'] = {
                'objective_to_minimize': float('inf'),
                'info_dict': {}
            }
        
        all_results.append(config_results)
    
    summary_file = results_dir / "summary_results.yaml"
    with open(summary_file, 'w') as f:
        yaml.safe_dump(all_results, f, default_flow_style=False)
    
    logger.info(f"All runs completed! Results saved to {results_dir}")
    logger.info(f"Summary results saved to {summary_file}")
    
    print("\n" + "="*80)
    print("CORRELATION ANALYSIS SUMMARY")
    print("="*80)
    
    for config_result in all_results:
        print(f"\nConfiguration {config_result['config_name']}:")
        print(f"  Original HPO Objective: {config_result['original_objective']:.4f}")
        print(f"  Hyperparameters: epochs={config_result['epochs']}, batch_size={config_result['batch_size']}, lr={config_result['lr']}")
        print("  All Datasets Combined Result:")
        
        result = config_result['all_datasets_result']
        obj_val = result['objective_to_minimize']
        if obj_val != float('inf'):
            print(f"    Combined MTL: objective={obj_val:.4f}")
            for dataset in ALL_DATASETS:
                roc_key = f"{dataset}_roc_auc_score"
                if roc_key in result['info_dict']:
                    print(f"      {dataset:>10} ROC AUC: {result['info_dict'][roc_key]:.4f}")
            if 'test_mean_roc_auc' in result['info_dict']:
                print(f"      {'Mean Test':>10} ROC AUC: {result['info_dict']['test_mean_roc_auc']:.4f}")
        else:
            print(f"    Combined MTL: FAILED")
    
    print("="*80)

if __name__ == "__main__":
    main()
