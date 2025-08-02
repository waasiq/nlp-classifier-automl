import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from automl.models import LSTMClassifier, TransformerMultiTaskClassifier
from automl.utils import SimpleTextDataset
from pathlib import Path
import logging
from typing import Tuple
from collections import Counter
from automl.normalization import remove_markdowns
from itertools import chain
import random

try:
    from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)

class TextAutoML:
    def __init__(
        self,
        seed=42,
        approach='auto',
        vocab_size=10000,
        token_length=128,
        epochs=5,
        batch_size=64,
        lr=1e-4,
        weight_decay=0.0,
        model_name="distilroberta-base",
        ffnn_hidden=128,
        lstm_emb_dim=128,
        lstm_hidden_dim=128,
        fraction_layers_to_finetune: float=1.0,
    ):
        self.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.model_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.approach = approach
        self.vocab_size = vocab_size
        self.token_length = token_length
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay

        self.ffnn_hidden = ffnn_hidden
        self.lstm_emb_dim = lstm_emb_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.fraction_layers_to_finetune = fraction_layers_to_finetune

        self.model = None
        self.tokenizer = None
        self.vectorizer = None
        self.num_classes = None
        self.train_texts = []
        self.train_labels = []
        self.val_texts = []
        self.val_labels = []

    def fit(
        self,
        train_dfs: dict[str, pd.DataFrame],
        val_dfs: dict[str, pd.DataFrame],
        num_classes: dict[str, int],
        approach=None,
        vocab_size=None,
        token_length=None,
        epochs=None,
        batch_size=None,
        lr=None,
        weight_decay=None,
        ffnn_hidden=None,
        lstm_emb_dim=None,
        lstm_hidden_dim=None,
        fraction_layers_to_finetune=None,
        load_path: Path=None,
        save_path: Path=None,
    ):
        """
        Fits a model to the given dataset.

        Parameters:
        - train_df (pd.DataFrame): Training data with 'text' and 'label' columns.
        - val_df (pd.DataFrame): Validation data with 'text' and 'label' columns.
        - num_classes (int): Number of classes in the dataset.
        - seed (int): Random seed for reproducibility.
        - approach (str): Model type - 'lstm', or 'transformer'. Default is 'auto'.
        - vocab_size (int): Maximum vocabulary size.
        - token_length (int): Maximum token sequence length.
        - epochs (int): Number of training epochs.
        - batch_size (int): Batch size for training.
        - lr (float): Learning rate.
        - weight_decay (float): Weight decay for optimizer.
        - ffnn_hidden (int): Hidden dimension size for FFNN.
        - lstm_emb_dim (int): Embedding dimension size for LSTM.
        - lstm_hidden_dim (int): Hidden dimension size for LSTM.
        """
        if approach is not None: self.approach = approach
        if vocab_size is not None: self.vocab_size = vocab_size
        if token_length is not None: self.token_length = token_length
        if epochs is not None: self.epochs = epochs
        if batch_size is not None: self.batch_size = batch_size
        if lr is not None: self.lr = lr
        if weight_decay is not None: self.weight_decay = weight_decay
        if ffnn_hidden is not None: self.ffnn_hidden = ffnn_hidden
        if lstm_emb_dim is not None: self.lstm_emb_dim = lstm_emb_dim
        if lstm_hidden_dim is not None: self.lstm_hidden_dim = lstm_hidden_dim
        if fraction_layers_to_finetune is not None: self.fraction_layers_to_finetune = fraction_layers_to_finetune
        
        logger.info("Loading and preparing data...")

        self.train_texts = {dataset: train_df['text'].tolist() for dataset, train_df in train_dfs.items()}
        self.train_labels = {dataset: train_df['label'].tolist() for dataset, train_df in train_dfs.items()}
        self.val_texts = {dataset: val_df['text'].tolist() for dataset, val_df in val_dfs.items()}
        self.val_labels = {dataset: val_df['label'].tolist() for dataset, val_df in val_dfs.items()}
        self.num_classes = num_classes
        train_class_dist = {dataset: Counter(train_label) for dataset, train_label  in self.train_labels.items()}
        val_class_dist = {dataset: Counter(val_label) for dataset, val_label  in self.val_labels.items()}
        logger.info(f"Train class distribution: {train_class_dist}")
        logger.info(f"Val class distribution: {val_class_dist}")

        dataset = None
        if self.approach in ['lstm', 'transformer']:
            if self.approach == 'transformer':
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)

            self.vocab_size = self.tokenizer.vocab_size
            datasets = {
                task: SimpleTextDataset(
                    self.train_texts[task], self.train_labels[task], self.tokenizer, self.token_length
                ) for task in self.train_texts.keys()
            }
            train_loaders = {
                task:
                DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
                for task, dataset in datasets.items()
            }
            _datasets = {
                task: SimpleTextDataset(
                    self.val_texts[task], self.val_labels[task], self.tokenizer, self.token_length
                ) for task in self.val_texts.keys()
            }
            val_loaders = {
                task: DataLoader(_dataset, batch_size=self.batch_size, shuffle=True)
                    for task, _dataset in _datasets.items()
            }

            match self.approach:
                case "lstm":
                    self.model = LSTMClassifier(
                        len(self.tokenizer),
                        self.lstm_emb_dim,
                        self.lstm_hidden_dim,
                        self.num_classes
                    )
                case "transformer":
                    if TRANSFORMERS_AVAILABLE:
                        task_num_classes = {
                                "dbpedia": 14,
                                "ag_news": 4,
                                "amazon": 5,
                                "imdb": 2,
                                "yelp": 5
                        }

                        backbone = AutoModel.from_pretrained(self.model_name)

                        self.model = TransformerMultiTaskClassifier(
                            backbone=backbone,
                            task_num_classes=task_num_classes,
                            num_bert_layers_to_unfreeze=2 
                        )
                    else:
                        raise ValueError(
                            "Need transformer package."
                        )
                case _:
                    raise ValueError("Unsupported approach or missing transformers.")
        else:
            raise ValueError(f"Unrecognized approach: {self.approach}")
        
        # Training and validating
        self.model.to(self.device)
        # assert dataset is not None, f"`dataset` cannot be None here!"
        # loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        val_acc = self._train_loop(
            train_loaders,
            val_loaders,
            load_path=load_path,
            save_path=save_path,
        )

        return 1 - val_acc

    def _train_loop(
        self, 
        train_loaders: dict[str, DataLoader],
        val_loaders: dict[str, DataLoader],
        load_path: Path=None,
        save_path: Path=None,
    ):
        if self.approach == "transformer":
            head_params = chain(
                            self.model.shared_common_backbone.parameters(),
                            self.model.classification_heads.parameters()
                        )
            optimizer_grouped_parameters = [
                {"params": self.model.backbone.parameters(), "lr": 5e-5}, # BERT backbone LR
                {"params": head_params, "lr": 2e-5}                       # Custom head LR
            ]
            optimizer = torch.optim.AdamW(optimizer_grouped_parameters, weight_decay=self.weight_decay)
            num_training_steps = sum(len(loader) for loader in train_loaders.values()) * self.epochs
            num_warmup_steps = int(num_training_steps * 0.1) 

            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps
            )
        else:
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        criterion = nn.CrossEntropyLoss()

        start_epoch = 0
        # handling checkpoint resume
        if load_path is not None:
            _states = torch.load(load_path / "checkpoint.pth", map_location='cpu')  # Load to CPU first
            self.model.load_state_dict(_states["model_state_dict"])
            optimizer.load_state_dict(_states["optimizer_state_dict"])
            start_epoch = _states["epoch"]
            logger.info(f"Resuming from checkpoint at {start_epoch}")

        for epoch in range(start_epoch, self.epochs):            
            train_iters = {task: iter(loader) for task, loader in train_loaders.items()}
            total_loss = 0
            while len(train_iters.keys()) > 0:
                task = random.choice(list(train_iters))
                try:
                    batch = next(train_iters[task])
                except StopIteration as e:
                    logger.info(f"completed one iteration over {task} datset. {repr(e)}")
                    train_iters.pop(task, None)
                    continue
                        
                self.model.train()
                optimizer.zero_grad()

                # if isinstance(batch, dict):
                if isinstance(self.model, AutoModel):
                    inputs = {k: v.to(self.device) for k, v in batch.items()}
                    outputs = self.model(**inputs)
                    loss = outputs.loss
                    labels = inputs['labels']
                else:
                    match self.approach:
                        case "lstm":
                            inputs = {k: v.to(self.device) for k, v in batch.items()}
                            outputs = self.model(**inputs, task=task)
                            labels = inputs["labels"]
                        case "transformer":
                            inputs = {k: v.to(self.device) for k, v in batch.items()}
                            model_inputs = {
                                'input_ids': inputs['input_ids'],
                                'attention_mask': inputs['attention_mask']
                            }
                            outputs = self.model(**model_inputs, task_name=task)
                            labels = inputs["labels"]
                        case _:
                            raise ValueError("Oops! Wrong approach.")

                    #outputs = outputs.logits if self.approach == "transformer" else outputs
                    loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                if scheduler:
                    scheduler.step()
                    
                total_loss += loss.item()

            logger.info(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")
            
            if self.val_texts:
                for task in train_loaders.keys():
                    val_preds, val_labels = self._predict(val_loaders[task], task)
                    val_acc = accuracy_score(val_labels, val_preds)
                    logger.info(f"Epoch {epoch + 1}, Validation Accuracy for dataset {task} is: {val_acc:.4f}")

        if save_path is not None:
            save_path = Path(save_path) if not isinstance(save_path, Path) else save_path
            save_path.mkdir(parents=True, exist_ok=True)

            torch.save(
                {
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                },
                save_path / "checkpoint.pth"
            )   
        torch.cuda.empty_cache()
        return val_acc or 0.0

    def _predict(self, val_loader: DataLoader, task):
        self.model.eval()
        preds = []
        labels = []
        with torch.no_grad():
            for batch in val_loader:
                if isinstance(self.model, AutoModel):
                    inputs = {k: v.to(self.device) for k, v in batch.items() if k != 'labels'}
                    outputs = self.model(**inputs).logits
                    labels.extend(batch["labels"])
                else:
                    match self.approach:
                        case "lstm" "transformer":
                            inputs = {k: v.to(self.device) for k, v in batch.items()}
                            outputs = self.model(**inputs, task=task)
                            labels.extend(inputs["labels"])
                        case "transformer":
                            inputs = {k: v.to(self.device) for k, v in batch.items()}
                            model_inputs = {
                                'input_ids': inputs['input_ids'],
                                'attention_mask': inputs['attention_mask']
                            }
                            outputs = self.model(**model_inputs, task_name=task)
                            labels.extend(inputs["labels"])
                        case _:
                            raise ValueError("Oops! Wrong approach.")
                            
                preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
        
        if isinstance(preds, list):
            preds = [p.item() for p in preds]
            labels = [l.item() for l in labels]
            return np.array(preds), np.array(labels)
        else:
            return preds.cpu().numpy(), labels.cpu().numpy()


    def predict(self, test_data: pd.DataFrame | DataLoader, task) -> Tuple[np.ndarray, np.ndarray]:

        assert isinstance(test_data, DataLoader) or isinstance(test_data, pd.DataFrame), \
            f"Input data type: {type(test_data)}; Expected: pd.DataFrame | DataLoader"

        if isinstance(test_data, DataLoader):
            return self._predict(test_data, task)
        
        if self.approach in ['lstm', 'transformer']:
            _dataset = SimpleTextDataset(
                test_data['text'].tolist(),
                test_data['label'].tolist(),
                self.tokenizer,
                self.token_length
            )
            _loader = DataLoader(_dataset, batch_size=self.batch_size, shuffle=True)
        else:
            raise ValueError(f"Unrecognized approach: {self.approach}")
            # handling any possible tokenization
        
        return self._predict(_loader, task)
