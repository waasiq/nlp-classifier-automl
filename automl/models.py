import numpy as np
import torch
import torch.nn as nn
from transformers import DistilBertModel
from transformers import RobertaModel


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dims: dict[str, int]):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        # self.dropout = nn.Dropout(0.3)
        self.heads   = nn.ModuleDict({
            t: nn.Linear(2*hidden_dim, c)           # 2* for bi‑directional
            for t, c in output_dims.items()
        })

    def forward(self, task, **kwargs):
        ids      = kwargs["input_ids"].long()          # <- cast
        lengths  = kwargs["lengths"].cpu()             # safer for packing

        x = self.embed(ids)                            # embed once
        x = nn.utils.rnn.pack_padded_sequence(
                x, lengths, batch_first=True, enforce_sorted=False
            )
        _, (h, _) = self.lstm(x)
        h = torch.cat((h[0], h[1]), dim=-1)            # bi‑LSTM
        return self.heads[task](h)


class CustomClassificationHead(nn.Module):
    def __init__(self, input_size, num_classes, dropout_prob=0.1):
        super(CustomClassificationHead, self).__init__()
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(input_size, num_classes) 

    def forward(self, features):
        x = self.dropout(features)
        logits = self.classifier(x)
        return logits

class TransformerMultiTaskClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, task_num_classes: dict,
                 common_layer_hidden_size: int = 768, common_dropout_prob: float = 0.1,
                 num_bert_layers_to_unfreeze: int = 0,
                 head_dropout_prob: float = 0.1
                ):
        super().__init__()
        self.backbone = backbone

        for param in self.backbone.parameters():
            param.requires_grad = False

        layer_module = None
        if isinstance(self.backbone, RobertaModel):
            layer_module = self.backbone.encoder.layer
        elif isinstance(self.backbone, DistilBertModel):
            layer_module = self.backbone.transformer.layer
        else:
            raise ValueError("Unsupported backbone type for unfreezing logic.")

        if num_bert_layers_to_unfreeze > 0:
            total_layers = len(layer_module)
            for i, layer in enumerate(layer_module):
                if i >= (total_layers - num_bert_layers_to_unfreeze):
                    for param in layer.parameters():
                        param.requires_grad = True
                    print(f"Unfreezing layer {i} of {self.backbone.config.model_type}")

        bert_output_size = self.backbone.config.hidden_size

        self.shared_common_backbone = nn.Sequential(
            nn.Linear(bert_output_size, common_layer_hidden_size),
            nn.ReLU(),
            nn.Dropout(common_dropout_prob),
            nn.Linear(common_layer_hidden_size, common_layer_hidden_size),
            nn.ReLU(),
            nn.Dropout(common_dropout_prob)
        )

        self.classification_heads = nn.ModuleDict()
        for task_name, num_classes in task_num_classes.items():
            self.classification_heads[task_name] = CustomClassificationHead(
                input_size=common_layer_hidden_size,
                num_classes=num_classes,
                dropout_prob=head_dropout_prob
            )

    def forward(self, input_ids, attention_mask, task_name):
        outputs = self.backbone(input_ids=input_ids,
                                attention_mask=attention_mask,
                                return_dict=True)
        cls_output = outputs.last_hidden_state[:, 0, :]
        shared_features = self.shared_common_backbone(cls_output)
        logits = self.classification_heads[task_name](shared_features)
        return logits

