from transformers import AutoTokenizer
from transformers import AutoModel

import pandas as pd
import numpy as np
from tqdm.auto import tqdm

import torch
from torch import optim
from torch import nn

import pytorch_lightning as pl
from pytorch_lightning.tuner.tuning import Tuner
from torchmetrics import Accuracy

from typing import Dict
from common_utils import asp_mapper, sent_mapper
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

model_name = "DeepPavlov/rubert-base-cased"
bert = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["token"], is_split_into_words=True,
        padding='max_length', max_length=400
    )

    for col in ["sentiment", "token", "aspect", "begin", "end"]:
        try:
            all_ids = []
            for i, label in enumerate(examples["token"]):
                word_ids = tokenized_inputs.word_ids(batch_index=i)
                previous_word_idx = None
                ids = []
                for word_idx in word_ids:
                    blank = -100 if col != "token" else "-100"
                    if word_idx is None:
                        ids.append(blank)
                    elif word_idx != previous_word_idx:
                        ids.append(examples[col][i][word_idx])
                    else:
                        ids.append(blank)
                    previous_word_idx = word_idx
                all_ids.append(ids)
            tokenized_inputs[col] = all_ids
        except:
            pass
    
    return tokenized_inputs


class AspectBERT(pl.LightningModule):
    
    loss = nn.CrossEntropyLoss(ignore_index=-100)
    
    def __init__(
        self, model,
        optimizer=optim.SGD,
        optimizer_kwargs=dict(lr=0.1),
        scheduler=None,
        scheduler_kwargs=dict(),
        n_of_aspects=6,
        n_of_sentiments=6,
        weights=[0.33, 0.33, 0.33]
    ):
        
        super().__init__()
        self.save_hyperparameters()
        self.model = model
        self.weights = weights
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.optimizer_kwargs = optimizer_kwargs
        self.scheduler_kwargs = scheduler_kwargs
        
        self.n_of_aspects = n_of_aspects
        self.n_of_sentiments = n_of_sentiments
        self.fc_aspect = nn.Linear(768, n_of_aspects)
        self.fc_sentiment = nn.Linear(768, n_of_sentiments)
        self.fc_clf = nn.Sequential(
            nn.Linear(768, 5),
            nn.Conv1d(400, 6, 1)
        )
        
    def forward(self, *args, **kwargs):
        attn_output = self.model(*args, **kwargs)[0]
        aspects = self.fc_aspect(attn_output)
        sentiments = self.fc_sentiment(attn_output)
        clf = self.fc_clf(attn_output)
        return aspects, sentiments, clf.view(-1, 5, self.n_of_sentiments)
    
    def compute_loss(self, logits, b_labels, num_labels, b_attn_mask=None):
        if b_attn_mask is None:
            loss = self.loss(logits.view(-1, num_labels), b_labels.view(-1))
        else:
            active_loss = b_attn_mask.view(-1) == 1
            active_logits = logits.view(-1, num_labels)
            active_labels = torch.where(
                active_loss, b_labels.view(-1), torch.tensor(self.loss.ignore_index).type_as(b_labels)
            )
            loss = self.loss(active_logits, active_labels.long())
        return loss
    
    def compute_acc(self, pred, true, num_classes, what="multiclass"):
        
        scorer = Accuracy(
            what, num_classes=num_classes, ignore_index=num_classes-1
        ).to(self.device)
        mask = true != -100
        pred = nn.Softmax(dim=-1)(pred).argmax(-1)[mask]
        true = true[mask]
        
        return {"accuracy": scorer(pred, true), "pred": pred, "true": true} 
    
    def step(
        self, batch, batch_idx, subset="train", log=True, **log_params
    ):

        aspects, sentiments, clf, tokens, token_type_ids, attn_mask = batch.values()
        tokens, aspects, sentiments = tokens.long(), aspects.long(), sentiments.long()
        pred_aspects, pred_sentiments, pred_clf = self(
            input_ids=tokens,
            token_type_ids=token_type_ids,
            attention_mask=attn_mask
        )
        
        aspect_loss = self.compute_loss(pred_aspects, aspects, self.n_of_aspects)
        aspect_acc = self.compute_acc(pred_aspects, aspects, self.n_of_aspects)
                
        sentiment_loss = self.compute_loss(pred_sentiments, sentiments, self.n_of_sentiments)
        sentiment_acc = self.compute_acc(pred_sentiments, sentiments, self.n_of_sentiments)
        
        clf_loss = self.compute_loss(pred_clf, clf, self.n_of_sentiments)
        clf_acc = self.compute_acc(pred_clf, clf, self.n_of_sentiments)

        loss = (self.weights[0]*aspect_loss +
                self.weights[1]*sentiment_loss +
                self.weights[2]*clf_loss)
        res = {
            "loss": loss,
            "aspect_accuracy": aspect_acc["accuracy"],
            "sentiment_accuracy": sentiment_acc["accuracy"],
            "clf_accuracy": clf_acc["accuracy"],
        }
            
        if log:
            self.log(f"{subset}_loss", loss, **log_params)
            self.log(f"{subset}_asp_acc", aspect_acc["accuracy"], **log_params)
            self.log(f"{subset}_sent_acc", sentiment_acc["accuracy"], **log_params)
            self.log(f"{subset}_clf_acc", clf_acc["accuracy"], **log_params)
        else:
            res.update({
                "aspects_true": aspect_acc["true"],
                "aspects_pred": aspect_acc["pred"],
                "sentiments_true": sentiment_acc["true"],
                "sentiments_pred": sentiment_acc["pred"],
                "clf_true": clf_acc["true"],
                "clf_pred": clf_acc["pred"],
            })

        return res
        
    def training_step(self, batch, batch_idx):
        
        return self.step(
            batch, batch_idx, "train",
            on_step=False, on_epoch=True,
            prog_bar=True, logger=True
        )
    
    def validation_step(self, batch, batch_idx):
        
        return self.step(
            batch, batch_idx, "valid",
            on_step=False, on_epoch=True,
            prog_bar=True, logger=True
        )
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self.step(batch, batch_idx, log=False)

    def configure_optimizers(self):
        
        config = {}
        optimizer = self.optimizer(
            self.parameters(), **self.optimizer_kwargs
        )
        config["optimizer"] = optimizer
        
        if self.scheduler is not None:
            scheduler = self.scheduler(
                optimizer, **self.scheduler_kwargs
            )
            config["lr_scheduler"] = scheduler
            
        return config
    
    
def train(
    optimizer_params: Dict = None,
    training_arguments: Dict = None,
    *trainer_args
):
    model = AspectBERT(bert, **optimizer_params)
    trainer = pl.Trainer(**training_arguments)
    trainer.fit(model, *trainer_args)
    

def predict_step(model, batch):
    pred_aspects, pred_sentiments, pred_clf = model(
        input_ids=batch["input_ids"],
        token_type_ids=batch["token_type_ids"],
        attention_mask=batch["attention_mask"]
    )
    # pred_aspects[pred_aspects != -100]
    # pred_sentiments[pred_sentiments != -100]
    # pred_clf[pred_clf != -100]
    pred_aspects = nn.Softmax(dim=-1)(pred_aspects).argmax(-1)
    pred_sentiments = nn.Softmax(dim=-1)(pred_sentiments).argmax(-1)
    pred_clf = nn.Softmax(dim=-1)(pred_clf).argmax(-1)
    return {
        "aspects_pred": pred_aspects[batch["begin"] != -100],
        "sentiments_pred": pred_sentiments[batch["begin"] != -100],
        "clf_pred": pred_clf,
    }
    

def evaluate(
    model_ckpt, loader,
    output_file=None,
):

    model = AspectBERT.load_from_checkpoint(model_ckpt).to("cpu")
    res = [predict_step(model, batch) for batch in loader]
    aspects = ["aspect_pred"]
    sentiments = ["sentiment_pred"]
    clfs = ["clf_pred"]
    bert_res = pd.DataFrame(columns=aspects+sentiments)
    clf_bert = pd.DataFrame(columns=clfs)

    for batch in res:
        batch["clf_pred"] = batch["clf_pred"].reshape(-1, 5)
        for row in tqdm(zip(*list(batch.values())[:-1])):
            bert_res.loc[bert_res.shape[0]] = row
        for row in zip(*list(batch.values())[-1:]):
            clf_bert.loc[clf_bert.shape[0]] = row

    bert_res[aspects+sentiments] = bert_res[aspects+sentiments].applymap(lambda x: x.item())
    bert_res[aspects] = bert_res[aspects].applymap(
        lambda x: {v: k for k, v in asp_mapper.items()}[x]
    )
    bert_res[sentiments] = bert_res[sentiments].applymap(
        lambda x: {v: k for k, v in sent_mapper.items()}[x]
    )
    clf_bert[clfs] = clf_bert[clfs].applymap(
        lambda x: [y.item() for y in x]
    )
    clf_bert[clfs] = clf_bert[clfs].applymap(
        lambda x: list(map({v: k for k, v in sent_mapper.items()}.get, x))
    )
    
    return bert_res, clf_bert