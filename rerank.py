import argparse
import glob
import json
import logging
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    AlbertConfig,
    AlbertForSequenceClassification,
    AlbertTokenizer,
    BertConfig,
    BertForSequenceClassification,
    BertTokenizer,
    DistilBertConfig,
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    RobertaConfig,
    RobertaForSequenceClassification,
    RobertaTokenizer,
    XLMConfig,
    XLMForSequenceClassification,
    XLMRobertaConfig,
    XLMRobertaForSequenceClassification,
    XLMRobertaTokenizer,
    XLMTokenizer,
    XLNetConfig,
    XLNetForSequenceClassification,
    XLNetTokenizer,
    get_linear_schedule_with_warmup,
)

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    "bert": (BertConfig, BertForSequenceClassification, BertTokenizer),
    "xlnet": (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    "xlm": (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
    "roberta": (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    "distilbert": (
        DistilBertConfig,
        DistilBertForSequenceClassification,
        DistilBertTokenizer,
    ),
}


class Rerank:
    model = None
    tokenizer = None
    device = None
    model_type = None

    def __init__(
        self,
        model,
        device="cuda",
        model_type="bert",
    ):

        # Load a trained model and vocabulary that you have fine-tuned
        config_class, model_class, tokenizer_class = MODEL_CLASSES[model_type]
        self.model = model_class.from_pretrained(model)
        self.tokenizer = tokenizer_class.from_pretrained(model)
        self.model.to(device)
        self.device = device
        self.model_type = model_type

    # Functions to do inference
    def tokenize(self, query, sentences, max_length=128, DEFAULT_LABEL=0):
        all_input_ids, all_attention_mask, all_token_type_ids, all_labels = (
            [],
            [],
            [],
            [],
        )
        for s in sentences:
            tok = self.tokenizer.encode_plus(
                text=query, text_pair=s, max_length=128, pad_to_max_length=True
            )
            all_input_ids.append(tok["input_ids"])
            all_attention_mask.append(tok["attention_mask"])
            all_token_type_ids.append(tok["token_type_ids"])
            all_labels.append(DEFAULT_LABEL)

        all_input_ids = torch.tensor(all_input_ids, dtype=torch.long)
        all_attention_mask = torch.tensor(all_attention_mask, dtype=torch.long)
        all_token_type_ids = torch.tensor(all_token_type_ids, dtype=torch.long)
        all_labels = torch.tensor(all_labels, dtype=torch.long)

        dataset = TensorDataset(
            all_input_ids, all_attention_mask, all_token_type_ids, all_labels
        )
        return dataset

    def inference(self, query, sentences):
        dataset = self.tokenize(query, sentences)
        preds, out_label_ids = None, None
        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=16)
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for batch in tqdm(eval_dataloader):
            self.model.eval()
            batch = tuple(t.to(self.device) for t in batch)

            with torch.no_grad():
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "labels": batch[3],
                }
                inputs["token_type_ids"] = (
                    batch[2] if self.model_type in ["bert", "xlnet"] else None
                )  # XLM, DistilBERT and RoBERTa don't use segment_ids
            outputs = self.model(**inputs)
            tmp_eval_loss, logits = outputs[:2]

            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs["labels"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(
                    out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0
                )
        s = torch.nn.Softmax(dim=1)  # just to make it easier to predict
        preds = s(torch.tensor(preds))
        return preds

    def rerank(self, query: str, sentences: list, topn=5):
        query = query.replace(r"?", "")
        n_candidates = min(topn, len(sentences))
        preds = self.inference(query, sentences)
        top_n = np.argsort(np.array(preds[:, 1]))

        top_sents = []
        for n in range(1, n_candidates + 1):
            ts = sentences[top_n[-n]]
            top_sents.append(ts)

        sorted_preds = np.array(preds[:, 1])
        sorted_preds.sort()
        sorted_preds = sorted_preds[::-1]

        to_return = {
            "query": query,
            "top_sentences": top_sents,
            "preds": sorted_preds[:topn],
        }

        return to_return
