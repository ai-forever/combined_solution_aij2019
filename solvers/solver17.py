# qbic

import attr
import numpy as np
import random
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import wraps
from torch.utils.data import DataLoader
from transformers.optimization import AdamW, WarmupLinearSchedule

from solvers.torch_helpers import (
    ModelTrainer,
    Field,
    BatchCollector,
    AccuracyCounter,
    init_bert,
    load_bert,
    load_bert_tokenizer,
)
from solvers.solver_helpers import fix_spaces, singleton


@attr.s(frozen=True)
class Example(object):
    tokens = attr.ib()
    token_ids = attr.ib()
    segment_ids = attr.ib()
    mask = attr.ib()
    position = attr.ib()
    label_id = attr.ib(default=None)

    def __len__(self):
        return len(self.token_ids)


@attr.s(frozen=True)
class TrainConfig(object):
    learning_rate = attr.ib(default=1e-5)
    train_batch_size = attr.ib(default=32)
    test_batch_size = attr.ib(default=32)
    epoch_count = attr.ib(default=10)
    warm_up = attr.ib(default=0.1)


def _get_optimizer(model, train_size, config):
    num_total_steps = int(train_size / config.train_batch_size * config.epoch_count)
    num_warmup_steps = int(num_total_steps * config.warm_up)

    optimizer = AdamW(model.parameters(), lr=config.learning_rate, correct_bias=False)
    scheduler = WarmupLinearSchedule(
        optimizer, warmup_steps=num_warmup_steps, t_total=num_total_steps
    )

    return optimizer, scheduler


def _get_batch_collector(device, is_train=True):
    if is_train:
        vector_fields = [Field("label_id", torch.float32)]
    else:
        vector_fields = []

    vector_fields.append(Field("position", torch.long))

    return BatchCollector(
        matrix_fields=[
            Field("token_ids", np.int64),
            Field("segment_ids", np.int64),
            Field("mask", np.int64),
        ],
        vector_fields=vector_fields,
        device=device,
    )


class RubertClassifier(nn.Module):
    def __init__(self, bert, output_name, bert_output_dim=768):
        super(RubertClassifier, self).__init__()

        self._bert = bert
        self._dropout = nn.Dropout(0.3)
        self._predictor = nn.Linear(bert_output_dim, 1)
        self._output_name = output_name

    def forward(self, batch):
        outputs, pooled_outputs = self._bert(
            input_ids=batch["token_ids"],
            attention_mask=batch["mask"],
            token_type_ids=batch["segment_ids"],
        )

        row_positions = torch.arange(0, outputs.shape[0])
        outputs = outputs[row_positions, batch["position"]]

        status_logits = self._predictor(self._dropout(outputs)).squeeze(-1)

        loss = 0.0
        if self._output_name + "_id" in batch:
            loss = F.binary_cross_entropy_with_logits(
                status_logits, batch[self._output_name + "_id"]
            )

        return {self._output_name + "_logits": status_logits, "loss": loss}


class ClassifierTrainer(ModelTrainer):
    def on_epoch_begin(self, *args, **kwargs):
        super(ClassifierTrainer, self).on_epoch_begin(*args, **kwargs)

        self._accuracies = [AccuracyCounter()]

    def on_epoch_end(self):
        info = super(ClassifierTrainer, self).on_epoch_end()

        return "{}, {}".format(
            info, ", ".join(str(score) for score in self._accuracies)
        )

    def forward_pass(self, batch):
        outputs = self._model(batch)

        predictions = (outputs["label_logits"] > 0.0).float()
        assert predictions.shape == batch["label_id"].shape

        self._accuracies[0].update(predictions, batch["label_id"])

        info = ", ".join(str(score) for score in self._accuracies)
        return outputs["loss"], info


class Solver(object):
    def __init__(self, seed=42, bert_path="data/models/bert/rubert/qbic"):
        self.seed = seed
        self.init_seed()
        self.is_loaded = False
        self.bert_path = bert_path
        self._tokenizer = load_bert_tokenizer(self.bert_path)
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._batch_collector = _get_batch_collector(self._device, is_train=False)
        self._model = RubertClassifier(init_bert(self.bert_path), output_name="label").to(
            self._device
        )

    def init_seed(self):
        return random.seed(self.seed)

    @staticmethod
    def _get_sentences(text):
        for sent in text.split("."):
            if re.search("\(\d+\)", sent):
                yield fix_spaces(sent.strip()) + "."

    def _convert_task(self, task):
        text = task["text"].replace("?", ".").replace("\xa0", "")
        sentence = " ".join(list(self._get_sentences(text)))

        correct = None
        if "solution" in task:
            if "correct" in task["solution"]:
                correct = sorted([int(idx) - 1 for idx in task["solution"]["correct"]])
            else:
                correct = sorted(
                    [int(idx) - 1 for idx in task["solution"]["correct_variants"][0]]
                )

        tokens, positions = ["[CLS]"], []
        prev_match_end = 0
        for match in re.finditer("\(\d+\)", sentence):
            tokens.extend(
                self._tokenizer.tokenize(
                    sentence[prev_match_end : match.start()].strip()
                )
            )
            positions.append(len(tokens))
            tokens.append("[MASK]")
            prev_match_end = match.end()

        tokens.extend(self._tokenizer.tokenize(sentence[prev_match_end:].strip()))
        tokens.append("[SEP]")
        assert all(tokens[position] == "[MASK]" for position in positions)

        token_ids = self._tokenizer.convert_tokens_to_ids(tokens)
        assert token_ids[0] == 101 and token_ids[-1] == 102
        assert len(token_ids) == len(tokens)

        for position_id, position in enumerate(positions):
            label_id = None
            if correct is not None:
                label_id = int(position_id in correct)

            yield Example(
                tokens=tokens,
                token_ids=token_ids,
                segment_ids=[0] * len(token_ids),
                mask=[1] * len(token_ids),
                position=position,
                label_id=label_id,
            )

    def _prepare_examples(self, tasks):
        examples = []
        for task in tasks:
            for example in self._convert_task(task):
                examples.append(example)
        return examples

    def fit(self, tasks, test_tasks=None):
        print("START FIT")
        train_examples = self._prepare_examples(tasks)
        test_examples = None
        if test_tasks is not None:
            test_examples = self._prepare_examples(test_tasks)

        config = TrainConfig()
        self._model = RubertClassifier(load_bert(self.bert_path), output_name="label").to(
            self._device
        )

        batch_collector = _get_batch_collector(self._device, is_train=True)

        train_loader = DataLoader(
            train_examples,
            batch_size=config.train_batch_size,
            shuffle=True,
            collate_fn=batch_collector,
            pin_memory=False,
        )
        test_loader, test_batches_per_epoch = None, 0
        if test_examples is not None:
            test_loader = DataLoader(
                test_examples,
                batch_size=config.test_batch_size,
                shuffle=False,
                collate_fn=batch_collector,
                pin_memory=False,
            )
            test_batches_per_epoch = int(len(test_examples) / config.test_batch_size)

        optimizer, scheduler = _get_optimizer(self._model, len(train_examples), config)

        trainer = ClassifierTrainer(self._model, optimizer, scheduler, use_tqdm=True)
        trainer.fit(
            train_iter=train_loader,
            train_batches_per_epoch=int(len(train_examples) / config.train_batch_size),
            val_iter=test_loader,
            val_batches_per_epoch=test_batches_per_epoch,
            epochs_count=config.epoch_count,
        )

    def save(self, path="data/models/solvers/solver17/punctuator.pkl"):
        torch.save(self._model.state_dict(), path)

    @singleton
    def load(self, path="data/models/solvers/solver17/punctuator.pkl"):
        weights = torch.load(path, map_location=self._device)
        self._model.load_state_dict(weights)
        self._model.eval()
        self.is_loaded = True

    def predict_from_model(self, task):

        examples = list(self._convert_task(task))
        model_inputs = self._batch_collector(examples)

        with torch.no_grad():
            model_prediction = self._model(model_inputs)["label_logits"].cpu().numpy()

        choices = task["question"]["choices"]
        prediction = [
            str(choices[ind]["id"])
            for ind, logit in enumerate(model_prediction)
            if logit > 0.0
        ]

        return prediction
