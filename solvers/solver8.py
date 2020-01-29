# coding: utf-8
# qbic

import attr
import numpy as np
import pymorphy2
import random
import re
import torch
from string import punctuation
from torch.utils.data import DataLoader
from transformers.optimization import AdamW, WarmupLinearSchedule

from solvers.torch_helpers import (
    ModelTrainer,
    Field,
    BatchCollector,
    F1ScoreCounter,
    AccuracyCounter,
    RubertMulticlassClassifier,
    init_bert,
    load_bert,
    load_bert_tokenizer,
)
from solvers.solver_helpers import fix_spaces, AbstractSolver

_ERRORS = [
    "деепричастный оборот",
    "косвенный речь",
    "несогласованный приложение",
    "однородный член",
    "причастный оборот",
    "связь подлежащее сказуемое",
    "сложноподчинённый",
    "сложный",
    "соотнесенность глагольный форма",
    "форма существительное",
    "числительное",
    "None",
]


@attr.s(frozen=True)
class Example(object):
    text = attr.ib()
    token_ids = attr.ib()
    segment_ids = attr.ib()
    mask = attr.ib()
    error_type = attr.ib()
    error_type_id = attr.ib()

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
        vector_fields = [Field("error_type_id", torch.long)]
    else:
        vector_fields = []

    return BatchCollector(
        matrix_fields=[
            Field("token_ids", np.int64),
            Field("segment_ids", np.int64),
            Field("mask", np.int64),
        ],
        vector_fields=vector_fields,
        device=device,
    )


class ClassifierTrainer(ModelTrainer):
    def on_epoch_begin(self, *args, **kwargs):
        super(ClassifierTrainer, self).on_epoch_begin(*args, **kwargs)

        self._f1_scores = [
            F1ScoreCounter(str(i), compact=True) for i in range(len(_ERRORS) - 1)
        ]
        self._accuracies = [
            AccuracyCounter(),
            AccuracyCounter(masked_values=[len(_ERRORS) - 1]),
        ]

    def on_epoch_end(self):
        info = super(ClassifierTrainer, self).on_epoch_end()

        return "{}, {}".format(
            info, ", ".join(str(score) for score in self._f1_scores + self._accuracies)
        )

    def forward_pass(self, batch):
        outputs = self._model(batch)

        predictions = outputs["error_type_logits"].argmax(-1)
        assert predictions.shape == batch["error_type_id"].shape

        for i in range(len(_ERRORS) - 1):
            self._f1_scores[i].update(predictions == i, batch["error_type_id"] == i)

        self._accuracies[0].update(predictions, batch["error_type_id"])
        self._accuracies[1].update(predictions, batch["error_type_id"])

        info = ", ".join(str(score) for score in self._f1_scores + self._accuracies)
        return outputs["loss"], info


class Solver(AbstractSolver):
    def __init__(self, bert_path="data/models/bert/rubert/qbic"):
        self.is_loaded = False
        self.bert_path = bert_path
        self._model = RubertMulticlassClassifier(
            init_bert(self.bert_path),
            class_count=len(_ERRORS),
            output_name="error_type",
        )
        self._tokenizer = load_bert_tokenizer(self.bert_path)
        self.morph = pymorphy2.MorphAnalyzer()
        use_cuda = torch.cuda.is_available()
        self._device = torch.device("cuda:0" if use_cuda else "cpu")
        self._batch_collector = _get_batch_collector(self._device, is_train=False)

    def normalize_category(self, cond):
        condition = cond["text"].lower().strip(punctuation)
        condition = re.sub(r"([а-я])\)", r"\1) ", condition).replace("ё", "е")
        condition = re.sub("[a-дabв]\)\s", "", condition).replace("членами.", "член")
        norm_cat = ""
        for token in condition.split():
            lemma = self.morph.parse(token)[0].normal_form
            if lemma not in [
                "неправильный",
                "построение",
                "предложение",
                "с",
                "ошибка",
                "имя",
                "видовременной",
                "видо-временной",
                "предложно-падежный",
                "падежный",
                "неверный",
                "выбор",
                "между",
                "нарушение",
                "в",
                "и",
                "у",
                "употребление",
                "предлог",
                "видовременный",
                "временной",
            ]:
                norm_cat += lemma + " "
        return norm_cat

    def parse_task(self, task):
        assert task["question"]["type"] == "matching"
        conditions = task["question"]["left"]
        normalized_conditions = [
            self.normalize_category(cond).rstrip() for cond in conditions
        ]

        choices = []
        for choice in task["question"]["choices"]:
            choice["text"] = re.sub("[0-9]\\s?\)", "", choice["text"])
            choices.append(choice)

        return choices, normalized_conditions

    def _get_examples_from_task(self, task):
        choices, conditions = self.parse_task(task)
        if "correct_variants" in task["solution"]:
            answers = task["solution"]["correct_variants"][0]
        else:
            answers = task["solution"]["correct"]

        choice_index_to_error_type = {
            int(answers[option]) - 1: conditions[option_index]
            for option_index, option in enumerate(sorted(answers))
        }
        choice_index_to_error_type = {
            choice_index: choice_index_to_error_type.get(choice_index, _ERRORS[-1])
            for choice_index, choice in enumerate(choices)
        }

        assert len(answers) == sum(
            1
            for error_type in choice_index_to_error_type.values()
            if error_type != _ERRORS[-1]
        )

        for choice_index, choice in enumerate(choices):
            error_type = choice_index_to_error_type[choice_index]

            text = fix_spaces(choices[choice_index]["text"])

            tokenization = self._tokenizer.encode_plus(text, add_special_tokens=True)
            assert len(tokenization["input_ids"]) == len(tokenization["token_type_ids"])

            yield Example(
                text=text,
                token_ids=tokenization["input_ids"],
                segment_ids=tokenization["token_type_ids"],
                mask=[1] * len(tokenization["input_ids"]),
                error_type=error_type,
                error_type_id=_ERRORS.index(error_type),
            )

    def _prepare_examples(self, tasks):
        examples = [
            example for task in tasks for example in self._get_examples_from_task(task)
        ]

        print("Examples:")
        for example in examples[0:100:10]:
            print(example)

        print("Examples count:", len(examples))
        for idx, error in enumerate(_ERRORS):
            print(
                "Examples for error {}: {}".format(
                    error, sum(1 for ex in examples if ex.error_type_id == idx)
                )
            )

        return examples

    def fit(self, tasks):
        examples = self._prepare_examples(tasks)

        config = TrainConfig()
        self._model = RubertMulticlassClassifier(
            load_bert(self.bert_path),
            class_count=len(_ERRORS),
            output_name="error_type",
        ).to(self._device)

        batch_collector = _get_batch_collector(self._device, is_train=True)

        train_loader = DataLoader(
            examples,
            batch_size=config.train_batch_size,
            shuffle=True,
            collate_fn=batch_collector,
            pin_memory=False,
        )
        print(next(iter(train_loader)))

        optimizer, scheduler = _get_optimizer(self._model, len(examples), config)

        trainer = ClassifierTrainer(self._model, optimizer, scheduler, use_tqdm=True)
        trainer.fit(
            train_iter=train_loader,
            train_batches_per_epoch=int(len(examples) / config.train_batch_size),
            epochs_count=config.epoch_count,
        )

    def load(self, path="data/models/solvers/solver8/solver8.pkl"):
        model_checkpoint = torch.load(path, map_location=self._device)
        self._model.load_state_dict(model_checkpoint)
        self._model.to(self._device)
        self._model.eval()
        self.is_loaded = True

    def save(self, path="data/models/solvers/solver8/solver8.pkl"):
        torch.save(self._model.state_dict(), path)

    def predict_from_model(self, task):
        choices, conditions = self.parse_task(task)
        examples = []
        for choice in choices:
            text = fix_spaces(choice["text"])
            tokenization = self._tokenizer.encode_plus(text, add_special_tokens=True)

            examples.append(
                Example(
                    text=text,
                    token_ids=tokenization["input_ids"],
                    segment_ids=tokenization["token_type_ids"],
                    mask=[1] * len(tokenization["input_ids"]),
                    error_type=None,
                    error_type_id=None,
                )
            )

        model_inputs = self._batch_collector(examples)

        target_condition_indices = [
            _ERRORS.index(condition) for condition in conditions
        ]

        pred_dict = {}
        with torch.no_grad():
            model_prediction = self._model(model_inputs)
            model_prediction = model_prediction["error_type_logits"][
                :, target_condition_indices
            ]
            model_prediction = model_prediction.argmax(0)

            for i, condition in enumerate(task["question"]["left"]):
                pred_dict[condition["id"]] = choices[model_prediction[i]]["id"]

        return pred_dict
