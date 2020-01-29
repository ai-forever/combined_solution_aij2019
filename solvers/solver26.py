# qbic


import attr
import re

import numpy as np
import torch
from nltk.tokenize import sent_tokenize
from string import punctuation
from torch.utils.data import DataLoader
from transformers.optimization import AdamW, WarmupLinearSchedule

from solvers.torch_helpers import (
    ModelTrainer,
    BertExample,
    Field,
    BatchCollector,
    F1ScoreCounter,
    AccuracyCounter,
    RubertMulticlassClassifier,
    init_bert,
    load_bert,
    load_bert_tokenizer,
)
from solvers.solver_helpers import singleton, AbstractSolver
from utils import read_config


@attr.s
class Example(BertExample):
    term = attr.ib(default=None)
    term_id = attr.ib(default=None)

    @classmethod
    def build(cls, text, tokenizer, term=None, term_id=None):
        example = super(Example, cls).build(text, tokenizer)
        example.term = term
        example.term_id = term_id
        return example


@attr.s(frozen=True)
class TrainConfig(object):
    learning_rate = attr.ib(default=1e-5)
    train_batch_size = attr.ib(default=32)
    test_batch_size = attr.ib(default=32)
    epoch_count = attr.ib(default=20)
    warm_up = attr.ib(default=0.1)


def _get_optimizer(model, train_size, config):
    num_total_steps = int(train_size / config.train_batch_size * config.epoch_count)
    num_warmup_steps = int(num_total_steps * config.warm_up)
    optimizer = AdamW(
        [
            {"params": model._bert.parameters(), "lr": 1e-5},
            {"params": model._predictor.parameters(), "lr": 1e-4},
        ],
        lr=config.learning_rate,
        correct_bias=False,
    )
    scheduler = WarmupLinearSchedule(
        optimizer, warmup_steps=num_warmup_steps, t_total=num_total_steps
    )
    return optimizer, scheduler


def _get_batch_collector(device, is_train=True):
    if is_train:
        vector_fields = [Field("term_id", torch.long)]
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
        self._accuracies = [AccuracyCounter()]

    def on_epoch_end(self):
        info = super(ClassifierTrainer, self).on_epoch_end()
        return "{}, {}".format(
            info, ", ".join(str(score) for score in self._accuracies)
        )

    def forward_pass(self, batch):
        outputs = self._model(batch)

        predictions = outputs["term_logits"].argmax(-1)
        self._accuracies[0].update(predictions, batch["term_id"])

        info = ", ".join(str(score) for score in self._accuracies)
        return outputs["loss"], info


class Solver(AbstractSolver):
    def __init__(
        self, model_config="data/models/solvers/solver26/solver26.json"
    ):
        self.model_config = model_config
        self.config = read_config(self.model_config)
        self.unified_substrings = self.config["unified_substrings"]
        self.replacements = self.config["replacements"]
        self.duplicates = self.config["duplicates"]
        self._labels = {}
        self._label_inv = []
        self._tokenizer = load_bert_tokenizer("data/models/bert/rubert/qbic")
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._batch_collector = _get_batch_collector(self._device, is_train=False)

    def predict_from_model(self, task):
        decisions, phrases = dict(), self.extract_phrases(task)
        used_answers, choices = (
            set(),
            [self.unify_type(choice["text"]) for choice in task["question"]["choices"]],
        )
        for letter in "ABCD":
            if len(phrases[letter]) == 0:
                decisions[letter] = "1"
            else:
                examples = []
                for phrase in phrases[letter]:
                    examples.append(
                        Example.build(text=phrase, tokenizer=self._tokenizer)
                    )

                model_inputs = self._batch_collector(examples)
                with torch.no_grad():
                    logits = self._model(model_inputs)["term_logits"]
                    model_prediction = logits.sum(0).cpu().numpy()

                options = [
                    self._label_inv[index]
                    for index in reversed(np.argsort(model_prediction))
                ]
                try:
                    answer = next(
                        option
                        for option in options
                        if option in choices and option not in used_answers
                    )
                except StopIteration:
                    decisions[letter] = "1"
                    continue
                used_answers.add(answer)
                answer_id = str(choices.index(answer) + 1)
                decisions[letter] = answer_id
        return decisions

    def unify_type(self, type_):
        type_ = re.sub("^\d(?:.|\))?\s+", "", type_)
        type_ = re.sub("\([^)]+(?:$|\))", "", type_)
        type_ = type_.strip(" \t\n\v\f\r-–—−()").replace("и ", "")
        type_ = type_.strip(punctuation + " ")
        for key, value in self.unified_substrings.items():
            if key in type_:
                return value
        for key, value in self.replacements.items():
            type_ = re.sub(key + r"\b", value, type_)
        for duplicate_list in self.duplicates:
            if type_ in duplicate_list:
                return duplicate_list[0]
        return type_

    @staticmethod
    def get_sent_num(sent: str):
        match = re.search(r"\(([\dЗбOО]{1,2})\)", sent)
        if match:
            num = match.group(1)
            table = str.maketrans("ЗбОO", "3600")
            num = num.translate(table)
            num = int(num)
            return num
        match = re.search(r"([\dЗбOО]{1,2})\)", sent)
        if match:
            num = match.group(1)
            table = str.maketrans("ЗбОO", "3600")
            num = num.translate(table)
            num = int(num)
            return num

    def extract_phrases(self, task):
        result, text = {key: list() for key in "ABCD"}, task["text"]
        text = text.replace("\xa0", " ")
        citations = [
            sent
            for sent in sent_tokenize(text.split("Список терминов")[0])
            if re.search(r"\([А-Г]\)|\(?[А-Г]\)?_{2,}", sent)
        ]
        text = [x for x in re.split(r"[АA]БВГ\.?\s*", text) if x != ""][-1]
        text = re.sub(r"(\([\dЗбOО]{1,2}\))", r" \1 ", text)
        sents = sent_tokenize(text)
        sents = [x.strip() for sent in sents for x in re.split(r"…|\.\.\.", sent)]
        sents = [x.strip() for sent in sents for x in re.split(" (?=\([\dЗбОO])", sent)]
        sents = [sent for sent in sents if re.match(r"\s*\(?[\dЗбОO]{1,2}\)", sent)]
        assert all(
            re.search(
                r"\({}\)|\(?{}\)?_{2,}".replace("{}", letter), " ".join(citations)
            )
            for letter in "АБВГ"
        ), "Not all letters found in {}".format(citations)
        citations = " ".join(citations)
        citations = re.split("\([А-Г]\)|\(?[А-Г]\)?_{2,}", citations)[1:]
        assert len(citations) == 4, "Expected 4 (not {}) citations: {}".format(
            len(citations), citations
        )
        for citation, letter in zip(citations, "ABCD"):
            sent_nums = list()
            matches = re.finditer(r"предложени\w{,3}\s*(\d[\d\-— ,]*)", citation)
            for match in matches:
                sent_nums_str = match.group(1)
                for part in re.split(r",\s*", sent_nums_str):
                    part = part.strip(" \t\n\v\f\r-–—−")
                    if len(part) > 0:
                        if part.isdigit():
                            sent_nums.append(int(part))
                        else:
                            from_, to = re.split(r"[-–—−]", part)
                            extension = range(int(from_), int(to) + 1)
                            sent_nums.extend(extension)
            sents_ = [sent for sent in sents if self.get_sent_num(sent) in sent_nums]
            sents_ = [re.sub(r"(\([\dЗбOО]{1,2}\))\s*", "", sent) for sent in sents_]
            result[letter].extend(sents_)
            matches = re.finditer(r"[«\"](.*?)[»\"]", citation)
            for match in matches:
                result[letter].append(match.group(1))
        result = {key: list(set(value)) for key, value in result.items()}
        return result

    def _prepare_examples(self, tasks, is_train_tasks):
        examples = []
        for task in tasks:
            letters_to_phrases = self.extract_phrases(task)
            solution = task["solution"]["correct"]
            for key in "ABCD":
                questions = letters_to_phrases[key]
                answer_number = solution[key]
                answer = next(
                    answ["text"]
                    for answ in task["question"]["choices"]
                    if answ["id"] == answer_number
                )
                if answer.isdigit():
                    continue
                answer = self.unify_type(answer)
                if answer not in self._labels:
                    if is_train_tasks:
                        self._labels[answer] = len(self._labels)
                    else:
                        continue
                answer_id = self._labels[answer]
                for question in questions:
                    examples.append(
                        Example.build(
                            text=question,
                            tokenizer=self._tokenizer,
                            term=answer,
                            term_id=answer_id,
                        )
                    )
        print("Examples count:", len(examples))
        for term, term_id in self._labels.items():
            print(
                "Examples for term {}: {}".format(
                    term, sum(1 for ex in examples if ex.term_id == term_id)
                )
            )
        return examples

    def fit(self, tasks, test_tasks=None):
        examples = self._prepare_examples(tasks, is_train_tasks=True)
        self._label_inv = [
            label for label, _ in sorted(self._labels.items(), key=lambda pair: pair[1])
        ]

        config = TrainConfig()
        self._model = RubertMulticlassClassifier(
            load_bert("data/models/bert/rubert/qbic"),
            class_count=len(self._labels),
            output_name="term",
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

        test_loader, test_batches_per_epoch = None, 0
        if test_tasks:
            test_examples = self._prepare_examples(test_tasks, is_train_tasks=False)
            test_loader = DataLoader(
                test_examples,
                batch_size=config.test_batch_size,
                shuffle=False,
                collate_fn=batch_collector,
                pin_memory=False,
            )
            test_batches_per_epoch = int(len(test_examples) / config.test_batch_size)

        optimizer, scheduler = _get_optimizer(self._model, len(examples), config)

        trainer = ClassifierTrainer(self._model, optimizer, scheduler, use_tqdm=True)
        trainer.fit(
            train_iter=train_loader,
            train_batches_per_epoch=int(len(examples) / config.train_batch_size),
            val_iter=test_loader,
            val_batches_per_epoch=test_batches_per_epoch,
            epochs_count=config.epoch_count,
        )

    @singleton
    def load(self, path="data/models/solvers/solver26/solver26.pkl"):
        checkpoint = torch.load(path, map_location=self._device)
        self._labels = checkpoint["labels"]
        self._label_inv = [
            label for label, _ in sorted(self._labels.items(), key=lambda pair: pair[1])
        ]

        self._model = RubertMulticlassClassifier(
            init_bert("data/models/bert/rubert/qbic"),
            class_count=len(self._labels),
            output_name="term",
        ).to(self._device)
        self._model.load_state_dict(checkpoint["model"])
        self._model.eval()
        self.is_loaded = True

    def save(self, path="data/models/solvers/solver26/solver26.pkl"):
        torch.save({"model": self._model.state_dict(), "labels": self._labels}, path)
