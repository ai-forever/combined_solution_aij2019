
import os
import random
import re

import numpy as np
import razdel
import torch

from solvers.torch_helpers import RubertFor13
from solvers.solver_helpers import morph, AbstractSolver


class Together(Exception):
    pass


replaces = {
    "«": '"',
    "»": '"',
    "“": '"',
    "„": '"',
    "\n": " ",
    "–": " - ",
    "―": " - ",
    "—": " - ",
    "…": "...",
    "ё": "е",
    "Ё": "Е",
    "--": "-",
    "\xa0": " ",
    "\t": " ",
    "C": "1",
    "I": "1",
    "N": "1",
    "V": "1",
    "X": "1",
    "\u200e": "",
}

pattern = re.compile("\((не|ни)\)")


class Solver(AbstractSolver):
    def __init__(self):
        self.rubert = None
        self.words = None

    def prepare(self, sentence):
        tokens = [
            token
            for token in (token.strip() for _, _, token in razdel.tokenize(sentence))
            if token
        ]
        for i, token in enumerate(tokens):
            if token == "," or re.match(r"_\d_", token):
                continue
            target = None

            if i + 1 < len(tokens):
                next_token = tokens[i + 1]
                if token == "не_" or token == "ни_":
                    token = token.replace("_", "")
                    target = token + next_token
                    if (
                        not morph.word_is_known(next_token)
                        and not next_token in self.words
                    ):
                        raise Together(target)
            yield token, target

    def insert_target(self, x):
        included = (self.rubert.segment_size - 1) // 2
        X = [0] * included + x + [0] * included
        ATTENTION = [0] * included + [1] * len(x) + [0] * included
        X_result, attention_result = [], []
        for i in range(0, len(x)):
            i = included + i
            segment = X[i - included : i + included + 1]
            segment.insert(included + 1, 0)
            attention_segment = ATTENTION[i - included : i + included + 1]
            attention_segment.insert(included + 1, 1)
            X_result.append(segment)
            attention_result.append(attention_segment)
        return X_result, attention_result

    def encode_data(self, sentence):
        """
        Converts words to (BERT) tokens and punctuation to given encoding.
        Note that words can be composed of multiple tokens.
        """
        X, Y = [], []
        for word, target in self.prepare(sentence):
            tokens = self.rubert.tokenizer.tokenize(word)
            x = self.rubert.tokenizer.convert_tokens_to_ids(tokens)
            y = [target]
            if len(x) > 0:
                if len(x) > 1:
                    y = (len(x) - 1) * [None] + y
                X += x
                Y += y
        return X, Y

    def load(self, path="data/models/solvers/solver13"):
        self.rubert = RubertFor13()
        self.rubert.load_state_dict(torch.load(os.path.join(path, "solver13.config")))
        self.rubert.eval()
        with open(os.path.join(path, "solver13.txt"), "r", encoding="utf-8") as f:
            self.words = set(
                word.strip().lower().replace("ё", "е") for word in f.readlines()
            )
        self.is_loaded = True

    def predict_from_model(self, task):
        candidates = {}
        text = task["text"]
        for k, v in replaces.items():
            text = text.replace(k, v)
        text = text.replace(".", ". ")
        all_variants = []

        for _, _, sentence in razdel.sentenize(text):
            sentence = sentence.lower()
            if not re.search(pattern, sentence):
                continue

            sentence = re.sub(pattern, r" \1_ ", sentence)
            try:
                encoded, is_targets = self.encode_data(sentence)
            except Together as e:
                word = e.args[0]
                return word
            targets, attentions = self.insert_target(encoded)
            for encode, X, attention, is_target in zip(
                encoded, targets, attentions, is_targets
            ):
                if is_target:
                    all_variants.append(is_target)
                    X = torch.from_numpy(np.array([X]))
                    attention = torch.from_numpy(np.array([attention]))
                    output = self.rubert(X, attention)
                    y_pred = output.argmax(dim=1).cpu().data.numpy().flatten()[0]
                    if y_pred:
                        candidates[is_target] = (
                            output.max(dim=1).values.cpu().data.numpy().flatten()[0]
                        )

        if candidates:
            return sorted(candidates.items(), key=lambda x: x[1], reverse=True)[0][0]
        else:
            return random.choice(all_variants)
