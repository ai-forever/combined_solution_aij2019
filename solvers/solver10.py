# Idyllium

import os
import random
import re

from solvers.utils import ALPHABET, morph, standardize_task
from utils import read_config


class Solver(object):
    def __init__(self, seed=42):
        self.is_loaded = False
        self.alphabet = ALPHABET
        self.seed = seed
        self.init_seed()
        self.morph = morph
        self.dictionary = None
        self.prefixes = None
        self.int_prefixes = None
        self.universal_prefixes = None

    def init_seed(self):
        return random.seed(self.seed)

    def predict_from_model(self, task):
        task["text"] = re.sub("[^а-яА-яЁё\.\,\! ]|_", "", task["text"])
        result, task = [], standardize_task(task)
        match = re.search(r"буква ([ЭОУАЫЕЁЮЯИ])*", task["text"], re.IGNORECASE)
        if match:
            letter = match.group(1)
            return self.get_answer_by_letter(
                task["question"]["choices"], letter.lower()
            )
        elif "одна и та же буква" in task["text"]:
            for vowel in self.alphabet:
                result_with_this_vowel = self.get_answer_by_letter(
                    task["question"]["choices"], vowel
                )
                result.extend(result_with_this_vowel)
        return sorted(list(set(result)))

    def get_answer_by_letter(self, choices, letter):
        result = list()
        for choice in choices:
            parts = [
                re.sub(
                    r"^\d\) ?| ?\(.*?\) ?", "", re.sub(r"^\)\d\) ?| ?\(.*?\) ?", "", x)
                ).replace(".. ", "..")
                for x in choice["parts"]
            ]
            parts = [x.replace("..", letter) for x in parts]
            if all(self.check_word_is_known(word) for word in parts):
                result.append(choice["id"])
        return sorted(result)

    def word_is_known(self, word):
        is_known = self.morph.word_is_known(word)
        if not is_known:
            is_known = word in self.dictionary
        return is_known

    def check_word_is_known(self, word):
        if self.word_is_known(word):
            return True
        else:
            letters = "яюёе"
            if self.word_is_known(word.replace("ъ", "ь")):
                return False
            else:
                for prefix in self.prefixes + self.int_prefixes:
                    prefix_index = word.find(prefix)
                    if prefix_index == 0:
                        try:
                            if (
                                word[prefix_index + len(prefix) + 1] in letters
                                and word[prefix_index + len(prefix)] == "ъ"
                            ):
                                if self.word_is_known(
                                    word[prefix_index + len(prefix) + 1 :]
                                ):
                                    return True
                        except IndexError:
                            continue

            for prefix in self.prefixes:
                if prefix not in [
                    "меж",
                    "сверх",
                    "двух",
                    "трех",
                    "трёх",
                    "четырех",
                    "четырёх",
                ]:
                    prefix_index = word.find(prefix)
                    if prefix_index == 0:
                        try:
                            if word[prefix_index + len(prefix)] == "ы":
                                if self.word_is_known(
                                    "и" + word[prefix_index + len(prefix) + 1 :]
                                ):
                                    if word != "взымал" and word != "взымать":
                                        return True
                        except IndexError:
                            continue
                else:
                    prefix_index = word.find(prefix)
                    if prefix_index == 0:
                        try:
                            if word[prefix_index + len(prefix)] == "и":
                                if self.word_is_known(
                                    "и" + word[prefix_index + len(prefix) + 1 :]
                                ):
                                    return True
                        except IndexError:
                            continue

            for prefix in self.int_prefixes:
                prefix_index = word.find(prefix)
                if prefix_index == 0:
                    try:
                        if word[prefix_index + len(prefix)] == "и":
                            if self.word_is_known(
                                "и" + word[prefix_index + len(prefix) + 1 :]
                            ):
                                return True
                    except IndexError:
                        continue

            for prefix in self.universal_prefixes:
                prefix_index = word.find(prefix)
                if prefix_index == 0:
                    try:
                        if self.word_is_known(word[prefix_index + len(prefix) :]):
                            return True
                    except IndexError:
                        continue
        return False

    def load(self, path="data/models/solvers/solver10"):
        model = read_config(os.path.join(path, "solver10.json"))
        self.dictionary = model["dictionary"]
        self.prefixes = model["prefixes"]
        self.int_prefixes = model["int_prefixes"]
        self.universal_prefixes = model["universal_prefixes"]
        self.is_loaded = True

    def save(self, path=""):
        pass

    def fit(self, path=""):
        pass
