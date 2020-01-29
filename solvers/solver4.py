# borsden

import json
import random
import re

from solvers.solver_helpers import AbstractSolver


class Solver(AbstractSolver):
    def __init__(self):
        self.is_loaded = False

    def get_prediction(self, variants):
        predictions = []
        for variant in variants:
            word = variant.lower()
            accent = self.accents.get(word)
            if not accent:
                add_end = word.endswith("шись")
                if add_end:
                    word = word.rstrip("шись")
                match = word + "ший"
                accent = self.accents.get(match)
                if accent:
                    accent = accent.rstrip("ший")
                    if add_end:
                        accent += "шись"
            predictions.append(accent)
        return predictions

    def compare_text_with_variants(self, variants, incorrect):
        predictions = self.get_prediction(variants)

        result = [
            v
            for z, v in zip(variants, predictions)
            if ((z != v) if incorrect else (z == v))
        ]

        if not result:
            result = random.choice(variants)
        else:
            result = random.choice(result)
        return result.lower().strip()

    def load(self, path="data/models/solvers/solver4/accents.json"):
        with open(path, "r", encoding="utf-8") as f:
            self.accents = json.load(f)
        self.is_loaded = True

    def get_variants(self, task):
        task_text = re.split(r"\n", task["text"])[1:]
        for variant in task_text:
            variant = variant.strip().replace("ё", "е").replace("Ё", "Е")
            matched = re.search("[а-яё]*[АЕИОУЫЭЮЯ][а-яё]*", variant)
            if "выпишите" not in variant.lower() and matched:
                yield matched[0]

    def get_type(self, task):
        task_text = re.split(r"\n", task["text"])
        if "Выпишите" in task_text[-1]:
            task = task_text[0] + task_text[-1]
        else:
            task = task_text[0]
        return "неверно" in task.lower()

    def predict_from_model(self, task):
        variants = list(self.get_variants(task))
        task_type = self.get_type(task)
        result = self.compare_text_with_variants(variants, task_type)
        return result
