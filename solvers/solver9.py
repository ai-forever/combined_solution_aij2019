# Idyllium


import os
import random
import re

from solvers.solver_helpers import standardize_task, AbstractSolver
from utils import read_config, save_config


class Solver(AbstractSolver):
    def __init__(self):
        self.known_examples = None
        self.rules = None
        self.exceptions = None
        self.prefixes = None
        self.endings = None

    def predict_from_model(self, task):
        task = standardize_task(task)
        text, choices = task["text"], task["question"]["choices"]
        alt, unver = "чередующаяся", "непроверяемая"
        type_ = (
            "alternations"
            if alt in text
            else "unverifiable"
            if unver in text
            else "verifiable"
        )
        nice_option_ids = list()
        for option in choices:
            parsed_option = re.sub(r"^\d\)", "", option["text"]).split(", ")
            parsed_option = [
                re.sub(
                    r"^[\w|\)]\d\)",
                    "",
                    re.sub(
                        r"^\d\)",
                        "",
                        re.sub(r" *(?:^\d\)|\(.*?\)) *", "", word.strip()).strip(),
                    ).strip(),
                ).strip()
                for word in parsed_option
            ]
            if all(self.is_of_type(word, type_) for word in parsed_option):
                nice_option_ids.append(option["id"])
        if choices[0]["text"].count(", ") == 0:
            if len(nice_option_ids) == 0:
                return [random.choice([str(i + 1) for i in range(5)])]
            elif len(nice_option_ids) == 1:
                return nice_option_ids
            else:
                return [random.choice(nice_option_ids)]
        else:
            if len(nice_option_ids) == 0:
                return sorted(random.sample([str(i + 1) for i in range(5)], 2))
            elif len(nice_option_ids) == 1:
                return sorted(
                    nice_option_ids
                    + [
                        random.choice(
                            [
                                str(i + 1)
                                for i in range(5)
                                if str(i + 1) != nice_option_ids[0]
                            ]
                        )
                    ]
                )
            elif len(nice_option_ids) in [2, 3, 4]:
                return sorted(nice_option_ids)
            else:
                return sorted(random.sample(nice_option_ids, 2))

    def is_of_type(self, word, type_):
        if word in self.known_examples[type_]:
            return True
        else:
            if type_ == "alternations":
                for center in self.rules["alternations"]:
                    for prefix in self.prefixes:
                        for ending in self.endings:
                            if (
                                ((prefix + center + ending) == word)
                                or ((prefix + center + ending + "ся") == word)
                                or (("не" + prefix + center + ending) == word)
                                or (("не" + prefix + center + ending + "ся") == word)
                            ):
                                return True
            if type_ == "unverifiable":
                if "ивать" in word or "ывать" in word:
                    return True
            if type_ == "verifiable":
                if (not self.is_of_type(word, "unverifiable")) and (
                    not self.is_of_type(word, "alternations")
                ):
                    return True
            return False

    def fit(self, tasks):
        alt, unver = "чередующаяся", "непроверяемая"
        for task in tasks:
            task = standardize_task(task)
            text = task["text"]

            if alt in text:
                type_ = "alternations"
            elif unver in text:
                type_ = "unverifiable"
            else:
                type_ = "verifiable"

            correct = (
                task["solution"]["correct_variants"][0]
                if "correct_variants" in task["solution"]
                else task["solution"]["correct"]
            )
            for correct_id in correct:
                for word in task["choices"][int(correct_id) - 1]["parts"]:
                    word_sub = re.sub(
                        r"^[\w|\)]\d\)",
                        "",
                        re.sub(
                            r"^\d\)",
                            "",
                            re.sub(r" *(?:^\d\)|\(.*?\)) *", "", word.strip()).strip(),
                        ).strip(),
                    ).strip()
                    self.known_examples[type_].append(word_sub)

    def load(self, path="data/models/solvers/solver9"):
        model = read_config(os.path.join(path, "solver9.json"))
        self.known_examples = model["known_examples"]
        self.rules = model["rules"]
        self.exceptions = model["exceptions"]
        self.prefixes = model["prefixes"]
        self.endings = model["endings"]
        self.is_loaded = True

    def save(self, path="data/models/solvers/solver9"):
        model = {}
        model["known_examples"] = self.known_examples
        model["rules"] = self.rules
        model["exceptions"] = self.exceptions
        model["prefixes"] = self.prefixes
        model["endings"] = self.endings
        save_config(os.path.join(path, "solver9.json"), model)
