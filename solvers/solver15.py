import random
import re

from solvers.utils import RubertForMasking, morph


class Solver(object):
    def __init__(self, seed=42):
        self.seed = seed
        self.init_seed()
        self.morph = morph
        self.known_words = []
        self.model = RubertForMasking()
        self.is_loaded = False

    def init_seed(self):
        return random.seed(self.seed)

    def fit(self, tasks):
        pass

    def save(self, path=""):
        pass

    def load(self, filename_path="data/models/solvers/solver15/dic_known_word.csv"):
        with open(filename_path, encoding="utf-8") as f:
            for line in f:
                self.known_words.append(line.replace("\n", ""))
        self.is_loaded = True

    def predict_from_model(self, task):
        words1, words_z, del_word1, del_word2 = [], [], [], []

        text = (
            task["text"]
            .replace(",", " ,")
            .replace(".", " .")
            .replace(";", " ;")
            .replace(":", " :")
            .replace("»", " »")
            .replace("«", "« ")
            .replace("(З)", "(3)")
        )
        if "пишется НН.".lower() in task["text"].lower():
            to_replace_one = "нн"
            to_replace_two = "н"
        elif (
            "пишется Н.".lower() in task["text"].lower()
            or "одна буква Н".lower() in task["text"].lower()
            or "одно Н".lower() in task["text"].lower()
            or "одна Н.".lower() in task["text"].lower()
        ):
            to_replace_one = "н"
            to_replace_two = "нн"
        else:
            to_replace_one = "нн"
            to_replace_two = "н"
        words = text.split()
        for j, word in enumerate(words):
            if re.search("\([0-9]\)", word) is not None:

                id_t = re.search("[0-9]", word).group(0)

                word1 = re.sub("\([0-9]\)", to_replace_one, word)
                word2 = re.sub("\([0-9]\)", to_replace_two, word)

                est_word1 = self.morph.word_is_known(word1) or word1 in self.known_words
                est_word2 = self.morph.word_is_known(word2) or word2 in self.known_words
                if est_word1 and not est_word2:
                    words1.append(id_t)
                    words[j] = word1
                elif est_word1 and est_word2:
                    words[j] = "[MASK]"
                    words_z.append(id_t)
                    del_word1.append(word1)
                    del_word2.append(word2)
                else:
                    words[j] = word2
        text = " ".join(words)
        text = re.sub("\([0-9]\)", "", text)
        text = re.sub("  ", " ", text)

        del_word1.extend(del_word2)
        if to_replace_one == "нн":
            delta = 0.047
        else:
            delta = -0.06
        if del_word1:
            results = self.model.masking_task_15(text + text, del_word1, delta)
            for result in results:
                if del_word1.index(result) < len(words_z):
                    words1.append(words_z[del_word1.index(result)])

        words1.sort()
        answer = words1
        return answer
