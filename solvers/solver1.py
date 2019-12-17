# nice

import pandas as pd
import random
import re
from nltk.tokenize import ToktokTokenizer
from sklearn.metrics.pairwise import cosine_similarity

from solvers.utils import BertEmbedder, morph


class Solver(object):
    def __init__(self, seed=42):
        self.seed = seed
        self.init_seed()
        self.morph = morph
        self.toktok = ToktokTokenizer()
        self.bert = BertEmbedder()
        self.is_loaded = True

    def init_seed(self):
        random.seed(self.seed)

    def get_num(self, text):
        lemmas = [
            self.morph.parse(word)[0].normal_form for word in self.toktok.tokenize(text)
        ]
        if "указывать" in lemmas and "предложение" in lemmas:
            w = lemmas[lemmas.index("указывать") + 1]
            d = {"один": 1, "два": 2, "три": 3, "четыре": 4, "предложение": 1}
            if w in d:
                return d[w]
        elif "указывать" in lemmas and "вариант" in lemmas:
            return 2
        return 2

    def compare_text_with_variants(self, variants):
        variant_vectors = self.bert.sentence_embedding(variants)
        predicts = []
        for i in range(0, len(variant_vectors)):
            for j in range(i + 1, len(variant_vectors)):
                sim = cosine_similarity(
                    variant_vectors[i].reshape(1, -1), variant_vectors[j].reshape(1, -1)
                ).flatten()[0]
                predicts.append(pd.DataFrame({"sim": sim, "i": i, "j": j}, index=[1]))
        predicts = pd.concat(predicts)
        indexes = predicts[predicts.sim == predicts.sim.max()][["i", "j"]].values[0]
        return sorted([str(i + 1) for i in indexes])

    def sent_split(self, text):
        reg = r"\(*\d+\)"
        return re.split(reg, text)

    def process_task(self, task):
        first_phrase, task_text = re.split(r"\(*1\)", task["text"])[:2]
        variants = [t["text"] for t in task["question"]["choices"]]
        text, task = "", ""
        if "Укажите" in task_text:
            text, task = re.split("Укажите ", task_text)
            task = "Укажите " + task
        elif "Укажите" in first_phrase:
            text, task = task_text, first_phrase
        return text, task, variants

    def fit(self, tasks):
        pass

    def load(self, path=""):
        pass

    def save(self, path=""):
        pass

    def predict_from_model(self, task):
        text, task, variants = self.process_task(task)
        result = self.compare_text_with_variants(variants)
        return result
