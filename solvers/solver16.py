# mihtw

import joblib
import numpy as np
from solvers.solver_helpers import morph
import random
import re
from catboost import CatBoostClassifier
from nltk.tokenize.toktok import ToktokTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

from solvers.solver_helpers import standardize_task


class Solver(object):
    def __init__(self, seed=42):
        self.seed = seed
        self.init_seed()
        self.is_loaded = False
        self.tokenizer = ToktokTokenizer()
        self.morph = morph
        self.count_vectorizer = CountVectorizer(ngram_range=(1, 4), tokenizer=str.split)
        self.classifier = CatBoostClassifier(verbose=0, use_best_model=True)

    def init_seed(self):
        return random.seed(self.seed)

    def strs_to_pos_tags(self, texts):
        result = []
        for text in texts:
            result.append(
                " ".join(
                    [
                        "PNCT"
                        if self.morph.parse(word)[0].tag.POS is None
                        else self.morph.parse(word)[0].tag.POS
                        for word in self.tokenizer.tokenize(text)
                    ]
                )
            )
        return result

    def save(self, path="data/models/solvers/solver16/solver16.pkl"):
        model = {
            "count_vectorizer": self.count_vectorizer,
            "classifier": self.classifier,
        }
        joblib.dump(model, path)

    def load(self, path="data/models/solvers/solver16/solver16.pkl"):
        model = joblib.load(path)
        self.count_vectorizer = model["count_vectorizer"]
        self.classifier = model["classifier"]
        self.is_loaded = True

    def fit(self, tasks):
        X, y = [], []
        for task in tasks:
            task = standardize_task(task)
            correct = (
                task["solution"]["correct_variants"][0]
                if "correct_variants" in task["solution"]
                else [task["solution"]["correct"]]
            )
            sentences = [
                re.sub(r"^\d\) ?", "", sentence["text"])
                for sentence in task["question"]["choices"]
            ]
            sentences = self.strs_to_pos_tags(sentences)
            X.extend(sentences)
            y.extend([1 if str(i + 1) in correct else 0 for i in range(5)])
        X = self.count_vectorizer.fit_transform(X).toarray()
        X_train, X_dev, y_train, y_dev = train_test_split(X, y, train_size=0.9)
        self.classifier.fit(X_train, y_train, eval_set=(X_dev, y_dev))

    def predict_from_model(self, task):
        task = standardize_task(task)
        sentences = [
            re.sub(r"^\d\) ?", "", sentence["text"])
            for sentence in task["question"]["choices"]
        ]
        sentences = self.strs_to_pos_tags(sentences)
        vector = self.count_vectorizer.transform(sentences).toarray()
        proba = self.classifier.predict_proba(vector)[:, 1]
        two_highest = sorted([str(i + 1) for i in np.argsort(proba)[-2:]])
        return two_highest
