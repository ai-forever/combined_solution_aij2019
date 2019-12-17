# exotol

import joblib
import numpy as np
import pandas as pd
import random
import re
import scipy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from string import punctuation


def clean(text, remove_stop_words=True):
    text = re.sub(r"\d+\)", "", text)
    text = re.sub(r"\(\d+\)", "", text)
    text = re.sub(r"др.", "", text)
    text = re.sub(r"<...>|<…>", "", text)
    text = re.sub(r"<....>|\( ..... \)|\( ... \)", "", text)
    text = re.sub(r"«|—|»|iii|…|xiv", "", text)

    text = " ".join([c for c in word_tokenize(text) if c not in punctuation])

    if remove_stop_words:
        text = text.split()
        text = [w.lower().strip() for w in text if not w in stopwords.words("russian")]
        text = " ".join(text)
    return text


class Solver(object):
    def __init__(self, seed=42):
        self.seed = seed
        self.init_seed()
        self.clf = SVC(kernel="linear", probability=True)
        self.count_vec = CountVectorizer(
            analyzer="word", token_pattern=r"\w{1,}", ngram_range=(1, 3)
        )
        self.is_loaded = False

    def init_seed(self):
        random.seed(self.seed)

    def transform_vec(self, train, test=None, type_transform="count_vec"):
        if type_transform == "count_vec":
            self.count_vec.fit(pd.concat((train["pq1"], train["pq2"])).unique())
        trainq1_trans = self.count_vec.transform(train["pq1"].values)
        trainq2_trans = self.count_vec.transform(train["pq2"].values)
        X_train = scipy.sparse.hstack((trainq1_trans, trainq2_trans))
        if "target" not in train.columns:
            return X_train
        y_train = train["target"].values
        if not (test is None):
            trainq1_trans = self.count_vec.transform(test["pq1"].values)
            trainq2_trans = self.count_vec.transform(test["pq2"].values)
            labels = test["target"].values
            X_valid = scipy.sparse.hstack((trainq1_trans, trainq2_trans))
            y_valid = labels
            return X_train, y_train, X_valid, y_valid
        return X_train, y_train

    def fit(self, tasks):
        train = self.create_dataset(tasks, is_train=True)
        train["pq1"] = train["q1"].apply(clean)
        train["pq2"] = train["q2"].apply(clean)
        X_train, y_train = self.transform_vec(train, None, type_transform="count_vec")
        self.clf.fit(X_train, y_train)

    def load(self, path="data/models/solvers/solver22/solver22.pkl"):
        model = joblib.load(path)
        self.clf = model["classifier"]
        self.count_vec = model["count_vec"]
        self.is_loaded = True

    def save(self, path="data/models/solvers/solver22/solver22.pkl"):
        model = {"classifier": self.clf, "count_vec": self.count_vec}
        joblib.dump(model, path)

    def create_dataset(self, tasks, is_train=True):
        data = []
        if is_train:
            for task in tasks:
                if "correct_variants" in task["solution"]:
                    solution = task["solution"]["correct_variants"][0]
                if "correct" in task["solution"]:
                    solution = task["solution"]["correct"]
                _tmp = []
                for choice in task["question"]["choices"]:
                    row = [
                        choice["id"],
                        task["text"],
                        choice["text"],
                        1 if choice["id"] in solution else 0,
                    ]
                    _tmp.append(choice["text"])
                    data.append(row)

            return pd.DataFrame(data, columns=["id", "q1", "q2", "target"])
        else:
            for task in tasks:
                _tmp = []
                for choice in task["question"]["choices"]:
                    row = [choice["id"], task["text"], choice["text"]]
                    _tmp.append(choice["text"])
                    data.append(row)

            return pd.DataFrame(data, columns=["id", "q1", "q2"])

    def predict_from_model(self, task):
        tasks = [task]
        data = self.create_dataset(tasks, is_train=False)
        data["pq1"] = data["q1"].apply(clean)
        data["pq2"] = data["q2"].apply(clean)
        trainq1_trans = self.count_vec.transform(data["pq1"].values)
        trainq2_trans = self.count_vec.transform(data["pq2"].values)
        sdata = scipy.sparse.hstack((trainq1_trans, trainq2_trans))

        data["res"] = self.clf.predict(sdata)
        ncount = int(np.sum(data["res"].values))

        if ncount > 0:
            result = list(zip(data["id"].values, data["res"].values))
            indexes = sorted(result, key=lambda tup: tup[1], reverse=True)[:ncount]
            return sorted([str(i[0]).strip(punctuation) for i in indexes])
        else:
            data["prob"] = self.clf.predict_proba(sdata)[:, 1]
            result = list(zip(data["id"].values, data["prob"].values))
            indexes = sorted(result, key=lambda tup: tup[1], reverse=True)[:1]
            return sorted([str(i[0]).strip(punctuation) for i in indexes])
