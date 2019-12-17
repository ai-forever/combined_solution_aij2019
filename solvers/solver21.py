# Magic City

import json
import lightgbm as lgb
import numpy as np
import os
import pandas as pd
import random
import re
from sklearn.feature_extraction.text import TfidfVectorizer

from solvers.utils import morph
from utils import load_pickle, save_pickle


class Solver(object):
    def __init__(self, seed=42):
        self.seed = seed
        self.init_seed()
        self.morph = morph
        self.clf = None
        self.is_loaded = False
        self.tfidf = TfidfVectorizer(ngram_range=(1, 1), max_features=15)

    def init_seed(self):
        return random.seed(self.seed)

    def fit(self, tasks):
        target_list = []
        seq_1_list = []
        seq_2_list = []

        for num_j, json_file in enumerate(tasks):
            with open(json_file, "rb") as f:
                task_json = json.load(f)
            if "tasks" in task_json:
                task_json = task_json["tasks"]
            task = task_json[20]
            first_seq = task["text"].split(".")[0]
            answer_key = []
            answer_type = []
            if "тире" in first_seq:
                for seq in re.split(r"\.|\?", task["text"])[1:]:
                    seq = seq.strip()
                    number = re.findall(r"\d\)", seq)
                    if number:
                        if "−" in seq or "–" in seq or "—" in seq:
                            answer_key += [number[0][0]]
                            answer_type += [
                                " ".join(
                                    str(morph.parse(word)[0].tag.POS)
                                    for word in seq.lower().split(" ")
                                )
                            ]
            elif "двоеточие" in first_seq:
                for seq in re.split(r"\.|\?", task["text"])[1:]:
                    seq = seq.strip()
                    number = re.findall(r"\d\)", seq)
                    if number:
                        if ":" in seq:
                            answer_key += [number[0][0]]
                            answer_type += [
                                " ".join(
                                    str(morph.parse(word)[0].tag.POS)
                                    for word in seq.lower().split(" ")
                                )
                            ]
            elif "запятые" in first_seq or "запятая" in first_seq:
                for seq in re.split(r"\.|\?", task["text"])[1:]:
                    seq = seq.strip()
                    number = re.findall(r"\d\)", seq)
                    if number:
                        if "," in seq:
                            answer_key += [number[0][0]]
                            answer_type += [
                                " ".join(
                                    str(morph.parse(word)[0].tag.POS)
                                    for word in seq.lower().split(" ")
                                )
                            ]

            true_answer = task["solution"]

            if "correct" in true_answer:
                true_answer = true_answer["correct"]
            else:
                true_answer = true_answer["correct_variants"][0]

            target = []
            seq_1 = []
            seq_2 = []
            for key_1, s_1 in zip(answer_key, answer_type):
                for key_2, s_2 in zip(answer_key, answer_type):
                    if key_1 > key_2:
                        if key_1 in true_answer and key_2 in true_answer:
                            target += [1]
                        else:
                            target += [0]
                        seq_1 += [s_1]
                        seq_2 += [s_2]

            target_list += target
            seq_1_list += seq_1
            seq_2_list += seq_2

        df = pd.DataFrame()
        df["target"] = target_list
        df["seq_1"] = seq_1_list
        df["seq_2"] = seq_2_list

        self.tfidf.fit(df["seq_1"].tolist() + df["seq_2"].tolist())

        X_1 = self.tfidf.transform(df["seq_1"]).toarray()
        X_2 = self.tfidf.transform(df["seq_2"]).toarray()

        X = pd.DataFrame(np.abs(X_1 - X_2))
        X.columns = ["X_" + str(x) for x in X.columns]
        df = pd.concat([df, X], axis=1)
        drop_cols = ["target", "seq_1", "seq_2", "seq_1_pos", "seq_2_pos"]
        train_cols = [col for col in df.columns if col not in drop_cols]

        df = df.drop_duplicates(train_cols + ["target"])

        param_lgb = {
            "boosting_type": "gbdt",
            "objective": "binary",
            "metric": "auc",
            "num_leaves": 32,
            "learning_rate": 0.1,
        }
        tr = lgb.Dataset(np.array(df[train_cols]), np.array(df["target"]))

        self.clf = lgb.train(param_lgb, tr, 60)

    def load(self, path="data/models/solvers/solver21/solver21.pickle"):
        self.clf, self.tfidf = load_pickle(path)
        self.is_loaded = True

    def save(self, path="data/models/solvers/solver21/solver21.pickle"):
        save_pickle([self.clf, self.tfidf], path)

    def predict_from_model(self, task):
        first_seq = task["text"].split(".")[0]
        answer_key = []
        answer_type = []
        if "тире" in first_seq:
            for seq in re.split(r"\.|\?", task["text"])[1:]:
                seq = seq.strip()
                number = re.findall(r"\d\)", seq)
                if number:
                    if "−" in seq or "–" in seq or "—" in seq:
                        answer_key += [number[0][0]]
                        answer_type += [
                            " ".join(
                                str(morph.parse(word)[0].tag.POS)
                                for word in seq.lower().split(" ")
                            )
                        ]
        elif "двоеточие" in first_seq:
            for seq in re.split(r"\.|\?", task["text"])[1:]:
                seq = seq.strip()
                number = re.findall(r"\d\)", seq)
                if number:
                    if ":" in seq:
                        answer_key += [number[0][0]]
                        answer_type += [
                            " ".join(
                                str(morph.parse(word)[0].tag.POS)
                                for word in seq.lower().split(" ")
                            )
                        ]
        elif "запятые" in first_seq or "запятая" in first_seq:
            for seq in re.split(r"\.|\?", task["text"])[1:]:
                seq = seq.strip()
                number = re.findall(r"\d\)", seq)
                if number:
                    if "," in seq:
                        answer_key += [number[0][0]]
                        answer_type += [
                            " ".join(
                                str(morph.parse(word)[0].tag.POS)
                                for word in seq.lower().split(" ")
                            )
                        ]

        seq_1, seq_2, keys = [], [], []
        for key_1, s_1 in zip(answer_key, answer_type):
            for key_2, s_2 in zip(answer_key, answer_type):
                if key_1 > key_2:
                    keys += [(key_1, key_2)]
                    seq_1 += [s_1]
                    seq_2 += [s_2]

        df = pd.DataFrame()
        df["seq_1"] = seq_1
        df["seq_2"] = seq_2

        X_1 = self.tfidf.transform(df["seq_1"]).toarray()
        X_2 = self.tfidf.transform(df["seq_2"]).toarray()

        X = pd.DataFrame(np.abs(X_1 - X_2))
        X.columns = ["X_" + str(x) for x in X.columns]
        df = pd.concat([df, X], axis=1)
        drop_cols = ["target", "seq_1", "seq_2", "seq_1_pos", "seq_2_pos"]
        train_cols = [col for col in df.columns if col not in drop_cols]

        pred = self.clf.predict(df[train_cols])

        tresh = np.max(pred)
        if tresh > 0.5:
            tresh = 0.5
        checks = np.where(pred >= tresh)

        answer = sorted(set(np.array(keys)[checks].flatten()))

        return answer
