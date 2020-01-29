# lamoda

import os
import random
import re

import numpy as np
import pandas as pd
from nltk.tokenize import ToktokTokenizer
from pymystem3 import Mystem
from sklearn.metrics.pairwise import cosine_distances
from string import punctuation

from utils import load_pickle
from solvers.solver_helpers import Word2vecProcessor, morph, singleton, AbstractSolver


class Solver(AbstractSolver):
    def __init__(self):
        self.morph = morph
        self.mystem = Mystem()
        self.tokenizer = ToktokTokenizer()
        self.w2v = Word2vecProcessor()
        self.synonyms = None
        self.antonyms = None
        self.phraseology = None
        self.phraseologisms = None
        self.prep_synon = None
        self.set_f = None
        self.verbs_dict = None
        self.chasti_rechi = None
        self.set_f_2 = None

    def lemmatize(self, text):
        return [
            self.morph.parse(word)[0].normal_form
            for word in self.tokenizer.tokenize(text.strip())
        ]

    def get_word(self, text):
        try:
            return re.split("»", re.split("«", text)[1])[0]
        except IndexError:
            return ""

    def get_pos(self, text):
        lemmas = [l for l in self.lemmatize(text) if l != " "]
        if "фразеологизм" in lemmas:
            pos = "PHR"
        elif "синоним" in lemmas:
            pos = "SYN"
        elif "антоним" in lemmas:
            pos = "ANT"
        elif "антонимический" in lemmas:
            pos = "ANT"
        elif "синонимический" in lemmas:
            pos = "SYN"
        else:
            pos = "DEF"
        return pos

    def full_intersection(self, small_lst, big_lst):
        if sum([value in big_lst for value in small_lst]) == len(small_lst):
            return True
        return False

    def sent_split(self, text):
        reg = r"\(*\n*\d+\n*\)"
        return re.split(reg, text)

    def search(self, text_lemmas, lst):
        for l in lst:
            if self.full_intersection(l, text_lemmas):
                return "".join(l)
        return ""

    @singleton
    def load(self, path="data/models/solvers/solver24"):
        self.synonyms = open(
            os.path.join(path, r"synonyms.txt"), "r", encoding="utf8"
        ).readlines()
        self.synonyms = [
            re.sub("\.", "", t.lower().strip("\n")).split(" ") for t in self.synonyms
        ]
        self.synonyms = [[t for t in l if t] for l in self.synonyms]
        self.antonyms = open(
            os.path.join(path, r"antonyms.txt"), "r", encoding="utf8"
        ).readlines()
        self.antonyms = [t.strip(" \n").split(" - ") for t in self.antonyms]
        self.phraseology = open(
            os.path.join(path, r"phraseologisms.txt"), "r", encoding="utf8",
        ).readlines()
        self.phraseology = [
            [
                l
                for l in self.lemmatize(l)
                if l not in ["\n", " ", "...", "", ",", "-", ".", "?", r" (", r"/"]
            ]
            for l in self.phraseology
        ]
        self.phraseologisms = load_pickle(os.path.join(path, "phraseologisms.pckl"))
        self.prep_synon = pd.read_csv(os.path.join(path, "prep_synonyms.csv"))
        self.sber_phraseologs = pd.read_csv(
            os.path.join(path, "prep_phraseologisms.csv")
        )
        self.set_f, self.verbs_dict, self.chasti_rechi, self.set_f_2 = load_pickle(
            os.path.join(path, "solver24.pkl")
        )
        self.is_loaded = True

    @staticmethod
    def parse_task(task):
        regex = "(\([0-9]{1,2}\)|\s[0-9]{1,2}\)|[.!?-][0-9]{1,2}\))"
        p1 = "из предлож[а-яё]+\s+\(?[0-9]{1,2}\)?\s*[–—−-]\s*\(?[0-9]{1,2}\)?"
        p2 = "из предлож[а-яё]+\s+\(?[0-9]{1,2}\)?\s*"

        task = task["text"].lower()
        selector = None

        if re.findall(p1, task):
            q = re.findall(p1, task)[0]
            q = q.replace("(", "")
            q = q.replace(")", "")
            task = re.sub(p1, q, task)
            numbers = re.findall("[0-9]{1,2}", q)
            selector = list(range(int(numbers[0]), int(numbers[1]) + 1))
        elif re.findall(p2, task):
            q = re.findall(p2, task)[0]
            q = q.replace("(", "")
            q = q.replace(")", "")
            q = "." + q
            task = re.sub(p2, q, task)
            numbers = re.findall("[0-9]{1,2}", q)
            selector = [int(numbers[0])]

        l = re.split("[.!?…]", task)
        l = [re.split(regex, x) for x in l]
        l = sum(l, [])
        l = [x.strip() for x in l]
        l = [x for x in l if len(x) > 0]

        text = []
        i = 0
        while i < len(l):
            line = [l[i]]
            i += 1
            while (
                re.match(regex, line[0])
                and (i < len(l))
                and not (re.match(regex, l[i]))
            ):
                line += [l[i]]
                i += 1
            text.append(line)

        question = [x[0] for x in text if not re.match(regex, x[0])]
        if len(text[-1]) > 2:
            question += text[-1][2:]
            text[-1] = text[-1][:2]

        question = " ".join(question)

        text = [(x[0], " ".join(x[1:])) for x in text]
        text = [x for x in text if re.match(regex, x[0])]
        text_df = pd.DataFrame(text)
        text_df[0] = text_df[0].map(lambda x: int(x.replace("(", "").replace(")", "")))
        if selector:
            tmp = text_df[text_df[0].isin(selector)]
            if tmp.shape[0] > 0:
                text_df = tmp
            else:
                print(">>>>> SELECTOR ERROR")
        return question, list(text_df[1])

    def lemm_and_clear(self, text, morph):
        analyze = morph.analyze(text)
        lemm_text = [
            (x["analysis"][0]["lex"] if x.get("analysis") else x["text"])
            for x in analyze
        ]
        lemm_text = [
            self.verbs_dict[x] if x in self.verbs_dict else x for x in lemm_text
        ]

        analyze = list(zip(lemm_text, [x["text"] for x in analyze]))
        lemm_text = [x for x in lemm_text if not re.match("\s+", x)]

        lemm_text = [x for x in lemm_text if re.match("\w+", x)]
        return lemm_text, analyze

    @staticmethod
    def find_subarray(arr1, anal_arr2):
        arr2 = [x[0] for x in anal_arr2]
        sourse_arr2 = [x[1] for x in anal_arr2]
        for i_arr2 in range(len(arr2) - 1, -1, -1):
            positions = []
            last_positions = 0
            for j_arr1, word1 in enumerate(arr1):
                for j_arr2, word2 in enumerate(arr2[i_arr2:]):
                    if (word1 == word2) and (last_positions <= j_arr2):
                        last_positions = j_arr2
                        positions.append(j_arr2)
                        break
                if len(arr1) == len(positions):
                    return sourse_arr2[i_arr2:][positions[0] : positions[-1] + 1]

    def suggest_prediction(self, task):
        question_task, text_task = self.parse_task(task)

        question_task_re = re.sub("[^а-яё]", "", question_task.lower())

        if "фразеологизм" in question_task_re:
            lemm_text_task = [self.lemm_and_clear(x, self.mystem) for x in text_task]
            for num_source in range(0, self.phraseologisms[1].max() + 1):
                for seq, annotated_seq in lemm_text_task:
                    for i in range(0, len(seq)):
                        for j in range(1, self.phraseologisms[2].map(len).max() + 1):
                            if (i + j) <= len(seq):
                                if any(
                                    [
                                        set(seq[i : i + j]) == set_f
                                        for set_f in self.phraseologisms[
                                            self.phraseologisms[1] == num_source
                                        ][3]
                                    ]
                                ):
                                    find_elements = seq[i : i + j]
                                    return (
                                        "".join(
                                            self.find_subarray(
                                                find_elements, annotated_seq
                                            )
                                        )
                                        .lower()
                                        .replace(" ", "")
                                    )

        elif "синоним" in question_task_re:
            if type(text_task) == list:
                text_task = " ".join(text_task)
            norm_text_task = self.lemm_and_clear(text_task, self.mystem)

            if "синонимкслов" in question_task_re:
                word = re.findall(r"(?<=к слову).*", question_task)[0]
                words = re.findall("\w+", word)
                words = [x.lower() for x in words]

                set_seq = set(norm_text_task[0])

                select_syn = self.prep_synon[
                    self.prep_synon["MAIN"].isin(words)
                    & self.prep_synon["Синоним"].isin(set_seq)
                ]
                select_syn = select_syn[select_syn["MAIN"] != select_syn["Синоним"]]
                select_syn = select_syn.sort_values("number")
                synon_result = select_syn[["MAIN", "Синоним"]].to_dict("split")["data"]

                if synon_result:
                    tmp = [x for x in synon_result if x[0] == words[0]]
                    if tmp:
                        synon_result = tmp[0]
                    else:
                        synon_result = synon_result[0]

                for norm_w, real_w in norm_text_task[1]:
                    if norm_w == synon_result[1]:
                        return real_w.lower()

            elif re.match(".*синонимич.*пар.*", question_task_re) or (
                "синонимы" in question_task_re
            ):
                result = []

                set_seq = set(norm_text_task[0])
                try:
                    select_syn = self.prep_synon[
                        self.prep_synon["prep_MAIN"].isin(set_seq)
                        & self.prep_synon["prep_Синоним"].isin(set_seq)
                    ]
                    select_syn = select_syn[
                        select_syn["prep_MAIN"] != select_syn["prep_Синоним"]
                    ]
                    select_syn = select_syn.sort_values("number")
                    synon_result = set(
                        select_syn[["prep_MAIN", "prep_Синоним"]].to_dict("split")[
                            "data"
                        ][0]
                    )

                    for norm_w, real_w in norm_text_task[1]:
                        if norm_w in synon_result:
                            result.append(real_w)
                            if len(synon_result) == len(result):
                                break
                    return "".join(result).lower()

                except:
                    pass

            result = []
            set_seq = set(norm_text_task[0])
            list_seq = list(set_seq)
            list_seq_w2v = [self.w2v.word_vector(i) for i in list_seq]
            list_seq = [x[0] for x in zip(list_seq, list_seq_w2v) if x[1] is not None]
            list_seq_w2v = [x for x in list_seq_w2v if x is not None]
            tmp = cosine_distances(np.stack(list_seq_w2v))
            for i in range(tmp.shape[0]):
                tmp[i, i] += 1000
            n1, n2 = np.unravel_index(tmp.argmin(), tmp.shape)

            synon_result = set((list_seq[n1], list_seq[n2]))

            for norm_w, real_w in norm_text_task[1]:
                if norm_w in synon_result:
                    result.append(real_w)
                    if len(synon_result) == len(result):
                        break
            return "".join(result).lower()

    def predict_from_model(self, task):
        prediction = self.suggest_prediction(task)
        if not prediction:
            task_description, sentences = self.parse_task(task)
            prediction = "".join(
                random.choices(
                    [
                        w.strip(punctuation)
                        for w in self.tokenizer.tokenize(random.choice(sentences))
                        if w not in punctuation and not w.isdigit()
                    ],
                    k=2,
                )
            )
        return prediction
