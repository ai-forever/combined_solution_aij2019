# lamoda

import json
import numpy as np
import os
import pandas as pd
import pickle
import re
from collections import defaultdict

from solvers.utils import NgramManager, morph, Word2vecProcessor, CommonData, ALPHABET, AbstractSolver


class Solver(AbstractSolver):
    def __init__(self, seed=42):
        self.seed = seed
        self.init_seed(seed)
        self.morph = morph
        self.ngram_manager = NgramManager()
        self.w2v = Word2vecProcessor()
        self.common_data = CommonData()
        self.is_loaded = False
        self.suffixes = {}
        self.paronyms = defaultdict(list)
        self.word2root = {}
        self.root2word = defaultdict(list)
        self.scaler = None
        self.lr = None
        self.alphabet = ALPHABET

    def predict_from_model(self, task):
        words = get_words(task)
        candidates = self.calc_features(words)
        rows = self.score_candidates(candidates)

        df = pd.DataFrame(rows)
        feature_names = list(range(5, df.shape[1]))
        df["id"] = 0
        _, df = filter_dataframe(df)
        feature_names.append("rank")

        x = self.scaler.transform(df[feature_names])
        df["pred"] = self.lr.predict_proba(x)[:, 1]
        ind = np.argsort(self.lr.predict_proba(x)[:, 1])[-1]
        return rows[ind][3]

    def load(self, path="data/models/solvers/solver5"):

        with open(os.path.join(path, "solver5_logreg.pkl"), "rb") as f:
            self.lr = pickle.load(f)

        with open(os.path.join(path, "solver5.pkl"), "rb") as f:
            self.scaler = pickle.load(f)

        with open(os.path.join(path, "suffixes.txt"), "r", encoding="utf-8") as f:
            for line in f:
                suffix = line.strip()
                if suffix.upper() == suffix or len(suffix) == 0:
                    continue
                self.suffixes[suffix.replace("-", "").replace("j", "")] = 1

        with open(os.path.join(path, "paronyms.csv"), "r", encoding="utf-8") as f:
            for line in f:
                array = line.strip().split()
                for i in range(len(array)):
                    for j in range(len(array)):
                        if i != j:
                            self.paronyms[array[i].lower()].append(array[j].lower())

        with open(
            os.path.join(path, "morph_annotated.txt"), "r", encoding="utf-8"
        ) as f:
            self.annotated = json.load(f)

        for k, v in self.annotated.items():
            array = v.lower().replace("}", "|").split("|")
            root = ""
            for e in array[1:]:
                if (
                    len(e) > 0
                    and e[0].isalpha()
                    and e[-1].isalpha()
                    and e.find("=") == -1
                ):
                    root += e
            self.word2root[k.lower()] = root
            self.root2word[root].append(k.lower())

        self.is_loaded = True

    def get_freq(self, t):
        return self.ngram_manager.gram_freq[
            tuple([self.ngram_manager.word2num.get(e, -1) for e in t])
        ]

    def calc_features(self, words):
        candidates = []
        for word, context in words:
            pars = self.find_paronyms(word.lower())

            features0 = self.calc_ngr_score(
                " ".join(context[0][-3:] + [word.lower()] + context[1][:3]),
                word.lower(),
                word.lower(),
            )
            p0 = self.morph.parse(word.lower())[0].tag
            word_norm = self.morph.normal_forms(word.lower())[0]
            s = " ".join(context[0][-3:] + [word.lower()] + context[1][:3])
            norm_cand_features = dict()
            w_candidates = []
            context_vector = self.w2v.text_vector(
                " ".join(context[0][-6:] + context[1][:6])
            )
            sim0 = self.w2v.distance(self.w2v.word_vector(word.lower()), context_vector)
            features0.append(sim0)

            t0 = (
                p0.aspect,
                p0.case,
                p0.gender,
                p0.number,
                p0.person,
                p0.tense,
                p0.transitivity,
                p0.voice,
            )

            for cand in pars:
                cand_vector = self.w2v.word_vector(cand)
                sim = self.w2v.distance(cand_vector, context_vector)
                cand_norm = self.morph.normal_forms(cand)[0]
                w_info = [
                    word,
                    str((context[0][-3:], context[1][:3])),
                    word_norm,
                    cand,
                    cand_norm,
                ]
                features = []
                ngr_features = self.calc_ngr_score(s, word.lower(), cand)
                features += ngr_features
                features.append(sim)
                features += [
                    custom_damerau_levenshtein_distance(word.lower(), cand)
                    / min(len(word), len(cand)),
                    custom_damerau_levenshtein_distance(word.lower()[-2:], cand[-2:]),
                ]
                p = self.morph.parse(cand)[0].tag
                coef = (
                    1
                    if p.POS == p0.POS
                    or p.POS in {"PRTF", "ADJF"}
                    and p0.POS in {"PRTF", "ADJF"}
                    else 0.01
                )
                score0 = features0[0]
                score = ngr_features[0]
                features += [
                    coef,
                    coef
                    * pars[cand]
                    * (
                        1 * (score - score0) / (1 + score0)
                        + 0.1 * (max(0, score - score0))
                    ),
                ]
                features += [int(pars[cand] == e) for e in range(1, 6)]

                t = (
                    p.aspect,
                    p.case,
                    p.gender,
                    p.number,
                    p.person,
                    p.tense,
                    p.transitivity,
                    p.voice,
                )
                for e1, e2 in zip(t0, t):
                    features.append(int(e1 == e2))
                if cand_norm not in norm_cand_features:
                    norm_cand_features[cand_norm] = features[:5].copy()
                else:
                    for i in range(5):
                        norm_cand_features[cand_norm][i] += features[i]
                w_candidates.append(w_info + features0 + features)
            for i in range(len(w_candidates)):
                w_candidates[i] += norm_cand_features[w_candidates[i][4]]
            candidates += w_candidates
        return candidates

    def score_candidates(self, candidates):
        return sorted(candidates, key=lambda x: -x[-1])

    def calc_ngr_score(self, s, from_w, to_w):
        s = (
            (" " + s + " ")
            .lower()
            .replace(" " + from_w + " ", " " + to_w + " ")
            .strip()
        )
        array = s.split()

        score1 = 0
        for i in range(0, len(array)):
            if array[i].lower().replace("ё", "е") == to_w.lower().replace("ё", "е"):
                score1 += np.log1p(min(100, self.get_freq((array[i],))))
        score2 = 0
        score2f = 0
        cc2 = 0
        for i in range(1, len(array)):
            if array[i].lower().replace("ё", "е") == to_w.lower().replace(
                "ё", "е"
            ) or array[i - 1].lower().replace("ё", "е") == to_w.lower().replace(
                "ё", "е"
            ):
                score2 += np.log1p(
                    min(1000, self.get_freq((array[i - 1], "", array[i])))
                )

                score2f += np.log1p(
                    min(1000, self.get_freq(("-", array[i - 1][-2:], "", array[i])))
                )
                score2f += np.log1p(
                    min(1000, self.get_freq(("-", array[i - 1], "", array[i][-2:])))
                )
                cc2 += 1
        if cc2 > 0:
            score2 /= cc2
            score2f /= cc2
        score3 = 0
        cc3 = 0
        for i in range(2, len(array)):
            if (
                array[i].lower().replace("ё", "е") == to_w.lower().replace("ё", "е")
                or array[i - 1].lower().replace("ё", "е")
                == to_w.lower().replace("ё", "е")
                or array[i - 2].lower().replace("ё", "е")
                == to_w.lower().replace("ё", "е")
            ):
                score3 += np.log1p(
                    min(
                        1000,
                        self.get_freq((array[i - 2], "", array[i - 1], "", array[i])),
                    )
                )
                cc3 += 1
        if cc3 > 0:
            score3 /= cc3

        score = 100 * score2 + 1000 * score3
        if score != -1:
            score += 10 * score2f
        if score == 0:
            score = 0.1 * score1
        return [score, score1, score2, score2f, score3]

    def find_paronyms(self, word):
        res = dict()
        if word in self.paronyms:
            for p in self.paronyms[word]:
                res[p] = 1
        for nf in self.morph.normal_forms(word):
            if nf in self.paronyms:
                for p in self.paronyms[nf]:
                    res[p] = 1

        for e in self.common_data.prefixes:
            if e in ("не", "анти"):
                continue
            if word.startswith(e):
                for e1 in self.common_data.prefixes:
                    if e != e1:
                        wn = e1 + word[len(e) :]
                        if self.morph.word_is_known(wn):
                            res[wn] = 2

        for e in self.suffixes:
            ind = word.rfind(e)
            if ind != -1:
                for e1 in self.suffixes:
                    if e != e1:
                        wn = word[:ind] + e1 + word[ind + len(e) :]
                        if self.morph.word_is_known(wn):
                            res[wn] = 3
                        if self.morph.word_is_known(wn.replace("ый", "ий")):
                            res[wn.replace("ый", "ий")] = 3
                        if self.morph.word_is_known(wn.replace("ий", "ый")):
                            res[wn.replace("ий", "ый")] = 3

        for ww in [word, self.morph.normal_forms(word)[0]]:
            if ww in self.word2root:
                root = self.word2root[ww]
                for root in [root, root.replace("ь", "")]:
                    k = 0
                    for w in self.root2word[root]:
                        if k == 1000:
                            break
                        if self.morph.word_is_known(w):
                            res[w] = 4
                        k += 1

        for ww in [word]:
            for ch in [""] + list(self.alphabet):
                for i in range(len(ww)):
                    wn = ww[:i] + ch + ww[i:]
                    if self.morph.word_is_known(wn):
                        res[wn] = 1
                    wn = ww[:i] + ch + ww[i + 1 :]
                    if self.morph.word_is_known(wn):
                        res[wn] = 5

        if word in res:
            del res[word]

        nfs = dict()
        for w in res:
            nf = self.morph.normal_forms(w)[0]
            nfs[nf] = min(nfs.get(nf, 6), res[w])
        for nf in nfs:
            if nf in self.common_data.norm2word:
                for w in {e.lower() for e in self.common_data.norm2word[nf]}:
                    res[w] = min(res.get(w, 6), nfs[nf])

        res_final = dict()
        init_nfs = set(self.morph.normal_forms(word))
        for w in res:
            fl = 1
            for e in self.morph.normal_forms(w):
                if e in init_nfs:
                    fl = 0
                    break
            if fl == 1:
                res_final[w] = res[w]

        return res_final


def clean_with_slot(word):
    return " ".join(re.sub("[^а-яё\. ]+", " ", word.lower()).split())


def clean(word):
    return " ".join(re.sub("[^а-яё ]+", " ", word.lower()).split())


def get_words(task):
    array = (
        task["text"]
        .replace("\xa0", "\n")
        .replace("НЕВЕРНО", "неверно")
        .replace("НЕПРАВИЛЬНО", "неправильно")
        .split(".")
    )

    words = []
    for e in array:
        context = [[]]
        e = (
            e.replace("?", ".")
            .replace("!", ".")
            .replace(".", " . ")
            .replace(",", " , ")
            .replace(":", " ")
            .replace("–", " ")
            .replace("—", " ")
            .replace('"', " ")
            .replace("«", " ")
            .replace("»", " ")
            .replace("( ", "(")
            .replace(" )", ")")
            .strip()
            .strip(".")
        )
        longest_word = ""
        for w in e.split():
            if len(w) > 1 and w.upper() == w and not w.isnumeric():
                if len(w) > len(longest_word):
                    longest_word = w
        word = None
        for w in e.split():
            if (
                len(w) > 1
                and w.upper() == w
                and not w.isnumeric()
                and w == longest_word
            ):
                word = w
                context.append([])
            else:
                context[len(context) - 1].append(w.lower())
        if word is not None:
            words.append((word, context))
    return words


def build_dataset(self, task):
    words = get_words(task)
    candidates = self.calc_features(words)
    return candidates


def filter_dataframe(df):
    f = 13
    df["pre_pred"] = df[f]
    df["rank"] = (
        df[["id", 0, "pre_pred"]]
        .groupby(["id", 0])["pre_pred"]
        .rank(ascending=False, method="first")
    )
    ind = (df["rank"] <= 10) & (df[19] == 1) & (df[29] == 1)
    ind |= df[4].isin(df[ind][4].unique()) & (df["rank"] <= 50)
    return ind, df


def remove_common_prefix_suffix(s1, s2):
    left = 0
    while s1[left] == s2[left]:
        left += 1
        if left == len(s1) or left == len(s2):
            break
    s1 = s1[left:]
    s2 = s2[left:]
    if len(s1) > 0 and len(s2) > 0:
        right = 0
        while s1[len(s1) - right - 1] == s2[len(s2) - right - 1]:
            right += 1
            if right == len(s1) or right == len(s2):
                break
        s1 = s1[: len(s1) - right]
        s2 = s2[: len(s2) - right]
    return s1, s2


def custom_damerau_levenshtein_distance(s1, s2):
    s1, s2 = remove_common_prefix_suffix(s1, s2)

    len1 = len(s1)
    len2 = len(s2)

    res = 0
    score = [[0] * (len2 + 1) for _ in range(len1 + 1)]

    for i in range(len1 + 1):
        for j in range(len2 + 1):
            if i == 0 or j == 0:
                if i == 0:
                    score[i][j] = j
                else:
                    score[i][j] = i
            elif s1[i - 1] == s2[j - 1]:
                score[i][j] = score[i - 1][j - 1]
            else:
                score[i][j] = min(
                    score[i - 1][j] + 1, score[i][j - 1] + 1, score[i - 1][j - 1] + 1
                )
                if (
                    i > 1
                    and j > 1
                    and s1[i - 2] == s2[j - 1]
                    and s1[i - 1] == s2[j - 2]
                ):
                    score[i][j] = min(score[i][j], score[i - 2][j - 2] + 1)

        res = score[len1][len2]
    return res
