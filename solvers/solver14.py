# lamoda

import random
import numpy as np
from solvers.utils import NgramManager, morph


def parse_sentence(e):
    last_word = ""
    last_c = ""
    arr = []
    fl = 0
    s = []
    for c in e:
        if c == "(" and last_c != ")":
            fl = 1
            if last_word != "":
                s[-1] = s[-1].rstrip()[: -len(last_word.strip())]
                arr.append(last_word.strip() + "?")
            else:
                arr.append("")
        elif c in {" ", ",", ".", "?", "!", ":", '"', "–", "«", "»"}:
            if fl == 1:
                if arr[-1].strip("?").find("?") != -1:
                    fl = 0
            else:
                fl = 0
        else:
            if fl == 1 and (c == ")" or c.isalpha()):
                arr[-1] += c.replace(")", "?")
        if c.isalpha() and c.upper() == c:
            last_word += c
        else:
            last_word = ""
        last_c = c
        if fl == 0:
            while len(arr) >= len(s):
                s.append("")
            s[len(arr)] += c
    arr = [e.strip("?") for e in arr]
    s.append("")
    s.append("")
    s.append("")
    s.append("")
    return arr, s


class Solver(object):
    def __init__(self, seed=42):
        self.seed = seed
        self.init_seed()
        self.morph = morph
        self.ngram_manager = NgramManager()
        self.is_loaded = True

    def init_seed(self):
        return random.seed(self.seed)

    def get_freq(self, t):
        return self.ngram_manager.gram_freq[tuple([self.ngram_manager.word2num.get(e, -1) for e in t])]

    def predict_from_model(self, task):
        sentences = task["text"].replace("\xa0", "\n").replace("ё", "е").split("\n")
        if len(sentences) <= 2:
            sentences = (
                task["text"]
                .replace("\xa0", "\n")
                .replace("?", ".")
                .replace("!", ".")
                .split(".")
            )
        sign = 1
        if sentences[0].lower().find("раздельно") != -1:
            sign = -1
        final_scores = dict()
        for e in sentences:
            try:
                arr, s = parse_sentence(e)
            except:
                print("PARSE_FAIL")
                print(e)
                continue

            s_left = s.copy()
            s_right = s.copy()
            for i in range(len(arr) - 1):
                if s[i + 1].strip() in {",", ""}:
                    s_right[i + 1] = s_right[i + 1] + arr[i + 1].replace("?", "")
            for i in range(1, len(arr)):
                if s[i].strip() in {",", ""}:
                    s_left[i] = arr[i - 1].replace("?", "") + s_left[i]

            scores = [0] * len(arr)
            for i in range(len(arr)):
                v1 = (arr[i].lower().replace("?", ""),)
                v21 = (arr[i].lower().replace("?", "-"),)
                v2 = []
                for _ in arr[i].lower().split("?"):
                    if len(v2) > 0:
                        v2.append("")
                    v2.append(_)
                v2 = tuple(v2)

                reg1 = 1000
                scores[i] += 0.5 * (
                    np.log1p(min(reg1, self.get_freq(v1)))
                    - np.log1p(min(reg1, self.get_freq(v2)))
                    - np.log1p(min(reg1, self.get_freq(v21)))
                )

                scores[i] += 1.0 * int(self.morph.word_is_known(v1[0])) - int(
                    self.morph.word_is_known(v2[0])
                )

                left = (s_left[i] if i == 1 else s[i]).lower()
                left = (
                    left.replace("?", ".")
                    .replace("!", ".")
                    .replace(".", " . ")
                    .replace(",", " , ")
                    .replace("–", " ")
                    .replace(":", " ")
                    .replace('"', " ")
                    .replace("«", " ")
                    .replace("»", " ")
                    .strip()
                    .strip(".")
                    .split()
                )
                right = (s_right[i + 1] if i == 0 else s[i + 1]).lower()
                right = (
                    right.replace("?", ".")
                    .replace("!", ".")
                    .replace(".", " . ")
                    .replace(",", " , ")
                    .replace("–", " ")
                    .replace(":", " ")
                    .replace('"', " ")
                    .replace("«", " ")
                    .replace("»", " ")
                    .strip()
                    .strip(".")
                    .split()
                )

                kk = 0
                if len(left) > 0:
                    kk += 1
                if len(right) > 0:
                    kk += 1
                reg2 = 1e9
                if len(left) > 0:
                    t = (left[-1], "")
                    if left[-1] == "," and len(left) >= 2:
                        t = (left[-2], ",")
                    scores[i] += (
                        np.log1p(min(reg2, self.get_freq(t + v1)))
                        - np.log1p(min(reg2, self.get_freq(t + v2)))
                        - np.log1p(min(reg2, self.get_freq(t + v21)))
                    ) / kk

                if len(right) > 0:
                    t = ("", right[0])
                    if right[0] == "," and len(right) >= 2:
                        t = (",", right[1])
                    scores[i] += (
                        np.log1p(min(reg2, self.get_freq(v1 + t)))
                        - np.log1p(min(reg2, self.get_freq(v2 + t)))
                        - np.log1p(min(reg2, self.get_freq(v21 + t)))
                    ) / kk
                scores[i] *= sign
            if len(arr) > 0:
                final_scores[
                    "".join([e.replace("?", "").lower() for e in arr])
                ] = np.min(scores)

            if len(arr) != 2 and len(arr) != 0:
                pass

        pred = sorted(final_scores.items(), key=lambda x: -x[1])[0][0]
        return pred

    def fit(self, tasks):
        pass

    def load(self, path=""):
        pass

    def save(self, path=""):
        pass
