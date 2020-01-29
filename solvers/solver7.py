# lamoda

import json
import numpy as np
import os

from solvers.solver_helpers import NgramManager, CommonData, morph, ALPHABET, AbstractSolver


class Solver(AbstractSolver):
    def __init__(self):
        self.alphabet = ALPHABET
        self.ngram_manager = NgramManager()
        self.common_data = CommonData()
        self.morph = morph
        self.solutions = {}
        self.from_tasks = {}

    def get_neighbours(self, word):
        res = []
        for ch in list(self.alphabet) + [""]:
            for i in range(max(4, int(len(word)) - 3), len(word)):
                nw = word[:i].lower() + ch + word[i + 1:].lower().replace("ё", "е")
                if self.morph.word_is_known(nw):
                    res.append(nw)
            if len(word) >= 5:
                nw = word[:-2].lower() + ch
                if self.morph.word_is_known(nw):
                    res.append(nw)
        return res

    def predict_from_model(self, task, return_row=False):
        task_text = task["text"].replace("\xa0", "\n")
        array = [e.strip() for e in task_text.split("\n")]
        words = []
        prediction = ""
        row = ""
        for word in array:
            if word in self.solutions:
                if self.solutions[word] != "":
                    prediction = "+" + self.solutions[word]
                continue
            q, pred_w = self.score_row(word)
            if q is not None:
                words.append(q)
                if pred_w != "" and not prediction.startswith("+"):
                    prediction = pred_w
                    row = word

        prediction = prediction.strip("+")

        if not prediction:
            scores = dict()
            for word in array:
                q = self.extract_words(word)
                w_scores = {}
                if len(q[0]) > 0:
                    longest = sorted(q[0], key=lambda x: -len(x))[0].lower()
                    if longest == "их":
                        continue
                    score0 = max(
                        self.calc_ngr_score(word, longest, longest),
                        self.calc_ngr_score(
                            word.lower().replace("ё", "е"), longest, longest
                        ),
                    )
                    candidates = set(
                        [
                            e.lower().replace("ё", "е")
                            for e in self.common_data.norm2word.get(
                                self.morph.normal_forms(longest)[0], []
                            )
                        ]
                    )
                    candidates = {e: 1 for e in candidates}
                    for e in set(self.get_neighbours(longest.lower())):
                        if e not in candidates:
                            candidates[e] = 0.5
                    p0 = self.morph.parse(longest)[0].tag.POS
                    for ww0 in sorted(candidates):
                        ww = ww0.lower()
                        if ww == longest.replace("ё", "е"):
                            continue
                        p = self.morph.parse(word)[0].tag.POS
                        coef = 1 if p == p0 else 0.01
                        if longest.lower() in set(self.from_tasks.values()):
                            coef /= 100
                        score = self.calc_ngr_score(word, longest, ww)
                        w_scores[(word, ww)] = (
                            score,
                            score0,
                            coef
                            * candidates[ww0]
                            * (
                                1 * (score - score0) / (1 + score0)
                                + 0.1 * (max(0, score - score0))
                            ),
                        )
                        scores[(word, ww)] = w_scores[(word, ww)]

            row, prediction = sorted(scores.items(), key=lambda x: -(x[1][2]))[0][0]
        if return_row:
            return prediction, row
        return prediction

    def load(self, path="data/models/solvers/solver7"):
        with open(os.path.join(path, "solver7.json"), "r", encoding="utf-8") as f:
            self.solutions = {k.strip(): v for k, v in json.load(f).items()}
        with open(os.path.join(path, "solver7.txt"), "r", encoding="utf-8") as f:
            for line in f:
                array = line.strip().lower().replace("ё", "е").split("\t")
                if array[1].find("(те)") != -1:
                    self.from_tasks[array[0]] = array[1].replace("(те)", "")
                    self.from_tasks[array[0] + "те"] = array[1].replace("(те)", "те")
                else:
                    self.from_tasks[array[0]] = array[1]
        self.is_loaded = True

    def get_freq(self, t):
        return self.ngram_manager.gram_freq[tuple([self.ngram_manager.word2num.get(e, -1) for e in t])]

    def calc_ngr_score(self, task_text, from_word, to_word):
        task_text = (
            (" " + task_text + " ")
            .lower()
            .replace(" " + from_word + " ", " " + to_word + " ")
            .strip()
        )

        array = task_text.split()

        score1 = 0
        for i in range(0, len(array)):
            if array[i].lower().replace("ё", "е") == to_word.lower().replace("ё", "е"):
                score1 += np.log1p(min(100, self.get_freq((array[i],))))
        score2 = 0
        score2f = 0
        cc2 = 0
        for i in range(1, len(array)):
            if array[i].lower().replace("ё", "е") == to_word.lower().replace(
                "ё", "е"
            ) or array[i - 1].lower().replace("ё", "е") == to_word.lower().replace(
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
                # array[i].lower().replace('ё', 'е') == to_word.lower().replace('ё', 'е')
                array[i - 1].lower().replace("ё", "е")
                == to_word.lower().replace("ё", "е")
                # or array[i - 2].lower().replace('ё', 'е') == to_word.lower().replace('ё', 'е')
            ):
                score3 += np.log1p(
                    min(
                        1000,
                        self.get_freq((array[i - 2], "", array[i - 1], "", array[i])),
                    )
                )
                cc3 += 1
            elif len(array) == 3:
                if (
                    # array[i].lower().replace('ё', 'е') == to_word.lower().replace('ё', 'е')
                    array[i - 1].lower().replace("ё", "е")
                    == to_word.lower().replace("ё", "е")
                    # or array[i - 2].lower().replace('ё', 'е') == to_word.lower().replace('ё', 'е')
                ):
                    score3 += np.log1p(
                        min(
                            1000,
                            self.get_freq(
                                (array[i - 2], "", array[i - 1], "", array[i])
                            ),
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
        return score

    def score_row(self, word):
        q = self.extract_words(word)
        if len(q[0]) == 0:
            return None, None

        prediction = ""

        is_verb = 0
        is_noun = 0
        for e in word.split():
            try:
                if self.morph.parse(e)[0].tag.POS in {"VERB", "INFN"}:
                    is_verb = 1
                if self.morph.parse(e)[0].tag.POS == "NOUN":
                    is_noun = 1
                    noun_gender = self.morph.parse(e)[0].tag.gender
                    if "вода" in self.morph.normal_forms(e):
                        noun_gender = "femn"
            except:
                pass

        fl_cur = 0
        for e in q[0]:
            if prediction != "":
                continue
            if (
                    self.morph.word_is_known(e.replace("ё", "е")) == 0
                    and self.morph.word_is_known(e) == 0
            ):
                swp = [
                    ("к", "ч"),
                    ("к", "чь"),
                    ("г", "ж"),
                    ("жь", "г"),
                    ("ех", "ем"),
                    ("ёх", "ём"),
                    ("яя", "ив"),
                    ("ти", "ть"),
                    ("ти", "тью"),
                    ("ть", "тью"),
                    ("ше", "ее"),
                    ("нул", ""),
                    ("ну", ""),
                    ("й", ""),
                    ("ок", "ков"),
                    ("х", "мя"),
                    ("и", "ее"),
                    ("ан", "ен"),
                    ("ьми", "емь"),
                    ("еми", "емью"),
                    ("ёх", "ьмя"),
                    ("я", "ью"),
                    ("и", "ью"),
                    ("ста", "сот"),
                    ("нуло", "ло"),
                    ("ьше", "ее"),
                    ("ьше", "нее"),
                    ("ова", "у"),
                ]
                for c1, c2 in swp:
                    if prediction != "":
                        break
                    nw = e.lower().replace(c1, c2)
                    if self.morph.word_is_known(nw.replace("ё", "е")):
                        prediction = nw
                        fl_cur = 1
                        break
                    if c2 != "":
                        nw = e.lower().replace(c2, c1)
                        if self.morph.word_is_known(nw.replace("ё", "е")):
                            prediction = nw
                            fl_cur = 1
                            break
                for c1 in ["ов", "нул"]:
                    if prediction != "":
                        break
                    if e.lower().endswith(c1):
                        nw = e.lower()[: -len(c1)]
                        if self.morph.word_is_known(nw.replace("ё", "е")):
                            prediction = nw
                            fl_cur = 1
                            break
                    nw = e.lower() + c1
                    if self.morph.word_is_known(nw.replace("ё", "е")):
                        prediction = nw
                        fl_cur = 1
                        break

                for ch in list(self.alphabet) + [""]:
                    if prediction != "":
                        break
                    nw = e.lower()[:-1] + ch
                    if self.morph.word_is_known(nw.replace("ё", "е")):
                        prediction = nw
                        fl_cur = 1
                        break

                for i in range(len(e)):
                    if prediction != "":
                        break
                    nw = e[:i].lower() + e[i + 1:].lower()
                    if self.morph.word_is_known(nw.replace("ё", "е")):
                        prediction = nw
                        fl_cur = 1
                        break

                for i in range(len(e)):
                    if prediction != "":
                        break
                    for ch in list(self.alphabet):
                        nw = e[:i].lower() + ch + e[i + 1:].lower()
                        if self.morph.word_is_known(nw.replace("ё", "е")):
                            prediction = nw
                            fl_cur = 1
                            break
                        if prediction != "":
                            break
        if prediction != "":
            if not self.morph.word_is_known(prediction) and self.morph.word_is_known(
                    prediction.replace("ё", "е")
            ):
                prediction = prediction.replace("ё", "е")
            if prediction.lower().replace("ё", "е") == q[0][0].lower().replace("ё", "е"):
                prediction = ""

        if prediction != "" and fl_cur == 1:
            try:
                pred_p = self.morph.parse(prediction)[0]
                if pred_p.tag.POS in {"ADJF", "NUMR"}:
                    for e in word.split():
                        if e.lower() == e:
                            p = self.morph.parse(e)[0]
                            if p.tag.POS == "NOUN":
                                case = p.tag.case
                                prediction = pred_p.inflect({case}).word
                elif pred_p.tag.POS in {"NOUN"}:
                    for e in word.split():
                        if e.lower() == e:
                            p = self.morph.parse(e)[0]
                            if p.tag.POS == "ADJF":
                                prediction = pred_p.inflect({p.tag.case, p.tag.number}).word
                            elif p.tag.POS == "NOUN":
                                if p.tag.case == "nomn":
                                    prediction = pred_p.inflect({"gent", "plur"}).word
                            else:
                                for prep in self.preps:
                                    if (" " + word + " ").find(" " + prep + " ") != -1:
                                        #                                        print('%%%', prep, self.preps[prep][-1])
                                        prediction = pred_p.inflect(
                                            {self.preps[prep][-1]}
                                        ).word

            except:
                pass

        if (" " + word).lower().find(" ихн") != -1:
            prediction = "их"
        if word.lower().find("тюлью") != -1:
            prediction = "тюлем"
        if word.lower().find("поклади") != -1:
            prediction = "положи"
        if (" " + word.lower() + " ").find(" ложи ") != -1:
            prediction = "клади"
        if word.lower().find("покладите") != -1:
            prediction = "положите"
        if (" " + word.lower() + " ").find(" ложите ") != -1:
            prediction = "кладите"

        if (" " + word.lower() + " ").find(" езжай ") != -1:
            prediction = "поезжай"
        if (" " + word.lower() + " ").find(" езжайте ") != -1:
            prediction = "поезжайте"

        if word.find("ПОЛТОРАСТА") != -1 and (" " + word + " ").find("ПОЛТОРАСТА ") == -1:
            if is_noun and noun_gender not in {"nomn", "accs"}:
                prediction = "+полутораста"

        for pr in [""] + list(self.common_data.prefixes):
            for end in ["", "те", "тесь"]:
                for bw in ["едь", "ехай"]:
                    if (" " + word.lower() + " ").find(" " + pr + bw + end + " ") != -1:
                        if pr != "":
                            prediction = pr + "езжай" + end
                        else:
                            prediction = "поезжай" + end
                    if (" " + word.lower() + " ").find(
                            " " + pr + "ъ" + bw + end + " "
                    ) != -1:
                        if pr != "":
                            prediction = pr + "ъ" + "езжай" + end
                        else:
                            prediction = "поезжай" + end

        if word.lower().find("яблоней") != -1:
            prediction = "яблонь"
        if word.lower().find("туфлей") != -1:
            prediction = "туфель"
        if word.lower().find("напополам") != -1:
            prediction = "пополам"

        for e in self.from_tasks:
            if (" " + word.replace("Ё", "Е") + " ").find(" " + e.upper() + " ") != -1:
                prediction = "+" + self.from_tasks[e]

        if (" " + word.lower() + " ").find(" ста ") != -1 and (
                " " + word.lower() + " "
        ).find("ого ") != -1:
            prediction = "+сто"
        if (" " + word + " ").find(" СТАХ ") != -1 and (" " + word + " ").find("ах ") != -1:
            prediction = "+ста"
        if (" " + word + " ").find(" в ТЫСЯЧУ ") != -1 and (" " + word.lower() + " ").find(
                "ом "
        ) != -1:
            prediction = "+тысяча"

        for z in ["тысяч", "десятк", "сотн", "сотен"]:
            if word.find(f"более ДВЕ {z}") != -1:
                prediction = "+двух"
            if word.find(f"более ТРИ {z}") != -1:
                prediction = "+трех"
            if word.find(f"более ЧЕТЫРЕ {z}") != -1:
                prediction = "+четырех"

        if word.find(f"более ДВЕСТИ") != -1:
            prediction = "+двухсот"
        if word.find(f"более ТРИСТА") != -1:
            prediction = "+трехсот"
        if word.find(f"более ЧЕТЫРЕСТА") != -1:
            prediction = "+четырехсот"
        if word.find(f"более ПЯТЬСОТ") != -1:
            prediction = "+пятисот"
        if word.find(f"более ШЕСТЬСОТ") != -1:
            prediction = "+шестисот"
        if word.find(f"более СЕМЬСОТ") != -1:
            prediction = "+семисот"
        if word.find(f"более ВОСЕМЬСОТ") != -1:
            prediction = "+восьмисот"
        if word.find(f"более ДЕВЯТЬСОТ") != -1:
            prediction = "+девятисот"

        if word.find("на ШКАФЕ") != -1:
            prediction = "+шкафу"

        if (" " + word.replace("ё", "е") + " ").find(" СМОТРЕВ вперед ") != -1:
            prediction = "смотря"

        if word.lower().find("двух тысячи") != -1:
            prediction = "две"
        if word.lower().replace("ё", "е").find("трех тысячи") != -1:
            prediction = "три"
        if word.lower().replace("ё", "е").find("четырех тысячи") != -1:
            prediction = "четыре"
        if word.lower().find("в двух тысяч ") != -1:
            prediction = "дветысячи"
        if word.lower().replace("ё", "е").find("в трех тысяч ") != -1:
            prediction = "тритысячи"
        if word.lower().replace("ё", "е").find("в четырех тысяч ") != -1:
            prediction = "четыретысячи"
        if (
                prediction
                not in {
            "их",
            "две",
            "три",
            "четыре",
            "тюлем",
            "дветысячи",
            "тритысячи",
            "четыретысячи",
            "положи",
            "клади",
            "положите",
            "кладите",
            "яблонь",
            "туфель",
            "пополам",
        }
                and prediction.find("езжа") == -1
                and (prediction == "" or prediction[0] != "+")
        ):
            t_pred = ""
            if word.lower().find("менее") != -1 or word.lower().find("более") != -1:
                is_supr = 0
                for e in q[0]:
                    if e in ["МЕНЕЕ", "НАИМЕНЕЕ", "БОЛЕЕ", "НАИБОЛЕЕ"]:
                        continue
                    for p in self.morph.parse(e):
                        if str(p.tag).find("Supr") != -1:
                            t_pred = p.normal_form
                            is_supr = 1
                        if str(p.tag).find("Cmp2") != -1:
                            t_pred = p.normal_form
                        if p.tag.POS == "COMP":
                            t_pred = p.normal_form
                        if e.lower() == "выше":
                            t_pred = "высокий"
                        if e.lower() == "позднее":
                            t_pred = "поздний"
                        if t_pred != "":
                            if is_verb == 1:
                                t_pred = t_pred[:-2] + "о"
                            elif is_noun == 1:
                                try:
                                    fl = 0
                                    for ee in ["МЕНЕЕ", "НАИМЕНЕЕ", "БОЛЕЕ", "НАИБОЛЕЕ"]:
                                        if ee in q[0]:
                                            fl = 1
                                    if fl == 0 or t_pred in {"высокий", "поздний"}:
                                        t_pred = (
                                            self.morph.parse(t_pred)[0]
                                                .inflect({noun_gender})
                                                .word
                                        )
                                    else:
                                        t_pred = p.inflect({noun_gender}).word
                                except:
                                    print("RAISE")
                                    pass
                            if is_supr == 0:
                                for ee in ["МЕНЕЕ", "НАИМЕНЕЕ", "БОЛЕЕ", "НАИБОЛЕЕ"]:
                                    if ee in q[0]:
                                        t_pred = ee.lower() + t_pred
                                        break
                            prediction = t_pred
                            break
        return q, prediction

    @staticmethod
    def extract_words(t):
        if t.lower().startswith("в одном из"):
            return ([], t)
        if t.lower().find("номер в банке") != -1:
            return ([], t)
        return ([e for e in t.split() if e.upper() == e and e.isalpha()], t)
