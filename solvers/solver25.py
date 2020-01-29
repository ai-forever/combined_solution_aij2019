# niw

import os
import random
import re

from solvers.solver_helpers import morph


class Solver(object):
    def __init__(self, seed=42):
        self.seed = seed
        self.init_seed()
        self.morph = morph
        self.is_loaded = False
        self.word_dictionary = {}
        self.synonyms = None

    def init_seed(self):
        random.seed(self.seed)

    def fit(self, tasks):
        pass

    def load(self, path="data/models/solvers/solver25"):
        with open(os.path.join(path, "words.csv"), encoding="utf-8") as f:
            for line in f:
                line = line.replace("\n", "").split("|")
                self.word_dictionary[line[0]] = line[1]
        self.synonyms = open(
            os.path.join(path, "synmaster.txt"), "r", encoding="utf8"
        ).readlines()
        self.synonyms = [
            re.sub("\.", "", t.lower().strip("\n")).split("|")
            for t in self.synonyms
            if len(t) > 5
        ]
        self.is_loaded = True

    def save(self, path="data/models/solvers/solver25"):
        pass

    def predict_from_model(self, task):
        words = (
            task["text"]
            .replace(",", " ,")
            .replace(".", " .")
            .replace(":", " :")
            .replace("!", " !")
            .replace("?", " ?")
            .replace("\n", " ")
            .replace("  ", " ")
            .replace("\t", " ")
            .replace("«", "")
            .replace("»", "")
        )
        z = []
        lik_pov, synonym, form_word, takoe, odn = False, False, False, False, False
        kol_task = 0
        if "частиц" in task["text"]:
            kol_task += 1
            z.extend([2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
        if "союзное слово" in task["text"]:
            kol_task += 1
            z.extend([12, 13, 14])
        if "сочинительн" in task["text"]:
            kol_task += 1
            z.extend([15, 16, 17])
        if "подчинительн" in task["text"]:
            kol_task += 1
            z.append(14)
        if "противительн" in task["text"]:
            kol_task += 1
            z.append(16)
        if "союз" in task["text"] and not 14 in z and not 16 in z:
            kol_task += 1
            z.extend([14, 15, 16, 17])
        if "предлог" in task["text"]:
            kol_task += 1
            z.append(19)
        if "указательн" in task["text"] and not "нареч" in task["text"]:
            kol_task += 1
            z.append(18)
        if "относительное местоимение" in task["text"]:
            kol_task += 1
            z.append(12)
        if "определительн" in task["text"]:
            kol_task += 1
            z.append(20)
        if "личного" in task["text"]:
            kol_task += 1
            z.append(24)
        if "личных местоимений" in task["text"]:
            kol_task += 2
            z.append(24)
            z.append(24)
        if "вводное слово" in task["text"] or "вводного слова" in task["text"]:
            kol_task += 1
            z.append(21)
        if "лексическ" in task["text"]:
            kol_task += 1
            lik_pov = True
        if "фразеологизм" in task["text"]:
            kol_task += 1
            z.append(23)
        if "притяжательн" in task["text"]:
            kol_task += 1
            z.append(22)
        if "синоним" in task["text"]:
            kol_task += 1
            synonym = True
        if "формы слова" in task["text"] or "форм слова" in task["text"]:
            form_word = True
        if "однокоренных слов" in task["text"]:
            odn = True
        if not z and "союза" in task["text"]:
            z.extend([14, 15, 16, 17])
            kol_task += 1
        if "нареч" in task["text"]:
            z.append(1)
            kol_task += 1
        all_my_dect_word = []
        my_dect_word = []
        type_my_dect_word = []
        for i in range(len(z)):
            if (
                z[i] == 1
                or z[i] == 2
                or z[i] == 12
                or (z[i] == 15 and not 14 in z)
                or z[i] > 17
            ):
                my_dect_word = []
            for key in self.word_dictionary.keys():
                if int(self.word_dictionary[key]) == z[i]:
                    my_dect_word.append(key.strip())
            if z[i] == 1 or z[i] == 11 or (z[i] == 14 and not 15 in z) or z[i] > 16:
                all_my_dect_word.append(my_dect_word)
                type_my_dect_word.append(z[i])
            if z[i] == 16 and not 17 in z:
                all_my_dect_word.append(my_dect_word)
                type_my_dect_word.append(z[i])
        num_sen = re.search("(?<=предложений )[0-9\–\-\—\−]*", words).group(0)
        if "–" in num_sen:
            num_sen = num_sen.split("–")
        elif "-" in num_sen:
            num_sen = num_sen.split("-")
        elif "—" in num_sen:
            num_sen = num_sen.split("—")
        elif "−" in num_sen:
            num_sen = num_sen.split("−")
        else:
            num_sen[1, 10]
        num_sen1 = int(num_sen[0])
        num_sen2 = int(num_sen[1])
        words = re.sub("\([^0-9]*?\)", "", words)
        sent = {}
        words = re.findall("[0-9 ][0-9 ]*\).*?\(", words)
        for word in words:
            z = (
                re.search("[0-9 ][0-9 ]*\)", word)
                .group(0)
                .replace(")", "")
                .replace(" ", "")
            )
            sent[z] = (
                re.sub("[0-9]*?", "", word)
                .replace("(", "")
                .replace(")", "")
                .replace("–", "")
            )
        words2 = sent
        indexes = []
        maybe1 = []
        maybe2 = []
        maybe0 = []
        maybe3 = []
        takoe_id = 1000
        takoe_id_min = 1000
        for j in words2:
            if int(j) > num_sen1 and int(j) <= num_sen2:
                conditions = 0
                conditions2 = 0
                words2[j] = words2[j].lower()
                s = " " + words2[j][:60] + " "
                mem_w = ""
                for my_dect_word in all_my_dect_word:
                    est_vo_vsem = False
                    if "и" in my_dect_word:
                        s = " " + words2[j][:21] + " "
                    else:
                        s = " " + words2[j][:50] + " "
                    for my_word in my_dect_word:
                        if my_word == "тем" and "тем , как" in s.lower():
                            continue
                        if my_word == "своего" and "он" in s.lower():
                            continue
                        if my_word == "его" and ":" in s.lower():
                            continue
                        if my_word == "ты" and "—" in s.lower():
                            continue
                        if my_word == "недавно":
                            continue
                        if " " + my_word + " " in (
                            " " + words2[j].lower() + " "
                        ) or " " + my_word + " " in (
                            " " + words2[j].lower().replace("ё", "е") + " "
                        ):
                            est_vo_vsem = True
                        if mem_w != my_word and (
                            " " + my_word + " " in s.lower()
                            or " " + my_word + " " in s.lower().replace("ё", "е")
                        ):
                            takoe_id = s.lower().find(" " + my_word + " ")
                            mem_w = my_word
                            if takoe_id == -1:
                                takoe_id = (
                                    s.lower()
                                    .replace("ё", "е")
                                    .find(" " + my_word + " ")
                                )
                            zap_id = s.lower().find(",")
                            if takoe_id < zap_id or zap_id == -1:
                                conditions += 1
                                s = s.lower().replace(my_word + " ", "", 1)
                                words2[j] = words2[j].replace(
                                    " " + my_word + " ", " ", 1
                                )
                                break
                    if est_vo_vsem:
                        conditions2 += 1
                s = " " + words2[j] + " "
                if lik_pov:
                    s_chs = s.split()
                    for s_ch in s_chs:
                        chast_rechi = str(self.morph.parse(s_ch)[0].tag.POS)
                        if s_ch.lower() == "существования":
                            break
                        if (
                            chast_rechi == "NOUN"
                            or chast_rechi == "VERB"
                            or chast_rechi == "INFN"
                            or chast_rechi == "NPRO"
                            or chast_rechi == "ADVB"
                            or s_ch.lower() == "все"
                            or chast_rechi == "PRED"
                        ):
                            if str(int(j) - 1) in words2:
                                if (
                                    " " + s_ch.lower() + " "
                                    in " " + words2[str(int(j) - 1)].lower() + " "
                                    or len(s_ch) > 6
                                    and " " + s_ch[:-2].lower()
                                    in " " + words2[str(int(j) - 1)].lower()
                                ):
                                    conditions += 1
                                    conditions2 += 1
                                    break
                            elif str(int(j) - 2) in words2:
                                if (
                                    " " + s_ch.lower() + " "
                                    in " " + words2[str(int(j) - 2)].lower() + " "
                                    or len(s_ch) > 6
                                    and " " + s_ch[:-2].lower()
                                    in " " + words2[str(int(j) - 2)].lower()
                                ):
                                    conditions += 1
                                    conditions2 += 1
                                    break
                if form_word and conditions2:
                    s_chs = s.split()
                    sent2 = [str(self.morph.parse(ok)[0].normal_form) for ok in s_chs]
                    pos = [str(self.morph.parse(ok)[0].tag.POS) for ok in s_chs]
                    if str(int(j) - 1) in words2:
                        s_chs = words2[str(int(j) - 1)].split()
                    elif str(int(j) - 2) in words2:
                        s_chs = words2[str(int(j) - 2)].split()
                    sent3 = [str(self.morph.parse(ok)[0].normal_form) for ok in s_chs]

                    for ij, s_ch in enumerate(sent2):
                        if s_ch in sent3 and (pos[ij] == "NOUN" or pos[ij] == "ADJS"):
                            conditions += 1
                            conditions2 += 1
                            break
                if synonym:
                    kol_syn = False
                    for syns in self.synonyms:
                        for syn in syns:
                            if (
                                syn
                                and not syn.isspace()
                                and " " + syn + " " in s.lower()
                            ):
                                mem_sinonim = syn
                                for syn in syns:
                                    if str(int(j) - 1) in words2:
                                        if (
                                            " " + syn + " "
                                            in " "
                                            + words2[str(int(j) - 1)].lower()
                                            + " "
                                            and syn != mem_sinonim
                                        ):
                                            kol_syn = True
                                            break
                                    elif str(int(j) - 2) in words2:
                                        if (
                                            " " + syn + " "
                                            in " "
                                            + words2[str(int(j) - 2)].lower()
                                            + " "
                                            and syn != mem_sinonim
                                        ):
                                            kol_syn = True
                                            break
                        if kol_syn:
                            conditions += 1
                            conditions2 += 1
                            break
                if conditions > 1:
                    maybe1.append(j)
                if conditions > 0:
                    maybe2.append(j)
                if conditions2 >= kol_task and kol_task != 0:
                    maybe0.append(j)
                if conditions > kol_task and kol_task != 0:
                    indexes.append(j)
                if conditions == kol_task and kol_task != 0:
                    maybe3.append(j)
        if indexes and (form_word or odn):
            indexes = [random.choice(indexes)]
        if not indexes and maybe3:
            indexes = [random.choice(maybe3)]
        if not indexes and maybe0:
            indexes = [random.choice(maybe0)]
        if not indexes and maybe1:
            indexes = [random.choice(maybe1)]
        if not indexes and maybe2:
            indexes = [random.choice(maybe2)]
        if not indexes:
            indexes = [str(random.choice(range(num_sen1 + 1, num_sen2 + 1)))]
        answer = indexes
        return answer
