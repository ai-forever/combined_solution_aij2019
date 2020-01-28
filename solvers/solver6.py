# Idyllium

import nltk
import os
import re
from nltk.util import ngrams
from sklearn.metrics.pairwise import cosine_similarity
from string import punctuation

from solvers.utils import morph, BertEmbedder, AbstractSolver


class Solver(AbstractSolver):
    def __init__(self, seed=42):
        self.seed = seed
        self.init_seed(seed)
        self.morph = morph
        self.bert = BertEmbedder()
        self.task_mode = 1
        self.is_loaded = False
        self.tautologisms = None

    def exclude_word(self, task_sent):
        tokens = [
            token.strip('.,";!:?><)«»') for token in task_sent.split(" ") if token != ""
        ]

        to_tokens = []
        for token in tokens:
            parse_res = self.morph.parse(token)[0]
            if parse_res.tag.POS not in [
                "CONJ",
                "PREP",
                "PRCL",
                "INTJ",
                "PRED",
                "NPRO",
            ]:
                if parse_res.normal_form != "быть":
                    to_tokens.append((parse_res.word, parse_res.tag.POS))

        bigrams = list(ngrams(to_tokens, 2))

        results = []
        for bigram in bigrams:
            if bigram[0] != bigram[1]:
                b1 = self.bert.sentence_embedding([bigram[0][0]])[0].reshape(1, -1)
                b2 = self.bert.sentence_embedding([bigram[1][0]])[0].reshape(1, -1)
                sim = cosine_similarity(b1, b2)[0][0]
                results.append(
                    (sim, bigram[0][0], bigram[1][0], bigram[0][1], bigram[0][1])
                )
        results = sorted(results)
        final_pair = results[-1]
        if final_pair[-1] == "NOUN" and final_pair[-2] == "NOUN":
            return results[-1][2], tokens
        else:
            return results[-1][1], tokens

    def load(self, path="data/models/solvers/solver6"):
        with open(os.path.join(path, "tautologisms.txt"), "r") as f:
            self.tautologisms = f.read().splitlines()
        self.is_loaded = True

    def predict_from_model(self, task):
        description = task["text"]
        task_description = ""
        if re.match(".*(замените|заменив|подберите).*", description.lower()):
            self.task_mode = 2
        else:
            self.task_mode = 1
        for par in description.split("\n"):
            for sentence in nltk.sent_tokenize(par):
                sentence = sentence.lower().rstrip(punctuation).replace("6.", "")
                if re.match(
                    ".*(отредактируйте|выпишите|запишите|исправьте|исключив|исключите|замените|заменив|подберите).*",
                    sentence,
                ):
                    continue
                else:
                    task_description += sentence
        found_tautologism = ""
        for pattern in self.tautologisms:
            task_description_no_punct = re.sub("\.\?!,…", "", task_description)
            if re.search(pattern, task_description_no_punct):
                span = re.search(pattern, task_description_no_punct).span()
                found_tautologism = task_description_no_punct[span[0] : span[1]]
                break

        tautologism_list = found_tautologism.split()
        tautology_tags = [
            self.morph.parse(token)[0].tag.POS for token in tautologism_list
        ]
        if self.task_mode == 1:
            if "COMP" in tautology_tags:
                return tautologism_list[tautology_tags.index("COMP")]
            if "ADVB" in tautology_tags:
                return tautologism_list[tautology_tags.index("ADVB")]
            if "ADJS" in tautology_tags:
                return tautologism_list[tautology_tags.index("ADJS")]
            if "PRTF" in tautology_tags:
                return tautologism_list[tautology_tags.index("PRTF")]
            if "PRTS" in tautology_tags:
                return tautologism_list[tautology_tags.index("PRTS")]
            if "ADJF" in tautology_tags:
                return tautologism_list[tautology_tags.index("ADJF")]
            if "PRED" in tautology_tags:
                return tautologism_list[tautology_tags.index("PRED")]
            if tautology_tags == ["NOUN", "NOUN"]:
                return tautologism_list[1]
        result, tokens = self.exclude_word(task_description)
        return result.strip(punctuation)
