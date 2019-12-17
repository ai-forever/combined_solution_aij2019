# MagicCity

import re

from solvers.utils import morph
from utils import read_config


class Solver(object):
    def __init__(self, config_path="data/models/solvers/solver27/solver27.json"):
        self.is_loaded = True
        self.morph = morph
        self.config = read_config(config_path)
        self.theme2keywords = self.config["theme2keywords"]
        self.tag2theme = self.config["tag2theme"]
        self.thesis = self.config["thesis"]
        self.arguments = self.config["arguments"]
        self.conclusion = self.config["conclusion"]
        self.patterns = self.config["patterns"]

    def func(self, x, theme_set):
        score = 0
        for word in x:
            for seed_word in theme_set:
                if word.startswith(seed_word) > 0:
                    if seed_word in ["воспит", "язык", "патриот"]:
                        score += 2.5
                    else:
                        score += 1
        return score

    def define_theme(self, text):
        text = text.split()
        if self.func(text, self.theme2keywords["kniga"]) > 3:
            return "kniga"
        if self.func(text, self.theme2keywords["lang"]) > 5:
            return "lang"
        if self.func(text, self.theme2keywords["voyna"]) > 1:
            return "voyna"
        if self.func(text, self.theme2keywords["patriot"]) > 1:
            return "patriot"
        if self.func(text, self.theme2keywords["iskus"]) > 4:
            return "iskus"
        if self.func(text, self.theme2keywords["money"]) > 2:
            return "money"
        if self.func(text, self.theme2keywords["life"]) > 4:
            return "life"
        if self.func(text, self.theme2keywords["elderly"]) > 2:
            return "elderly"
        if self.func(text, self.theme2keywords["hist_memory"]) > 3:
            return "hist_memory"
        if self.func(text, self.theme2keywords["ger"]) > 1:
            return "ger"
        if self.func(text, self.theme2keywords["sov"]) > 4:
            return "sov"
        if self.func(text, self.theme2keywords["fair"]) > 1:
            return "fair"
        if self.func(text, self.theme2keywords["kids"]) > 4:
            return "kids"
        if self.func(text, self.theme2keywords["kras"]) > 2:
            return "kras"
        if self.func(text, self.theme2keywords["ficus"]) > 3:
            return "ficus"
        if self.func(text, self.theme2keywords["happy"]) > 2:
            return "happy"
        if self.func(text, self.theme2keywords["dream"]) > 3:
            return "dream"
        if self.func(text, self.theme2keywords["friends"]) > 7:
            return "friends"
        if self.func(text, self.theme2keywords["teach"]) > 4:
            return "teach"
        if self.func(text, self.theme2keywords["otkr"]) > 2:
            return "otkr"
        if self.func(text, self.theme2keywords["nature"]) > 3:
            return "nature"
        if self.func(text, self.theme2keywords["progr"]) > 3:
            return "progr"
        if self.func(text, self.theme2keywords["zeml"]) > 4:
            return "zeml"
        if self.func(text, self.theme2keywords["dobr"]) > 4:
            return "dobr"
        if self.func(text, self.theme2keywords["dobro"]) > 5:
            return "dobro"
        if self.func(text, self.theme2keywords["lie"]) > 2:
            return "lie"
        return "life"

    def preprocess_text(self, task):
        text = (
            task["text"]
            .replace("(З", "(3")
            .replace("З)", "3)")
            .replace("з)", "3)")
            .replace("(з", "(3")
        )
        text = text.replace("(б", "(6").replace("б)", "6)")
        text = (
            text.replace("(О", "(0")
            .replace("О)", "0)")
            .replace("(о", "(0")
            .replace("о)", "0)")
        )
        text = re.split("\(\d+\)", text)[1:]
        return text

    def extract_author(self, text):
        author = re.findall("\([^\(]+\)", text[-1])
        if len(author) > 0:
            author = re.sub("[!\"#$%&'()*+,/:;<=>?@[\\]^_`{|}~]|\d+", " ", author[0])
            author = re.sub("По |по ", " ", author).strip()
            author = author.split(" ")
            author = (
                " ".join(author[:-1])
                + " "
                + self.morph.parse(author[-1])[0].normal_form.capitalize()
            )
        else:
            author = "автор"
        return author

    def normalize_sentences(self, sentences):
        normalized_sentences = [
            re.sub("\W+", " ", sentence.lower()) for sentence in sentences
        ]
        normalized_sentences = [
            [
                self.morph.parse(word.strip())[0].normal_form
                for word in sentence.split(" ")
                if self.morph.parse(word.strip())[0].tag.POS in {"NOUN", "VERB", "ADJF"}
            ]
            for sentence in normalized_sentences
        ]
        return normalized_sentences

    def normalize_text(self, text):
        text = re.sub("[!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~]|\d+", " ", " ".join(text))
        text = re.sub("\W+", " ", text.lower())
        text = " ".join(
            [
                self.morph.parse(word.strip())[0].normal_form
                for word in text.split(" ")
                if self.morph.parse(word.strip())[0].tag.POS in {"NOUN", "VERB", "ADJF"}
            ]
        )
        return text

    def predict_from_model(self, task):
        text = self.preprocess_text(task)
        # author = self.extract_author(text)
        text = text[:-1] + [text[-1].split("(")[0]]
        sentences = list(text)
        normalized_sentences = self.normalize_sentences(sentences)
        normalized_text = self.normalize_text(text)
        theme = self.define_theme(normalized_text)
        scores = [
            (self.func(normalized_sentence, self.theme2keywords[theme]), sentences[i])
            for i, normalized_sentence in enumerate(normalized_sentences)
        ]
        essay = []
        essay.append(
            " ".join(
                [
                    self.patterns["introduction"][0] + f"{self.tag2theme[theme]}.",
                    self.patterns["introduction"][1],
                ]
            )
        )
        for score in sorted(scores, reverse=True):
            if len(score[1].split()) > 6:
                essay.append(self.patterns["author_position"] + f'"{score[1].strip()}"')
                break
        essay.append(self.patterns["opinion"])
        if theme in self.thesis:
            essay.append(self.patterns["thesis"] + f"{self.thesis[theme]}")
        essay.append(self.patterns["arguments"][0] + f"{self.arguments[theme][0]}")
        essay.append(self.patterns["arguments"][1] + f"{self.arguments[theme][1]}")
        essay.append(self.conclusion[theme])
        return "\n\t".join(essay).strip()

    def load(self, path=""):
        pass

    def save(self, path=""):
        pass

    def fit(self, path=""):
        pass
