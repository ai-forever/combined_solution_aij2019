# lamoda

import random
import re

from solvers.utils import Word2vecProcessor, tokenize, clean


class Solver(object):
    def __init__(self, seed=42):
        self.seed = seed
        self.init_seed()
        self.w2v = Word2vecProcessor()
        self.is_loaded = True

    def init_seed(self):
        random.seed(self.seed)

    def predict_from_model(self, task):
        try:
            word = (
                task["text"]
                .split(" значения слова", 1)[1]
                .strip()
                .split(" ")[0]
                .strip(".")
                .lower()
            )
            lemma = self.w2v.get_normal_form(word)
        except:
            word = lemma = None

        text = (
            task["text"].split(")", 1)[1].split("Определите")[0].split("Прочитайте")[0]
        )
        number = -1
        text_array = task["text"].split("предложении")

        if len(text_array) == 2:
            tokens = [
                self.w2v.get_normal_form(token)
                for token in tokenize(
                    text_array[0][-20:] + text_array[1][:20], clean_method=clean
                )
            ]
            number_dict = {
                "один": 1,
                "1": 1,
                "первое": 1,
                "первый": 1,
                "два": 2,
                "2": 2,
                "второе": 2,
                "второй": 2,
                "три": 3,
                "3": 3,
                "третье": 3,
                "третий": 3,
                "четыре": 4,
                "4": 4,
                "четвертое": 4,
                "четвертый": 4,
            }
            for token in tokens:
                if token in number_dict:
                    number = number_dict[token] - 1
                    break

        text = (
            text.replace("(З)", "(3)")
            .replace(
                "Выпишите цифру, соответствующую этому значению в приведённом фрагменте словарной статьи",
                "",
            )
            .replace("предложении текста", "")
        )
        sentences = [token for token in self.sent_split(text) if len(token) > 10]

        text_vector = self.w2v.text_vector(text)
        target_sentence_vector = None
        if number != -1 and number < len(sentences):
            target_sentence_vector = self.w2v.text_vector(sentences[number])

        score = dict()
        for i, choice in enumerate(task["question"]["choices"]):
            text = choice["text"]
            if word is not None:
                text = text.lower().replace(f" {word[0]}.", "")
            for s in [
                "разг",
                "офиц",
                "перен",
                "знач",
                "сущ",
                "ед",
                "мн",
                "-н",
                "жен",
                "муж",
                "прост",
            ]:
                text = ((text + " ").replace(s + " ", "").replace(s + ".", "")).strip()

            tokens = [
                token
                for token in tokenize(text)
                if self.w2v.get_normal_form(token) != lemma
            ]
            text_without_target = " ".join(tokens)

            text_without_target_vector = self.w2v.text_vector(text_without_target)
            score[i] = self.w2v.distance(
                text_vector, text_without_target_vector
            ) + 2.0 * self.w2v.distance(
                target_sentence_vector, text_without_target_vector
            )

        return [
            str(
                task["question"]["choices"][
                    sorted(score.items(), key=lambda x: (x[1], x[0]))[0][0]
                ]["id"]
            )
        ]

    def sent_split(self, text):
        reg = r"\(\n*\d+\n*\)"
        return re.split(reg, text)

    def fit(self, tasks):
        pass

    def load(self, path=""):
        pass

    def save(self, path=""):
        pass
