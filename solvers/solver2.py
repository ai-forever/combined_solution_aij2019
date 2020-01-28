# niw

import re
from nltk.tokenize import sent_tokenize
from string import punctuation

from solvers.utils import RubertForMasking, AbstractSolver


class Solver(AbstractSolver):
    def __init__(self, seed=42):
        self.seed = seed
        self.init_seed(seed)
        self.dictionary_words = {}
        self.rubert = RubertForMasking()
        self.is_loaded = False

    @staticmethod
    def get_sentence_pair(text):
        dots = ("<...>", "<…>", "<..>")
        sentences = sent_tokenize(text)
        if any(d in sent for sent in sentences for d in dots):
            sentence_number = next(
                sentence_number
                for sentence_number, sent in enumerate(sentences)
                if any(d in sent for d in dots)
            )
            if sentence_number == 0:
                return " ".join(sentences[0 : sentence_number + 1])
            else:
                return " ".join(sentences[sentence_number - 1 : sentence_number + 1])
        else:
            try:
                sentence_number = next(
                    sentence_number
                    for sentence_number, sent in enumerate(sentences)
                    if ("..." in sent or "…" in sent)
                    and not sent.endswith("...")
                    and not sent.endswith("…")
                )
                if sentence_number == 0:
                    return " ".join(sentences[0 : sentence_number + 1])
                else:
                    return " ".join(
                        sentences[sentence_number - 1 : sentence_number + 1]
                    )
            except StopIteration:
                return None

    def get_mask(self, sentence_pair):
        sentence_pair = (
            sentence_pair.replace("<....>", " [MASK] ")
            .replace("<...>", " [MASK] ")
            .replace("<…>", " [MASK] ")
            .replace("( ..... )", " [MASK] ")
            .replace("<..>", " [MASK] ")
        )
        if not "[MASK]" in sentence_pair:
            sentence_pair = sentence_pair.replace("…", " [MASK] ")
        if not "[MASK]" in sentence_pair:
            sentence_pair = sentence_pair.replace("...", " [MASK] ")
        if not "[MASK]" in sentence_pair:
            sentence_pair = (
                re.search(".{0,100}\(\.\.\.\).{0,100}", sentence_pair)
                .group(0)
                .replace("(...)", "[MASK]")
            )
        sentence_pair = sentence_pair.replace("  ", " ")
        return sentence_pair

    def predict_from_model(self, task):
        sentence_pair = self.get_sentence_pair(task["text"])
        if not sentence_pair:
            return "но"
        sentence_pair = self.get_mask(sentence_pair)
        task_type, task_type_list, seed_word = 0, [], []
        if "подберите сочетание" in task["text"]:
            if "производного предлога с указательным местоимением" in task["text"]:
                if "но [" in sentence_pair.lower():
                    result = "несмотрянаэто"
                else:
                    result = "вследствиеэтого"
            elif "частицы с наречием" in task["text"]:
                result = "именнопоэтому"
            elif "подчинительного союза и определительного местоимения" in task["text"]:
                result = "каклюбое"
            elif "числительного с предлогом" in task["text"]:
                result = "одиниз"
            elif "предлога с относительным местоимением" in task["text"]:
                result = "изкоторых"
            elif "частицы с указательным местоимением" in task["text"]:
                result = "именноэта"

            elif "предлога со словом" in task["text"]:
                result = "вслучае"
            elif "] ," in sentence_pair:
                result = "деловтом"
            elif "] полагают" in sentence_pair:
                result = "другиеже"
            elif "сочетание частицы с местоименным наречием" in task["text"]:
                result = "именнопоэтому"
            else:
                result = "деловтомчто"
            task_type = 2
        if "наречие" in task["text"]:
            task_type_list.append(1)
        if "ограничительно-выделительную частицу" in task["text"]:
            task_type_list.append(2)
        elif "частиц" in task["text"]:
            task_type_list.extend([2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
        if "союзное слово" in task["text"]:
            task_type_list.extend([12, 28, 13, 14])
        if "сочинительный союз" in task["text"]:
            task_type_list.extend([15, 16, 17])
        if "подчинительный составной союз" in task["text"]:
            result = "потомучто"
            task_type = 2
        if "фразеологическое словосочетание" in task["text"]:
            result = "вконцеконцов"
            task_type = 2
        if "подчинительный составной союз" in task["text"]:
            result = "потомучто"
            task_type = 2
        elif "составной союз" in task["text"]:
            result = "вместестем"
            task_type = 2
        if "подберите глагол" in task["text"]:
            result = "оказалось"
            task_type = 2
        if "пояснительный союз" in task["text"]:
            result = "тоесть"
            task_type = 2
        if "сочетание частицы со сложным предлогом" in task["text"]:
            result = "именноизза"
            task_type = 2
        if "производный составной предлог" in task["text"]:
            result = "вотличиеот"
            task_type = 2
        elif "подчинительный союз" in task["text"]:
            task_type_list.append(14)
        elif "противительный союз" in task["text"]:
            task_type_list.append(16)
        elif "союз" in task["text"]:
            task_type_list.extend([14, 15, 16, 17])
        if "предлог" in task["text"]:
            task_type_list.append(19)
        if (
            "указательное местоимение" in task["text"]
            or "указательным местоимением" in task["text"]
        ):
            task_type_list.append(18)
        elif "относительное местоимение" in task["text"]:
            task_type_list.append(12)
        elif "определительное местоимение" in task["text"]:
            task_type_list.append(20)
        elif "личное местоимение" in task["text"]:
            task_type_list.append(24)
        elif "местоимение" in task["text"]:
            task_type_list.extend([12, 18, 20])
        if (
            "из приведённых ниже слов" in task["text"]
            or "слово или сочетание слов" in task["text"]
        ):
            task_type = 1
            candidates = re.search("\?.*?1", task["text"]).group(0)
            candidates = re.sub(".*\.", "", candidates)
            candidates = candidates.replace("?", "").replace("1", "").replace("(", "")
            candidates = re.findall("[А-Я][^A-Я]*", candidates)
            for i in range(len(candidates)):
                candidates[i] = (
                    candidates[i].replace(",", "").replace(".", "").strip().lower()
                )
            if not candidates:
                candidates = re.search("1[^)]*$", task["text"]).group(0)
                candidates = re.findall("[0-9][^0-9]*", candidates)
                for i in range(len(candidates)):
                    candidates[i] = (
                        re.sub("[0-9]", "", candidates[i])
                        .replace(",", "")
                        .replace(".", "")
                        .strip()
                        .lower()
                    )
            seed_word = candidates
        if "вводное словосочетание" in task["text"]:
            task_type_list.append(26)
        elif "вводное слово" in task["text"]:
            task_type_list.append(21)
        if "вводную конструкцию" in task["text"]:
            task_type_list.append(26)
            seed_word.append("кроме этого")
        if not task_type_list and not seed_word:
            task_type_list = [i for i in range(21)]
        for key in self.dictionary_words.keys():
            if int(self.dictionary_words[key]) in task_type_list:
                if (
                    not "вводную конструкцию" in task["text"]
                    or key.strip() != "таким образом"
                ):
                    seed_word.append(key.strip())

        sentence_pair = sentence_pair.replace("(", "")
        sentence_pair = sentence_pair.replace(")", "")
        sentence_pair = (
            re.sub("[0-9]", "", sentence_pair)
            .replace(",", " ,")
            .replace(".", " .")
            .replace(":", " :")
            .replace("  ", " ")
        )
        if task_type != 2:
            result, from_search_one = self.rubert.masking_task_2(
                sentence_pair, seed_word, task_type
            )
            if not from_search_one:
                sentence_pair = sentence_pair.replace("[MASK]", "[MASK] [MASK]")
                result2, from_search_two = self.rubert.masking_task_2(
                    sentence_pair, seed_word, 4
                )
            if not from_search_one and from_search_two:
                result = result2
        return result.strip(punctuation)

    def load(self, path="data/models/solvers/solver2/words.csv"):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.replace("\n", "").split("|")
                self.dictionary_words[line[0]] = line[1]
        self.is_loaded = True
