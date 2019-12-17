import json
import numpy as np
import os
import re
import torch
from collections import defaultdict, OrderedDict
from functools import wraps
from gensim.models import KeyedVectors
from pymorphy2 import MorphAnalyzer
from pytorch_pretrained_bert import BertModel, BertTokenizer, BertForMaskedLM
from scipy.spatial.distance import cosine


morph = MorphAnalyzer()
ALPHABET = "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"


def fix_spaces(text):
    space_fix_pattern = re.compile("\s+")
    return space_fix_pattern.sub(" ", text)


def singleton(cls):
    instance = None

    @wraps(cls)
    def inner(*args, **kwargs):
        nonlocal instance
        if instance is None:
            instance = cls(*args, **kwargs)
        return instance

    return inner


def standardize_task(task):
    if "choices" not in task:
        if "question" in task and "choices" in task["question"]:
            task["choices"] = task["question"]["choices"]
        else:
            parts = task["text"].split("\n")
            task["text"] = parts[0]
            task["choices"] = []
            for i in range(1, len(parts)):
                task["choices"].append({"id": str(i), "text": parts[i]})
    for i in range(len(task["choices"])):
        parts = [x.strip() for x in task["choices"][i]["text"].split(",")]
        task["choices"][i]["parts"] = parts
    return task


class BertEmbedder(object):
    """
    Embedding Wrapper on Bert Multilingual Cased
    """
    def __init__(self, bert_path="data/models/bert/multilingual"):
        self.bert_path = bert_path
        self.model_file = os.path.join(bert_path, "bert-base-multilingual-cased.tar.gz")
        self.vocab_file = os.path.join(
            bert_path, "bert-base-multilingual-cased-vocab.txt"
        )
        self.model = self.bert_model()
        self.tokenizer = self.bert_tokenizer()
        self.embedding_matrix = self.get_bert_embed_matrix()

    @singleton
    def bert_model(self):
        model = BertModel.from_pretrained(self.model_file).eval()
        return model

    @singleton
    def bert_tokenizer(self):
        tokenizer = BertTokenizer.from_pretrained(self.vocab_file, do_lower_case=False)
        return tokenizer

    @singleton
    def get_bert_embed_matrix(self):
        bert_embeddings = list(self.model.children())[0]
        bert_word_embeddings = list(bert_embeddings.children())[0]
        matrix = bert_word_embeddings.weight.data.numpy()
        return matrix

    def sentence_embedding(self, text_list):
        embeddings = []
        for text in text_list:
            token_list = self.tokenizer.tokenize("[CLS] " + text + " [SEP]")
            segments_ids, indexed_tokens = (
                [1] * len(token_list),
                self.tokenizer.convert_tokens_to_ids(token_list),
            )
            segments_tensors, tokens_tensor = (
                torch.tensor([segments_ids]),
                torch.tensor([indexed_tokens]),
            )
            with torch.no_grad():
                encoded_layers, _ = self.model(tokens_tensor, segments_tensors)
            sent_embedding = torch.mean(encoded_layers[11], 1)
            embeddings.append(sent_embedding)
        return embeddings

    def token_embedding(self, token_list):
        token_embedding = []
        for token in token_list:
            ontoken = self.tokenizer.tokenize(token)
            segments_ids, indexed_tokens = (
                [1] * len(ontoken),
                self.tokenizer.convert_tokens_to_ids(ontoken),
            )
            segments_tensors, tokens_tensor = (
                torch.tensor([segments_ids]),
                torch.tensor([indexed_tokens]),
            )
            with torch.no_grad():
                encoded_layers, _ = self.model(tokens_tensor, segments_tensors)
            ontoken_embeddings = []
            for subtoken_i in range(len(ontoken)):
                hidden_layers = []
                for layer_i in range(len(encoded_layers)):
                    vector = encoded_layers[layer_i][0][subtoken_i]
                    hidden_layers.append(vector)
                ontoken_embeddings.append(hidden_layers)
            cat_last_4_layers = [
                torch.cat((layer[-4:]), 0) for layer in ontoken_embeddings
            ]
            token_embedding.append(cat_last_4_layers)
        token_embedding = (
            torch.stack(token_embedding[0], 0)
            if len(token_embedding) > 1
            else token_embedding[0][0]
        )
        return token_embedding


class RubertForMasking(object):
    """
    DeepPavlov Rubert Wrapper for Masking in Tasks 2, 15
    by team Niw
    """

    def __init__(self, bert_path="data/models/bert/rubert/deeppavlov"):
        self.bert_path = bert_path
        self.model_file = os.path.join(
            self.bert_path, "ru_conversational_cased_L-12_H-768_A-12.tar.gz"
        )
        self.vocab_file = os.path.join(self.bert_path, "vocab.txt")
        self.model = self.bert_model()
        self.tokenizer = self.bert_tokenizer()

    @singleton
    def bert_model(self):
        model = BertForMaskedLM.from_pretrained(self.model_file).eval()
        return model

    @singleton
    def bert_tokenizer(self):
        tokenizer = BertTokenizer.from_pretrained(self.vocab_file, do_lower_case=False)
        return tokenizer

    def token_embedding(self, token_list):
        token_embedding = []
        for token in token_list:
            ontoken = self.tokenizer.tokenize(token)
            segments_ids, indexed_tokens = (
                [1] * len(ontoken),
                self.tokenizer.convert_tokens_to_ids(ontoken),
            )
            segments_tensors, tokens_tensor = (
                torch.tensor([segments_ids]),
                torch.tensor([indexed_tokens]),
            )
            with torch.no_grad():
                encoded_layers, _ = self.model(tokens_tensor, segments_tensors)
            ontoken_embeddings = []
            for subtoken_i in range(len(ontoken)):
                hidden_layers = []
                for layer_i in range(len(encoded_layers)):
                    vector = encoded_layers[layer_i][0][subtoken_i]
                    hidden_layers.append(vector)
                ontoken_embeddings.append(hidden_layers)
            cat_last_4_layers = [
                torch.cat((layer[-4:]), 0) for layer in ontoken_embeddings
            ]
            token_embedding.append(cat_last_4_layers)
        token_embedding = (
            torch.stack(token_embedding[0], 0)
            if len(token_embedding) > 1
            else token_embedding[0][0]
        )
        return token_embedding

    def masking_task_15(self, text, w_x, delta):
        w_y = []
        for i in range(len(w_x)):
            w_y.append(self.tokenizer.tokenize(w_x[i].lower()))
            w_y[i] = self.tokenizer.convert_tokens_to_ids(w_y[i])
        if text[-1] == "]":
            text = text + " . ."
        text = "[CLS] " + text + " [SEP]"
        tokenized_text = self.tokenizer.tokenize(text)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        indexed_tokens = indexed_tokens[:500]
        mask_input = []
        for i in range(len(indexed_tokens)):
            if indexed_tokens[i] == 103:
                mask_input.append(i)
        segments_ids = [0] * len(indexed_tokens)
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])
        with torch.no_grad():
            predictions = self.model(tokens_tensor, segments_tensors)
        predictions_one, predictions_two, predictsx3 = [], [], []
        for i in range(len(mask_input)):
            predictions_one.append(predictions[0, mask_input[i], :].numpy() + 10)
            predictions_two.append(predictions[0, mask_input[i] + 1, :].numpy() + 10)
            predictsx3.append(predictions[0, mask_input[i] + 2, :].numpy() + 10)
        ver_w_y, ver_w2, ver_w3 = [], [], []
        output = []
        for i in range(len(w_x)):
            if len(w_y[i]) > 2:
                ver_w_y.append(
                    abs(
                        predictions_one[i][w_y[i][0]]
                        * predictions_two[i][w_y[i][1]]
                        * predictsx3[i][w_y[i][2]]
                    )
                    ** (1 / 3)
                )
            elif len(w_y[i]) > 1:
                ver_w_y.append(
                    abs(predictions_one[i][w_y[i][0]] * predictions_two[i][w_y[i][1]])
                    ** (1 / 2)
                )
            else:
                ver_w_y.append(predictions_one[i][w_y[i][0]])
            if (
                "ковану" in w_x[i]
                or "кованы" in w_x[i]
                or "золочёно" in w_x[i].replace("е", "ё")
                or "золочёны" in w_x[i].replace("е", "ё")
                or "золочёну" in w_x[i].replace("е", "ё")
            ):
                ver_w_y[i] += 5
        for i in range(len(w_x) // 2):
            ver_w2.append(ver_w_y[i] / ver_w_y[len(w_x) // 2 + i])
        for i in range(len(ver_w2)):
            if ver_w2[i] > 1 + delta:
                output.append(w_x[i])
            else:
                output.append(w_x[len(w_x) // 2 + i])
        return output

    def masking_task_2(self, text, seed_word, task_type=0):
        text = "[CLS] " + text + " [SEP]"
        tokenized_text = self.tokenizer.tokenize(text)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        indexed_tokens = indexed_tokens[:500]
        mask_input = []
        for i in range(len(indexed_tokens)):
            if indexed_tokens[i] == 103:
                mask_input.append(i)
        segments_ids = [0] * len(tokenized_text)
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])
        model_output_two = ""
        with torch.no_grad():
            predictions = self.model(tokens_tensor, segments_tensors)
            predictions_one = predictions[0, mask_input[0], :].argsort()[-50:].numpy()
            model_output_one = self.tokenizer.convert_ids_to_tokens(predictions_one)
            if len(mask_input) > 1:
                predictions_two = (
                    predictions[0, mask_input[1], :].argsort()[-50:].numpy()
                )
                model_output_two = self.tokenizer.convert_ids_to_tokens(predictions_two)
        result, from_search = "", True
        for i in range(len(model_output_one) - 1, -1, -1):
            if task_type == 3:
                break
            model_output_one[i] = model_output_one[i].strip().lower()
            if model_output_two:
                for w in seed_word:
                    if task_type == 3:
                        break
                    if model_output_one[i].lower() in w:
                        w2 = w.replace(model_output_one[i], "")
                        for j in range(len(model_output_two) - 1, -1, -1):
                            if model_output_two[j].lower() in w2:
                                w3 = w2.replace(model_output_two[j], "").replace(
                                    " ", ""
                                )
                                if not w3:
                                    task_type = 3
                                    result = w.replace(" ", "")
                                    break
        if task_type == 2:
            result = "не знаю"
        elif task_type == 4:
            for w in seed_word:
                if "-" in w or " " in w:
                    result = w.replace(" ", "").replace("-", "")
                    break
            if not result:
                from_search = False
                if seed_word:
                    result = seed_word[0]
                else:
                    result = model_output_one[-1]
        else:
            for i in range(len(model_output_one) - 1, -1, -1):
                if (
                    len(model_output_one[i]) < 2
                    and (model_output_one[i] != "и" and model_output_one[i] != "а")
                ) or model_output_one[i] == "вот":
                    model_output_one.pop(i)
                elif model_output_one[i] in seed_word:
                    result = model_output_one[i]
                    break
            if not result and task_type == 1:
                for w in seed_word:
                    if "-" in w or " " in w:
                        result = w.replace(" ", "").replace("-", "")
                        break
            if not result:
                from_search = False
                if seed_word:
                    result = seed_word[0]
                else:
                    result = model_output_one[-1]
        if result == "ксожалению":
            result = "такимобразом"
        if result == "вовсене":
            result = "именно"
        return result, from_search


def clean(word):
    return " ".join(re.sub("[^а-яa-z0-9ё ]+", " ", word.lower()).split())


def tokenize(text, clean_method=clean):
    text = clean_method(text)
    return text.split()


class CommonData(object):
    def __init__(self, common_data_path="data/models/utils/common_files"):
        self.common_data_path = common_data_path
        self.prefixes_path = os.path.join(self.common_data_path, "prefixes.txt")
        self.norm2word_path = os.path.join(self.common_data_path, "norm2word.json")
        self.prepositions_path = os.path.join(self.common_data_path, "prepositions.txt")
        self.prefixes = self.load_prefixes()
        self.prepositions = self.load_prepositions()
        self.norm2word = self.load_norm2word()

    @singleton
    def load_prefixes(self):
        prefixes = {}
        with open(self.prefixes_path, "r", encoding="utf-8") as f:
            for line in f:
                if line[0] == "=":
                    k = line[1:].strip()
                    continue
                for e in line.replace("-", "").split("/"):
                    if e.strip() not in prefixes:
                        prefixes[e.strip()] = k
        return prefixes

    @singleton
    def load_prepositions(self):
        prepositions = defaultdict(list)
        with open(self.prepositions_path, "r", encoding="utf-8") as f:
            for line in f:
                if line[0] == "=":
                    k = line[2:].strip()
                    continue
                for e in line.replace("-", "").split("/"):
                    if e.lower() == e:
                        prepositions[e.strip()].append(k)
        for prep in ["с", "со"]:
            prepositions[prep] = ["ablt"]
        return prepositions

    @singleton
    def load_norm2word(self):
        with open(self.norm2word_path, "r", encoding="utf-8") as f:
            norm2word = json.load(f)
        return norm2word


class Word2vecProcessor(object):
    def __init__(self, w2v_path="data/models/utils/w2v"):
        self.w2v_path = w2v_path
        self.w2v_model_filename = os.path.join(
            self.w2v_path, "ruwikiruscorpora_0_300_20.bin"
        )
        self.verbs_filename = os.path.join(self.w2v_path, "verbs.txt")
        self.word2vec = self.load_w2v()
        self.lemma2word = self.build_lemma2word()
        self.verbs = self.load_verbs()
        self.norm_cache = dict()

    @singleton
    def load_w2v(self):
        return KeyedVectors.load_word2vec_format(
            self.w2v_model_filename, binary=True, datatype=np.float32
        )

    @singleton
    def build_lemma2word(self):
        lemma2word = dict()
        for word in self.word2vec.index2word:
            q = word.split("_")[0]
            if q not in lemma2word:
                lemma2word[q] = word
        return lemma2word

    @singleton
    def load_verbs(self):
        with open(os.path.join(self.verbs_filename), encoding="utf-8") as fin:
            verbs = json.load(fin)
        return verbs

    def warmup_cache(self, word2norm_filename):
        with open(word2norm_filename, encoding="utf-8") as fin:
            self.norm_cache = json.load(fin)
        self.norm2word_cache = defaultdict(set)
        for word, word_norm in self.norm_cache.items():
            self.norm2word_cache[word_norm].add(word)

    def get_all_forms(self, word_norm):
        return {
            self.verbs.get(lemma, lemma) for lemma in self.norm2word_cache[word_norm]
        }

    def get_normal_form(self, word):
        if word not in self.norm_cache:
            self.norm_cache[word] = morph.normal_forms(word)[0]
        lemma = self.norm_cache[word]
        lemma = self.verbs.get(lemma, lemma)
        lemma = lemma.replace("ё", "е")
        return lemma

    def prepare_word(self, word):
        lemma = self.get_normal_form(word)
        word = self.lemma2word.get(lemma)
        return word

    def word_vector(self, word):
        word = self.prepare_word(word)
        return self.word2vec[word] if word in self.word2vec else None

    def text_vector(self, text):
        word_vectors = [
            self.word_vector(token)
            for token in tokenize(text.lower())
            if token.isalpha()
        ]
        word_vectors = [vec for vec in word_vectors if vec is not None]
        return (
            np.mean(np.vstack(word_vectors), axis=0) if len(word_vectors) != 0 else None
        )

    def distance(self, vec1, vec2):
        if vec1 is None or vec2 is None:
            return 2
        try:
            return cosine(vec1, vec2)
        except:
            return 2

    def get_most_similar(self, text, topn=100):
        vec = self.text_vector(text)

        res = OrderedDict()
        try:
            neighbours = dict(self.word2vec.most_similar([vec], topn=topn))
            for e in sorted(neighbours.items(), key=lambda x: -x[1]):
                word = e[0].split("_")[0].split("::")[0]
                if word not in res:
                    res[word] = 1 - e[1]
        except:
            pass
        return res


class NgramManager(object):
    def __init__(self, ngram_path="data/models/utils/ngram"):
        self.ngram_path = ngram_path
        self.ngram_files = [
            os.path.join(self.ngram_path, "{}grams.txt".format(i + 1)) for i in range(4)
        ]
        self.word2num, self.gram_freq = self.load_all()

    @singleton
    def load_all(self):
        with open(
            os.path.join(self.ngram_path, "word2num.json"), "r", encoding="utf-8"
        ) as fin:
            word2num = json.load(fin)
        gram_freq = defaultdict(int)
        for filename in self.ngram_files:
            with open(filename, "r", encoding="utf-8") as fin:
                for line in fin:
                    tokens = line.strip().split("\t")
                    f = int(tokens[0])
                    tokens = [e.lower() for e in tokens[1:]]
                    for e in tokens:
                        if e not in word2num:
                            word2num[e] = len(word2num)
                    gram_freq[tuple([word2num[e] for e in tokens])] += f
                    if filename == "./data/2grams-3.txt":
                        for i in range(1, len(tokens)):
                            if len(tokens[i]) > 1:
                                tt = tokens[i][-2:]
                                tokens_fixed = (
                                    ["-"] + tokens[:i] + [tt] + tokens[i + 1 :]
                                )
                                if tt not in word2num:
                                    word2num[tt] = len(word2num)
                                gram_freq[
                                    tuple([word2num[e] for e in tokens_fixed])
                                ] += f

        return word2num, gram_freq
