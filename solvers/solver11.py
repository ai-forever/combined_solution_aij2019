# lamoda

import json
import os
import re
from collections import defaultdict, Counter

from solvers.solver_helpers import morph, CommonData, ALPHABET, AbstractSolver


def clean_with_slot(word):
    return " ".join(re.sub("[^а-яё\. ]+", " ", word.lower()).split())


def clean(word):
    return " ".join(re.sub("[^а-яё ]+", " ", word.lower()).split())


class Solver(AbstractSolver):
    def __init__(self):
        self.common_data = CommonData()
        self.morph = morph
        self.alphabet = ALPHABET
        self.vowels = "уеыаоэёяию"
        self.accents_dict = defaultdict(set)
        self.solutions = {}

    def predict_from_model(self, task):
        def _find_chars(word, d, char_from_text=None, w0=''):
            res = set()
            for char in self.alphabet:
                w = word.replace('..', char)
                w = w.replace('.', '')
                if w in d or morph.word_is_known(w):
                    res.add(char)
            if len(res) > 1:
                noun_number = None
                for e in w0.strip().split(' '):
                    if e.find('..') == -1:
                        try:
                            for p in morph.parse(clean(e)):
                                if p.tag.POS in {'NOUN', 'NPRO'} and p.tag.case in {'nomn'}:
                                    noun_number = p.tag.number
                                    break
                        except:
                            pass
                    if noun_number is not None:
                        break
                res_new = set()
                for ch in res:
                    w = word.replace('..', ch)
                    w = w.replace('.', '')
                    try:
                        p = morph.parse(w)[0]
                        if not (p.tag.POS in {'VERB',
                                              'INFN'} and p.tag.number != noun_number and noun_number is not None):
                            res_new.add(ch)
                    except:
                        pass
                if len(res_new) != 0:
                    res = res_new

            elif len(res) == 0:
                if morph.word_is_known(word.replace('.', '')):
                    return {word}
                for char in self.alphabet:
                    try:
                        slot_ind = word.index('.')
                    except:
                        print(word)
                        raise
                    w = word.replace('..', char)
                    # Ы:
                    # after russian prefixes ending with consonant, except for МЕЖ И СВЕРХ .
                    # examples: безынтересный, подыграть, разыскивать.
                    for pref in self.common_data.prefixes:
                        if len(pref) == slot_ind:
                            if self.common_data.prefixes[pref] == 'ru' and pref[-1] not in self.vowels and pref not \
                                    in ['меж', 'сверх']:
                                if w.startswith(pref) and len(w) >= len(pref) + 2:
                                    if char == 'ы' and morph.word_is_known('и' + w[slot_ind + 1:]):
                                        print('======1', word, w, char)
                                        res.add(char)
                    # И:
                    # after russian prefixes ending with vowel (поиграть, поискать)
                    # after prefixes МЕЖ- and СВЕРХ- (сверхинтересный, межинститутский)
                    # in word ВЗИМАТЬ
                    # in compositional words (пединститут, спортинвентарь)
                    # after borrowed prefixes (пан-, суб-, транс-, контр- etc.)
                    #           (панисламизм, субинспектор, трансиордания, контригра)
                    # after numerals двух-, трех-, четырех- (двухигольный, трехимпульсный)

                    for pref in self.common_data.prefixes:
                        if len(pref) == slot_ind:
                            if self.common_data.prefixes[pref] == 'ru' and pref[-1] in self.vowels or \
                                    self.common_data.prefixes[pref] != 'ru':
                                if w.startswith(pref) and len(w) >= len(pref) + 2:
                                    if w[len(pref) + 1] not in self.vowels and char == 'и' and morph.word_is_known(
                                            w[slot_ind:]):
                                        print('======2', word, w, char)
                                        res.add(char)

                    for pref in ['меж', 'сверх']:
                        if len(pref) == slot_ind:
                            if w.startswith(pref) and len(w) >= len(pref) + 2:
                                if w[len(pref) + 1] not in self.vowels and char == 'и':
                                    print('======', word, w, char)
                                    res.add(char)

                    # letter Ъ is placed:
                    # after russian prefixes ending with consonant, before letters Е, Ё, Ю, Я. (Подъем, разъезд.)
                    # after numerals двух- трех-, четырех-, перед Е, Ё, Ю, Я. (Трехъярусный)
                    # after borrowed prefixes, which are not considered as prefixed in Russian. (объем, адъютант и т.д.)

                    for pref in self.common_data.prefixes:
                        if len(pref) == slot_ind:
                            if pref[-1] not in self.vowels:
                                if w.startswith(pref) and len(w) >= len(pref) + 2:
                                    if w[len(pref) + 1] in 'еёюя' and char == 'ъ':
                                        print('======', word, w, char)
                                        res.add(char)

                    for pref in ['двух', 'трех', 'трёх', 'четырех', 'четырёх']:
                        if len(pref) == slot_ind:
                            if w.startswith(pref) and len(w) >= len(pref) + 2:
                                if w[len(pref) + 1] in 'еёюя' and char == 'ъ' or \
                                        w[len(pref) + 1] not in self.vowels and char == 'и':
                                    print('======', word, w, char)
                                    res.add(char)

            if len(res) == 0:
                print(('NO chars', word))
            if char_from_text is not None:
                res = {char_from_text} & res
            return res

        def _solve_one_row(row, char_from_text=None):
            row = ' '.join(row.split()).replace(' ... ', '..').replace('.. .', '..') \
                .replace('. ..', '..').replace('.. ', '..').replace(' ..', '..')

            cc = Counter()
            ww = []
            words = row.strip(';').replace(';', ',').split(',')
            for word in words:
                word = clean_with_slot(word)
                if not word.endswith('..'):
                    word = word.rstrip('.')
                w_clean = ''
                for e in word.strip().split(' '):
                    if e.find('..') != -1:
                        w_clean = e
                        break
                if w_clean == '':
                    continue
                ww.append(w_clean)
                chars = None
                if word in self.solutions:
                    chars = {self.solutions[word]}
                elif word.replace('ё', 'е') in self.solutions:
                    chars = {self.solutions[word.replace('ё', 'е')]}
                elif w_clean in self.solutions:
                    chars = {self.solutions[w_clean]}
                elif w_clean.replace('ё', 'е') in self.solutions:
                    chars = {self.solutions[w_clean.replace('ё', 'е')]}

                if chars is None:
                    chars = _find_chars(w_clean, self.accents_dict, char_from_text, word.strip())
                    cc.update(chars)
                else:
                    if char_from_text is not None:
                        chars = {char_from_text} & chars
                    cc.update(chars)

            best_char = cc.most_common(1)[0][0] if len(cc) > 0 else ''
            www = ''.join([e.replace('..', best_char) for e in ww])
            return www, [max(cc.values()) / len(words) if len(cc) > 0 else 0, best_char]

        char_from_text = None
        if task['text'].find('одна и та же буква') == -1:
            for ch in self.alphabet:
                if (
                        (task['text'] + ' ').find(f'буква {ch} ') != -1
                        or (task['text'] + ' ').find(f'буква {ch.upper()} ') != -1
                        or (task['text'] + ' ').find(f'буква {ch}.') != -1
                        or (task['text'] + ' ').find(f'буква {ch.upper()}.') != -1
                ):
                    char_from_text = ch

        if task['question']['type'] == 'text':
            rows = task['text'].split('\n')[1:]

            score = dict()
            for row in rows:
                kk, vv = _solve_one_row(row, char_from_text)
                score[kk] = vv
            return sorted(score.items(), key=lambda x: (-x[1][0], x[0]))[0][0]
        else:
            score = dict()
            for row in task['question']['choices']:
                kk, vv = _solve_one_row(row['text'], char_from_text)
                score[row['id']] = vv

            max_v = max([e[0] for e in score.values()])

            return sorted([e[0] for e in sorted(score.items(), key=lambda x: (-x[1][0], x[0])) if e[1][0] == max_v])

    def load(self, path="data/models/solvers/solver11"):
        self.filenames = [os.path.join(path, f"solver1{i}.json") for i in range(3)]
        for filename in self.filenames:
            with open(filename, "r", encoding="utf-8") as fin:
                from_tasks = json.load(fin)
                for k, v in from_tasks.items():
                    self.solutions[k.strip()] = v[k.find("..")]

        for w0 in list(from_tasks):
            if w0.find(" ") == -1:
                nw = self.morph.normal_forms(from_tasks[w0.strip()])[0]
                arr = w0.strip().split("..")
                for word in self.common_data.norm2word.get(nw, [nw]):
                    ww = word.lower().replace("ё", "е")
                    if ww not in self.solutions:
                        if (
                                ww.startswith(arr[0])
                                and ww[len(arr[0])] == self.solutions[w0]
                        ):
                            ww = ww[: len(arr[0])] + ".." + ww[len(arr[0]) + 1:]
                            self.solutions[ww] = self.solutions[w0]
        self.is_loaded = True
