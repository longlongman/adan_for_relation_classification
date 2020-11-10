import json
import os
import numpy as np


class Vocab(object):
    def __init__(self, config):
        self.config = config
        self.id2rel = {}
        self.rel2id = {}
        self.en2id = {}
        self.id2en = {}
        self.ch2id = {}
        self.id2ch = {}
        self.type2id = {}
        self.id2type = {}
        self.ch_char2id = {}
        self.id2ch_char = {}
        self.en_pre_emb = None
        self.ch_pre_emb = None
        self.ch_char_pre_emb = None
        self.pad = '<PAD>'
        self.unk = '<UNK>'
        self.build()
        self.load_pre_emb()

    def build(self):
        # build the id2rel and rel2id
        rel_file_path = os.path.join(self.config.data_path, 'labels.txt')
        with open(rel_file_path, 'r') as rel_fr:
            for line in rel_fr:
                line = line.replace('\n', '')
                line = line.replace('"', '')
                line = line.strip()
                self.add(line, 'rel')
        # build the en2id, id2en, ch2id, id2ch, type2id, id2type
        self.add(self.pad, 'en')
        self.add(self.pad, 'ch')
        self.add(self.pad, 'ch_char')
        self.add(self.unk, 'en')
        self.add(self.unk, 'ch')
        self.add(self.unk, 'ch_char')
        en_with_label_train_path = os.path.join(self.config.data_path, self.config.en_with_label_train + '.txt')
        ch_with_label_train_path = os.path.join(self.config.data_path, self.config.ch_with_label_train + '.txt')
        en_without_label_train_path = os.path.join(self.config.data_path, self.config.en_without_label_train + '.txt')
        ch_without_label_train_path = os.path.join(self.config.data_path, self.config.ch_without_label_train + '.txt')
        ch_with_label_dev_path = os.path.join(self.config.data_path, self.config.ch_with_label_dev + '.txt')
        ch_with_label_test_path = os.path.join(self.config.data_path, self.config.ch_with_label_test + '.txt')
        for p in [en_with_label_train_path, en_without_label_train_path]:
            with open(p, 'r') as fr:
                for line in fr:
                    ins = json.loads(line)
                    en_tokens = ins['tokens']
                    h_type = ins['h']['type']
                    t_type = ins['t']['type']
                    for en_token in en_tokens:
                        self.add(en_token, 'en')
                    for t in [h_type, t_type]:
                        self.add(t, 'type')
        for p in [ch_with_label_train_path, ch_without_label_train_path, ch_with_label_dev_path, ch_with_label_test_path]:
            with open(p, 'r') as fr:
                for line in fr:
                    ins = json.loads(line)
                    ch_tokens = ins['tokens']
                    h_type = ins['h']['type']
                    t_type = ins['t']['type']
                    for ch_token in ch_tokens:
                        self.add(ch_token, 'ch')
                        for char in ch_token:
                            self.add(char, 'ch_char')
                    for t in [h_type, t_type]:
                        self.add(t, 'type')

    def getsize(self, which):
        if which == 'ch':
            return len(self.ch2id)
        if which == 'en':
            return len(self.en2id)
        if which == 'rel':
            return len(self.rel2id)
        if which == 'type':
            return len(self.type2id)
        if which == 'ch_char':
            return len(self.ch_char2id)

    def add(self, item, which):
        def in_add(item, xx2id, id2xx):
            if item not in xx2id:
                xx2id[item] = len(xx2id)
                id2xx[len(id2xx)] = item

        if which == 'ch':
            in_add(item, self.ch2id, self.id2ch)
        if which == 'en':
            in_add(item, self.en2id, self.id2en)
        if which == 'rel':
            in_add(item, self.rel2id, self.id2rel)
        if which == 'type':
            in_add(item, self.type2id, self.id2type)
        if which == 'ch_char':
            in_add(item, self.ch_char2id, self.id2ch_char)

    def getitem(self, idx, which):
        def in_getitem(idx, id2xx):
            if idx in id2xx:
                return id2xx[idx]
            else:
                return self.unk

        if which == 'ch':
            return in_getitem(idx, self.id2ch)
        if which == 'en':
            return in_getitem(idx, self.id2en)
        if which == 'rel':
            return in_getitem(idx, self.id2rel)
        if which == 'type':
            return in_getitem(idx, self.id2type)
        if which == 'ch_char':
            return in_getitem(idx, self.id2ch_char)

    def getid(self, item, which):
        def in_getid(item, xx2id):
            if item in xx2id:
                return xx2id[item]
            else:
                return xx2id[self.unk]

        if which == 'ch':
            return in_getid(item, self.ch2id)
        if which == 'en':
            return in_getid(item, self.en2id)
        if which == 'rel':
            return in_getid(item, self.rel2id)
        if which == 'type':
            return in_getid(item, self.type2id)
        if which == 'ch_char':
            return in_getid(item, self.ch_char2id)

    def load_pre_emb(self):
        first_line = True
        en_hit = 0
        ch_hit = 0
        ch_char_hit = 0
        word2vec = {}
        with open(self.config.pre_vec_path, 'r') as fr:
            for line in fr:
                if first_line:
                    pre_vec_num, pre_vec_dim = line.split()
                    assert int(pre_vec_dim) == self.config.word_dim
                    first_line = False
                else:
                    tmp = line.split()
                    word = tmp[0]
                    vec = tmp[1:]
                    word2vec[word] = vec
        en_tmp = []
        for item in self.en2id:
            if item is self.pad:
                vec = np.zeros(self.config.word_dim)
            else:
                if item.lower() in word2vec:
                    vec = np.array(word2vec[item.lower()], dtype=float)
                    en_hit += 1
                else:
                    vec = np.random.normal(0, 0.1, self.config.word_dim)
            en_tmp.append(vec)
        self.en_pre_emb = np.array(en_tmp, dtype=float)

        ch_tmp = []
        for item in self.ch2id:
            if item is self.pad:
                vec = np.zeros(self.config.word_dim)
            else:
                if item.lower() in word2vec:
                    vec = np.array(word2vec[item.lower()], dtype=float)
                    ch_hit += 1
                else:
                    vec = np.random.normal(0, 0.1, self.config.word_dim)
            ch_tmp.append(vec)
        self.ch_pre_emb = np.array(ch_tmp, dtype=float)

        ch_char_tmp = []
        for item in self.ch_char2id:
            if item is self.pad:
                vec = np.zeros(self.config.word_dim)
            else:
                if item.lower() in word2vec:
                    vec = np.array(word2vec[item.lower()], dtype=float)
                    ch_char_hit += 1
                else:
                    vec = np.random.normal(0, 0.1, self.config.word_dim)
            ch_char_tmp.append(vec)
        self.ch_char_pre_emb = np.array(ch_char_tmp, dtype=float)

        print('EN hit rate: ', en_hit / len(self.en2id))
        print('CH hit rate: ', ch_hit / len(self.ch2id))
        print('CH char hit rate: ', ch_char_hit / len(self.ch_char2id))
