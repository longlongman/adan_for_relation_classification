from torch.utils.data import DataLoader, Dataset
import json
import os
import numpy as np
import torch


class MyDataset(Dataset):
    def __init__(self, config, prefix, language, vocab):
        self.config = config
        self.vocab = vocab
        self.language = language
        self.prefix = prefix
        self.raw_data = self.load_raw_data(os.path.join(self.config.data_path, prefix + '.txt'))

    def load_raw_data(self, path):
        ret = []
        with open(path, 'r') as fr:
            for line in fr:
                ret.append(json.loads(line))
        return ret

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, idx):
        def distance(dist):
            if dist < -60:
                return 1
            elif -60 <= dist <= 60:
                return dist + 62
            return 123

        ins = self.raw_data[idx]
        tokens = ins['tokens']
        h_pos = ins['h']['pos']
        t_pos = ins['t']['pos']
        h_type = ins['h']['type']
        t_type = ins['t']['type']
        r = ins['r']

        if self.language == 'en':
            token_ids = [self.vocab.getid(token, 'en') for token in tokens]
        elif self.language == 'ch':
            token_ids = [self.vocab.getid(token, 'ch') for token in tokens]
        elif self.language == 'ch_char':
            token_ids = [self.vocab.getid(token, 'ch_char') for token in tokens]
        else:
            raise NotImplementedError
        token_ids = np.array(token_ids)

        h_dist = [distance(i - h_pos[0]) for i in range(len(tokens))]
        t_dist = [distance(i - t_pos[0]) for i in range(len(tokens))]
        h_dist = np.array(h_dist)
        t_dist = np.array(t_dist)

        h_ner = self.vocab.getid(h_type, 'type')
        t_ner = self.vocab.getid(t_type, 'type')

        rel = self.vocab.getid(r, 'rel')

        text_len = len(tokens)

        return token_ids, h_dist, t_dist, h_ner, t_ner, rel, text_len


def my_collate_fn(batch):
    batch.sort(key=lambda x: x[6], reverse=True)
    token_ids, h_dist, t_dist, h_ner, t_ner, rel, text_len = zip(*batch)
    cur_batch = len(batch)
    max_len = max(text_len)
    batch_token_ids = torch.LongTensor(cur_batch, max_len).zero_()
    batch_h_dist = torch.LongTensor(cur_batch, max_len).zero_()
    batch_t_dist = torch.LongTensor(cur_batch, max_len).zero_()
    batch_h_ner = torch.LongTensor(cur_batch, 1).zero_()
    batch_t_ner = torch.LongTensor(cur_batch, 1).zero_()
    batch_rel = torch.LongTensor(cur_batch).zero_()
    batch_text_len = torch.LongTensor(cur_batch).zero_()

    for i in range(cur_batch):
        batch_token_ids[i, :text_len[i]].copy_(torch.from_numpy(token_ids[i]))
        batch_h_dist[i, :text_len[i]].copy_(torch.from_numpy(h_dist[i]))
        batch_t_dist[i, :text_len[i]].copy_(torch.from_numpy(t_dist[i]))
        batch_h_ner[i, 0] = h_ner[i]
        batch_t_ner[i, 0] = t_ner[i]
        batch_rel[i] = rel[i]
        batch_text_len[i] = text_len[i]

    return {
        'token_ids': batch_token_ids,
        'h_dist': batch_h_dist,
        't_dist': batch_t_dist,
        'h_ner': batch_h_ner,
        't_ner': batch_t_ner,
        'rel': batch_rel,
        'text_len': batch_text_len
    }


def get_loader(config, prefix, vocab, language, num_workers=0, collate_fn=my_collate_fn):
    dataset = MyDataset(config, prefix, language, vocab)
    data_loader = DataLoader(dataset=dataset, batch_size=config.batch_size, shuffle=True, num_workers=num_workers,
                             collate_fn=collate_fn)
    return data_loader
