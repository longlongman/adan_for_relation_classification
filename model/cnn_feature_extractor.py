import torch.nn as nn
import torch.nn.functional as functional
import torch


class CNNFeatureExtractor(nn.Module):
    def __init__(self,
                 config,
                 vocab,
                 num_layers,
                 hidden_size,
                 kernel_num,
                 kernel_sizes,
                 dropout):
        super(CNNFeatureExtractor, self).__init__()
        self.config = config

        self.en_word_emb = nn.Embedding(vocab.getsize('en'), config.word_dim, padding_idx=0)
        self.en_word_emb.weight = nn.Parameter(torch.from_numpy(vocab.en_pre_emb.astype('float32')),
                                               requires_grad=config.tune_emb)

        self.ch_word_emb = nn.Embedding(vocab.getsize('ch'), config.word_dim, padding_idx=0)
        self.ch_word_emb.weight = nn.Parameter(torch.from_numpy(vocab.ch_pre_emb.astype('float32')),
                                               requires_grad=config.tune_emb)

        self.dist_emb = nn.Embedding(config.dist_size, config.dist_dim, padding_idx=0)

        self.ner_emb = nn.Embedding(vocab.getsize('type'), config.ner_dim)

        self.kernel_num = kernel_num
        self.kernel_sizes = kernel_sizes

        self.convs = nn.ModuleList([nn.Conv2d(1, kernel_num, (K, config.word_dim + 2 * config.dist_dim)) for K in kernel_sizes])

        assert num_layers >= 0, 'Invalid layer numbers'
        self.fcnet = nn.Sequential()
        for i in range(num_layers):
            if dropout > 0:
                self.fcnet.add_module('f-dropout-{}'.format(i), nn.Dropout(p=dropout))
            if i == 0:
                self.fcnet.add_module('f-linear-{}'.format(i),
                                      nn.Linear(len(kernel_sizes) * kernel_num + 2 * config.ner_dim, hidden_size))
            else:
                self.fcnet.add_module('f-linear-{}'.format(i), nn.Linear(hidden_size, hidden_size))
            self.fcnet.add_module('f-relu-{}'.format(i), nn.ReLU())

    def forward(self, batch, language):
        # [batch_size, seq_len]
        token_ids = batch['token_ids']
        h_dist = batch['h_dist']
        t_dist = batch['t_dist']
        # [batch_size, 1]
        h_ner = batch['h_ner']
        t_ner = batch['t_ner']

        # [batch_size, seq_len, word_dim(300)]
        if language == 'en':
            embeds = self.en_word_emb(token_ids)
        elif language == 'ch':
            embeds = self.ch_word_emb(token_ids)
        else:
            raise NotImplementedError

        # [batch_size, seq_len, dist_dim(20)]
        h_dist_embeds = self.dist_emb(h_dist)
        t_dist_embeds = self.dist_emb(t_dist)

        # [batch_size, seq_len, word_dim + 2 * dist_dim]
        embeds = torch.cat([embeds, h_dist_embeds, t_dist_embeds], dim=-1)

        # conv
        embeds = embeds.unsqueeze(1)  # batch_size, 1, seq_len, emb_size
        x = [functional.relu(conv(embeds)).squeeze(3) for conv in self.convs]
        x = [functional.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        # [batch_size, len(kernel_size) * kernel_num]
        x = torch.cat(x, 1)

        # [batch_size, ner_dim(20)]
        h_ner_embeds = self.ner_emb(h_ner).squeeze(1)
        t_ner_embeds = self.ner_emb(t_ner).squeeze(1)

        # [batch_size, len(kernel_size) * kernel_num + 2 * ner_dim]
        x = torch.cat([x, h_ner_embeds, t_ner_embeds], dim=-1)

        # fcnet
        # [batch_size, hidden_size]
        return self.fcnet(x)
