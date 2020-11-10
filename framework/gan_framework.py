import os
import pickle
from utils import Vocab
from utils.utils import freeze_net, unfreeze_net
import torch
from data import get_loader
import model
import torch.optim as optim
import torch.nn.functional as functional
from tqdm import tqdm
import logging
import sys
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import warnings

# turn off the sklearn warning
warnings.filterwarnings('ignore')


class GANFramework(object):
    def __init__(self, config):
        self.config = config

        if not os.path.exists(self.config.checkpoint_dir):
            os.mkdir(self.config.checkpoint_dir)

        if not os.path.exists(self.config.log_dir):
            os.mkdir(self.config.log_dir)

        logging.basicConfig(stream=sys.stderr, level=logging.DEBUG if self.config.debug else logging.INFO)
        fh = logging.FileHandler(os.path.join(self.config.log_dir, self.config.log_file))
        self.log = logging.getLogger(__name__)
        self.log.addHandler(fh)
        self.log.info('Configuration: ')
        self.log.info(self.config.args)

    def preprocess(self):
        self.log.info('Getting Vocabulary...')
        if os.path.exists(os.path.join(self.config.data_path, 'vocab.pkl')):
            with open(os.path.join(self.config.data_path, 'vocab.pkl'), 'rb') as fr:
                vocab = pickle.load(fr)
        else:
            if not self.config.debug:
                with open(os.path.join(self.config.data_path, 'vocab.pkl'), 'wb') as fw:
                    vocab = Vocab(self.config)
                    pickle.dump(vocab, fw)
            else:
                vocab = Vocab(self.config)
        return vocab

    def batch_to_cuda(self, batch):
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = batch[k].cuda()
        return batch

    def get_metric(self, label, predict):
        pos_rel = [i for i in range(self.config.rel_num)]

        macro_f1 = f1_score(label, predict, average='macro', labels=pos_rel)
        macro_precision = precision_score(label, predict, average='macro', labels=pos_rel)
        macro_recall = recall_score(label, predict, average='macro', labels=pos_rel)

        micro_f1 = f1_score(label, predict, average='micro', labels=pos_rel)
        micro_precision = precision_score(label, predict, average='micro', labels=pos_rel)
        micro_recall = recall_score(label, predict, average='micro', labels=pos_rel)

        acc = accuracy_score(label, predict)
        return macro_f1, macro_precision, macro_recall, micro_f1, micro_precision, micro_recall, acc

    def train(self, feature_extractor, vocab):
        if feature_extractor == 'cnn':
            F = model.CNNFeatureExtractor(self.config, vocab, self.config.F_layers, self.config.hidden_dim,
                                          self.config.kernel_num, self.config.kernel_sizes, self.config.dropout).cuda()
        else:
            raise NotImplementedError

        P = model.RelationClassifier(self.config.P_layers, self.config.hidden_dim, self.config.rel_num,
                                     self.config.dropout, self.config.P_bn).cuda()

        Q = model.LanguageDetector(self.config.Q_layers, self.config.hidden_dim, self.config.dropout,
                                   self.config.Q_bn).cuda()

        optimizer = optim.Adam(list(F.parameters()) + list(P.parameters()), lr=self.config.learning_rate)
        optimizerQ = optim.Adam(Q.parameters(), lr=self.config.Q_learning_rate)

        # the English data for training F and P
        en_with_label_train_data_loader = get_loader(self.config, prefix=self.config.en_with_label_train, language='en',
                                                     vocab=vocab)

        # the Chinese data for training F
        ch_with_label_train_data_loader = get_loader(self.config, prefix=self.config.ch_with_label_train, language='ch',
                                                     vocab=vocab)
        ch_with_label_train_data_iter = iter(ch_with_label_train_data_loader)

        # the English data for training Q
        en_without_label_train_data_loader = get_loader(self.config, prefix=self.config.en_without_label_train,
                                                        language='en', vocab=vocab)
        en_without_label_train_data_iter = iter(en_without_label_train_data_loader)

        # the Chinese data for training Q
        ch_without_label_train_data_loader = get_loader(self.config, prefix=self.config.ch_without_label_train,
                                                        language='ch', vocab=vocab)
        ch_without_label_train_data_iter = iter(ch_without_label_train_data_loader)

        # the Chinese data for validate F and P
        ch_with_label_dev_data_loader = get_loader(self.config, prefix=self.config.ch_with_label_dev, language='ch',
                                                   vocab=vocab)
        best_macro_f1 = 0.0
        for epoch in range(self.config.max_epoch):
            F.train()
            P.train()
            Q.train()
            golds = []
            preds = []
            for i, en_with_label_train_batch in tqdm(enumerate(en_with_label_train_data_loader),
                                                     total=len(en_with_label_train_data_loader.dataset) // self.config.batch_size + 1):
                en_with_label_train_batch = self.batch_to_cuda(en_with_label_train_batch)

                try:
                    ch_with_label_train_batch = next(ch_with_label_train_data_iter)
                except StopIteration:
                    ch_with_label_train_data_iter = iter(ch_with_label_train_data_loader)
                    ch_with_label_train_batch = next(ch_with_label_train_data_iter)
                ch_with_label_train_batch = self.batch_to_cuda(ch_with_label_train_batch)

                n_critic = self.config.n_critic
                if n_critic > 0 and ((epoch == 0 and i <= 25) or (i % 500 == 0)):
                    n_critic = 10

                freeze_net(F)
                freeze_net(P)
                unfreeze_net(Q)

                for qiter in range(n_critic):
                    # clip Q parameters to make it a Lipschitz function
                    for p in Q.parameters():
                        p.data.clamp_(self.config.clip_lower, self.config.clip_upper)
                    Q.zero_grad()

                    try:
                        en_without_label_train_batch = next(en_without_label_train_data_iter)
                    except StopIteration:
                        en_without_label_train_data_iter = iter(en_without_label_train_data_loader)
                        en_without_label_train_batch = next(en_without_label_train_data_iter)
                    en_without_label_train_batch = self.batch_to_cuda(en_without_label_train_batch)

                    try:
                        ch_without_label_train_batch = next(ch_without_label_train_data_iter)
                    except StopIteration:
                        ch_without_label_train_data_iter = iter(ch_without_label_train_data_loader)
                        ch_without_label_train_batch = next(ch_without_label_train_data_iter)
                    ch_without_label_train_batch = self.batch_to_cuda(ch_without_label_train_batch)

                    # English -> J_q
                    features_en = F(en_without_label_train_batch, 'en')
                    o_en_ad = Q(features_en)
                    l_en_ad = torch.mean(o_en_ad)
                    (-l_en_ad).backward()

                    # Chinese -> -J_q
                    features_ch = F(ch_without_label_train_batch, 'ch')
                    o_ch_ad = Q(features_ch)
                    l_ch_ad = torch.mean(o_ch_ad)
                    l_ch_ad.backward()

                    optimizerQ.step()

                unfreeze_net(F)
                unfreeze_net(P)
                freeze_net(Q)

                if not self.config.tune_emb:
                    freeze_net(F.en_word_emb)
                    freeze_net(F.ch_word_emb)

                for p in Q.parameters():
                    p.data.clamp_(self.config.clip_lower, self.config.clip_upper)

                F.zero_grad()
                P.zero_grad()

                # training the F and P with English labeled data
                features_en = F(en_with_label_train_batch, 'en')
                o_en_sent = P(features_en)
                l_en_sent = functional.nll_loss(o_en_sent, en_with_label_train_batch['rel'])
                l_en_sent.backward(retain_graph=True)
                o_en_ad = Q(features_en)
                l_en_ad = torch.mean(o_en_ad)
                # English -> \lambda * -J_q
                (self.config.lambd * l_en_ad).backward(retain_graph=True)

                _, pred = torch.max(o_en_sent, 1)
                golds += en_with_label_train_batch['rel'].cpu().tolist()
                preds += pred.cpu().tolist()

                features_ch = F(ch_with_label_train_batch, 'ch')
                o_ch_ad = Q(features_ch)
                l_ch_ad = torch.mean(o_ch_ad)
                # Chinese -> \lambda * J_q
                (-self.config.lambd * l_ch_ad).backward()

                optimizer.step()

            self.log.info('Ending epoch {}'.format(epoch + 1))

            macro_f1, macro_precision, macro_recall, micro_f1, micro_precision, micro_recall, acc = \
                self.get_metric(golds, preds)
            self.log.info('Train macro F1: {:4.3f}, macro P: {:4.3f}, macro R: {:4.3f}, micro F1: {:4.3f}, micro P: {:4.3}, micro R: {:4.3f}, Acc: {:4.3f}'
                          .format(macro_f1, macro_precision, macro_recall, micro_f1, micro_precision, micro_recall, acc))

            macro_f1, macro_precision, macro_recall, micro_f1, micro_precision, micro_recall, acc = \
                self.test(ch_with_label_dev_data_loader, F, P)
            self.log.info('Dev macro F1: {:4.3f}, macro P: {:4.3f}, macro R: {:4.3f}, micro F1: {:4.3f}, micro P: {:4.3}, micro R: {:4.3f}, Acc: {:4.3f}'
                          .format(macro_f1, macro_precision, macro_recall, micro_f1, micro_precision, micro_recall, acc))

            if macro_f1 > best_macro_f1:
                self.log.info('New best Chinese dev macro F1: {:4.3}'.format(macro_f1))
                best_macro_f1 = macro_f1
                torch.save(F.state_dict(), '{}/{}_F.pth'.format(self.config.checkpoint_dir, self.config.checkpoint_file))
                torch.save(P.state_dict(), '{}/{}_P.pth'.format(self.config.checkpoint_dir, self.config.checkpoint_file))
                torch.save(Q.state_dict(), '{}/{}_Q.pth'.format(self.config.checkpoint_dir, self.config.checkpoint_file))

    def test(self, loader, F, P):
        F.eval()
        P.eval()
        it = iter(loader)
        golds = []
        preds = []
        with torch.no_grad():
            for batch in tqdm(it):
                batch = self.batch_to_cuda(batch)
                outputs = P(F(batch, 'ch'))
                _, pred = torch.max(outputs, 1)
                golds += batch['rel'].cpu().tolist()
                preds += pred.cpu().tolist()
        return self.get_metric(golds, preds)

    def test_all(self, vocab):
        if self.config.args.feature_extractor == 'cnn':
            F = model.CNNFeatureExtractor(self.config, vocab, self.config.F_layers, self.config.hidden_dim,
                                          self.config.kernel_num, self.config.kernel_sizes, self.config.dropout).cuda()
        else:
            raise NotImplementedError
        P = model.RelationClassifier(self.config.P_layers, self.config.hidden_dim, self.config.rel_num,
                                     self.config.dropout, self.config.P_bn).cuda()
        F.load_state_dict(torch.load('{}/{}_F.pth'.format(self.config.checkpoint_dir, self.config.checkpoint_file)))
        P.load_state_dict(torch.load('{}/{}_P.pth'.format(self.config.checkpoint_dir, self.config.checkpoint_file)))
        ch_with_label_test_data_loader = get_loader(self.config, prefix=self.config.ch_with_label_test, language='ch',
                                                    vocab=vocab)
        macro_f1, macro_precision, macro_recall, micro_f1, micro_precision, micro_recall, acc = self.test(
            ch_with_label_test_data_loader, F, P)
        self.log.info('Test macro F1: {:4.3f}, macro P: {:4.3f}, macro R: {:4.3f}, micro F1: {:4.3f}, micro P: {:4.3}, micro R: {:4.3f}, Acc: {:4.3f}'
                      .format(macro_f1, macro_precision, macro_recall, micro_f1, micro_precision, micro_recall, acc))
