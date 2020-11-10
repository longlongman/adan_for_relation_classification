import config
import framework
import argparse
import os
import torch
import numpy as np
import random

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

parser = argparse.ArgumentParser()
parser.add_argument('--debug', type=bool, default=True)
parser.add_argument('--feature_extractor', type=str, default='cnn')
parser.add_argument('--dataset', type=str, default='ACE2005')
parser.add_argument('--en_with_label_train', type=str, default='en_with_label_train')
parser.add_argument('--ch_with_label_train', type=str, default='ch_with_label_train')
parser.add_argument('--en_without_label_train', type=str, default='en_without_label_train')
parser.add_argument('--ch_without_label_train', type=str, default='ch_without_label_train')
parser.add_argument('--ch_with_label_dev', type=str, default='ch_with_label_dev')
parser.add_argument('--ch_with_label_test', type=str, default='ch_with_label_test')
parser.add_argument('--tune_emb', type=bool, default=True)
parser.add_argument('--max_epoch', type=int, default=30)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--seed', type=int, default=100)
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--lr_q', type=float, default=5e-4)
parser.add_argument('--clip_lower', type=float, default=-0.01)
parser.add_argument('--clip_upper', type=float, default=0.01)
parser.add_argument('--n_critic', type=int, default=5)
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--P_bn', type=bool, default=True)
parser.add_argument('--Q_bn', type=bool, default=True)
parser.add_argument('--lambd', type=int, default=10)
parser.add_argument('--framework', type=str, default='GAN')
args = parser.parse_args()

seed = args.seed
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

con = config.Config(args)

if args.framework == 'GAN':
    fw = framework.GANFramework(con)
else:
    raise NotImplementedError

vocab = fw.preprocess()
fw.train(args.feature_extractor, vocab)
