class Config(object):
    def __init__(self, args):
        self.args = args
        self.debug = args.debug

        # path and file
        self.dataset = args.dataset
        self.dataset = args.dataset
        self.root = '/home/data_ti4_c/longsy/adan_for_relation_classification'
        self.data_path = self.root + '/data/' + self.dataset
        self.en_with_label_train = args.en_with_label_train
        self.ch_with_label_train = args.ch_with_label_train
        self.en_without_label_train = args.en_without_label_train
        self.ch_without_label_train = args.ch_without_label_train
        self.ch_with_label_dev = args.ch_with_label_dev
        self.ch_with_label_test = args.ch_with_label_test
        self.checkpoint_dir = self.root + '/checkpoint/' + self.dataset
        self.log_dir = self.root + '/log/' + self.dataset
        self.analysis_dir = self.root + '/analysis/' + self.dataset
        self.pre_vec_name = 'wiki.zh.align.vec'
        self.pre_vec_path = '/home/data_ti4_c/longsy/feature_adaptation4RC-reimplement' + '/pre-vec/' + self.pre_vec_name
        self.log_file = 'log' + \
                        '_fe_' + args.feature_extractor + \
                        '_ds_' + args.dataset + \
                        '_lambd_' + str(args.lambd) + \
                        '_n_critic_' + str(args.n_critic) + \
                        '.txt'

        self.checkpoint_file = 'model' + \
                               '_fe_' + args.feature_extractor + \
                               '_ds_' + args.dataset + \
                               '_lamba_' + str(args.lambd) + \
                               '_n_critic_' + str(args.n_critic)

        # public configuration
        self.rel_num = 6
        self.tune_emb = args.tune_emb

        self.max_epoch = args.max_epoch
        self.word_dim = 300
        self.hidden_dim = 900
        self.batch_size = args.batch_size
        self.dist_size = 124
        self.dist_dim = 50
        self.ner_dim = 50
        self.dropout = args.dropout

        # F and P configuration
        self.F_layers = 1
        self.P_layers = 2
        self.P_bn = args.P_bn
        self.learning_rate = args.lr
        self.lambd = args.lambd  # [30， 10]

        # Q configuration
        self.Q_layers = 2
        self.Q_bn = args.Q_bn
        self.Q_learning_rate = args.lr_q
        self.clip_lower = args.clip_lower
        self.clip_upper = args.clip_upper
        self.n_critic = args.n_critic

        # cnn feature extractor configuration
        self.kernel_num = 400
        self.kernel_sizes = [3, 4, 5]
