import tensorflow as tf
import os
import sys

sys.path.append('./')

from model.dl_model.model_semantic_match.data_preprocess import Intent_Slot_Data
from model.dl_model.model_semantic_match.model_fun import embedding, sent_encoder, self_attention, loss_function, \
    intent_acc, cosin_com, \
    label_sent_attention, output_layers, sigmiod_layer, last_relevant_output, match_attention
from model.dl_model.model_one_shot.focal_loss import focal_loss
from IntentConfig import Config
import numpy as np
import logging
from entity_recognition.ner import entity_dict
import pickle
from sklearn.metrics import classification_report, precision_recall_fscore_support, confusion_matrix
import gc

path = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, path)
from basic_model import BasicModel

base_path = os.path.split(os.path.split(os.path.split(os.path.split(os.path.realpath(__file__))[0])[0])[0])[0]

from entity_recognition.ner import EntityRecognition
from xmlrpc.server import SimpleXMLRPCServer

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename='modellog.log',
                    filemode='w')
_logger = logging.getLogger("intent_bagging")

intent_config = Config()

gpu_id = 0

os.environ["CUDA_VISIBLE_DEVICES"] = "%s" % gpu_id


class Config_lstm(object):
    '''
    默认配置
    '''
    learning_rate = 0.001
    batch_size = 128
    label_max_len = 16
    sent_len = 30  # 句子长度
    embedding_dim = 100  # 词向量维度
    hidden_dim = 400
    train_dir = './data/train_out_%s.txt'
    dev_dir = './data/dev_out.txt'
    test_dir = './data/test.txt'
    model_dir = base_path + '/save_model/model_lstm_mask/%s.ckpt' % 12
    print('config_lstm', model_dir)
    if not os.path.exists(base_path + '/save_model/model_lstm_mask'):
        os.makedirs(base_path + '/save_model/model_lstm_mask')
    use_cpu_num = 16
    keep_dropout = 0.7
    summary_write_dir = "./tmp/r_net.log"
    epoch = 90
    use_auto_buckets = False
    lambda1 = 0.01
    model_mode = 'bilstm_attention_crf'  # 模型选择：bilstm bilstm_crf bilstm_attention bilstm_attention_crf,cnn_crf


config = Config_lstm()
tf.app.flags.DEFINE_float("mask_lambda1", config.lambda1, "l2学习率")
tf.app.flags.DEFINE_float("mask_learning_rate", config.learning_rate, "学习率")
tf.app.flags.DEFINE_float("mask_keep_dropout", config.keep_dropout, "dropout")
tf.app.flags.DEFINE_integer("mask_batch_size", config.batch_size, "批处理的样本数量")
tf.app.flags.DEFINE_integer("mask_max_len", config.sent_len, "句子长度")
tf.app.flags.DEFINE_integer("mask_max_label_len", config.label_max_len, "句子长度")
tf.app.flags.DEFINE_integer("mask_embedding_dim", config.embedding_dim, "词嵌入维度.")
tf.app.flags.DEFINE_integer("mask_hidden_dim", config.hidden_dim, "中间节点维度.")
tf.app.flags.DEFINE_integer("mask_use_cpu_num", config.use_cpu_num, "限定使用cpu的个数")
tf.app.flags.DEFINE_integer("mask_epoch", config.epoch, "epoch次数")
tf.app.flags.DEFINE_string("mask_summary_write_dir", config.summary_write_dir, "训练数据过程可视化文件保存地址")
tf.app.flags.DEFINE_string("mask_train_dir", config.train_dir, "训练数据的路径")
tf.app.flags.DEFINE_string("mask_dev_dir", config.dev_dir, "验证数据文件路径")
tf.app.flags.DEFINE_string("mask_test_dir", config.test_dir, "测试数据文件路径")
tf.app.flags.DEFINE_string("mask_model_dir", config.model_dir, "模型保存路径")
tf.app.flags.DEFINE_boolean('mask_use Encoder2Decoder', False, '')
tf.app.flags.DEFINE_string("mask_mod", "infer_dev", "默认为训练")  # true for prediction
tf.app.flags.DEFINE_string('mask_model_mode', config.model_mode, '模型类型')
tf.app.flags.DEFINE_boolean('mask_use_auto_buckets', config.use_auto_buckets, '是否使用自动桶')
tf.app.flags.DEFINE_string('mask_only_mode', 'intent', '执行哪种单一任务')
FLAGS = tf.app.flags.FLAGS


def get_sent_mask(sent_ids, entity_ids):
    sent_mask = np.zeros_like(sent_ids, dtype=np.float32)
    for i in range(sent_ids.shape[0]):
        for j in range(sent_ids.shape[1]):
            if sent_ids[i, j] > 0 and sent_ids[i, j] not in entity_ids:
                sent_mask[i, j] = 1.0
            elif sent_ids[i, j] > 0 and sent_ids[i, j] in entity_ids:
                sent_mask[i, j] = 1.0
    return sent_mask


class LstmMask(BasicModel):

    def __init__(self, scope='lstm', mod='train'):
        self.scope = scope

        if mod == 'train':
            flag = 'train_new'
        else:
            flag = 'test'
        with tf.device('/gpu:%s' % gpu_id):
            self.dd = Intent_Slot_Data(train_path=base_path + "/corpus_data/%s" % intent_config.train_name,
                                       test_path=base_path + "/corpus_data/%s" % intent_config.dev_name,
                                       dev_path=base_path + "/corpus_data/%s" % intent_config.dev_name,
                                       batch_size=FLAGS.mask_batch_size,
                                       max_length=FLAGS.mask_max_len, flag=flag,
                                       use_auto_bucket=FLAGS.mask_use_auto_buckets)

            self.word_vocab = self.dd.vocab
            self.word_num = self.dd.vocab_num
            self.id2intent = self.dd.id2intent
            self.intent_num = len(self.id2intent)

            self.entity_id = []
            for k, v in self.word_vocab.items():
                if k in entity_dict.keys():
                    self.entity_id.append(v)

    def _decay(self):
        costs = []
        # 遍历所有可训练变量
        for var in tf.trainable_variables():
            # 只计算标有“DW”的变量
            costs.append(tf.nn.l2_loss(var))
        # 加和，并乘以衰减因子
        return tf.multiply(0.0001, tf.add_n(costs))

    def loss_fun(self, pos_0, pos_1, neg_0, m):

        distance_pos_0_1 = tf.sqrt(tf.reduce_sum(tf.square(pos_0 - pos_1), 2))
        distance_pos_0_neg_0 = tf.sqrt(tf.reduce_sum(tf.square(pos_0 - neg_0), 2))
        distance_pos_1_neg_0 = tf.sqrt(tf.reduce_sum(tf.square(pos_1 - neg_0), 2))
        distance_1 = tf.nn.relu(m + distance_pos_0_1 - distance_pos_0_neg_0)
        distance_2 = tf.nn.relu(m + distance_pos_0_1 - distance_pos_1_neg_0)
        loss = distance_1 + distance_2
        tf.reduce_mean(loss)
        return loss

    def __build_model__(self, ):
        with tf.variable_scope(name_or_scope=self.scope):
            with tf.device('/device:GPU:%s' % gpu_id):
                self.mod = 1
                self.pos_0 = tf.placeholder(shape=(None, FLAGS.mask_max_len), dtype=tf.int32, name='pos_0')
                self.pos_1 = tf.placeholder(shape=(None, FLAGS.mask_max_len), dtype=tf.int32, name='pos_1')
                self.pos_0_len = tf.placeholder(shape=(None,), dtype=tf.int32, name='pos_0_len')
                self.pos_1_len = tf.placeholder(shape=(None,), dtype=tf.int32, name='pos_1_len')
                self.dropout = tf.placeholder(dtype=tf.float32)
                self.neg_0 = tf.placeholder(shape=(None, FLAGS.mask_max_len), dtype=tf.int32, name='neg_0')
                self.neg_0_len = tf.placeholder(shape=(None,), dtype=tf.int32, name='neg_0_len')

                self.pos_0_mask = tf.placeholder(shape=(None, FLAGS.mask_max_len), dtype=tf.float32, name='pos_0_mask')
                self.pos_1_mask = tf.placeholder(shape=(None, FLAGS.mask_max_len), dtype=tf.float32, name='pos_1_mask')
                self.neg_0_mask = tf.placeholder(shape=(None, FLAGS.mask_max_len), dtype=tf.float32, name='neg_0_mask')

                pos_0_emb, emb_0 = embedding(self.pos_0, self.word_num, FLAGS.mask_embedding_dim, 'sent_emb',
                                             reuse=False)
                pos_1_emb, emb_1 = embedding(self.pos_1, self.word_num, FLAGS.mask_embedding_dim, 'sent_emb',
                                             reuse=True)
                neg_0_emb, _ = embedding(self.neg_0, self.word_num, FLAGS.mask_embedding_dim, 'sent_emb', reuse=True)
                self.pos_0_emb = emb_0
                self.pos_1_emb = emb_1
                pos_0_emb = tf.nn.dropout(pos_0_emb, self.dropout)
                pos_1_emb = tf.nn.dropout(pos_1_emb, self.dropout)
                neg_0_emb = tf.nn.dropout(neg_0_emb, self.dropout)

                pos_0_enc = sent_encoder(sent_word_emb=pos_0_emb, hidden_dim=FLAGS.mask_hidden_dim,
                                         num=FLAGS.mask_max_len,
                                         sequence_length=self.pos_0_len, name='pos_0_enc', dropout=self.dropout,
                                         reuse=False)

                pos_1_enc = sent_encoder(sent_word_emb=pos_1_emb, hidden_dim=FLAGS.mask_hidden_dim,
                                         num=FLAGS.mask_max_len,
                                         sequence_length=self.pos_1_len, name='pos_0_enc', dropout=self.dropout,
                                         reuse=True)

                neg_0_enc = sent_encoder(sent_word_emb=neg_0_emb, hidden_dim=FLAGS.mask_hidden_dim,
                                         num=FLAGS.mask_max_len,
                                         sequence_length=self.neg_0_len, name='pos_0_enc', dropout=self.dropout,
                                         reuse=True)

                # pos_0_enc_new=match_attention(pos_0_enc,pos_1_enc,self.pos_0_mask,self.pos_1_mask,FLAGS.mask_hidden_dim,self.dropout,name='match_attention',seq_len=FLAGS.mask_max_len,reuse=False)
                # pos_0_enc_new_1=match_attention(pos_0_enc,pos_1_enc,self.pos_0_mask,self.pos_1_mask,FLAGS.mask_hidden_dim,self.dropout,name='match_attention',seq_len=FLAGS.mask_max_len,reuse=True)

                # self.pos_0_enc_new=pos_0_enc_new
                # self.pos_0_enc_new1 = pos_0_enc_new_1
                # pos_1_enc_new=match_attention(pos_1_enc,pos_0_enc,self.pos_1_mask,self.pos_0_mask,FLAGS.mask_hidden_dim,self.dropout,name='match_attention',seq_len=FLAGS.mask_max_len,reuse=True)
                # neg_0_enc_new=match_attention(neg_0_enc,pos_0_enc,self.neg_0_mask,self.pos_0_mask,FLAGS.mask_hidden_dim,self.dropout,name='match_attention',seq_len=FLAGS.mask_max_len,reuse=True)

                # pos_0_self_attention=self_attention(pos_0_enc,self.pos_0_mask,reuse=False)
                # pos_1_self_attention=self_attention(pos_1_enc,self.pos_1_mask,reuse=True)
                # neg_0_self_attention=self_attention(neg_0_enc,self.pos_0_mask,reuse=True)

                pos_0_enc_new = pos_0_enc
                pos_1_enc_new = pos_1_enc
                neg_0_enc_new = neg_0_enc

                pos_0_last_enc = last_relevant_output(pos_0_enc_new, self.pos_0_len)
                pos_1_last_enc = last_relevant_output(pos_1_enc_new, self.pos_1_len)
                neg_0_last_enc = last_relevant_output(neg_0_enc_new, self.neg_0_len)

                # pos_0_1_enc=tf.concat([pos_0_last_enc,pos_1_self_attention],-1)
                # pos_neg_0_0_enc=tf.concat([pos_0_last_enc,neg_0_self_attention],-1)
                #
                # pos_1_0_enc=tf.concat([pos_1_last_enc,pos_0_self_attention],-1)
                # neg_pos_0_0_enc=tf.concat([neg_0_last_enc,pos_0_self_attention],-1)

                # pos_0_enc=sigmiod_layer(pos_0_enc,FLAGS.mask_hidden_dim,'sigmiod_layer_pos_0_enc',reuse=False)
                # pos_1_enc=sigmiod_layer(pos_1_enc,FLAGS.mask_hidden_dim,'sigmiod_layer_pos_1_enc',reuse=tr)
                # neg_0_enc=sigmiod_layer(neg_0_enc,FLAGS.mask_hidden_dim,'sigmiod_layer_neg_0_enc',reuse=False)

                pos_0_enc = tf.nn.dropout(pos_0_last_enc, self.dropout)
                pos_1_enc = tf.nn.dropout(pos_1_last_enc, self.dropout)
                neg_0_enc = tf.nn.dropout(neg_0_last_enc, self.dropout)

                if self.mod == 0:
                    self.pos_0_encoder = pos_0_enc
                    self.pos_1_encoder = pos_1_enc
                    distance_pos_0_1 = tf.sqrt(tf.reduce_sum(tf.square(pos_0_enc - pos_1_enc), 1))
                    distance_pos_0_neg_0 = tf.sqrt(tf.reduce_sum(tf.square(pos_0_enc - neg_0_enc), 1))
                    distance_pos_1_neg_0 = tf.sqrt(tf.reduce_sum(tf.square(pos_1_enc - neg_0_enc), 1))

                    # distance_pos_0_1=tf.clip_by_value(distance_pos_0_1,1e-5,3.0)
                    # distance_pos_1_neg_0=tf.clip_by_value(distance_pos_1_neg_0,1e-5,3.0)
                    # distance_pos_0_neg_0=tf.clip_by_value(distance_pos_0_neg_0,1e-5,3.0)

                    distance_1 = tf.nn.relu(50.0 + distance_pos_0_1 - distance_pos_0_neg_0)
                    distance_2 = tf.nn.relu(50.0 + distance_pos_0_1 - distance_pos_1_neg_0)
                    loss = distance_1 + distance_2
                    self.loss = tf.reduce_mean(loss)

                    self.dis1_0 = tf.reduce_mean(distance_pos_0_1)
                    self.dis1_1 = tf.reduce_mean(distance_pos_0_neg_0)
                    self.dis1_2 = tf.reduce_mean(distance_pos_1_neg_0)
                    self.dis1 = tf.reduce_mean(distance_1)
                    self.dis2 = tf.reduce_mean(distance_2)
                    # self.loss=self.loss_fun(pos_0_enc,pos_1_enc,neg_0_enc,2)

                    # self.opt=tf.train.AdamOptimizer(FLAGS.mask_learning_rate).minimize(self.loss)
                    #
                    self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.mask_learning_rate)
                    grads_vars = self.optimizer.compute_gradients(self.loss)
                    capped_grads_vars = [[tf.clip_by_value(g, -1e-3, 1.0), v] for g, v in grads_vars]
                    self.opt = self.optimizer.apply_gradients(capped_grads_vars)
                elif self.mod == 1:

                    self.output_features_postive = tf.concat([pos_0_enc, pos_1_enc,
                                                              pos_0_enc - pos_1_enc,
                                                              pos_0_enc * pos_1_enc], axis=-1)

                    self.output_features_negative = tf.concat([pos_0_enc, neg_0_enc,
                                                               pos_0_enc - neg_0_enc,
                                                               pos_0_enc * neg_0_enc], axis=-1)

                    self.estimation_postive = tf.layers.dense(
                        self.output_features_postive,
                        units=2,
                        name="prediction_layer", reuse=False)

                    self.estimation_negative = tf.layers.dense(
                        self.output_features_negative,
                        units=2,
                        name="prediction_layer", reuse=True)

                    self.soft_pos = tf.nn.softmax(self.estimation_postive, 1)

                    y = tf.zeros_like(self.pos_0_len)

                    pos_y = tf.one_hot(y, 2, 1, 0, 1)
                    neg_y = tf.one_hot(y, 2, 0, 1, 1)

                    self.pred = tf.argmax(self.soft_pos, 1)
                    self.pos_y = tf.argmax(pos_y, 1)
                    self.neg_y = tf.argmax(neg_y, 1)
                    self.acc = tf.cast(tf.reduce_sum(tf.cast(tf.equal(self.pred, self.pos_y), tf.int32)), tf.float32) \
                               / tf.cast(tf.reduce_sum(tf.ones_like(self.pos_0_len)), tf.float32)

                    pos_loss = tf.losses.softmax_cross_entropy(pos_y, self.estimation_postive)
                    neg_loss = tf.losses.softmax_cross_entropy(neg_y, self.estimation_negative)

                    self.loss = tf.reduce_mean(pos_loss + neg_loss)

                    self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.mask_learning_rate)
                    grads_vars = self.optimizer.compute_gradients(self.loss)
                    capped_grads_vars = [[tf.clip_by_value(g, -1e-3, 1.0), v] for g, v in grads_vars]
                    self.opt = self.optimizer.apply_gradients(capped_grads_vars)

    def __deal_data(self,data):
        pos_0_array, pos_0_len, pos_1_array, pos_1_len, neg_0_array, neg_0_len = [], [], [], [], [], []

        for ele in data:
            pos_0_array.append(ele['pos_0']['word_arr'])
            pos_0_len.append(ele['pos_0']['real_len_arr'])

            pos_1_array.append(ele['pos_1']['word_arr'])
            pos_1_len.append(ele['pos_1']['real_len_arr'])

            neg_0_array.append(ele['neg_0']['word_arr'])
            neg_0_len.append(ele['neg_0']['real_len_arr'])

        pos_0_array = np.array(pos_0_array)
        pos_1_array = np.array(pos_1_array)
        pos_0_len = np.array(pos_0_len)
        pos_1_len = np.array(pos_1_len)
        neg_0_array = np.array(neg_0_array)
        neg_0_len = np.array(neg_0_len)

        train_pos_0_mask = get_sent_mask(pos_0_array, self.entity_id)
        train_pos_1_mask = get_sent_mask(pos_1_array, self.entity_id)
        train_neg_0_mask = get_sent_mask(neg_0_array, self.entity_id)


        return pos_0_array,pos_1_array,pos_0_len,pos_1_len,neg_0_array,neg_0_len,train_pos_0_mask,train_pos_1_mask,train_neg_0_mask


    def compute_acc(self,sess,sent_arr,sent_len,sent_intent,data_dict,n):
        '''

        :param sent_arr:
        :param sent_len:
        :param intent:
        :param data_dict: {k:list} k为意图类别 list为该类别下样本
        :param n:选择每个样本下n个样本作为标准句
        :return:
        '''
        pass

    def get_norm_array(self,data_dict,n=-1):
        '''
        获取标准句的矩阵
        :param data_dict:
        :param n:
        :return:
        '''

        pos_1_array,pos_1_len,pos_1_label=[],[],[]
        for k,v in data_dict.items():
            for ele in v[:n]:
                pos_1_array.append(ele['word_arr'])
                pos_1_len.append(ele['real_len_arr'])
                pos_1_label.append(k)
        pos_1_array=np.array(pos_1_array)
        pos_1_len=np.array(pos_1_len)
        pos_1_label=np.array(pos_1_label)
        pos_1_mask=get_sent_mask(pos_1_array,self.entity_id)
        return pos_1_array,pos_1_len,pos_1_mask,pos_1_label



    def __train__(self):

        config = tf.ConfigProto(allow_soft_placement=True)

        _logger.info("load data")
        # _logger.info('entity_words:%s'%(entity_dict.keyjr s()))
        with tf.Session(config=config) as sess:
            init_train_acc = 0.0
            init_dev_acc = 0.0
            init_train_loss = 9999.99
            init_dev_loss = 9999.99
            saver = tf.train.Saver()
            # if os.path.exists('./model.ckpt.meta' ):
            #     saver.restore(sess, './model.ckpt')
            # else:
            #     sess.run(tf.global_variables_initializer())

            sess.run(tf.global_variables_initializer())
            train_data = self.dd.train_data
            dev_data = self.dd.get_dev_data()

            pos_0_array, pos_1_array, pos_0_len, pos_1_len, neg_0_array, neg_0_len, train_pos_0_mask, train_pos_1_mask, train_neg_0_mask=self.__deal_data(train_data)

            dev_pos_0_array, dev_pos_1_array, dev_pos_0_len, dev_pos_1_len, dev_neg_0_array, dev_neg_0_len,dev_pos_0_mask, dev_pos_1_mask, dev_neg_0_mask=self.__deal_data(dev_data)

            # word_arr, real_len_arr, intent_arr=self.dd.get_dev_data_new()

            num_batch = int(pos_0_array.shape[0] / FLAGS.mask_batch_size)
            for _ in range(50):
                all_train_loss, all_train_acc = 0.0, 0.0
                for i in range(num_batch):
                    pos_0_array_batch = pos_0_array[i * FLAGS.mask_batch_size:(i + 1) * FLAGS.mask_batch_size]
                    pos_0_len_batch = pos_0_len[i * FLAGS.mask_batch_size:(i + 1) * FLAGS.mask_batch_size]
                    pos_1_array_batch = pos_1_array[i * FLAGS.mask_batch_size:(i + 1) * FLAGS.mask_batch_size]
                    pos_1_len_batch = pos_1_len[i * FLAGS.mask_batch_size:(i + 1) * FLAGS.mask_batch_size]
                    neg_0_array_batch = neg_0_array[i * FLAGS.mask_batch_size:(i + 1) * FLAGS.mask_batch_size]
                    neg_0_len_batch = neg_0_len[i * FLAGS.mask_batch_size:(i + 1) * FLAGS.mask_batch_size]
                    pos_0_mask_batch = train_pos_0_mask[i * FLAGS.mask_batch_size:(i + 1) * FLAGS.mask_batch_size]
                    pos_1_mask_batch = train_pos_1_mask[i * FLAGS.mask_batch_size:(i + 1) * FLAGS.mask_batch_size]
                    neg_0_mask_batch = train_neg_0_mask[i * FLAGS.mask_batch_size:(i + 1) * FLAGS.mask_batch_size]

                    losses, _, train_acc = sess.run(
                        [self.loss, self.opt, self.acc], feed_dict={
                            self.pos_0: pos_0_array_batch,
                            self.pos_0_len: pos_0_len_batch,
                            self.pos_1: pos_1_array_batch,
                            self.pos_1_len: pos_1_len_batch,
                            self.neg_0: neg_0_array_batch,
                            self.neg_0_len: neg_0_len_batch,
                            self.pos_0_mask: pos_0_mask_batch,
                            self.pos_1_mask: pos_1_mask_batch,
                            self.neg_0_mask: neg_0_mask_batch,
                            self.dropout: 1.0
                        })
                    all_train_loss += losses
                    all_train_acc += train_acc

                all_train_loss = all_train_loss / float(num_batch)
                all_train_acc = all_train_acc / float(num_batch)

                dev_losses, dev_acc = sess.run(
                    [self.loss, self.acc], feed_dict={
                        self.pos_0: dev_pos_0_array,
                        self.pos_0_len: dev_pos_0_len,
                        self.pos_1: dev_pos_1_array,
                        self.pos_1_len: dev_pos_1_len,
                        self.neg_0: dev_neg_0_array,
                        self.neg_0_len: dev_neg_0_len,
                        self.pos_0_mask: dev_pos_0_mask,
                        self.pos_1_mask: dev_pos_1_mask,
                        self.neg_0_mask: dev_neg_0_mask,
                        self.dropout: 1.0
                    })

                _logger.info('train_loss:%s train_acc:%s dev_loss:%s dec_acc:%s' % (
                all_train_loss, all_train_acc, dev_losses, dev_acc))

                if dev_losses < init_dev_loss:
                    init_dev_loss = dev_losses
                    saver.save(sess, './model.ckpt')
                    print('save........')

    def __infer__(self):

        config = tf.ConfigProto(  # limit to num_cpu_core CPU usage
            inter_op_parallelism_threads=8,
            intra_op_parallelism_threads=8,
            log_device_placement=False,
            allow_soft_placement=True,
        )

        id2intent = self.dd.id2intent
        id2sent = self.dd.id2sent

        train_data_dict = self.dd.get_train_data_dict()
        standard_sample = {}

        for k, v in train_data_dict.items():
            np.random.shuffle(v)
            standard_sample[k] = v[:3]
        saver = tf.train.Saver()
        with tf.Session(config=config) as sess:
            saver.restore(sess, './model.ckpt')


            while True:
                sent=input('输入：')

                sent_arr, sent_vec = self.dd.get_sent_char([sent])
                sent_mask = get_sent_mask(sent_arr, self.entity_id)
                opt_k=0
                for stand_k, v_ in standard_sample.items():
                    stand_es = standard_sample[stand_k]
                    alls_index = []
                    for stand_e in stand_es:
                        stand_word_arr = stand_e['word_arr']
                        stand_rel_arr = stand_e['real_len_arr']
                        stand_word_arr = stand_word_arr.reshape((1, 30))
                        stand_rel_arr = stand_rel_arr.reshape((1,))
                        stand_mask = get_sent_mask(stand_word_arr, self.entity_id)

                        pred, pos_y_, neg_y_ = sess.run(
                            [self.pred, self.pos_y, self.neg_y], feed_dict={
                                self.pos_0: sent_arr,
                                self.pos_0_len: sent_vec,
                                self.pos_1: stand_word_arr,
                                self.pos_1_len: stand_rel_arr,
                                self.neg_0: stand_word_arr,
                                self.neg_0_len: stand_rel_arr,
                                self.pos_0_mask: sent_mask,
                                self.pos_1_mask: stand_mask,
                                self.neg_0_mask: stand_mask,
                                self.dropout: 1.0})
                        soft_max = pred

                        alls_index.append(soft_max)
                    if alls_index.count(0) >= 2:
                        opt_k = id2intent[stand_k]
                        break

                print(''.join([id2sent[e] for e in sent_arr[0] if e not in [0]]), '\t\t', opt_k,)


    def infer_dev(self):
        config = tf.ConfigProto(  # limit to num_cpu_core CPU usage
            inter_op_parallelism_threads=8,
            intra_op_parallelism_threads=8,
            log_device_placement=False,
            allow_soft_placement=True,
        )

        id2intent = self.dd.id2intent
        id2sent = self.dd.id2sent
        saver = tf.train.Saver()
        with tf.Session(config=config) as sess:
            saver.restore(sess, './model.ckpt')

            dev_data_dict = self.dd.get_dev_data_dict()
            train_data_dict = self.dd.get_train_data_dict()

            dev_data_dict = train_data_dict

            standard_sample = {}

            for k, v in train_data_dict.items():
                np.random.shuffle(v)
                standard_sample[k] = v[:3]

            all_num = 0
            num = 0
            for k, v in dev_data_dict.items():

                for dev_e in v:
                    dev_word_arr = dev_e['word_arr']
                    dev_rel_arr = dev_e['real_len_arr']
                    dev_word_arr = dev_word_arr.reshape((1, 30))
                    dev_rel_arr = dev_rel_arr.reshape((1,))
                    dev_mask = get_sent_mask(dev_word_arr, self.entity_id)

                    min_dis = 100
                    opt_k = None
                    for stand_k, v_ in standard_sample.items():
                        stand_es = standard_sample[stand_k]
                        alls_index = []
                        for stand_e in stand_es:
                            stand_word_arr = stand_e['word_arr']
                            stand_rel_arr = stand_e['real_len_arr']
                            stand_word_arr = stand_word_arr.reshape((1, 30))
                            stand_rel_arr = stand_rel_arr.reshape((1,))
                            stand_mask = get_sent_mask(stand_word_arr, self.entity_id)

                            pred, pos_y_, neg_y_ = sess.run(
                                [self.pred, self.pos_y, self.neg_y], feed_dict={
                                    self.pos_0: dev_word_arr,
                                    self.pos_0_len: dev_rel_arr,
                                    self.pos_1: stand_word_arr,
                                    self.pos_1_len: stand_rel_arr,
                                    self.neg_0: stand_word_arr,
                                    self.neg_0_len: stand_rel_arr,
                                    self.pos_0_mask: dev_mask,
                                    self.pos_1_mask: stand_mask,
                                    self.neg_0_mask: stand_mask,
                                    self.dropout: 1.0})
                            soft_max = pred

                            alls_index.append(soft_max)
                        if alls_index.count(0) >= 2:
                            opt_k = id2intent[stand_k]
                            break

                    print(''.join([id2sent[e] for e in dev_word_arr[0] if e not in [0]]), '\t\t', opt_k, '\t\t',
                          id2intent[k])

                    if opt_k == id2intent[k]:
                        num += 1
                    all_num += 1

            print(float(num) / float(all_num))

    def __server__(self):

        config = tf.ConfigProto(device_count={"CPU": FLAGS.mask_use_cpu_num},  # limit to num_cpu_core CPU usage
                                inter_op_parallelism_threads=8,
                                intra_op_parallelism_threads=8,
                                log_device_placement=False,
                                allow_soft_placement=True,
                                )
        _logger.info("load data")

        with tf.Session(config=config) as sess:

            saver = tf.train.Saver()
            sess = tf.Session(config=config)
            if os.path.exists('%s.meta' % FLAGS.mask_model_dir):
                saver.restore(sess, '%s' % FLAGS.mask_model_dir)
            else:
                _logger.error('lstm没有模型')

            def intent(sent_list):
                sents = []
                _logger.info("%s" % len(sent_list))

                sents = []
                for sent in sent_list:
                    sents.append(self.dd.get_sent_char(sent))

                sent_arr, sent_vec = self.dd.get_sent_char(sents)
                infer_sent_mask = get_sent_mask(sent_arr, self.entity_id)

                intent_logit = sess.run(self.soft_logit, feed_dict={self.sent_word: sent_arr,
                                                                    self.sent_len: sent_vec,
                                                                    self.sent_mask: infer_sent_mask})

                res = []
                for ele in intent_logit:
                    ss = [[self.id2intent[index], str(e)] for index, e in enumerate(ele) if e >= 0.3]
                    if not ss:
                        ss = [[self.id2intent[np.argmax(ele)], str(np.max(ele))]]
                    res.append(ss)

                del sents
                gc.collect()
                _logger.info('process end')
                _logger.info('%s' % res)
                return res

            svr = SimpleXMLRPCServer((config_lstm.host, 8087), allow_none=True)
            svr.register_function(intent)
            svr.serve_forever()

    def matirx(self, pre_label, label, id2intent, id2sent, sent, indexs, type):

        pre_label_prob = np.max(pre_label, 1)
        pre_label = np.argmax(pre_label, 1)
        label = np.argmax(label, 1)

        pre_label = pre_label.flatten()
        label = label.flatten()

        ss = [[int(k), v] for k, v in id2intent.items()]
        ss.sort(key=lambda x: x[0], reverse=False)
        labels = [e[0] for e in ss]
        traget_name = [e[1] for e in ss]
        class_re = classification_report(label, pre_label, target_names=traget_name, labels=labels)
        print(class_re)
        # con_mat=confusion_matrix(y_true=label,y_pred=pre_label,labels=labels)

        sents = []
        for ele in sent:
            s = ' '.join([id2sent[e] for e in ele if e != 0])
            sents.append(s)

        confus_dict = {}
        for ele in traget_name:
            confus_dict[ele] = {}
            for e in traget_name:
                confus_dict[ele][e] = [0, []]

        for true, pred, sent, prob, ii in zip(label, pre_label, sents, pre_label_prob, indexs):
            true_name = id2intent[true]
            pred_name = id2intent[pred]

            data_list = confus_dict[true_name][pred_name]
            data_list[0] += 1
            data_list[1].append((ii, prob, sent))
            confus_dict[true_name][pred_name] = data_list
        # print(confus_dict)
        if type == 'train':
            pickle.dump(confus_dict, open('./train.p', 'wb'))
        elif type == 'dev':
            pickle.dump(confus_dict, open('./dev.p', 'wb'))


def main(_):
    lm = LstmMask()
    lm.__build_model__()
    if FLAGS.mask_mod == 'train':
        lm.__train__()

    elif FLAGS.mask_mod == 'infer_dev':
        lm.infer_dev()

    elif FLAGS.mask_mod == 'server':
        lm.__server__()
    elif FLAGS.mask_mod == 'infer_':
        lm.__infer__()


if __name__ == '__main__':
    tf.app.run()