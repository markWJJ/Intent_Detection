import tensorflow as tf
import os
import sys
sys.path.append('./')

from model.dl_model.model_s_lstm.data_preprocess import Intent_Slot_Data
from model.dl_model.model_s_lstm.model_fun import embedding,sent_encoder,self_attention,loss_function,intent_acc,cosin_com,label_sent_attention,output_layers
from model.dl_model.model_s_lstm.focal_loss import focal_loss
from model.dl_model.model_s_lstm.slstm import lstm_layer,Classifer
from IntentConfig import Config
import numpy as np
import logging
import pickle
from sklearn.metrics import classification_report,precision_recall_fscore_support,confusion_matrix
import gc
path=os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0,path)
from basic_model import BasicModel
base_path=os.path.split(os.path.split(os.path.split(os.path.split(os.path.realpath(__file__))[0])[0])[0])[0]
from model.dl_model.model_s_lstm.Config import Config_lstm
from entity_recognition.ner import EntityRecognition
from xmlrpc.server import SimpleXMLRPCServer
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename='modellog.log',
                    filemode='w')
_logger = logging.getLogger("intent_bagging")

intent_config=Config()

gpu_id=2


os.environ["CUDA_VISIBLE_DEVICES"] = "%s"%gpu_id



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
tf.app.flags.DEFINE_boolean('mask_use Encoder2Decoder',False,'')
tf.app.flags.DEFINE_string("mask_mod", "train", "默认为训练")  # true for prediction
tf.app.flags.DEFINE_string('mask_model_mode', config.model_mode, '模型类型')
tf.app.flags.DEFINE_boolean('mask_use_auto_buckets',config.use_auto_buckets,'是否使用自动桶')
tf.app.flags.DEFINE_string('mask_only_mode','intent','执行哪种单一任务')
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

    def __init__(self,scope='lstm',mod='train'):
        self.scope=scope
        self.Class=Classifer()
        if mod=='train':
            flag='train_new'
        else:
            flag='test'
        with tf.device('/gpu:%s'%gpu_id):
            self.dd = Intent_Slot_Data(train_path=base_path+"/corpus_data/%s"%intent_config.train_name,
                                  test_path=base_path+"/corpus_data/%s"%intent_config.dev_name,
                                  dev_path=base_path+"/corpus_data/%s"%intent_config.dev_name,
                                  batch_size=FLAGS.mask_batch_size,
                                  max_length=FLAGS.mask_max_len, flag=flag,
                                  use_auto_bucket=FLAGS.mask_use_auto_buckets,save_model=self.scope)

            self.word_vocab = self.dd.vocab
            self.word_num = self.dd.vocab_num
            self.id2intent = self.dd.id2intent
            self.intent_num = len(self.id2intent)

    def _decay(self):
        costs = []
        # 遍历所有可训练变量
        for var in tf.trainable_variables():
            # 只计算标有“DW”的变量
            costs.append(tf.nn.l2_loss(var))
        # 加和，并乘以衰减因子
        return tf.multiply(0.0001, tf.add_n(costs))

    def batch_norm(self, name, x):
        with tf.variable_scope(name):
            # 输入通道维数
            params_shape = [x.get_shape()[-1]]
            # offset
            beta = tf.get_variable('beta',
                                   params_shape,
                                   tf.float32,
                                   initializer=tf.constant_initializer(0.0, tf.float32))
            # scale
            gamma = tf.get_variable('gamma',
                                    params_shape,
                                    tf.float32,
                                    initializer=tf.constant_initializer(1.0, tf.float32))

            if FLAGS.mask_mod == 'train':
                # 为每个通道计算均值、标准差
                mean, variance = tf.nn.moments(x, [0, 1, 2], name='moments')
                # 新建或建立测试阶段使用的batch均值、标准差
                moving_mean = tf.get_variable('moving_mean',
                                              params_shape, tf.float32,
                                              initializer=tf.constant_initializer(0.0, tf.float32),
                                              trainable=False)
                moving_variance = tf.get_variable('moving_variance',
                                                  params_shape, tf.float32,
                                                  initializer=tf.constant_initializer(1.0, tf.float32),
                                                  trainable=False)
                # 添加batch均值和标准差的更新操作(滑动平均)
                # moving_mean = moving_mean * decay + mean * (1 - decay)
                # moving_variance = moving_variance * decay + variance * (1 - decay)
                self._extra_train_ops.append(moving_averages.assign_moving_average(
                    moving_mean, mean, 0.9))
                self._extra_train_ops.append(moving_averages.assign_moving_average(
                    moving_variance, variance, 0.9))
            else:
                # 获取训练中积累的batch均值、标准差
                mean = tf.get_variable('moving_mean',
                                       params_shape, tf.float32,
                                       initializer=tf.constant_initializer(0.0, tf.float32),
                                       trainable=False)
                variance = tf.get_variable('moving_variance',
                                           params_shape, tf.float32,
                                           initializer=tf.constant_initializer(1.0, tf.float32),
                                           trainable=False)
                # 添加到直方图总结
                tf.summary.histogram(mean.op.name, mean)
                tf.summary.histogram(variance.op.name, variance)

            # BN层：((x-mean)/var)*gamma+beta
            y = tf.nn.batch_normalization(x, mean, variance, beta, gamma, 0.001)
            y.set_shape(x.get_shape())

        return y


    def __build_model__(self,):
        with tf.variable_scope(name_or_scope=self.scope):
            with tf.device('/device:GPU:%s'%gpu_id):

                self.save_model=base_path+'/save_model/model_lstm_mask/%s.ckpt'%self.scope
                ee=EntityRecognition()
                self.entity_id = []
                for k, v in self.word_vocab.items():
                    if k in ee.entity_dict.keys():
                        self.entity_id.append(v)


                self.sent_word = tf.placeholder(shape=(None, FLAGS.mask_max_len), dtype=tf.int32,name='sent_word')
                self.sent_len = tf.placeholder(shape=(None,), dtype=tf.int32,name='sent_len')
                self.sent_mask = tf.placeholder(shape=(None, FLAGS.mask_max_len), dtype=tf.float32,name='sent_mask')
                self.dropout=tf.placeholder(dtype=tf.float32)
                self.intent_y = tf.placeholder(shape=(None, self.intent_num), dtype=tf.int32,name='intent_y')



                initial_hidden_states = embedding(self.sent_word, self.word_num, FLAGS.mask_embedding_dim, 'sent_emb')
                initial_cell_states = tf.identity(initial_hidden_states)
                initial_hidden_states = tf.nn.dropout(initial_hidden_states, self.dropout)
                initial_cell_states = tf.nn.dropout(initial_cell_states, self.dropout)

                # create layers
                new_hidden_states, new_cell_state, dummynode_hidden_states = self.Class.slstm_cell("word_slstm",
                                                                                             FLAGS.mask_embedding_dim,
                                                                                             self.sent_len,
                                                                                             initial_hidden_states,
                                                                                             initial_cell_states,
                                                                                             3,FLAGS.mask_max_len)
                softmax_w = tf.Variable(
                    tf.random_normal([2 * FLAGS.mask_embedding_dim, self.intent_num], mean=0.0, stddev=0.1,
                                     dtype=tf.float32), dtype=tf.float32, name="softmax_w")
                softmax_b = tf.Variable(
                    tf.random_normal([self.intent_num], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32,
                    name="softmax_b")
                # representation=dummynode_hidden_states
                representation = tf.reduce_mean(
                    tf.concat([new_hidden_states, tf.expand_dims(dummynode_hidden_states, axis=1)], axis=1), axis=1)

                softmax_w2 = tf.Variable(
                    tf.random_normal([FLAGS.mask_embedding_dim, self.intent_num], mean=0.0, stddev=0.1,
                                     dtype=tf.float32), dtype=tf.float32, name="softmax_w2")
                softmax_b2 = tf.Variable(
                    tf.random_normal([self.intent_num], mean=0.0, stddev=0.1, dtype=tf.float32),
                    dtype=tf.float32, name="softmax_b2")
                representation = tf.nn.tanh(tf.matmul(representation, softmax_w2) + softmax_b2)

                self.soft_logit=tf.nn.softmax(representation,1)
                self.loss = tf.losses.softmax_cross_entropy(onehot_labels=self.intent_y, logits=representation)


                # label_emb = tf.Variable(tf.random_uniform(shape=(self.intent_num, 300),maxval=1.0,minval=-1.0, dtype=tf.float32), trainable=True)
                # sen_enc = sent_encoder(sent_word_emb=sent_emb, hidden_dim=FLAGS.mask_hidden_dim, num=FLAGS.mask_max_len,
                #                        sequence_length=self.sent_len, name='sent_enc',dropout=self.dropout)
                #
                #
                # sent_attention = self_attention(sen_enc, self.sent_mask)
                # stack_sent_enc = tf.stack([tf.concat((ele, sent_attention), 1) for ele in sen_enc], 1,name='stack_sent_enc')
                #
                # stack_sent_enc=tf.nn.dropout(stack_sent_enc,self.dropout)
                #
                # out = label_sent_attention(stack_sent_enc, label_emb, self.sent_mask)
                #
                # logit = output_layers(out, self.intent_num, name='out_layers', reuse=False)
                # self.soft_logit = tf.nn.softmax(logit, 1,name='mask_soft_logit')
                #
                # loss=focal_loss(self.soft_logit,tf.cast(self.intent_y,tf.float32))
                #
                # class_y = tf.constant(name='class_y', shape=[self.intent_num, self.intent_num], dtype=tf.float32,
                #                       value=np.identity(self.intent_num), )
                #
                # logit_label = output_layers(label_emb, self.intent_num, name='out_layers', reuse=True)
                # label_loss = tf.losses.softmax_cross_entropy(onehot_labels=class_y, logits=logit_label)
                # #
                # # loss = tf.losses.softmax_cross_entropy(onehot_labels=self.intent_y, logits=logit)
                # l2_loss=self._decay()
                # self.loss = 0.7 * loss + 0.3 * label_loss
                # ss=tf.concat((sent_attention,tf.stack_sent_enc),1)
                # cosin=cosin_com(ss,label_emb,intent_num)

                # loss, soft_logit=loss_function(cosin,intent_y)

                # loss,soft_logit=loss_function(cosin,intent_y)
                # ss=tf.concat((sent_attention,sen_enc[0]),1)
                # logit=tf.layers.dense(ss,intent_num)
                # soft_logit=tf.nn.softmax(logit,1)
                # loss=focal_loss(soft_logit,tf.cast(intent_y,tf.float32))
                # # loss=tf.losses.softmax_cross_entropy(intent_y,logit)
                # #
                self.optimizer = tf.train.AdamOptimizer(FLAGS.mask_learning_rate).minimize(self.loss)

                # opt=tf.train.AdamOptimizer(0.3)
                # grad_var=opt.compute_gradients(loss)
                # cappd_grad_varible=[[tf.clip_by_value(g,1e-5,1.0),v] for g,v in grad_var]
                # optimizer=opt.apply_gradients(grads_and_vars=cappd_grad_varible)

    def __train__(self):

        config = tf.ConfigProto(allow_soft_placement=True)

        _logger.info("load data")
        # _logger.info('entity_words:%s'%(entity_dict.keys()))
        with tf.Session(config=config) as sess:
            num_batch = self.dd.num_batch
            init_train_acc = 0.0
            init_dev_acc = 0.0
            init_train_loss=9999.99
            init_dev_loss=9999.99
            saver = tf.train.Saver()
            if os.path.exists('%s.meta' % self.save_model):
                saver.restore(sess, self.save_model)
            else:
                sess.run(tf.global_variables_initializer())
            dev_sent, dev_slot, dev_intent, dev_rel_len,_ = self.dd.get_dev()
            dev_sent_mask = get_sent_mask(dev_sent, self.entity_id)
            train_sent, train_slot, train_intent, train_rel_len,_ = self.dd.get_train()
            train_sent_mask = get_sent_mask(train_sent, self.entity_id)
            for i in range(FLAGS.mask_epoch):
                for _ in range(num_batch):
                    sent, slot, intent_label, rel_len, cur_len = self.dd.next_batch()
                    batch_sent_mask = get_sent_mask(sent, self.entity_id)

                    soft_logit_, loss_, _ = sess.run([self.soft_logit, self.loss, self.optimizer],
                                                     feed_dict={self.sent_word: sent,
                                                                self.sent_len: rel_len,
                                                                self.intent_y: intent_label,
                                                                self.sent_mask: batch_sent_mask,
                                                                self.dropout:FLAGS.mask_keep_dropout
                                                                })


                train_soft_logit_, train_loss_ = sess.run([self.soft_logit, self.loss],
                                                          feed_dict={self.sent_word: train_sent,
                                                                     self.sent_len: train_rel_len,
                                                                     self.intent_y: train_intent,
                                                                     self.sent_mask: train_sent_mask,
                                                                     self.dropout:1.0
                                                                     })
                train_acc = intent_acc(train_soft_logit_, train_intent, self.id2intent)

                dev_soft_logit_, dev_loss_ = sess.run([self.soft_logit, self.loss],
                                                      feed_dict={self.sent_word: dev_sent,
                                                                 self.sent_len: dev_rel_len,
                                                                 self.intent_y: dev_intent,
                                                                 self.sent_mask: dev_sent_mask,
                                                                 self.dropout:1.0
                                                                 })
                dev_acc = intent_acc(dev_soft_logit_, dev_intent, self.id2intent)

                # if train_acc > init_train_acc and dev_acc > (init_dev_acc-0.05):
                if  dev_acc>init_dev_acc:
                    init_dev_loss=dev_loss_
                    init_dev_acc=dev_acc
                    # init_train_acc = train_acc
                    # init_dev_acc = dev_acc
                    _logger.info('save')
                    saver.save(sess,self.save_model)

                _logger.info('第 %s 次迭代  train_loss:%s train_acc:%s dev_loss:%s dev_acc:%s' % (
                i, train_loss_, train_acc, dev_loss_, dev_acc))

    def __infer__(self,sents,sess):


        sent_arr, sent_vec = self.dd.get_sent_char(sents)
        sent_mask = get_sent_mask(sent_arr, self.entity_id)

        intent_logit = sess.run(self.soft_logit, feed_dict={self.sent_word: sent_arr,
                                                            self.sent_len: sent_vec,
                                                            self.sent_mask:sent_mask,
                                                            self.dropout:1.0})

        res = []
        for ele in intent_logit:

            ss = [[self.id2intent[index], str(e)] for index, e in enumerate(ele) if e >= 0.3]
            if not ss:
                ss = [[self.id2intent[np.argmax(ele)], str(np.max(ele))]]
            res.append(ss)
        return res

    def infer_dev(self):
        config = tf.ConfigProto(  # limit to num_cpu_core CPU usage
                                inter_op_parallelism_threads=8,
                                intra_op_parallelism_threads=8,
                                log_device_placement=False,
                                allow_soft_placement=True,
                                )

        id2intent = self.dd.id2intent
        id2sent=self.dd.id2sent
        saver = tf.train.Saver()
        with tf.Session(config=config) as sess:
            if os.path.exists('%s.meta' % self.save_model):
                saver.restore(sess, '%s' % self.save_model)
            else:
                sess.run(tf.global_variables_initializer())
            dev_sent, dev_slot, dev_intent, dev_rel_len,dev_index = self.dd.get_dev()
            train_sent, train_slot, train_intent, train_rel_len,train_index = self.dd.get_train()

            dev_sent_mask = get_sent_mask(dev_sent, self.entity_id)
            train_sent_mask = get_sent_mask(train_sent, self.entity_id)

            dev_softmax_logit, dev_loss = sess.run([self.soft_logit, self.loss], feed_dict={self.sent_word: dev_sent,
                                                                                               self.sent_len: dev_rel_len,
                                                                                               self.intent_y: dev_intent,
                                                                                               self.sent_mask: dev_sent_mask,
                                                                                            self.dropout:1.0
                                                                                               })



            self.matirx(dev_softmax_logit, dev_intent, id2intent,id2sent,dev_sent,dev_index, 'dev')

            train_softmax_logit, train_loss = sess.run([self.soft_logit, self.loss],
                                                       feed_dict={self.sent_word: train_sent,
                                                                  self.sent_len: train_rel_len,
                                                                  self.intent_y: train_intent,
                                                                  self.sent_mask: train_sent_mask,
                                                                  self.dropout:1.0
                                                                  })
            self.matirx(train_softmax_logit, train_intent, id2intent,id2sent,train_sent,train_index, 'train')

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


    def matirx(self,pre_label,label,id2intent,id2sent,sent,indexs,type):

        pre_label_prob=np.max(pre_label,1)
        pre_label=np.argmax(pre_label,1)
        label=np.argmax(label,1)

        pre_label=pre_label.flatten()
        label=label.flatten()

        ss = [[int(k), v] for k, v in id2intent.items()]
        ss.sort(key=lambda x: x[0], reverse=False)
        labels = [e[0] for e in ss]
        traget_name=[e[1] for e in ss]
        class_re=classification_report(label,pre_label,target_names=traget_name,labels=labels)
        print(class_re)
        # con_mat=confusion_matrix(y_true=label,y_pred=pre_label,labels=labels)

        sents=[]
        for ele in sent:
            s=' '.join([id2sent[e]for e in ele if e!=0])
            sents.append(s)


        confus_dict={}
        for ele in traget_name:
            confus_dict[ele]={}
            for e in traget_name:
                confus_dict[ele][e]=[0,[]]

        for true,pred,sent,prob,ii in zip(label,pre_label,sents,pre_label_prob,indexs):

            true_name=id2intent[true]
            pred_name=id2intent[pred]

            data_list=confus_dict[true_name][pred_name]
            data_list[0]+=1
            data_list[1].append((ii,prob,sent))
            confus_dict[true_name][pred_name]=data_list
        # print(confus_dict)
        if type=='train':
            pickle.dump(confus_dict,open('./train.p','wb'))
        elif type=='dev':
            pickle.dump(confus_dict,open('./dev.p','wb'))



def main(_):

    lm=LstmMask()
    lm.__build_model__()
    if FLAGS.mask_mod=='train':
        lm.__train__()

    elif FLAGS.mask_mod=='infer_dev':
        lm.infer_dev()

    elif FLAGS.mask_mod=='server':
        lm.__server__()










if __name__ == '__main__':
    tf.app.run()