import tensorflow as tf
import os
import sys
sys.path.append('./')

from model.dl_model.model_lstm_res.data_preprocess import Intent_Slot_Data
from model.dl_model.model_lstm_mask.model_fun import embedding,sent_encoder,self_attention,loss_function,intent_acc,cosin_com,label_sent_attention,output_layers
from model.dl_model.model_lstm_mask.focal_loss import focal_loss
from IntentConfig import Config
import numpy as np
import logging
import pickle
from sklearn.metrics import classification_report,precision_recall_fscore_support,confusion_matrix
import gc
path=os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0,path)
import collections
slim=tf.contrib.slim
base_path=os.path.split(os.path.split(os.path.split(os.path.split(os.path.realpath(__file__))[0])[0])[0])[0]

from tensorflow.python.training import moving_averages
from entity_recognition.ner import EntityRecognition
from xmlrpc.server import SimpleXMLRPCServer
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename='modellog.log',
                    filemode='w')
_logger = logging.getLogger("intent_bagging")

intent_config=Config()

gpu_id=3


os.environ["CUDA_VISIBLE_DEVICES"] = "%s"%gpu_id

class Block(collections.namedtuple('Block',['scope','unit_fn','args'])):
    pass



class Config_lstm(object):
    '''
    默认配置
    '''
    learning_rate = 0.001
    batch_size = 128
    label_max_len=16
    sent_len = 30  # 句子长度
    embedding_dim = 100  # 词向量维度
    hidden_dim = 400
    train_dir = './data/train_out_%s.txt'
    dev_dir = './data/dev_out.txt'
    test_dir = './data/test.txt'
    model_dir = base_path+'/save_model/model_lstm_mask/intent_lstm_%s.ckpt'%intent_config.save_model_name
    if not os.path.exists(base_path+'/save_model/model_lstm_mask'):
        os.makedirs(base_path+'/save_model/model_lstm_mask')
    use_cpu_num = 16
    keep_dropout = 0.7
    summary_write_dir = "./tmp/r_net.log"
    epoch = 100
    use_auto_buckets=False
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


class ResNet(object):

    def __init__(self):
        with tf.device('/gpu:%s'%gpu_id):
            self.dd = Intent_Slot_Data(train_path=base_path+"/corpus_data/%s"%intent_config.train_name,
                                  test_path=base_path+"/corpus_data/%s"%intent_config.dev_name,
                                  dev_path=base_path+"/corpus_data/%s"%intent_config.dev_name,
                                  batch_size=FLAGS.mask_batch_size,
                                  max_length=FLAGS.mask_max_len, flag="train_new",
                                  use_auto_bucket=FLAGS.mask_use_auto_buckets)

            self.word_vocab = self.dd.vocab
            self.word_num = self.dd.vocab_num
            self.id2intent = self.dd.id2intent
            self.intent_num = len(self.id2intent)
            print(self.word_num)
            self.en=EntityRecognition()
            self.entity_id = []
            for k, v in self.word_vocab.items():
                if k in self.en.entity_dict.keys():
                    self.entity_id.append(v)

            self._extra_train_ops = []

    def subsample(self,inputs,factor,ksizes,scope=None):

        with tf.variable_scope(name_or_scope=scope):
            if factor==1:
                return inputs
            else:
                return tf.nn.max_pool(inputs,ksize=[1,ksizes[0],ksizes[1],1],strides=[1,factor,factor,1],name=scope,padding='VALID')

    def conv2d_same(self,inputs,fliter_size,num_in,num_out,stride,scope=None):

        with tf.variable_scope(name_or_scope=scope):
            filter=tf.Variable(tf.random_uniform(shape=(fliter_size[0],fliter_size[1],num_in,num_out),dtype=tf.float32))

            return tf.nn.conv2d(inputs,filter=filter,strides=[1,stride,stride,1],padding='SAME')

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


    def resider(self,x, in_filter, out_filter, stride):
        '''
        默认前置激活
        :param x:
        :param in_filter:
        :param out_filter:
        :param stride:
        :return:
        '''
        with tf.variable_scope('shared'):
            x=self.batch_norm('init',x)
            x=tf.nn.relu(x)
        origin_x=x

        # 第1子层
        with tf.variable_scope('sub1'):
            # 3x3卷积，使用输入步长，通道数(in_filter -> out_filter)
            x=self.conv2d_same(inputs=x,fliter_size=[3,3],num_in=in_filter,num_out=out_filter,stride=stride,scope='conv1')
        # 第2子层
        with tf.variable_scope('sub2'):
            # BN和ReLU激活
            x = self.batch_norm('bn2', x)
            x = tf.nn.relu(x)
            # 3x3卷积，步长为1，通道数不变(out_filter)
            x=self.conv2d_same(inputs=x,fliter_size=[3,3],num_in=out_filter,num_out=out_filter,stride=stride,scope='conv2')

        # 合并残差层
        with tf.variable_scope('sub_add'):
            # 当通道数有变化时
            if in_filter != out_filter:
                # 均值池化，无补零
                origin_x = self.subsample(inputs=x,factor=1,ksizes=[1,1],scope='sub_add')
                # 通道补零(第4维前后对称补零)
                # origin_x = tf.pad(origin_x,
                #                 [[0, 0],
                #                  [0, 0],
                #                  [0, 0],
                #                  [(out_filter - in_filter) // 2, (out_filter - in_filter) // 2]
                #                  ])
            # 合并残差
            print('x',x)
            print('origin_x',origin_x)
            x += origin_x

        tf.logging.debug('image after unit %s', x.get_shape())
        return x


    def __build_model__(self):

        with tf.device('/device:GPU:%s' % gpu_id):
            self.inputs=tf.placeholder(shape=(None,FLAGS.mask_max_len),dtype=tf.int32)
            self.intent_y=tf.placeholder(shape=(None,self.intent_num),dtype=tf.int32)

            embed=tf.Variable(tf.random_normal(shape=(self.word_num,FLAGS.mask_embedding_dim),dtype=tf.float32))
            self.input_emb=tf.nn.embedding_lookup(embed,self.inputs)

            inputs=tf.expand_dims(self.input_emb,-1)

            # 第一次卷积
            input_conv_1=self.conv2d_same(inputs=inputs,fliter_size=[3,3],num_in=1,num_out=16,stride=1,scope='init_conv')

            print(input_conv_1)
            res_func = self.resider
            # 通道数量
            filters = [16, 16, 32, 64]

            # 第一组

            with tf.variable_scope('unit_1_0'):
                x=res_func(x=input_conv_1,in_filter=16,out_filter=32,stride=1)

            for i in range(4):
                with tf.variable_scope('unit_1_%s'%(i+1)):
                    x=res_func(x=x,in_filter=32,out_filter=32,stride=1)

            x=self.batch_norm('ll',x)

            #池化

            x_mean=tf.reduce_max(x,1)

            x_out=tf.reshape(x_mean,[-1,3200])

            x_out=tf.layers.dense(x_out,self.intent_num)

            self.soft_logit=tf.nn.softmax(x_out,1)
            self.loss=tf.losses.softmax_cross_entropy(self.intent_y,x_out)

            self.opt=tf.train.AdagradOptimizer(0.1).minimize(self.loss)

    def _decay(self):
        costs = []
        # 遍历所有可训练变量
        for var in tf.trainable_variables():
            # 只计算标有“DW”的变量
            costs.append(tf.nn.l2_loss(var))
        # 加和，并乘以衰减因子
        return tf.multiply(0.0001, tf.add_n(costs))

    def __train__(self):

        config = tf.ConfigProto(allow_soft_placement=True)

        _logger.info("load data")
        _logger.info('entity_words:%s'%(self.en.entity_dict.keys()))
        with tf.Session(config=config) as sess:
            num_batch = self.dd.num_batch
            init_train_acc = 0.0
            init_dev_acc = 0.0
            init_train_loss=9999.99
            init_dev_loss=9999.99
            saver = tf.train.Saver()
            if os.path.exists('%s.meta' % FLAGS.mask_model_dir):
                saver.restore(sess, FLAGS.mask_model_dir)
            else:
                sess.run(tf.global_variables_initializer())
            dev_sent, dev_slot, dev_intent, dev_rel_len,_ = self.dd.get_dev()
            train_sent, train_slot, train_intent, train_rel_len,_ = self.dd.get_train()
            for i in range(FLAGS.mask_epoch):
                for _ in range(num_batch):
                    sent, slot, intent_label, rel_len, cur_len = self.dd.next_batch()
                    batch_sent_mask = get_sent_mask(sent, self.entity_id)

                    soft_logit_, loss_, _ = sess.run([self.soft_logit, self.loss, self.opt],
                                                     feed_dict={self.inputs: sent,
                                                                self.intent_y: intent_label,
                                                                })



                dev_soft_logit_, dev_loss_ = sess.run([self.soft_logit, self.loss],
                                                      feed_dict={self.inputs: dev_sent,
                                                                 self.intent_y: dev_intent,
                                                                 })
                dev_acc = intent_acc(dev_soft_logit_, dev_intent, self.id2intent)

                # if train_acc > init_train_acc and dev_acc > (init_dev_acc-0.05):
                if  dev_acc>init_dev_acc:
                    init_dev_loss=dev_loss_
                    init_dev_acc=dev_acc
                    # init_train_acc = train_acc
                    # init_dev_acc = dev_acc
                    _logger.info('save')
                    saver.save(sess, FLAGS.mask_model_dir)

                _logger.info('第 %s 次迭代  dev_loss:%s dev_acc:%s' % (
                i, dev_loss_, dev_acc))

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
            if os.path.exists('%s.meta' % FLAGS.mask_model_dir):
                saver.restore(sess, '%s' % FLAGS.mask_model_dir)
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

    lm=ResNet()
    lm.__build_model__()
    if FLAGS.mask_mod=='train':
        lm.__train__()

    elif FLAGS.mask_mod=='infer_dev':
        lm.infer_dev()

    elif FLAGS.mask_mod=='server':
        lm.__server__()










if __name__ == '__main__':
    tf.app.run()