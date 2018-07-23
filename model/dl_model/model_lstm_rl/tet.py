

from model.dl_model.model_lstm_mask.data_preprocess import Intent_Slot_Data
from model.dl_model.model_lstm_mask.lstm_mask import get_sent_mask
import tensorflow as tf
from entity_recognition.ner import entity_dict

import os
import numpy as np
base_path=os.path.split(os.path.split(os.path.split(os.path.split(os.path.realpath(__file__))[0])[0])[0])[0]

# saver=tf.train.import_meta_graph(base_path+'/save_model/model_lstm_mask/intent_lstm_dn.ckpt.meta')
# config = tf.ConfigProto(allow_soft_placement=True)

# with tf.Session(config=config) as sess:
#
#     saver.restore(sess,base_path+'/save_model/model_lstm_mask/intent_lstm_dn.ckpt')
#
#     graph = tf.get_default_graph()
#     for v in tf.all_variables():
#         print(v)
#     # sent_enc = graph.get_operation_by_name('sent_enc')
#     for e in graph.get_operations():
#         print(e)
#
#     sent_word=graph.get_tensor_by_name('sent_word:0')
#     sent_len=graph.get_tensor_by_name('sent_len:0')
#     sent_mask=graph.get_tensor_by_name('sent_mask:0')
#     intent_y=graph.get_tensor_by_name('intent_y:0')
#

    # sent_emb=graph.get_tensor_by_name('stack_sent_enc:0')
    # # sent_enc = graph.get_tensor_by_name('sent_enc:0')
    # dd = Intent_Slot_Data(train_path=base_path + "/corpus_data/train_out_char.txt",
    #                            test_path=base_path + "/corpus_data/dev_out_char.txt",
    #                            dev_path=base_path + "/corpus_data/dev_out_char.txt", batch_size=128,
    #                            max_length=30, flag="train",
    #                            use_auto_bucket=False)
    #
    # dev_sent, dev_slot, dev_intent, dev_rel_len = dd.get_dev()
    # word_vocab = dd.vocab
    # entity_id = []
    # for k, v in word_vocab.items():
    #     if k in entity_dict.keys():
    #         entity_id.append(v)
    # dev_sent_mask = get_sent_mask(dev_sent, entity_id)
    #
    # sent_enc1 = sess.run(sent_emb, feed_dict={sent_word: dev_sent,
    #                                         sent_len: dev_rel_len,
    #                                         sent_mask: dev_sent_mask,
    #                                         })
    #
    # print(sent_enc1[0][0])

#
class test_class(object):


    def __init__(self,pre_sess,intent_num,dd):

        # init=tf.constant(value=)
        self.pre_sess=pre_sess
        self.dd=dd
        graph = tf.get_default_graph()

        # sent_enc = graph.get_operation_by_name('sent_enc')

        self.sent_word = graph.get_tensor_by_name('sent_word:0')
        self.sent_len = graph.get_tensor_by_name('sent_len:0')
        self.sent_mask = graph.get_tensor_by_name('sent_mask:0')
        self.intent_y = graph.get_tensor_by_name('intent_y:0')
        self.sent_emb = graph.get_tensor_by_name('stack_sent_enc:0')

        sent_encs = graph.get_tensor_by_name('stack_sent_enc:0')

        self.soft_logit=graph.get_tensor_by_name('mask_soft_logit:0')

        # sent_enc=tf.unstack(sent_encs,30,1)[0]
        #
        # logit=tf.layers.dense(sent_enc,intent_num)
        # self.soft_logit=tf.nn.softmax(logit,1,name='sss')
        # self.loss=tf.losses.softmax_cross_entropy(self.intent_y,logit)
        #
        # self.opt=tf.train.AdamOptimizer(0.01,name='ss').minimize(self.loss)

    def intent_acc(self,pre, label, id2intent):
        '''
        获取intent准确率
        :param pre:
        :param label:
        :return:
        '''
        pre_ = np.argmax(pre, 1)

        label_ = np.argmax(label, 1)
        ss = [[int(k), v] for k, v in id2intent.items()]
        ss.sort(key=lambda x: x[0], reverse=False)
        s1 = [e[1] for e in ss]
        # print(classification_report(y_true=label_,y_pred=pre_,target_names=s1))
        all_sum = len(label_)
        num = sum([1 for e, e1 in zip(pre_, label_) if e == e1])

        return float(num) / float(all_sum)

    def train(self):
        config = tf.ConfigProto(allow_soft_placement=True)

        sess=tf.Session(config=config)
        # sess.run(tf.global_variables_initializer())
        # self.pre_sess.run(tf.global_variables_initializer())
        dev_sent, dev_slot, dev_intent, dev_rel_len =self.dd.get_dev()
        print(dev_sent[0])
        word_vocab = dd.vocab
        entity_id = []
        for k, v in word_vocab.items():
            if k in entity_dict.keys():
                entity_id.append(v)
        dev_sent_mask = get_sent_mask(dev_sent, entity_id)
        for _ in range(50):
            dev_soft_logit = sess.run([self.soft_logit], feed_dict={self.sent_word: dev_sent,
                                                    self.sent_len: dev_rel_len,
                                                    self.sent_mask: dev_sent_mask,
                                                    self.intent_y:dev_intent
                                                    })

            # print(loss)
            acc=self.intent_acc(dev_soft_logit[0],dev_intent,self.dd.id2sent)
            print(acc)




if __name__ == '__main__':
    dd = Intent_Slot_Data(train_path=base_path + "/corpus_data/train_out_char.txt",
                               test_path=base_path + "/corpus_data/dev_out_char.txt",
                               dev_path=base_path + "/corpus_data/dev_out_char.txt", batch_size=128,
                               max_length=30, flag="train_new",
                               use_auto_bucket=False)
    intent_num=len(dd.id2intent)

    saver = tf.train.import_meta_graph(base_path + '/save_model/model_lstm_mask/intent_lstm_dn.ckpt.meta')
    config = tf.ConfigProto(allow_soft_placement=True)

    pre_sess=tf.Session(config=config)
    saver.restore(pre_sess, base_path + '/save_model/model_lstm_mask/intent_lstm_dn.ckpt')

    tc=test_class(pre_sess,intent_num,dd)
    tc.train()


