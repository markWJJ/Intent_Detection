import logging
import sys,os
sys.path.append('./')
import pickle
from model.dl_model.model_lstm_mask.data_preprocess import Intent_Slot_Data

from IntentConfig import Config
from model.dl_model.model_lstm_mask.lstm_mask import LstmMask,Config_lstm
import tensorflow as tf
base_path=os.path.split(os.path.split(os.path.split(os.path.split(os.path.realpath(__file__))[0])[0])[0])[0]

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename='modellog.log',
                    filemode='w')
_logger = logging.getLogger("intent_bagging")




class IntentDLB(object):

    def __init__(self,config):
        intent_config = config

        vocab = pickle.load(open(base_path + "/save_model/vocab_%s.p" % intent_config.save_model_name, 'rb'))  # 词典
        slot_vocab = pickle.load(
            open(base_path + "/save_model/slot_vocab_%s.p" % intent_config.save_model_name, 'rb'))  # 词典
        intent_vocab = pickle.load(
            open(base_path + "/save_model/intent_vocab_%s.p" % intent_config.save_model_name, 'rb'))  # 词典

        id2intent={}

        for k, v in intent_vocab.items():
            id2intent[v] = k

        self.nn_model = LstmMask(scope=intent_config.save_model_name,mod='infer')
        self.nn_model.word_vocab=vocab
        self.nn_model.word_num=len(vocab)
        self.nn_model.id2intent=id2intent
        self.nn_model.intent_num=len(id2intent)

        self.nn_model.dd.vocab=vocab
        self.nn_model.dd.slot_vocab=slot_vocab
        self.nn_model.dd.intent_vocab=intent_vocab

        self.nn_model.__build_model__()
        config = tf.ConfigProto(device_count={"CPU": 8},  # limit to num_cpu_core CPU usage
                                inter_op_parallelism_threads=8,
                                intra_op_parallelism_threads=8,
                                log_device_placement=False,
                                allow_soft_placement=True,
                                )
        saver=tf.train.Saver()
        self.sess=tf.Session(config=config)
        saver.restore(self.sess,self.nn_model.save_model)



    def get_intent(self,sents):
        '''

        :param sents:
        :return:
        '''
        return self.nn_model.__infer__(sents,self.sess)


if __name__ == '__main__':

    id=IntentDLB()
    s=id.get_intent(['感冒保不保'])
    print(s)