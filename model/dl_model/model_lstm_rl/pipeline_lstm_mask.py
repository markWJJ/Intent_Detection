import logging
import sys,os
sys.path.append('./')
from model.dl_model.model_lstm_mask.data_preprocess import Intent_Slot_Data

from IntentConfig import Config
from model.dl_model.model_lstm_mask.lstm_mask import LstmMask,Config_lstm
import tensorflow as tf
config_lstm=Config_lstm()

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename='modellog.log',
                    filemode='w')
_logger = logging.getLogger("intent_bagging")




class IntentDLB(object):

    def __init__(self):

        self.nn_model = LstmMask()
        self.nn_model.__build_model__()
        config = tf.ConfigProto(device_count={"CPU": 8},  # limit to num_cpu_core CPU usage
                                inter_op_parallelism_threads=8,
                                intra_op_parallelism_threads=8,
                                log_device_placement=False,
                                allow_soft_placement=True,
                                )
        saver=tf.train.Saver()
        self.sess=tf.Session(config=config)
        saver.restore(self.sess,config_lstm.model_dir)



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