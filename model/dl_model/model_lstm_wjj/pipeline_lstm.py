import logging
from model.dl_model.model_lstm_wjj.data_preprocess import Intent_Slot_Data
import sys,os
sys.path.append('../../')
from IntentConfig import Config
from model.dl_model.model_lstm_wjj.lstm import IntentLstm,Config_lstm
import tensorflow as tf
config_lstm=Config_lstm()

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename='modellog.log',
                    filemode='w')
_logger = logging.getLogger("intent_bagging")




class IntentDLA(object):

    def __init__(self):
        self.dd = Intent_Slot_Data(train_path=config_lstm.train_dir,
                              test_path=config_lstm.dev_dir,
                              dev_path=config_lstm.dev_dir, batch_size=config_lstm.batch_size,
                              max_length=config_lstm.max_len, flag="train",
                              use_auto_bucket=config_lstm.use_auto_buckets)

        self.nn_model = IntentLstm(slot_num_class=self.dd.slot_num, intent_num_class=self.dd.intent_num, vocab_num=self.dd.vocab_num)
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
        return self.nn_model.__infer__(self.dd,sents,self.sess)


if __name__ == '__main__':

    id=IntentDLA()
    s=id.get_intent(['今天天气'])
    print(s)