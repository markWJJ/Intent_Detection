import os
from IntentConfig import Config
base_path=os.path.split(os.path.split(os.path.split(os.path.split(os.path.realpath(__file__))[0])[0])[0])[0]
intent_config=Config()


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
    model_dir = base_path+'/save_model/model_lstm_mask/%s.ckpt'%intent_config.save_model_name
    print('config_lstm',model_dir)
    if not os.path.exists(base_path+'/save_model/model_lstm_mask'):
        os.makedirs(base_path+'/save_model/model_lstm_mask')
    use_cpu_num = 16
    keep_dropout = 0.7
    summary_write_dir = "./tmp/r_net.log"
    epoch = 90
    use_auto_buckets=False
    lambda1 = 0.01
    step=3
    model_mode = 'bilstm_attention_crf'  # 模型选择：bilstm bilstm_crf bilstm_attention bilstm_attention_crf,cnn_crf
    random_initialize=True