'''
配置类
'''

from collections import OrderedDict

class Config(object):

    def __init__(self):
        self.corpus_name='意图识别数据_all.txt'
        self.train_name='train_out_char.txt'
        self.dev_name='dev_out_char.txt'
        self.ml_classifiers=['SVM','LR']#可选的机器学习模型NB','SVM','KNN','LR','RF','DT'
        self.dl_classifiers='DLB'
        self.host='192.168.3.132'
        self.port=8082
        self.entity_level='level_1'
        self.root_intent=OrderedDict({0:["1宏观预测", "2产品诊断", "3知识方法", "4理财规划", "5时事分析"]})
        self.save_model_name='intent_all'
        self.stop_word_path='./corpus_data/stop_word.txt'
        self.classifier_dict={'BLSTM':1.0,'SVM':0.5,'LR':0.5,'RF':0.5}


        #{'intent_1':{}}
        # self.root_intent_list=['1宏观预测','2产品诊断','3知识方法','4理财规划','5时事分析']
        # if self.root_intent not in self.root_intent_list:
        #     raise ValueError('错误的 root_intent')

    _instance=None
    def __new__(cls, *args, **kwargs):

        if not cls._instance:
            cls._instance=super(Config,cls).__new__(cls,*args,**kwargs)
        return cls._instance

if __name__ == '__main__':
    config=Config()

    print(config.root_intent)