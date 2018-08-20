import numpy as np
import pickle
import os
PATH=os.path.split(os.path.split(os.path.split(os.path.split(os.path.realpath(__file__))[0])[0])[0])[0]
print(PATH)
import logging
import json
import random
import re
from IntentConfig import Config
from entity_recognition.ner import EntityRecognition
import jieba

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filemode='w')
_logger=logging.getLogger("data_preprocess")

config=Config()
er=EntityRecognition()

class Intent_Slot_Data(object):
    '''
    intent_slot 数据处理模块
    '''
    def __init__(self, train_path, dev_path, test_path, batch_size, flag,max_length,use_auto_bucket):

        #分隔符
        self.split_token='\t'
        self.train_path = train_path  # 训练文件路径
        self.dev_path = dev_path  # 验证文件路径
        self.test_path = test_path  # 测试文件路径
        self.batch_size = batch_size  # batch大小
        self.max_length = max_length
        self.use_auto_bucket=use_auto_bucket
        self.index=1
        self.word_vocab = {'none': 0}
        self.index = 0
        self.class_vocab = {}




        if flag == "train_new":
            self.datas_train = self.token_json(self.train_path)
            pickle.dump(self.datas_train, open('datas_train_json.p', 'wb'))
            pickle.dump(self.word_vocab, open(PATH+"/save_model/word_vocab.p", 'wb'))  # 词典
            pickle.dump(self.class_vocab, open( PATH+"/save_model/class_vocab.p", 'wb'))  # 词典
        elif flag == "test" or flag == "train":
            self.datas_train = pickle.load(open(PATH+"/save_model/datas_train_json.p", 'rb'))
            self.word_vocab = pickle.load(open(PATH+"/save_model/word_vocab.p", 'rb'))  # 词典
            self.class_vocab = pickle.load(open(PATH+"/save_model/class_vocab.p", 'rb'))  # 词典


    def get_vocab(self,args):
        '''

        :param args:
        :return:
        '''
        for ele in args:
            eles=list(jieba.cut(ele))
            eles_words=[]
            rel_len=len(eles)
            for e in eles:
                if e not in self.word_vocab:
                    self.word_vocab[e]=self.index
                    self.index+=1
                eles_words.append(self.word_vocab[e])
            eles_words.extend([0]*self.max_length)
            eles_words=eles_words[:self.max_length]
            eles_words.append(rel_len)
            yield eles_words

    def token_json(self,json_path):
        '''
        读取json文件并分词,
        :param json_path:
        :return:
        '''
        datas = json.load(open(json_path, 'r'))

        for k,v in datas.items():
            v_=list(self.get_vocab(v))
            if k not in self.class_vocab:
                self.class_vocab[k]=int(k)
            datas[k]=v_
            _logger.info('%s finish'%k)
        return datas


    def get_train_test(self,split=0.1):
        '''
        '''
        sent_data=[]
        label_data=[]
        for k,v in self.datas_train.items():
            for ele in v:
                sent_data.append([ele,k])

        length=len(sent_data)
        random.shuffle(sent_data)
        num=int(length*split)
        test_data=sent_data[:num]
        train_data=sent_data[num:]
        return train_data,test_data




def main():
    dd = Intent_Slot_Data(train_path="./../../../corpus_data/train.json",
                              test_path="./../../../corpus_data/dev.json",
                              dev_path="./../../../corpus_data/test.json", batch_size=20 ,max_length=20,
                          flag="train_new",use_auto_bucket=False)

    train,test=dd.get_train_test(0.1)

    print(test)

if __name__ == '__main__':
    main()



