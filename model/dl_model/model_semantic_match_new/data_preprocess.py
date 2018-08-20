import numpy as np
import pickle
import os
PATH=os.path.split(os.path.split(os.path.split(os.path.split(os.path.realpath(__file__))[0])[0])[0])[0]
print(PATH)
import logging
import random
import re
from IntentConfig import Config
from entity_recognition.ner import EntityRecognition
from collections import defaultdict
import os
base_path = os.path.split(os.path.split(os.path.split(os.path.split(os.path.realpath(__file__))[0])[0])[0])[0]

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
        self.split_token='\t\t'
        self.train_path = train_path  # 训练文件路径
        self.dev_path = dev_path  # 验证文件路径
        self.test_path = test_path  # 测试文件路径
        self.batch_size = batch_size  # batch大小
        self.max_length = max_length
        self.use_auto_bucket=use_auto_bucket
        self.bucket_len=[0,5,15,30]
        self.index=0
        if flag == "train_new":
            self.vocab= self.get_vocab()
            pickle.dump(self.vocab, open("vocab.p", 'wb'))  # 词典

        elif flag == "test" or flag == "train":
            self.vocab = pickle.load(open("vocab.p", 'rb'))  # 词典

        else:
            pass
        self.vocab_num=len(self.vocab)

        self.train_data= self.get_train_data()
        self.dev_data= self.get_dev_data()

        self.id2sent={}
        for k,v in self.vocab.items():
            self.id2sent[v]=k



    def get_vocab(self):
        '''
        构造字典 dict{NONE:0,word1:1,word2:2...wordn:n} NONE为未登录词
        :return:
        '''
        train_file = open(self.train_path, 'r')
        test_file = open(self.dev_path, 'r')
        dev_file = open(self.test_path, 'r')
        vocab = {"NONE": 0,'bos':1,'eos':2}
        intent_vocab = {}
        slot_vocab={}
        self.index = len(vocab)
        self.intent_label_index = len(intent_vocab)+1
        self.slot_label_index=len(slot_vocab)
        intent_num_vocab={}
        def _vocab(file):
            for ele in file:
                ele = ele.replace("\n", "")
                eles=ele.split(self.split_token)
                sent1=eles[0].split(' ')
                sent2=eles[1].split(' ')
                label=ele[2]

                for word in sent1:
                    word=word.lower()
                    if word not in vocab and word!='':
                        vocab[word]=self.index
                        self.index+=1

                for word in sent2:
                    word=word.lower()
                    if word not in vocab and word!='':
                        vocab[word]=self.index
                        self.index+=1

        _vocab(train_file)
        _vocab(dev_file)
        _vocab(test_file)

        return vocab


    def padd_sentences_no_buckets(self, sent_list):
        '''
        find the max length from sent_list , and standardation
        :param sent_list:
        :return:
        '''
        # slot_labels = [' '.join(str(sent).replace('\n', '').split('\t')[1].split(' ')[:-1]) for sent in sent_list]
        # intent_labels=[str(sent).replace('\n', '').split('\t')[1].split(' ')[-1] for sent in sent_list]

        words,slot_labels,intent_labels,loss_weights=[],[],[],[]
        sent1,sent2,label,sent1_len,sent2_len=[],[],[],[],[]
        for sent in sent_list:
            sent=sent.replace('\n','')
            sents=sent.split(self.split_token)

            s1,s2=[],[]
            for e in sents[0].split(' '):
                if e not in self.vocab:
                    s1.append(0)
                else:
                    s1.append(self.vocab[e])

            for e in sents[1].split(' '):
                if e not in self.vocab:
                    s2.append(0)
                else:
                    s2.append(self.vocab[e])
            sent1_len.append(len(s1))
            sent2_len.append(len(s2))
            s1.extend([0]*self.max_length)
            s2.extend([0]*self.max_length)

            sent1.append(s1[:self.max_length])
            sent2.append(s2[:self.max_length])
            label.append(sents[2])

        sent1,sent2,sent1_len,sent2_len,label=np.array(sent1),np.array(sent2),np.array(sent1_len),np.array(sent2_len),np.array(label)
        return sent1,sent2,sent1_len,sent2_len,label



    def get_train_data(self):
        '''
        对训练样本按照长度进行排序 分箱
        :return:
        '''
        train_flie = open(self.train_path, 'r')
        data_list = [line for line in train_flie.readlines()]
        data_list.sort(key=lambda x: len(x))  # sort not shuffle
        random.shuffle(data_list)
        sent1, sent2, sent1_len, sent2_len, label = self.padd_sentences_no_buckets(data_list)

        return [sent1,sent2,sent1_len,sent2_len,label]




    def get_dev_data(self):

        train_flie = open(self.dev_path, 'r')
        data_list = [line for line in train_flie.readlines()]
        data_list.sort(key=lambda x: len(x))  # sort not shuffle
        random.shuffle(data_list)
        sent1, sent2, sent1_len, sent2_len, label = self.padd_sentences_no_buckets(data_list)
        return [sent1, sent2, sent1_len, sent2_len, label]



    def get_sent_char(self,sent_list):

        pattern = '\d{1,3}(\\.|，|、|？)|《|》|？|。| '


        res=[]
        res_vec=[]
        sent_list = [re.subn(pattern, '', sent)[0] for sent in sent_list]
        sent_list=er.get_entity(sent_list)
        for sent in sent_list:
            ss=[]
            ss_new=['eos']
            ss_new.extend(sent)
            ss_new.append('bos')
            ss_=ss_new
            for word in ss_:
                word=word.lower()
                if word in self.vocab:
                    ss.append(self.vocab[word])
                else:
                    ss.append(self.vocab['NONE'])

            if len(ss)>=self.max_length:
                res_vec.append(self.max_length)
                ss=ss[:self.max_length]
            else:
                res_vec.append(len(ss))
                padd=[0]*(self.max_length-len(ss))
                ss.extend(padd)
            res.append(ss)
        sent_arr=np.array(res)
        sent_vec=np.array(res_vec)
        return sent_arr,sent_vec


    def get_origin_train_data(self):
        '''

        :return:
        '''
        sents,sents_len,intents=[],[],[]

        for line in open(base_path+'/corpus_data/train_out_char.txt','r').readlines():
            lines=line.replace('\n','').split('\t')

            sent=lines[1]
            s=[]
            for ele in sent.split(' '):
                if ele in self.vocab:
                    s.append(self.vocab[ele])
                else:
                    s.append(0)
            sents_len.append(len(s))
            s.extend([0]*self.max_length)
            sents.append(s[:self.max_length])
            intent=lines[3]
            intents.append(intent)
        sents=np.array(sents)
        return sents,sents_len,intents




if __name__ == '__main__':

    dd = Intent_Slot_Data(train_path="train_en.txt",
                              test_path="dev_en.txt",
                              dev_path="dev_en.txt", batch_size=20 ,max_length=30,
                          flag="train_new",use_auto_bucket=False)

    sent1, sent2, sent1_len, sent2_len, label=dd.dev_data

    # print(sent1.shape,sent2.shape,sent1_len.shape,sent2_len.shape)

    print(sent1[0])
    print(sent2[0])
    print(sent1_len[0])
    print(sent2_len[0])

    # for ele in dev_data:
    #     pos_0=''.join( id2sent[e] for e in ele['pos_0']['word_arr'] if e !=0)
    #     pos_1=''.join( id2sent[e] for e in ele['pos_1']['word_arr'] if e !=0)
    #     neg_0=''.join( id2sent[e] for e in ele['neg_0']['word_arr'] if e !=0)
    #
    #     print(pos_0,'\t\t',pos_1,'\t\t',neg_0,'\n')

    # dd.next_batch()
    # for k,v in dd.vocab.items():
    #     print(k,v)
    # ss=dd.get_sent_char(['康爱保保什么'])[0]
    # print(ss)
    # print(sent)
    # print(slot)
    # print(intent)
    # print(real_len)

    #
    #
    #
    #
    # print(sent)

    # print(loss_weight)
    # print(sent[0])
    # print(''.join([dd.id2sent[e] for e in sent[0]]))
    # #     print()
    # #     print(sent.shape,slot.shape,intent.shape,real_len.shape,cur_len.shape)
    #     # print(dd.slot_vocab)
    #     # print(cur_len)
    # sent='专业导医陪诊是什么服务'
    # sent_arr,sent_vec=dd.get_sent_char([sent])
    # # print(sent_arr)
    # print(sent_arr,sent_vec)

