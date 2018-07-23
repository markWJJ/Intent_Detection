'''

'''
import re
import random
import sys
sys.path.append('.')
from entity_recognition.ner import EntityRecognition
from IntentConfig import Config
er=EntityRecognition()
config=Config()
import logging
from jieba import analyse
import jieba
from collections import OrderedDict
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filemode='w')
_logger=logging.getLogger("Intent_Data_Deal")
jieba.load_userdict('./stop_word.txt')

tfidf=analyse.extract_tags
class Intent_Data_Deal(object):

    def __init__(self):

        self.stop_words = [ele.replace('\n', '') for ele in open('./stop_word.txt', 'r').readlines()]

        self.intent_indice=list(config.root_intent.keys())[-1]
        print(self.intent_indice)
        intent_dict_=config.root_intent
        self.intent_dict=OrderedDict()
        for k,v in intent_dict_.items():
            v_=[e.lower() for e in v]
            self.intent_dict[k]=v_

    def pre_data_deal(self,origin_sent):

        sub_pattern = '你好|.{0,4}老师|.{0,4}年|.{0,4}%|，|谢谢你|谢谢|。|！|？|～'

        replace_pattern = r'\d{6}'
        replace_pattern_0 = r'\d.'

        origin_sent=origin_sent.replace('\n','')
        try:
            index = origin_sent.split('\t\t')[0]
            sent = origin_sent.split('\t\t')[1]
            label = origin_sent.split('\t\t')[2]
        except Exception as ex:
            _logger.info('Error sent ',origin_sent)


        sent = re.subn(sub_pattern, '', sent)[0]
        sent = sent.split('?')[0].replace(' ', '')
        sent = re.subn(replace_pattern, 'stock', sent)[0]
        sent = re.subn(replace_pattern_0, '', sent)[0]

        res_sent='\t\t'.join([index,sent,label])
        return res_sent

    def deal_sent(self,line):
        '''
        对句子进行处理,实体识别,增加eos/bos和实体标签
        :param sent:
        :return:
        '''
        pattern = '\d{1,3}(\\.|，|、|？)|《|》|？|。'
        line = line.replace('\n', '').strip()
        line=self.pre_data_deal(line)
        sent=''
        ll=''
        index=''
        line=line.replace('\t\t','\t').replace('Other','other').lower()
        if '\t' in line:
            index=str(line).split('\t')[0].strip().replace(' ','')
            sent=str(line).split('\t')[1].strip().replace(' ','')
            ll=str(line).split('\t')[2].strip().replace(' ','')
        else:
            try:
                index = str(line).split(' ')[0].strip().replace(' ', '')
                sent=str(line).split(' ')[1].strip().replace(' ','')
                ll=str(line).split(' ')[2].strip().replace(' ','')
            except Exception as ex:
                _logger.info('%s:%s'%(ex,[line]))
        sent = re.subn(pattern, '', sent)[0]
        ss=er.get_entity([sent])[0] #实体识别功能
        # 结巴关键词提取实现
        # sent=''.join(ss)
        # keys=tfidf(sent)
        # ss=keys
        sent=' '.join(ss)
        label = ll
        sent = 'bos' + ' ' + sent + ' ' + 'eos'
        entity = ' '.join(['o'] * len(sent.split(' ')))
        res = str(index)+'\t'+sent + '\t' + entity + '\t' + label

        return sent,res

    def deal_file(self,input_file_name,train_file_name,dev_file_name,split_rate=0.2):
        '''
        将输入的标注数据 转换为带实体标签的char数据
        :param file_name:
        :return:
        '''
        fw_train=open(train_file_name,'w')
        fw_dev=open(dev_file_name,'w')
        num_dict={}
        data=set()
        with  open(input_file_name,'r') as fr:
            for ele in fr.readlines():
                e=self.deal_sent(ele)[1]
                data.add(e)

            fr.close()
        data=list(data)
        random.shuffle(data)
        split_num=int(len(data)*split_rate)



        # dev write
        for ele in data[:split_num]:
            eles=ele.split('\t')
            left=eles[:-1]
            labels=eles[-1].split('##')

            true_label=[]
            if self.intent_indice==0:
                if labels[self.intent_indice] in self.intent_dict[0] and labels[self.intent_indice]!='none':
                    left.append(labels[0])
                    true_label.append(labels[0])
                    sent='\t'.join(left)
                    fw_dev.write(sent)
                    fw_dev.write('\n')
            else:
                for i in range(self.intent_indice+1):
                    if labels[i] == 'none':
                        true_label=[]
                        break
                    elif labels[i] in self.intent_dict[i]:
                        true_label.append(labels[i])
                    else:
                        true_label=[]



                if true_label!=[]:
                    left.append('_'.join(true_label))
                    sent='\t'.join(left)
                    fw_dev.write(sent)
                    fw_dev.write('\n')


            if true_label != []:
                true_label='_'.join(true_label)
                if true_label not in num_dict:
                    num_dict[true_label]=1
                else:
                    s=num_dict[true_label]
                    s+=1
                    num_dict[true_label]=s

        '''train write'''

        for ele in data[split_num::]:
            eles=ele.split('\t')
            left=eles[:-1]
            labels=eles[-1].split('##')

            true_label=[]
            if self.intent_indice==0:
                if labels[self.intent_indice] in self.intent_dict[0] and labels[self.intent_indice]!='none':
                    left.append(labels[0])
                    true_label.append(labels[0])
                    sent='\t'.join(left)
                    fw_train.write(sent)
                    fw_train.write('\n')
            else:
                for i in range(self.intent_indice+1):
                    if labels[i] == 'none':
                        true_label=[]
                        break
                    elif labels[i] in self.intent_dict[i]:
                        true_label.append(labels[i])
                    else:
                        true_label=[]



                if true_label!=[]:
                    left.append('_'.join(true_label))
                    sent='\t'.join(left)
                    fw_train.write(sent)
                    fw_train.write('\n')


            if true_label != []:
                true_label='_'.join(true_label)
                if true_label not in num_dict:
                    num_dict[true_label]=1
                else:
                    s=num_dict[true_label]
                    s+=1
                    num_dict[true_label]=s

        print('label num dict')
        print(num_dict)



if __name__ == '__main__':


    idd=Intent_Data_Deal()
    # corpus=sys.argv[1]
    # train_out=sys.argv[2]
    # dev_out=sys.argv[3]
    # print(corpus,train_out,dev_out)
    idd.deal_file(config.corpus_name,config.train_name,config.dev_name)
