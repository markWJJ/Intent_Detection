#-*-
from model.ml_model.intent_ml import IntentMl



from model.dl_model.model_lstm_wjj.pipeline_lstm import IntentDLA
from model.dl_model.model_lstm_mask.pipeline_lstm_mask import IntentDLB
from entity_recognition.ner import EntityRecognition
from xmlrpc.client import  ServerProxy
from sklearn.metrics import classification_report,precision_recall_fscore_support
from corpus_data.data_deal import Intent_Data_Deal
import logging
from collections import defaultdict
from configparser import ConfigParser
import re
import gc
import pdb

from IntentConfig import Config
config=Config()


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filemode='w')
_logger=logging.getLogger("pipeline")

ML = IntentMl('Classifier')
if config.dl_classifiers=='DLA':
    DLA=IntentDLA()
elif config.dl_classifiers=='DLB':
    DLB=IntentDLB()
idd=Intent_Data_Deal()
# RE=IntentRe()
ER=EntityRecognition()

PATTERN='\\?|？|\\.|。'


# classifier_dict={'BLSTM':1.0}
classifier_dict={'BLSTM':1.0,'SVM':0.5,'LR':0.5,'RF':0.5}


class Pipeline(object):

    def __init__(self):
        self.faq_dict=self._get_FAQ()

    def _get_FAQ(self):
        '''
        获取人工标注的FAQ意图
        :return:
        '''


        FAQ_dict={}
        with open('./faq_data/FAQ.txt','r') as fr:
            fr=( line for line in fr.readlines())
            for line in fr:
                line=re.subn(PATTERN,'',line.replace('\n',''))[0].replace('\t\t','\t')
                try:
                    sent=line.split('\t')[0]
                    label=line.split('\t')[1].strip().split(' ')[0]
                    if label not in ['',' ']:
                        FAQ_dict[sent] = label
                except Exception as ex:
                    print(ex,[line])

        return FAQ_dict


    def get_ml_intent(self,sents:list)->dict:
        '''
        从ml模型获取意图
        :param sents:
        :return:
        '''

        ss=[' '.join(e) for e in ER.get_entity(sents)]
        _logger.info('data_deal_finish')
        return ML.infer(ss,classifier_name=config.ml_classifiers)


    def get_dlA_intent(self,sents:list)->list:
        '''
        从dl模型获取意图
        :param sents:
        :return:
        '''
        if config.dl_classifiers == 'DLA':
            for ele in DLA.get_intent(sents):
                yield ele[0][0]
        elif config.dl_classifiers == 'DLB':
            for ele in DLB.get_intent(sents):
                yield ele[0][0]



    def pipeline(self,sents:list)->dict:
        '''
        获取sents的意图
        :param sents:list
        :return:
        '''

        res_dict=defaultdict()

        dl_result=list(self.get_dlA_intent(sents))
        _logger.info('DL 意图识别完成')

        ml_result=self.get_ml_intent(sents)
        _logger.info('ML 意图识别完成')

        all_dict=ml_result
        all_dict['BLSTM']=dl_result
        print(all_dict.keys())
        for sent,ele in zip(sents,self.vote(all_dict)):
            res_dict[sent]=[[ele,1.0]]

        return res_dict


    def vote(self, class_result):
        '''
        投票
        :param class_result:
        :return:
        '''

        ss = []
        for k, v in dict(class_result).items():
            ele=[(e, classifier_dict[k]) for e in v]
            ss.append(ele)
        num_=len(ss[0])
        result=[]
        for i in range(num_):
            ss_i_dict={}
            for j in range(len(ss)):
                if isinstance(ss[j][i][0],str):
                    if ss[j][i][0].lower() not in ss_i_dict:
                        ss_i_dict[ss[j][i][0].lower()]=ss[j][i][1]
                    else:
                        num=ss_i_dict[ss[j][i][0].lower()]
                        num+=ss[j][i][1]
                        ss_i_dict[ss[j][i][0].lower()]=num
                else:
                    for ele in ss[j][i][0]:
                        if ele.lower() not in ss_i_dict:
                            ss_i_dict[ele.lower()]=ss[j][i][1]
                        else:
                            num=ss_i_dict[ele.lower()]
                            num+=ss[j][i][1]
                            ss_i_dict[ele.lower()]=num


            ss_sort=[[k,v] for k,v in ss_i_dict.items() if k not in ['',' ']]
            ss_sort.sort(key=lambda x:x[1],reverse=True)
            fin_res=ss_sort[0][0]
            result.append(fin_res)
        return result




if __name__ == '__main__':

    pipeline=Pipeline()
    # sents=['感冒是不是重大疾病','感冒保不保','合同生效后能退保吗','你们的总部在那个城市','保单查询','广东分公司的联系方式','你们公司在哪','患艾滋病是否赔偿','吸食毒品定义','北京有哪些营销服务部',
    #       '盐城营销服务部的联系方式','深圳营销服务部','代理人的姓名','深圳营销服务部电话号码是多少']
    # result = pipeline.pipeline(sents)

    data_dict={}
    for sent in open('./corpus_data/dn.txt','r').readlines():
        sent_eles=sent.replace('\n','').split('\t\t')
        data_dict[sent_eles[0].strip()]=[sent_eles[1],sent_eles[2]]
    sents=[]
    labels=[]
    label_dict={}
    for dev in open('./corpus_data/dev_out_char.txt','r').readlines():
        sent_eles=dev.replace('\n','').split('\t')
        id=sent_eles[0].strip()
        try:
            sents.append(data_dict[id][0])
            labels.append(data_dict[id][1])

            label_dict[data_dict[id][0]]=data_dict[id][1]
        except:
            print(id)
    s=pipeline.get_ml_intent(sents)
    result=pipeline.pipeline(sents)

    print(len(labels),len(result))
    right_sum=0
    all_sum=0
    for k,v in result.items():
        pred_label=v[0][0]
        true_label=label_dict[k]
        all_sum+=1
        if pred_label==true_label:
            right_sum+=1

    # for true_label,pred_label in zip(labels,result):
    #
    #     all_sum+=1
    #     if true_label==pred_label:
    #         right_sum+=1

    print(float(right_sum)/float(all_sum))


