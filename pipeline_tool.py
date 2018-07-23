'''
单个模型的pipeline
比如多层意图 只能对某一层意图进行infer
'''
from pre_data_deal import Pre_Data_Deal
from NameEntityRec import NameEntityRec
import random
from IntentConfig import Config
from collections import OrderedDict
from model.dl_model.model_lstm_mask.lstm_mask import LstmMask
from model.dl_model.model_lstm_mask.pipeline_lstm_mask import IntentDLB
from model.ml_model.intent_ml import IntentMl
from entity_recognition.ner import EntityRecognition
from collections import defaultdict
import math
import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filemode='w')
_logger=logging.getLogger("pipeline_tool")


class PipelineTool(object):

    def __init__(self,config):
        self.config=config
        self.corpus_path='./corpus_data/%s'%self.config.corpus_name #数据集路径
        self.pdd=Pre_Data_Deal() #数据预处理
        if config.save_model_name != 'intent_all':
            config.entity_level='level_2'
        self.ner=NameEntityRec(config) #
        self.origin_ner=EntityRecognition(config_=config)

    def pre_data_deal(self,sent):
        '''
        数据预处理 去停用词
        :return:
        '''
        return self.pdd.main(sent=sent)

    def entity_rec(self,sent):
        '''
        命名实体识别
        :param sent:
        :return:
        '''
        return self.ner.main(sent=sent)[1]

    def train_dev_split(self,datas,split_rate=0.2):
        '''
        构建 训练集/测试集
        :param datas:
        :return:
        '''
        self.intent_indice = int(len(self.config.root_intent.keys())-1)
        intent_dict_ = self.config.root_intent
        self.intent_dict = OrderedDict()
        for k, v in intent_dict_.items():
            v_ = [e.lower() for e in v]
            self.intent_dict[k] = v_

        fw_train = open('./corpus_data/%s'%self.config.train_name, 'w')
        fw_dev = open('./corpus_data/%s'%self.config.dev_name, 'w')
        num_dict = {}

        random.shuffle(datas)
        split_num = int(len(datas) * split_rate)

        # dev write
        for ele in datas[:split_num]:
            eles = ele.split('\t')
            left = eles[:-1]
            labels = eles[-1].lower().split('##')

            true_label = []
            if self.intent_indice == 0:
                if labels[self.intent_indice] in self.intent_dict[0] and labels[self.intent_indice] != 'none':
                    left.append(labels[0])
                    true_label.append(labels[0])
                    sent = '\t'.join(left)
                    fw_dev.write(sent)
                    fw_dev.write('\n')
            else:
                for i in range(self.intent_indice + 1):
                    if labels[i] == 'none':
                        true_label = []
                        break
                    elif labels[i] in self.intent_dict[i]:
                        true_label.append(labels[i])
                    elif labels[i] not in self.intent_dict[i]:
                        true_label = []
                        break
                    else:
                        true_label = []
                # print(true_label,'\t\t',ele)
                if true_label != []:
                    left.append('_'.join(true_label))
                    sent = '\t'.join(left)
                    fw_dev.write(sent)
                    fw_dev.write('\n')

            if true_label != []:
                true_label = '_'.join(true_label)
                if true_label not in num_dict:
                    num_dict[true_label] = 1
                else:
                    s = num_dict[true_label]
                    s += 1
                    num_dict[true_label] = s

        '''train write'''

        for ele in datas[split_num::]:
            eles = ele.split('\t')
            left = eles[:-1]
            labels = eles[-1].split('##')

            true_label = []
            if self.intent_indice == 0:
                if labels[self.intent_indice] in self.intent_dict[0] and labels[self.intent_indice] != 'none':
                    left.append(labels[0])
                    true_label.append(labels[0])
                    sent = '\t'.join(left)
                    fw_train.write(sent)
                    fw_train.write('\n')
            else:
                for i in range(self.intent_indice + 1):
                    if labels[i] == 'none':
                        true_label = []
                        break
                    elif labels[i] in self.intent_dict[i]:
                        true_label.append(labels[i])
                    elif labels[i] not in self.intent_dict[i]:
                        true_label = []
                        break
                    else:
                        true_label = []

                if true_label != []:
                    left.append('_'.join(true_label))
                    sent = '\t'.join(left)
                    fw_train.write(sent)
                    fw_train.write('\n')

            if true_label != []:
                true_label = '_'.join(true_label)
                if true_label not in num_dict:
                    num_dict[true_label] = 1
                else:
                    s = num_dict[true_label]
                    s += 1
                    num_dict[true_label] = s
        _logger.info(num_dict)



    def train(self):
        '''

        :return:
        '''
        datas=[]
        with open(self.corpus_path,'r') as fr:
            for line in fr.readlines():
                pre_line=self.pre_data_deal(sent=line)
                entity_sent=self.entity_rec(sent=pre_line)
                datas.append(entity_sent)
        fr.close()
        self.train_dev_split(datas,0.2)
        #
        if self.config.ml_classifiers:
            intent_ml=IntentMl(self.config.ml_classifiers)
            intent_ml.build_model()

            outs=intent_ml.train(self.config.ml_classifiers)
            for ele in outs:
                for e in ele:
                    _logger.info('%s'%e)

        if self.config.dl_classifiers=='DLB':
            lstm=LstmMask(scope=self.config.save_model_name)
            _logger.info('构建模型')
            lstm.__build_model__()
            _logger.info('LSTM mask is train')
            lstm.__train__()
            index=0
            # for ele in outs:
            #     _logger.info('第%s次迭代: 训练准确率:%s 测试准确率:%s'%(index,round(ele[0],2),round(ele[1],2)))
            #     index+=1
            _logger.info('模型存储在%s'%'./save_model/model_lstm_mask/%s'%self.config.save_model_name)



    '''#############################infer#########################'''


    def _get_FAQ(self):
        '''
        获取人工标注的FAQ意图
        :return:
        '''

        FAQ_dict = {}
        with open('./faq_data/FAQ.txt', 'r') as fr:
            fr = (line for line in fr.readlines())
            for line in fr:
                line = line.replace('\n', '').replace('\t\t', '\t')
                try:
                    sent = line.split('\t')[0]
                    label = line.split('\t')[1].strip().split(' ')[0]
                    if label not in ['', ' ']:
                        FAQ_dict[sent] = label
                except Exception as ex:
                    print(ex, [line])

        return FAQ_dict


    def get_ml_intent(self, sents: list) -> dict:
        '''
        从ml模型获取意图
        :param sents:
        :return:
        '''
        ml=IntentMl(class_name=self.config.ml_classifiers)
        datas=[' '.join(e) for e in self.origin_ner.get_entity(sents)]
        return ml.infer(datas, classifier_name=self.config.ml_classifiers)


    def get_dlA_intent(self, sents: list,config) -> list:
        '''
        从dl模型获取意图
        :param sents:
        :return:
        '''
        dlb=IntentDLB(config)
        if self.config.dl_classifiers == 'DLB':
            for ele in dlb.get_intent(sents):
                yield ele[0][0]


    def infer(self,sents,config):
        '''
        预测
        :return:
        '''
        print('infer save_model',self.config.save_model_name)
        res_dict=defaultdict()
        all_dict={}
        if self.config.dl_classifiers:
            dl_result = list(self.get_dlA_intent(sents,config))
            _logger.info('DL 意图识别完成 %s'%dl_result)
            all_dict['BLSTM'] = dl_result

        if self.config.ml_classifiers:
            ml_result = self.get_ml_intent(sents)
            _logger.info('ML 意图识别完成 %s'%ml_result)

            all_dict = ml_result

        for sent, ele in zip(sents, self.vote(all_dict)):
            res_dict[sent] = ele

        return res_dict


    def intent_hgyc_level2(self,sent):
        '''

        :param sent:
        :return:
        '''

    def vote(self, class_result):
        '''
        投票
        :param class_result:
        :return:
        '''

        ss = []
        for k, v in dict(class_result).items():
            ele = [(e, self.config.classifier_dict[k]) for e in v]
            ss.append(ele)
        num_ = len(ss[0])
        result = []
        for i in range(num_):
            ss_i_dict = {}
            for j in range(len(ss)):
                if isinstance(ss[j][i][0], str):
                    if ss[j][i][0].lower() not in ss_i_dict:
                        ss_i_dict[ss[j][i][0].lower()] = ss[j][i][1]
                    else:
                        num = ss_i_dict[ss[j][i][0].lower()]
                        num += ss[j][i][1]
                        ss_i_dict[ss[j][i][0].lower()] = num
                else:
                    for ele in ss[j][i][0]:
                        if ele.lower() not in ss_i_dict:
                            ss_i_dict[ele.lower()] = ss[j][i][1]
                        else:
                            num = ss_i_dict[ele.lower()]
                            num += ss[j][i][1]
                            ss_i_dict[ele.lower()] = num

            ss_sort = [[k, v] for k, v in ss_i_dict.items() if k not in ['', ' ']]
            ss_sort.sort(key=lambda x: x[1], reverse=True)
            fin_res = ss_sort[0][0]
            result.append(fin_res)
        return result



if __name__ == '__main__':

    config=Config()
    pipeline=PipelineTool(config)
    # pipeline.train()
    sent=['今天天气不词']
    res=pipeline.infer(sent,config)
    print(res)
    #
    # config.save_model_name='intent_1'
    #
    # res=pipeline.infer(sent)
    # print(res)

