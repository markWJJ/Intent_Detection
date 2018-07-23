'''
多层pipeline
可对多层意图进行infer或则
'''


from IntentConfig import Config
from pipeline_tool import PipelineTool
from collections import defaultdict
import json
#第一层意图infer
import time
import os
import logging
if not os.path.exists('./Log/'):
    os.mkdir('./Log')
import time
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    filename='./Log/intent.log',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filemode='w')
_logger=logging.getLogger("pipeline_tool")

class pipeline(object):


    def __init__(self,config_train_path,config_infer_path):

        self.config_train_json=json.load(open(config_train_path,'r'))
        self.config=Config()

        self.config_infer_json=json.load(open(config_infer_path,'r'))


    def train_multi(self):
        '''
        多层 意图 训练
        :return:
        '''

        for k,v in self.config_train_json.items():
            _logger.info('当前训练的意图层级为:%s  具体配置:%s'%(k,v))
            self.config.save_model_name=self.config_train_json[k]['save_model_name']
            self.config.root_intent={int(k):v for k,v in self.config_train_json[k]['root_intent'].items()}

            pipeline_tool=PipelineTool(self.config)
            pipeline_tool.train()


    def __infer__(self,sents,config):
        '''
        infer_multi方法类
        :param sents:
        :param config:
        :return:
        '''
        pipeline_tool_level1 = PipelineTool(config)
        res_dict = pipeline_tool_level1.infer(sents,config)
        sentlist_dict = defaultdict(list)
        for sent, intent in res_dict.items():
            sentlist_dict[intent].append(sent)
        return sentlist_dict

    def __change_dict__(self,list_dict):
        '''
        字典转换 'label':[sent1,sent2,...,]->'sent1':label,'sent2':label
        :param list_dict:
        :return:
        '''
        new_dict={}
        for k ,v in list_dict.items():
            for e in v:
                new_dict[e]=k
        return new_dict

    def infer_multi(self,sents):
        '''
        多层 意图 infer
        :return:
        '''
        fin_dict={}
        # 第一层意图识别
        index="0"
        save_model_name=self.config_train_json[self.config_infer_json[index]]['save_model_name']
        self.config.save_model_name = save_model_name
        _logger.info('第一层意图识别模型 :%s' %self.config.save_model_name)

        list_dict = self.__infer__(sents, self.config)

        if "1" in self.config_infer_json.keys():
            for k, v in list_dict.items():
                if k in list(self.config_infer_json['1']):
                    self.config.save_model_name = self.config_train_json[k]['save_model_name']
                    _logger.info('第二层意图识别  %s 的模型 :%s' % (k, self.config.save_model_name))
                    list_dict_1=self.__infer__(v,self.config)
                    if "2" in self.config_infer_json.keys():
                        for k_3, v_3 in list_dict_1.items():
                            if k_3 in list(self.config_infer_json['2']):
                                _logger.info('%s 的模型 :%s' % (k_3, self.config.save_model_name))
                                self.config.save_model_name = self.config_train_json[k_3]['save_model_name']
                                list_dict_2 = self.__infer__(v_3, self.config)
                                fin_dict.update(list_dict_2)
                            else:
                                fin_dict.update({k_3:v_3})
                    else:
                        fin_dict.update(list_dict_1)


                else:
                    fin_dict.update({k:v})
        else:
            fin_dict.update(list_dict)

        fin_dict=self.__change_dict__(fin_dict)
        return fin_dict
        # # 第一层意图识别
        # level_1_save_model=self.config_train_json[self.config_infer_json['0']]['save_model_name']
        # self.config.save_model_name=level_1_save_model
        # res_dict=self.__infer__(sents,self.config)
        #
        # sentlist_dict=defaultdict(list)
        # for sent,intent in res_dict.items():
        #     sentlist_dict[intent].append(sent)
        #
        #
        # for k,v in sentlist_dict.items():
        #     if k in list(self.config_infer_json['1']):
        #         self.config.save_model_name=self.config_train_json[k]['save_model_name']
        #         _logger.info('%s 的模型 :%s'%(k,self.config.save_model_name))
        #         model=PipelineTool(self.config)
        #         rr=model.infer(v)
        #         rr_dict = defaultdict(list)
        #         fin_dict.update(rr)
        #         for sent, intent in rr.items():
        #             rr_dict[intent].append(sent)
        #         for k_3,v_3 in rr_dict.items():
        #             if k_3 in list(self.config_infer_json['2']):
        #                 _logger.info('%s 的模型 :%s' % (k, self.config.save_model_name))
        #                 self.config.save_model_name = self.config_train_json[k_3]['save_model_name']
        #                 model = PipelineTool(self.config)
        #                 rr = model.infer(v)
        #                 fin_dict.update(rr)
        #
        #
        # print(fin_dict)


if __name__ == '__main__':

    pipeline=pipeline('./train_config.json','./infer_config.json')
    pipeline.train_multi()
    #
    # # datas=['2018年A股是牛市吗？','能不能分析一下原油后续的走势？','微信刚出的那款保险怎么样','如何选择P2P平台？']
    #

    datas=[]
    labels=[]

    origin_dict={}
    for ele in open('./corpus_data/意图识别数据_all.txt','r').readlines():
    #
        index=ele.replace('\n','').split('\t\t')[0]
        ele1=ele.replace('\n','').split('\t\t')[1]
        origin_dict[str(index)]=ele1

        # datas.append(ele1)
    #     label=ele.replace('\n','').split('\t\t')[2]
    #     label=label.replace('##None##None','').replace('##None','').replace('##','_')
    #     labels.append(label)

    for ele in open('./corpus_data/dev_out_char.txt','r').readlines():
        index=ele.replace('\n','').split('\t')[0]
        ele1=origin_dict[str(index)]
        # ele1=ele.replace('\n','').split('\t')[1].strip()
        datas.append(ele1)
        label=ele.replace('\n','').split('\t')[3]
        label=label.replace('##None##None','').replace('##None','').replace('##','_')
        labels.append(label)

    true_dict={}

    for e,label in zip(datas,labels):
        true_dict[e]=label
    # #
    # start_time=time.time()
    # print('开始预测',start_time)
    #
    import pickle
    pre_dict=pipeline.infer_multi(datas)
    pickle.dump(pre_dict,open('./pre_dict.p','wb'))


    end_time=time.time()

    # def com():
    #     avg=0.0
    #     total=0.0
    #     count=0.0
    #     while True:
    #         num=yield avg
    #         total+=num
    #         count+=1
    #         avg=float(total)/float(count)
    #
    # com_avg=com()
    # next(com_avg)
    # for k,v in pre_dict.items():
    #     if k in true_dict and v!=true_dict[k]:
    #         print(com_avg.send(0))
    #     elif k in true_dict and v == true_dict[k]:
    #         print(com_avg.send(1))


    import pickle

    # true_dict=pickle.load(open('./true_dict.p','rb'))
    pre_dict=pickle.load(open('./pre_dict.p','rb'))

    def com():
        avg=0.0
        total=0.0
        count=0.0
        while True:
            num=yield avg
            total+=num
            count+=1
            avg=float(total)/float(count)

    com_avg=com()
    next(com_avg)
    for k,v in pre_dict.items():


        if k in true_dict and v!=true_dict[k]:
            if true_dict[k].__contains__('5时事分析') and pre_dict[k].__contains__('5时事分析'):
                print(com_avg.send(1))
            elif true_dict[k].__contains__('4理财规划') and pre_dict[k].__contains__('4理财规划'):
                print(com_avg.send(1))
            elif true_dict[k].__contains__('2产品诊断') and pre_dict[k].__contains__('2产品诊断'):
                print(com_avg.send(1))
            elif true_dict[k].__contains__('1宏观预测_基金') and pre_dict[k].__contains__('1宏观预测_基金'):
                print(com_avg.send(1))
            elif true_dict[k].__contains__('1宏观预测_贵金属') and pre_dict[k].__contains__('1宏观预测_贵金属'):
                print(com_avg.send(1))
            else:
                print(com_avg.send(0))
                print(k, '\t\t', true_dict[k], '\t\t', pre_dict[k])
        elif k in true_dict and v == true_dict[k]:
            print(com_avg.send(1))


    pickle.dump(true_dict,open('./true_dict.p','wb'))
    pickle.dump(pre_dict,open('./pre_dict.p','wb'))

    # print(pre_dict)



    # print(end_time-start_time)


# config.save_model_name='intent_all'
# config.entity_level='level_1'
#
# pipeline_tool=PipelineTool(config=config)
#
# sents=['2018年A股是牛市吗？','能不能分析一下原油后续的走势？']
#
# res_level1=pipeline_tool.infer(sents)
#
# print(res_level1)


