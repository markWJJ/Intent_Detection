
import jieba
import re
entity_types=[e.replace('\n','') for e in open('./entity_type_level_2.txt','r').readlines()]
for e in entity_types:
    jieba.load_userdict('./entity_data/level_2/%s.txt'%e)


# astock=[e.replace('\n','') for e in open('./entity_data/level_2/Astock.txt','r').readlines()] #A股
# dzsp=[e.replace('\n','') for e in open('./entity_data/level_2/Dazongshanpin.txt','r').readlines()]#大宗商品
# fdc=[e.replace('\n','') for e in open('./entity_data/level_2/Fandichan.txt','r').readlines()]#房地产
# gjs=[e.replace('\n','') for e in open('./entity_data/level_2/Fandichan.txt','r').readlines()] #贵金属
# hlwlc=[e.replace('\n','') for e in open('./entity_data/level_2/Hulianwanglicai.txt','r').readlines()]#互联网理财
# hb=[e.replace('\n','') for e in open('./entity_data/level_2/Huobi.txt','r').readlines()]#货币
# jj=[e.replace('\n','') for e in open('./entity_data/level_2/Jijin.txt','r').readlines()]#基金
#
#
#
# class hgyc(object):
#
#
#     def __init__(self):
#         self.astock="|".join(astock)
#         self.dzsp="|".join(dzsp)
#         self.fdc="|".join(fdc)
#         self.gjs='|'.join(gjs)
#         self.hlwlc='|'.join(hlwlc)
#         self.hb='|'.join(hb)
#         self.jj='|'.join(jj)
#
#     def infer(self,sent):
#
#         '''
#
#         :param sent:
#         :return:
#         '''
#
#         # 贵金属
#         pattern_0='(%s)'%self.gjs
#         if re.search(pattern_0,sent):
#             return '贵金属'
#
#         # 大宗商品
#         pattern_1='%s'%self.dzsp
#         if re.search(pattern_1,sent):
#             return '大宗商品'
#
#         #房产
#         pattern_2='(%s).*房|买房'%self.fdc
#         if re.search(pattern_2,sent):
#             return '房产'

        #

if __name__ == '__main__':

    s={
       'level_1':{'save_model_name':'intent_all',
       'root_intent':{0:['1宏观预测','2产品诊断','3知识方法','4理财规划','5时事分析']}},

        'hgyc_level_1':{'save_model_name':'hgyc_level_1',
                        'root_intent':{0:['1宏观预测'],1:['A股','大宗商品','房产','房地产','个股','贵金属','海外房产','宏观','互联网理财','黄金','基金','数字货币','行业']}}
       }

    import json

    json.dump(s,open('./config.json','w'),ensure_ascii=False)
