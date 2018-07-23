import re
import sys
sys.path.append('.')
from IntentConfig import Config
import logging
from jieba import analyse
import jieba
from collections import OrderedDict
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filemode='w')
_logger=logging.getLogger("Intent_Data_Deal")
config=Config()

jieba.load_userdict(config.stop_word_path)

tfidf=analyse.extract_tags
class Pre_Data_Deal(object):
    '''
    数据预处理
    包括:去停用词/
    '''

    def __init__(self):
        print(config.port)
        self.stop_words = [ele.replace('\n', '') for ele in open(config.stop_word_path, 'r').readlines()]
        self.intent_indice=list(config.root_intent.keys())[-1]
        intent_dict_=config.root_intent
        self.intent_dict=OrderedDict()
        for k,v in intent_dict_.items():
            v_=[e.lower() for e in v]
            self.intent_dict[k]=v_

    def pre_data_deal(self,origin_sent):

        sub_pattern = '你好|.{0,4}老师|.{0,4}年|.{0,4}%|，|谢谢你|谢谢|。|！|？|～'

        replace_pattern = r'\d{6}'
        replace_pattern_0 = r'\d.'
        pattern = '\d{1,3}(\\.|，|、|？)|《|》|？|。'

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
        sent = re.subn(pattern,'',sent)[0]

        res_sent='\t\t'.join([index,sent,label])
        return res_sent

    def main(self,sent):
        return self.pre_data_deal(sent)


