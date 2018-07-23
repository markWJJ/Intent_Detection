'''
实体识别功能实现
1.实体识别并替换为代号
2.对非实体词进行字或则词切分
'''
import jieba
import os
import logging
from IntentConfig import Config


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filemode='w')
_logger=logging.getLogger("data_preprocess")

PATH_=os.path.split(os.path.realpath(__file__))[0]
PATH=os.path.split(PATH_)[0]




class EntityRecognition(object):

    def __init__(self,type='t',config_=None):
        self.type=type #type='w' 字切分 type='t' 词切分
        if config_:
            config=config_
        else:
            config = Config()

        type_list = [e.replace('\n', '') for e in open(PATH + '/entity_type_%s.txt' % config.entity_level).readlines()
                     if e]
        self.entity_dict = {}
        for ele in type_list:
            jieba.load_userdict(PATH + '/entity_data/%s/%s.txt' % (config.entity_level, ele))
            ele_list = [e.replace('\n', '') for e in
                        open(PATH + '/entity_data/%s/%s.txt' % (config.entity_level, ele), 'r')]
            self.entity_dict[ele] = ele_list

    def get_entity(self,sent_list):
        '''
        输入 sent list 识别每个sent的实体并用相应的代号进行替代
        :param sents:
        :return:
        '''
        ss=[]
        for sent in sent_list:

            ss_ele=[]
            if 'Shiyi' in self.entity_dict and sent in self.entity_dict['Shiyi']:
                ss_ele.append('Shiyi')
            else:
                sents=[e for e in jieba.cut(sent) if e]
                for e in sents :
                    entity_e=[k for k,v in self.entity_dict.items() if e in v and k not in ['Shiyi']]
                    if entity_e:
                        ss_ele.append(entity_e[0])
                    else:
                        if self.type=='w': #字切分
                            word=[ee for ee in e if ee]
                            ss_ele.extend(word)
                        elif self.type=='t': #词切分
                            ss_ele.append(e)
                        else:
                            raise ValueError('错误的切分类型')
            ss.append(ss_ele)
        return ss


if __name__ == '__main__':

    er=EntityRecognition()
    print(er.get_entity(['康爱保是什么']))
