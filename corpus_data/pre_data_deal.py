
import re
sub_pattern='你好|.{0,4}老师|.{0,4}年|.{0,4}%|，|谢谢你|谢谢|。|！|？|～'

replace_pattern=r'\d{6}'
replace_pattern_0=r'\d.'
# s='600620'
# if re.search(replace_pattern,s):
#     print(123)
#
# ss=re.subn(replace_pattern,'cpzd',s)[0]
# print(ss)


import jieba
jieba.load_userdict('./stop_word.txt')

stop_words=[ele.replace('\n','') for ele in open('./stop_word.txt','r').readlines()]


def data_clean(sent):

    sent=re.subn(sub_pattern,'',sent)[0]
    sent=''.join([ele for ele in jieba.cut(sent) if ele not  in []])
    return sent


fw=open('./dn.txt','w')


for line in open('意图识别数据_all.txt','r').readlines():
    line=line.replace('\n','')
    try:
        index=line.split('\t\t')[0]
        sent=line.split('\t\t')[1]
        label=line.split('\t\t')[2]
    except:
        print('error',line)
    sent=sent.split('?')[0].replace(' ','')
    sent=re.subn(replace_pattern,'stock',sent)[0]
    sent=re.subn(replace_pattern_0,'',sent)[0]
    new_sent=data_clean(sent)

    fw.write(index)
    fw.write('\t\t')
    fw.write(new_sent)
    fw.write('\t\t')
    fw.write(label)
    fw.write('\n')
