import os
import random
from collections import defaultdict
base_path = os.path.split(os.path.split(os.path.split(os.path.split(os.path.realpath(__file__))[0])[0])[0])[0]

train_path=base_path+'/corpus_data/train_out_char.txt'
dev_path=base_path+'/corpus_data/dev_out_char.txt'

data_type='train'

fw=open('%s.txt'%data_type,'w')

datas_dict=defaultdict(list)
for line in open(train_path,'r').readlines():
    lines=line.replace('\n','').split('\t')
    sent=lines[1]
    intent=lines[3]

    datas_dict[intent].append(sent)


pos_num=0
neg_num=0
get_pos_num=3
get_neg_num=3
keys=list(datas_dict.keys())
for k in datas_dict.keys():
    keys.remove(k)
    other_k=keys
    # print(other_k)
    v=datas_dict[k]
    for i in range(len(v)-get_pos_num):
        for j in range(i,i+get_pos_num):
            # print(v[i],'\t\t',v[j],'\t\t',1)
            pos_num+=1
            fw.write(v[i])
            fw.write('\t\t')
            fw.write(v[j])
            fw.write('\t\t')
            fw.write('1')
            fw.write('\n')
            # pass
        random.shuffle(other_k)
        for ok in other_k[0:get_neg_num]:
            ov=datas_dict[ok]
            random.shuffle(ov)
            for e in  ov[:1]:
                neg_num += 1
                # print(v[i],'\t\t',e,'\t\t',0)
                fw.write(v[i])
                fw.write('\t\t')
                fw.write(ov[0])
                fw.write('\t\t')
                fw.write('0')
                fw.write('\n')
            # pass
    keys.append(k)
print(pos_num,neg_num)
# data=[]
# for ele in open('./train.txt','r').readlines():
#     if ele not in data:
#         data.append(ele)
#
# fw1=open('train_1.txt','w')
# for e in data:
#     e=e.replace('\n','')
#     fw1.write(e)
#     fw1.write('\n')