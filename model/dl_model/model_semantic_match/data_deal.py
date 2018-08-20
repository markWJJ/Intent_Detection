import os
import random
from collections import defaultdict
base_path = os.path.split(os.path.split(os.path.split(os.path.split(os.path.realpath(__file__))[0])[0])[0])[0]

train_path=base_path+'/corpus_data/train_out_char.txt'
dev_path=base_path+'/corpus_data/dev_out_char.txt'

fw=open('dev.txt','w')

datas_dict=defaultdict(list)
for line in open(dev_path,'r').readlines():
    lines=line.replace('\n','').split('\t')
    sent=lines[1]
    intent=lines[3]

    datas_dict[intent].append(sent)

pos_num=0
neg_num=0
keys=list(datas_dict.keys())
for k in datas_dict.keys():
    keys.remove(k)
    other_k=keys
    # print(other_k)
    v=datas_dict[k]
    for i in range(len(v)-1):
        for j in range(len(v)):
            print(v[i],'\t\t',v[j],'\t\t',1)
            pos_num+=1
            fw.write(v[i])
            fw.write('\t\t')
            fw.write(v[j])
            fw.write('\t\t')
            fw.write('1')
            fw.write('\n')
            # pass
        for ok in other_k:
            ov=datas_dict[ok]
            random.shuffle(ov)
            for e in  ov[:1]:
                neg_num += 1
                print(v[i],'\t\t',e,'\t\t',0)
                fw.write(v[i])
                fw.write('\t\t')
                fw.write(ov[0])
                fw.write('\t\t')
                fw.write('0')
                fw.write('\n')
            # pass
    keys.append(k)

print(pos_num,neg_num)