from gensim.models import ldamodel
from gensim import corpora, models
from gensim.models import Doc2Vec
from gensim.models.doc2vec import LabeledSentence
import os
import re
base_path=os.path.split(os.path.split(os.path.split(os.path.realpath(__file__))[0])[0])[0]
stop_word=[e.replace('\n','') for e in open(base_path+'/corpus_data/stop_word.txt')]
tokens=[]
sub_pattern='eos|bos'
sents=[]
labels=[]

index=0
for ele in open(base_path+'/corpus_data/train_out_char.txt','r').readlines():
    ele=ele.replace('\n','')

    sent=re.subn(sub_pattern,'',ele.split('\t')[0].lower())[0]
    label=ele.split('\t')[2]
    labels.append(label)
    sents.append(sent)
    s=LabeledSentence(words=[ e for e in sent.split(' ') if e not in stop_word],labels=['SENT_%s'%index])

    index+=1
    tokens.append(s)
    # tokens.append([ e for e in sent.split(' ') if e not in stop_word])

model = models.Doc2Vec(size=300, window=10, min_count=5, workers=11,alpha=0.025, min_alpha=0.025)
model.build_vocab(tokens)
# model=Doc2Vec(documents=tokens,dm=1,epochs=50,size=50)
for epoch in range(10):
    model.train(tokens)
    model.alpha -= 0.002            # decrease the learning rate
    model.min_alpha = model.alpha       # fix the learning rate, no deca
    model.train(tokens)
model.save('./doc2vec.model')
print(model.most_similar('今天天气很好'))