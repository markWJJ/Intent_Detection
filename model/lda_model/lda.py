from gensim.models import ldamodel
from gensim import corpora, models
import os
import re
base_path=os.path.split(os.path.split(os.path.split(os.path.realpath(__file__))[0])[0])[0]
stop_word=[e.replace('\n','') for e in open(base_path+'/corpus_data/stop_word.txt')]


class LDA(object):

    def __init__(self):

        pass


    def data_get(self,data_path):

        tokens=[]
        sub_pattern='eos|bos'
        sents=[]
        labels=[]
        for ele in open(data_path,'r').readlines():
            ele=ele.replace('\n','')
            try:
                sent=re.subn(sub_pattern,'',ele.split('\t')[0].lower())[0]
                label=ele.split('\t')[2]
                labels.append(label)
                sents.append(sent)
                tokens.append([ e for e in sent.split(' ') if e not in stop_word])
            except:
                pass

        # 得到文档-单词矩阵 （直接利用统计词频得到特征）
        dictionary = corpora.Dictionary(tokens)  # 得到单词的ID,统计单词出现的次数以及统计信息
        # print type(dictionary)            # 得到的是gensim.corpora.dictionary.Dictionary的class类型

        texts = [dictionary.doc2bow(text) for text in tokens]  # 将dictionary转化为一个词袋，得到文档-单词矩阵
        texts_tf_idf = models.TfidfModel(texts)[texts]  # 文档的tf-idf形式(训练加转换的模式)
        lda=models.ldamodel.LdaModel.load('./lda.model')
        # lda = models.ldamodel.LdaModel(corpus=texts, id2word=dictionary, num_topics=6, update_every=0, passes=20,iterations=100)
        texts_lda = lda[texts_tf_idf]
        lda.save('lda.model')
        lda_word=lda.print_topics(num_topics=6, num_words=10)
        for ele in lda_word:
            print(ele)

        for ss,ll,doc in zip(sents,labels,texts_lda):
            print(ss,ll,doc)
            print('\n\n')


if __name__ == '__main__':
    lda=LDA()
    lda.data_get(base_path+'/corpus_data/train_out_char.txt')