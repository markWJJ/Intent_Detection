import tensorflow as tf
import os
import numpy as np
from sklearn.metrics import classification_report,precision_recall_fscore_support


def embedding(sent,num,emb_dim,name):
    '''
    词嵌入
    :param sent:
    :param num:
    :param emb_dim:
    :param name:
    :return:
    '''
    with tf.variable_scope(name_or_scope=name):
        embedding=tf.Variable(tf.random_uniform(shape=(num,emb_dim),dtype=tf.float32),name='sent_emb',trainable=True)
        emb=tf.nn.embedding_lookup(embedding,sent)
        return emb


def sent_encoder(sent_word_emb,num,hidden_dim,sequence_length,name,res_dim,sent_length,sent_mask,keep_dropout):
    '''
    句编码
    :param sent_word_emb:
    :param hidden_dim:
    :param name:
    :return:
    '''
    with tf.variable_scope(name_or_scope=name):
        sent_word_embs=tf.unstack(sent_word_emb,num,1)
        #前向lstm cell

        sent_mask=tf.expand_dims(sent_mask,-1)
        lstm_cell_fw=tf.nn.rnn_cell.DropoutWrapper(tf.contrib.rnn.BasicLSTMCell(hidden_dim,state_is_tuple=True),output_keep_prob=keep_dropout)
        zero_init=tf.zeros_like(sent_word_embs[0])

        fw_state=[(zero_init,zero_init)]
        fw_h=[]
        fw_res_dim=[(zero_init,zero_init)]
        for i in range(sent_length):
            if i > 0:
                tf.get_variable_scope().reuse_variables()
            if i%res_dim==0:
                h,state=lstm_cell_fw(sent_word_embs[i],fw_state[-1])
                state=(tf.add(state[0],fw_res_dim[-1][0]),tf.add(state[1],fw_res_dim[-1][1]))
                fw_h.append(h)
                fw_state.append(state)
            else:
                h, state = lstm_cell_fw(sent_word_embs[i], fw_state[-1])
                fw_h.append(h)
                fw_state.append(state)
        fw_out=tf.stack(fw_h,1)
        fw_out=tf.multiply(fw_out,tf.cast(sent_mask,tf.float32))


        bw_sent_embs=sent_word_embs[::-1]
        lstm_cell_bw=tf.nn.rnn_cell.DropoutWrapper(tf.contrib.rnn.BasicLSTMCell(hidden_dim,state_is_tuple=True),output_keep_prob=keep_dropout)
        bw_state = [(zero_init, zero_init)]
        bw_h = []
        bw_res_dim = [(zero_init, zero_init)]
        for i in range(sent_length):
            if i > 0:
                tf.get_variable_scope().reuse_variables()
            if i % res_dim == 0:
                h, state = lstm_cell_bw(bw_sent_embs[i], bw_state[-1])
                state = (tf.add(state[0], bw_res_dim[-1][0]), tf.add(state[1], bw_res_dim[-1][1]))
                # state = (bw_res_dim[-1][0],bw_res_dim[-1][1])
                bw_h.append(h)
                bw_state.append(state)
            else:
                h, state = lstm_cell_bw(bw_sent_embs[i], bw_state[-1])
                bw_h.append(h)
                bw_state.append(state)

        bw_h=bw_h[::-1]
        bw_out=tf.stack(bw_h,1)
        bw_out=tf.multiply(bw_out,tf.cast(bw_out,tf.float32))

        encoder=tf.concat((fw_out,bw_out),2)
        # encoder=tf.expand_dims(encoder,-1)
        # cnn_w=tf.Variable(tf.random_uniform(shape=(2,1,1,1)))
        # encoder=tf.nn.conv2d(encoder,cnn_w,strides=[1,2,1,1],padding='VALID')
        # encoder=tf.squeeze(encoder,-1)
        return tf.unstack(encoder,num,1)
        # encoder,_=tf.nn.bidirectional_dynamic_rnn(
        #     lstm_cell,
        #     lstm_cell_1,
        #     sent_word_emb,
        #     dtype=tf.float32,
        #     sequence_length=sequence_length, )
        # # encoder,_=tf.nn.static_rnn(lstm_cell,sent_word_embs,sequence_length=sequence_length,dtype=tf.float32)
        # encoder=tf.concat(encoder,2)
        # encoder=tf.unstack(encoder,num,1)
        # # encoder=tf.layers.dense(encoder,100,activation=tf.nn.tanh)
        # # encoder=tf.unstack(encoder,num,1)
        # return encoder

def self_attention(lstm_outs,sent_mask):
    '''
    attention
    :param lstm_outs:
    :param sent_mask:
    :return:
    '''
    with tf.variable_scope(name_or_scope='attention'):
        if isinstance(lstm_outs,list):
            lstm_outs=tf.stack(lstm_outs,1)
        V=tf.Variable(tf.random_uniform(shape=(300,1),dtype=tf.float32))
        logit=tf.layers.dense(lstm_outs,300,activation=tf.nn.tanh,use_bias=True)
        logit=tf.einsum('ijk,kl->ijl',logit,V)
        logit=tf.squeeze(logit,-1)
        logit=tf.multiply(logit,sent_mask)
        soft_logit=tf.nn.softmax(logit,1)
        soft_logit=tf.expand_dims(soft_logit,-1)
        attention_out=tf.einsum('ijk,ijl->ilk',lstm_outs,soft_logit)
        attention_out=tf.squeeze(attention_out,1)
        return attention_out



def cosin_com(sent_enc,label_enc,label_num):
    '''
    相似度计算
    :param sent_enc:
    :param label_enc:
    :return:
    '''
    sent=tf.layers.dense(sent_enc,300)
    label=tf.layers.dense(label_enc,300)
    sent_emb_norm = tf.sqrt(tf.reduce_sum(tf.square(sent), axis=1))
    label=tf.unstack(label,label_num,0)
    cosins = []
    # 内积
    for ele in label:
        intent_norm = tf.sqrt(tf.reduce_sum(tf.square(ele)))
        ele = tf.expand_dims(ele, -1)
        sent_intent = tf.matmul(sent, ele)
        sent_intent = tf.reshape(sent_intent, [-1, ])
        cosin = sent_intent / (sent_emb_norm * intent_norm)
        cosins.append(cosin)
    cosin = tf.stack(cosins, 1)

    return cosin

def label_sent_attention(sent_encoder,label_emb,sent_mask):


    sent_encoder=tf.layers.dense(sent_encoder,300)
    label_emb=tf.layers.dense(label_emb,300)
    sent_encoder=tf.multiply(sent_encoder,tf.expand_dims(sent_mask,-1))
    tran_label_emb=tf.transpose(label_emb,[1,0])

    sent_encoder=tf.nn.l2_normalize(sent_encoder,-1)
    tran_label_emb=tf.nn.l2_normalize(tran_label_emb,0)

    G=tf.einsum('ijk,kl->ijl',sent_encoder,tran_label_emb)

    G=tf.expand_dims(G,-1)
    fliter_w=tf.Variable(tf.random_uniform(shape=(8,1,1,1),dtype=tf.float32))
    max_G=tf.nn.relu(tf.nn.conv2d(G,filter=fliter_w,strides=[1,1,1,1],padding='SAME'))
    max_G=tf.squeeze(max_G,-1)

    max_G=tf.reduce_max(max_G,axis=-1,keep_dims=True)

    mask_G=tf.multiply(max_G,tf.expand_dims(sent_mask,-1))

    soft_mask_G=tf.clip_by_value(tf.nn.softmax(mask_G,1),1e-5,1.0)

    out=tf.einsum('ijk,ijl->ikl',sent_encoder,soft_mask_G)

    out=tf.squeeze(out,-1)

    return out


def output_layers(inputs,out_dim,name,reuse):

    with tf.variable_scope(name_or_scope=name,reuse=reuse):
        return tf.layers.dense(inputs,out_dim)





def loss_function(cosin,label):
    soft_logit = tf.nn.softmax(cosin, 1)
    intent = tf.cast(label, tf.float32)
    intent_loss = -tf.reduce_sum(intent * tf.log(tf.clip_by_value(soft_logit, 1e-5, 1.0, name=None)))
    return intent_loss,soft_logit

def intent_acc(pre,label,id2intent):
    '''
    获取intent准确率
    :param pre:
    :param label:
    :return:
    '''
    pre_ = np.argmax(pre, 1)

    label_ = np.argmax(label, 1)
    ss=[[int(k),v] for k,v in id2intent.items()]
    ss.sort(key=lambda x:x[0],reverse=False)
    s1=[e[1] for e in ss]
    # print(classification_report(y_true=label_,y_pred=pre_,target_names=s1))
    all_sum = len(label_)
    num = sum([1 for e, e1 in zip(pre_, label_) if e == e1])

    return float(num) / float(all_sum)


