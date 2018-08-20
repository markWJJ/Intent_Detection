import tensorflow as tf
import os
import numpy as np
from sklearn.metrics import classification_report,precision_recall_fscore_support


def embedding(sent,num,emb_dim,name,reuse=False):
    '''
    词嵌入
    :param sent:
    :param num:
    :param emb_dim:
    :param name:
    :return:
    '''
    with tf.variable_scope(name_or_scope=name,reuse=reuse):
        embedding=tf.get_variable(name='sent_emb',shape=(num,emb_dim),initializer=tf.random_normal_initializer,trainable=True)
        emb=tf.nn.embedding_lookup(embedding,sent)
        return emb,embedding


def sent_encoder(sent_word_emb,num,hidden_dim,sequence_length,name,dropout,reuse=False):
    '''
    句编码
    :param sent_word_emb:
    :param hidden_dim:
    :param name:
    :return:
    '''
    with tf.variable_scope(name_or_scope=name,reuse=reuse):
        lstm_cell=tf.contrib.rnn.BasicLSTMCell(hidden_dim)
        lstm_cell_1=tf.contrib.rnn.BasicLSTMCell(hidden_dim)
        lstm_cell=tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=dropout)
        lstm_cell_1=tf.nn.rnn_cell.DropoutWrapper(lstm_cell_1, output_keep_prob=dropout)
        encoder,_=tf.nn.bidirectional_dynamic_rnn(
            lstm_cell,
            lstm_cell_1,
            sent_word_emb,
            dtype=tf.float32,
            sequence_length=sequence_length, )
        # encoder,_=tf.nn.static_rnn(lstm_cell,sent_word_embs,sequence_length=sequence_length,dtype=tf.float32)
        encoder=tf.concat(encoder,2)
        # encoder_fw,encoder_bw=encoder[0],encoder[1]
        # encoder_fw=last_relevant_output(encoder_fw,sequence_length)
        # encoder_bw=last_relevant_output(encoder_bw,sequence_length)
        # encoder=tf.concat([encoder_fw,encoder_bw],1)
        # encoder=tf.unstack(encoder,num,1)
        # encoder=tf.layers.dense(encoder,100,activation=tf.nn.tanh)
        # encoder=tf.unstack(encoder,num,1)
        return encoder

def sigmiod_layer(input,hidden_dim,name,reuse=False):

    with tf.variable_scope(name_or_scope=name,reuse=reuse):
        w=tf.get_variable(name='w',initializer=tf.random_normal_initializer,shape=(2*hidden_dim,300))
        b=tf.get_variable(name='b',initializer=tf.random_normal_initializer,shape=(300,))

        return tf.nn.xw_plus_b(input,w,b)





def last_relevant_output(output, sequence_length):
    """
    Given the outputs of a LSTM, get the last relevant output that
    is not padding. We assume that the last 2 dimensions of the input
    represent (sequence_length, hidden_size).

    Parameters
    ----------
    output: Tensor
        A tensor, generally the output of a tensorflow RNN.
        The tensor index sequence_lengths+1 is selected for each
        instance in the output.

    sequence_length: Tensor
        A tensor of dimension (batch_size, ) indicating the length
        of the sequences before padding was applied.

    Returns
    -------
    last_relevant_output: Tensor
        The last relevant output (last element of the sequence), as retrieved
        by the output Tensor and indicated by the sequence_length Tensor.
    """
    with tf.name_scope("last_relevant_output"):
        batch_size = tf.shape(output)[0]
        max_length = tf.shape(output)[-2] #seq_len
        out_size = int(output.get_shape()[-1]) #dim
        index = tf.range(0, batch_size) * max_length + (sequence_length - 1)
        print(index)
        flat = tf.reshape(output, [-1, out_size])
        relevant = tf.gather(flat, index)
        return relevant


def mean_pool(input_tensor, sequence_length=None):
    """
    Given an input tensor (e.g., the outputs of a LSTM), do mean pooling
    over the last dimension of the input.

    For example, if the input was the output of a LSTM of shape
    (batch_size, sequence length, hidden_dim), this would
    calculate a mean pooling over the last dimension (taking the padding
    into account, if provided) to output a tensor of shape
    (batch_size, hidden_dim).

    Parameters
    ----------
    input_tensor: Tensor
        An input tensor, preferably the output of a tensorflow RNN.
        The mean-pooled representation of this output will be calculated
        over the last dimension.

    sequence_length: Tensor, optional (default=None)
        A tensor of dimension (batch_size, ) indicating the length
        of the sequences before padding was applied.

    Returns
    -------
    mean_pooled_output: Tensor
        A tensor of one less dimension than the input, with the size of the
        last dimension equal to the hidden dimension state size.
    """
    with tf.name_scope("mean_pool"):
        # shape (batch_size, sequence_length)
        input_tensor_sum = tf.reduce_sum(input_tensor, axis=-2)

        # If sequence_length is None, divide by the sequence length
        # as indicated by the input tensor.
        if sequence_length is None:
            sequence_length = tf.shape(input_tensor)[-2]

        # Expand sequence length from shape (batch_size,) to
        # (batch_size, 1) for broadcasting to work.
        expanded_sequence_length = tf.cast(tf.expand_dims(sequence_length, -1),
                                           "float32") + 1e-08

        # Now, divide by the length of each sequence.
        # shape (batch_size, sequence_length)
        mean_pooled_input = (input_tensor_sum /
                             expanded_sequence_length)
        return mean_pooled_input


def match_attention_ops(encoder_0_t,encoder_1,encoder_1_mask,hidden_dim,self_hidden_dim):
    '''
    match_attention的辅助函数
    :param encoder_0_t:
    :param encoder_1:
    :param encoder_1_mask:
    :return:
    '''

    encoder_0_t=tf.expand_dims(encoder_0_t,1)
    with tf.variable_scope(name_or_scope='ops'):
        w_0=tf.get_variable(name='w_0',shape=(2*hidden_dim,self_hidden_dim),initializer=tf.random_normal_initializer)
        w_1=tf.get_variable(name='w_1',shape=(2*hidden_dim,self_hidden_dim),initializer=tf.random_normal_initializer)

        v=tf.get_variable(name='v',shape=(self_hidden_dim,1),initializer=tf.random_normal_initializer)

        encoder_0_t_=tf.einsum('ijk,kl->ijl',encoder_0_t,w_0)
        encoder_1_=tf.einsum('ijk,kl->ijl',encoder_1,w_1)

        encoder=tf.nn.tanh(tf.add(encoder_1_,encoder_0_t_))
        logit=tf.einsum('ijk,kl->ijl',encoder,v)
        logit=tf.squeeze(logit,-1)
        logit=tf.multiply(logit,encoder_1_mask)
        soft_logit=tf.nn.softmax(logit,1)
        soft_logit=tf.expand_dims(soft_logit,-1)
        out=tf.einsum('ijk,ijl->ikl',encoder_1,soft_logit)
        out=tf.squeeze(out,-1)

        return out




def match_attention(encoder_0,encoder_1,encoder_0_mask,encoder_1_mask,hidden_dim,dropout,name,seq_len,reuse=False):
    '''
    match_attention, encoder_0的每个step 以encoder_1的attention作为输入
    :param encoder_0:
    :param encoder_1:
    :param encoder_0_mask:
    :param encoder_1_mask:
    :return:
    '''

    with tf.variable_scope(name_or_scope=name,reuse=reuse) as scope:
        cell=tf.contrib.rnn.BasicLSTMCell(hidden_dim)
        cell=tf.nn.rnn_cell.DropoutWrapper(cell=cell,input_keep_prob=dropout)
        encoder_0_list=tf.unstack(encoder_0,seq_len,1)
        encoder_ele=encoder_0_list[0][:,:hidden_dim]

        init_c=tf.zeros_like(encoder_ele)
        init_h=tf.zeros_like(encoder_ele)
        H=[init_h]
        C=[init_c]
        output=[]


        for t in range(len(encoder_0_list)):

            if t>0:
                scope.reuse_variables()

            encoder_0_t=encoder_0_list[t]
            h,c=H[-1],C[-1]
            state=(h,c)
            attention_t=match_attention_ops(encoder_0_t,encoder_1,encoder_1_mask,hidden_dim,hidden_dim)
            input_t=tf.concat([encoder_0_t,attention_t],1)
            out_t,state_t=cell(input_t,state)
            h_t,c_t=state_t[0],state_t[1]
            H.append(h_t)
            C.append(c_t)
            output.append(out_t)

        out=tf.stack(output,1)
        return out







def self_attention(lstm_outs,sent_mask,reuse=False):
    '''
    attention
    :param lstm_outs:
    :param sent_mask:
    :return:
    '''
    with tf.variable_scope(name_or_scope='attention',reuse=reuse):
        if isinstance(lstm_outs,list):
            lstm_outs=tf.stack(lstm_outs,1)

        V=tf.get_variable(name='v',shape=(300,1),initializer=tf.random_normal_initializer)
        logit=tf.layers.dense(lstm_outs,300,activation=tf.nn.tanh,use_bias=True,reuse=reuse)
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


