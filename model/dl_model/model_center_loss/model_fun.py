import tensorflow as tf
import os
import numpy as np
from sklearn.metrics import classification_report,precision_recall_fscore_support


def get_center_loss(features, labels, alpha, num_classes):
    """获取center loss及center的更新op

    Arguments:
        features: Tensor,表征样本特征,一般使用某个fc层的输出,shape应该为[batch_size, feature_length].
        labels: Tensor,表征样本label,非one-hot编码,shape应为[batch_size].
        alpha: 0-1之间的数字,控制样本类别中心的学习率,细节参考原文.
        num_classes: 整数,表明总共有多少个类别,网络分类输出有多少个神经元这里就取多少.

    Return：
        loss: Tensor,可与softmax loss相加作为总的loss进行优化.
        centers: Tensor,存储样本中心值的Tensor，仅查看样本中心存储的具体数值时有用.
        centers_update_op: op,用于更新样本中心的op，在训练时需要同时运行该op，否则样本中心不会更新
    """
    # 获取特征的维数，例如256维
    len_features = features.get_shape()[1]
    # 建立一个Variable,shape为[num_classes, len_features]，用于存储整个网络的样本中心，
    # 设置trainable=False是因为样本中心不是由梯度进行更新的
    centers = tf.get_variable('centers', [num_classes, len_features], dtype=tf.float32,
                              initializer=tf.constant_initializer(0), trainable=False)
    # 将label展开为一维的，输入如果已经是一维的，则该动作其实无必要
    labels = tf.reshape(labels, [-1])

    # 根据样本label,获取mini-batch中每一个样本对应的中心值
    centers_batch = tf.gather(centers, labels)
    # 计算loss
    loss = tf.nn.l2_loss(features - centers_batch)

    # 当前mini-batch的特征值与它们对应的中心值之间的差
    diff = centers_batch - features

    # 获取mini-batch中同一类别样本出现的次数,了解原理请参考原文公式(4)
    unique_label, unique_idx, unique_count = tf.unique_with_counts(labels)
    appear_times = tf.gather(unique_count, unique_idx)
    appear_times = tf.reshape(appear_times, [-1, 1])

    diff = diff / tf.cast((1 + appear_times), tf.float32)
    diff = alpha * diff

    centers_update_op = tf.scatter_sub(centers, labels, diff)

    return loss, centers, centers_update_op

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


def sent_encoder(sent_word_emb,num,hidden_dim,sequence_length,name):
    '''
    句编码
    :param sent_word_emb:
    :param hidden_dim:
    :param name:
    :return:
    '''
    with tf.variable_scope(name_or_scope=name):
        sent_word_embs=tf.unstack(sent_word_emb,num,1)
        lstm_cell=tf.contrib.rnn.BasicLSTMCell(hidden_dim)
        lstm_cell_1=tf.contrib.rnn.BasicLSTMCell(hidden_dim)
        encoder,_=tf.nn.bidirectional_dynamic_rnn(
            lstm_cell,
            lstm_cell_1,
            sent_word_emb,
            dtype=tf.float32,
            sequence_length=sequence_length, )
        # encoder,_=tf.nn.static_rnn(lstm_cell,sent_word_embs,sequence_length=sequence_length,dtype=tf.float32)
        encoder=tf.concat(encoder,2)
        encoder=tf.unstack(encoder,num,1)
        # encoder=tf.layers.dense(encoder,100,activation=tf.nn.tanh)
        # encoder=tf.unstack(encoder,num,1)
        return encoder

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

def accuray(pre_label_soft,label):
    pre_ = np.argmax(pre_label_soft, 1)
    num=np.sum(np.equal(pre_,label))
    acc=float(num)/float(pre_label_soft.shape[0])
    return acc

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


def positional_encoding(inputs,
                        num_units,
                        zero_pad=True,
                        scale=True,
                        scope="positional_encoding",
                        reuse=None):
    '''Sinusoidal Positional_Encoding.

    Args:
      inputs: A 2d Tensor with shape of (N, T).
      num_units: Output dimensionality
      zero_pad: Boolean. If True, all the values of the first row (id = 0) should be constant zero
      scale: Boolean. If True, the output will be multiplied by sqrt num_units(check details from paper)
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
        A 'Tensor' with one more rank than inputs's, with the dimensionality should be 'num_units'
    '''

    N, T = inputs.get_shape().as_list()
    with tf.variable_scope(scope, reuse=reuse):
        position_ind = tf.tile(tf.expand_dims(tf.range(T), 0), [N, 1])

        # First part of the PE function: sin and cos argument
        position_enc = np.array([
            [pos / np.power(10000, 2. * i / num_units) for i in range(num_units)]
            for pos in range(T)])

        # Second part, apply the cosine to even columns and sin to odds.
        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1

        # Convert to a tensor
        lookup_table = tf.convert_to_tensor(position_enc)

        if zero_pad:
            lookup_table = tf.concat((tf.zeros(shape=[1, num_units]),
                                      lookup_table[1:, :]), 0)
        outputs = tf.nn.embedding_lookup(lookup_table, position_ind)

        if scale:
            outputs = outputs * num_units ** 0.5

        return outputs

        # ?


def normalize(inputs,
              epsilon=1e-8,
              scope="ln",
              reuse=None):
    '''Applies layer normalization.

    Args:
      inputs: A tensor with 2 or more dimensions, where the first dimension has
        `batch_size`.
      epsilon: A floating number. A very small number for preventing ZeroDivision Error.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A tensor with the same shape and data dtype as `inputs`.
    '''
    with tf.variable_scope(scope, reuse=reuse):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]

        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta = tf.Variable(tf.zeros(params_shape))
        gamma = tf.Variable(tf.ones(params_shape))
        normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
        outputs = gamma * normalized + beta

    return outputs


def multihead_attention(queries,
                        keys,
                        num_units=None,
                        num_heads=8,
                        dropout_rate=0.1,
                        is_training=True,
                        causality=False,
                        scope="multihead_attention",
                        reuse=None):
    '''Applies multihead attention.

    Args:
      queries: A 3d tensor with shape of [N, T_q, C_q].
      keys: A 3d tensor with shape of [N, T_k, C_k].
      num_units: A scalar. Attention size.
      dropout_rate: A floating point number.
      is_training: Boolean. Controller of mechanism for dropout.
      causality: Boolean. If true, units that reference the future are masked.
      num_heads: An int. Number of heads.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns
      A 3d tensor with shape of (N, T_q, C)
    '''

    # 其中x，y的shape为[N,T]，N即batch_size的大小，T为最大句子长度maxlen，默认为10
    #
    #
    #
    with tf.variable_scope(scope, reuse=reuse):
        # Set the fall back option for num_units
        if num_units is None:
            num_units = queries.get_shape().as_list[-1]

        # Linear projections
        Q = tf.layers.dense(queries, num_units, activation=tf.nn.relu)  # (N, T_q, C)
        K = tf.layers.dense(keys, num_units, activation=tf.nn.relu)  # (N, T_k, C)
        V = tf.layers.dense(keys, num_units, activation=tf.nn.relu)  # (N, T_k, C)

        # Split and concat
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, C/h)
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)

        # Multiplication
        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # (h*N, T_q, T_k)

        # Scale
        outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

        # Key Masking
        key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1)))  # (N, T_k)
        key_masks = tf.tile(key_masks, [num_heads, 1])  # (h*N, T_k)
        key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1])  # (h*N, T_q, T_k)

        paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
        outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

        # Causality = Future blinding
        if causality:
            diag_vals = tf.ones_like(outputs[0, :, :])  # (T_q, T_k)
            tril = tf.contrib.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()  # (T_q, T_k)
            masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1])  # (h*N, T_q, T_k)

            paddings = tf.ones_like(masks) * (-2 ** 32 + 1)
            outputs = tf.where(tf.equal(masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

        # Activation
        outputs = tf.nn.softmax(outputs)  # (h*N, T_q, T_k)

        # Query Masking
        query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1)))  # (N, T_q)
        query_masks = tf.tile(query_masks, [num_heads, 1])  # (h*N, T_q)
        query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])  # (h*N, T_q, T_k)
        outputs *= query_masks  # broadcasting. (N, T_q, C)

        # Dropouts
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))

        # Weighted sum
        outputs = tf.matmul(outputs, V_)  # ( h*N, T_q, C/h)

        # Restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, C)

        # Residual connection
        outputs += queries

        # Normalize
        outputs = normalize(outputs)  # (N, T_q, C)

    return outputs


def feedforward(inputs,
                num_units=[2048, 512],
                scope="multihead_attention",
                reuse=None):
    '''Point-wise feed forward net.

    Args:
      inputs: A 3d tensor with shape of [N, T, C].
      num_units: A list of two integers.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A 3d tensor with the same shape and dtype as inputs
    '''
    with tf.variable_scope(scope, reuse=reuse):
        # Inner layer
        params = {"inputs": inputs, "filters": num_units[0], "kernel_size": 1,
                  "activation": tf.nn.relu, "use_bias": True}
        outputs = tf.layers.conv1d(**params)

        # Readout layer
        params = {"inputs": outputs, "filters": num_units[1], "kernel_size": 1,
                  "activation": None, "use_bias": True}
        outputs = tf.layers.conv1d(**params)

        # Residual connection
        outputs += inputs

        # Normalize
        outputs = normalize(outputs)

    return outputs


def label_smoothing(inputs, epsilon=0.1):
    '''Applies label smoothing. See https://arxiv.org/abs/1512.00567.

    Args:
      inputs: A 3d tensor with shape of [N, T, V], where V is the number of vocabulary.
      epsilon: Smoothing rate.

    For example,

    ```
    import tensorflow as tf
    inputs = tf.convert_to_tensor([[[0, 0, 1],
       [0, 1, 0],
       [1, 0, 0]],

      [[1, 0, 0],
       [1, 0, 0],
       [0, 1, 0]]], tf.float32)

    outputs = label_smoothing(inputs)

    with tf.Session() as sess:
        print(sess.run([outputs]))

    >>
    [array([[[ 0.03333334,  0.03333334,  0.93333334],
        [ 0.03333334,  0.93333334,  0.03333334],
        [ 0.93333334,  0.03333334,  0.03333334]],

       [[ 0.93333334,  0.03333334,  0.03333334],
        [ 0.93333334,  0.03333334,  0.03333334],
        [ 0.03333334,  0.93333334,  0.03333334]]], dtype=float32)]
    ```
    '''
    K = inputs.get_shape().as_list()[-1]  # number of channels
    return ((1 - epsilon) * inputs) + (epsilon / K)