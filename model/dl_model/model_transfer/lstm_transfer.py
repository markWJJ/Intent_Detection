import tensorflow as tf
import os
import numpy as np
from tensorflow.python import pywrap_tensorflow
from model.dl_model.model_transfer.data_preprocess import Intent_Slot_Data
from IntentConfig import Config
intent_config=Config()
from entity_recognition.ner import EntityRecognition
base_path=os.path.split(os.path.split(os.path.split(os.path.split(os.path.realpath(__file__))[0])[0])[0])[0]

config = tf.ConfigProto(allow_soft_placement=True)


# saver = tf.train.import_meta_graph(base_path+"/save_model/model_lstm_mask/lstm.ckpt.meta")
# with tf.Session(config=config) as sess:
#     saver.restore(sess, base_path+"/save_model/model_lstm_mask/lstm.ckpt")
#
#
#     checkpoint_path = os.path.join(base_path+"/save_model/model_lstm_mask/lstm.ckpt")
#     reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)  # tf.train.NewCheckpointReader
#     var_to_shape_map = reader.get_variable_to_shape_map()
#     for key in var_to_shape_map:
#         print("tensor_name: ", key)



def get_sent_mask(sent_ids, entity_ids):
    sent_mask = np.zeros_like(sent_ids, dtype=np.float32)
    for i in range(sent_ids.shape[0]):
        for j in range(sent_ids.shape[1]):
            if sent_ids[i, j] > 0 and sent_ids[i, j] not in entity_ids:
                sent_mask[i, j] = 1.0
            elif sent_ids[i, j] > 0 and sent_ids[i, j] in entity_ids:
                sent_mask[i, j] = 0.5
    return sent_mask

saver = tf.train.import_meta_graph(base_path+"/save_model/model_lstm_mask/lstm.ckpt.meta")

# We can now access the default graph where all our metadata has been loaded
graph = tf.get_default_graph()

for ele in graph.get_operations():
    print(ele.name)



# Finally we can retrieve tensors, operations, etc.
sent_word_operation = graph.get_tensor_by_name('lstm/sent_word:0')
sent_len_operation = graph.get_tensor_by_name('lstm/sent_len:0')
sent_mask_operation = graph.get_tensor_by_name('lstm/sent_mask:0')
sent_dropout_operation = graph.get_tensor_by_name('lstm/Placeholder:0')
sent_intent_y_operation = graph.get_tensor_by_name('lstm/intent_y:0')

sent_emb=graph.get_tensor_by_name('lstm/out_layers/dense/BiasAdd:0')
sent_emb=tf.layers.dense(sent_emb,79)
sent_soft=tf.nn.softmax(sent_emb,1)
sent_max=tf.argmax(sent_soft,1)
sent_y=tf.argmax(sent_intent_y_operation,1)

print(sent_max)
print(sent_y)
acc= tf.reduce_mean(tf.cast(tf.equal(sent_max,sent_y),tf.float32))

loss=tf.losses.softmax_cross_entropy(onehot_labels=sent_intent_y_operation,logits=sent_emb)

train_op=tf.train.AdamOptimizer(0.001).minimize(loss)

# sent_word_operation = graph.get_operation_by_name('lstm/sent_word')
# sent_len_operation = graph.get_operation_by_name('lstm/sent_len')
# sent_mask_operation = graph.get_operation_by_name('lstm/sent_mask')
# sent_dropout_operation = graph.get_operation_by_name('lstm/Placeholder')
# sent_intent_y_operation = graph.get_operation_by_name('lstm/intent_y')
#
#
# sent_emb=graph.get_operation_by_name('lstm/out_layers/dense/BiasAdd')

# train_op = graph.get_operation_by_name('loss/train_op')
# hyperparameters = tf.get_collection('hyperparameters')

saver_restore = tf.train.import_meta_graph(base_path+"/save_model/model_lstm_mask/lstm.ckpt.meta")
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    saver_restore.restore(sess,base_path+"/save_model/model_lstm_mask/lstm.ckpt")

    dd = Intent_Slot_Data(train_path=base_path + "/corpus_data/%s" % intent_config.train_name,
                               test_path=base_path + "/corpus_data/%s" % intent_config.dev_name,
                               dev_path=base_path + "/corpus_data/%s" % intent_config.dev_name,
                               batch_size=32,
                               max_length=30, flag='train_new',
                               use_auto_bucket=False, save_model='lstm')

    ee = EntityRecognition()
    entity_id = []
    for k, v in dd.vocab.items():
        if k in ee.entity_dict.keys():
            entity_id.append(v)

    for _ in range(10):
        sent, slot, intent_label, rel_len, cur_len = dd.next_batch()
        sent_mask = get_sent_mask(sent, entity_id)
        loss_, acc_,_ = sess.run([loss, acc,train_op], feed_dict={sent_word_operation: sent,
                                                       sent_len_operation: rel_len,
                                                       sent_mask_operation: sent_mask,
                                                       sent_dropout_operation: 1.0,
                                                       sent_intent_y_operation: intent_label
                                                       })
        print('train',loss_,acc_)

    # dev_sent, dev_slot, dev_intent, dev_rel_len, _ = dd.get_dev()
    # dev_sent_mask = get_sent_mask(dev_sent, entity_id)
    #
    # loss_,acc_=sess.run([loss,acc],feed_dict={sent_word_operation:dev_sent,
    #                                sent_len_operation:dev_rel_len,
    #                                sent_mask_operation:dev_sent_mask,
    #                                sent_dropout_operation:1.0,
    #                                sent_intent_y_operation:dev_intent
    #                                })
    #
    #
    # print('dev_loss,',loss_,acc_)