import tensorflow as tf
import os
import sys
sys.path.append('./')

from model.dl_model.model_lstm_rl.data_preprocess import Intent_Slot_Data
from model.dl_model.model_lstm_rl.model_fun import embedding,sent_encoder,self_attention,loss_function,intent_acc,cosin_com,label_sent_attention,output_layers
from model.dl_model.model_lstm_rl.focal_loss import focal_loss
from IntentConfig import Config
import numpy as np
import logging
import copy
import pickle
from sklearn.metrics import classification_report,precision_recall_fscore_support,confusion_matrix
import gc
path=os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0,path)
from collections import deque, namedtuple

base_path=os.path.split(os.path.split(os.path.split(os.path.split(os.path.realpath(__file__))[0])[0])[0])[0]
from random import random
from entity_recognition.ner import entity_dict
import itertools

from xmlrpc.server import SimpleXMLRPCServer
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename='modellog.log',
                    filemode='w')
_logger = logging.getLogger("intent_bagging")

config_lstm=Config()

gpu_id=3


os.environ["CUDA_VISIBLE_DEVICES"] = ""

class Config_lstm(object):
    '''
    默认配置
    '''
    learning_rate = 0.001
    batch_size = 128
    label_max_len=16
    sent_len = 40  # 句子长度
    embedding_dim = 100  # 词向量维度
    hidden_dim = 200
    train_dir = './data/train_out_%s.txt'
    dev_dir = './data/dev_out.txt'
    test_dir = './data/test.txt'
    model_dir = base_path+'/save_model/model_lstm_mask/intent_lstm_dn.ckpt'
    if not os.path.exists(base_path+'/save_model/model_lstm_mask'):
        os.makedirs(base_path+'/save_model/model_lstm_mask')
    use_cpu_num = 16
    keep_dropout = 0.5
    summary_write_dir = "./tmp/r_net.log"
    epoch = 100
    use_auto_buckets=False
    lambda1 = 0.01
    model_mode = 'bilstm_attention_crf'  # 模型选择：bilstm bilstm_crf bilstm_attention bilstm_attention_crf,cnn_crf


config = Config_lstm()
tf.app.flags.DEFINE_float("mask_lambda1", config.lambda1, "l2学习率")
tf.app.flags.DEFINE_float("mask_learning_rate", config.learning_rate, "学习率")
tf.app.flags.DEFINE_float("mask_keep_dropout", config.keep_dropout, "dropout")
tf.app.flags.DEFINE_integer("mask_batch_size", config.batch_size, "批处理的样本数量")
tf.app.flags.DEFINE_integer("mask_max_len", config.sent_len, "句子长度")
tf.app.flags.DEFINE_integer("mask_max_label_len", config.label_max_len, "句子长度")
tf.app.flags.DEFINE_integer("mask_embedding_dim", config.embedding_dim, "词嵌入维度.")
tf.app.flags.DEFINE_integer("mask_hidden_dim", config.hidden_dim, "中间节点维度.")
tf.app.flags.DEFINE_integer("mask_use_cpu_num", config.use_cpu_num, "限定使用cpu的个数")
tf.app.flags.DEFINE_integer("mask_epoch", config.epoch, "epoch次数")
tf.app.flags.DEFINE_string("mask_summary_write_dir", config.summary_write_dir, "训练数据过程可视化文件保存地址")
tf.app.flags.DEFINE_string("mask_train_dir", config.train_dir, "训练数据的路径")
tf.app.flags.DEFINE_string("mask_dev_dir", config.dev_dir, "验证数据文件路径")
tf.app.flags.DEFINE_string("mask_test_dir", config.test_dir, "测试数据文件路径")
tf.app.flags.DEFINE_string("mask_model_dir", config.model_dir, "模型保存路径")
tf.app.flags.DEFINE_boolean('mask_use Encoder2Decoder',False,'')
tf.app.flags.DEFINE_string("mask_mod", "infer_dev", "默认为训练")  # true for prediction
tf.app.flags.DEFINE_string('mask_model_mode', config.model_mode, '模型类型')
tf.app.flags.DEFINE_boolean('mask_use_auto_buckets',config.use_auto_buckets,'是否使用自动桶')
tf.app.flags.DEFINE_string('mask_only_mode','intent','执行哪种单一任务')
FLAGS = tf.app.flags.FLAGS


def get_sent_mask(sent_ids, entity_ids):
    sent_mask = np.zeros_like(sent_ids, dtype=np.float32)
    for i in range(sent_ids.shape[0]):
        for j in range(sent_ids.shape[1]):
            if sent_ids[i, j] > 0 and sent_ids[i, j] not in entity_ids:
                sent_mask[i, j] = 1.0
            elif sent_ids[i, j] > 0 and sent_ids[i, j] in entity_ids:
                sent_mask[i, j] = 1.5
    return sent_mask


class LstmRl(object):

    def __init__(self,scope):
        self.scope = scope
        with tf.variable_scope(name_or_scope=scope):
            with tf.device('/gpu:%s'%gpu_id):
                self.dd = Intent_Slot_Data(train_path=base_path+"/corpus_data/train_out_char.txt",
                                      test_path=base_path+"/corpus_data/dev_out_char.txt",
                                      dev_path=base_path+"/corpus_data/dev_out_char.txt", batch_size=FLAGS.mask_batch_size,
                                      max_length=FLAGS.mask_max_len, flag="train_new",
                                      use_auto_bucket=FLAGS.mask_use_auto_buckets)

                self.id2intent = self.dd.id2intent
                self.intent_num = len(self.id2intent)
                self.word_num = self.dd.vocab_num
                word_vocab = self.dd.vocab
                self.entity_id = []
                for k, v in word_vocab.items():
                    if k in entity_dict.keys():
                        self.entity_id.append(v)


                self.__build_model__()


    def __build_model__(self):
        with tf.device('/device:GPU:%s'%gpu_id):

            self.sent_word = tf.placeholder(shape=(None, FLAGS.mask_max_len), dtype=tf.int32,name='sent_word')
            self.sent_len = tf.placeholder(shape=(None,), dtype=tf.int32,name='sent_len')
            self.sent_mask = tf.placeholder(shape=(None, FLAGS.mask_max_len), dtype=tf.float32,name='sent_mask')
            self.sent_action=tf.placeholder(shape=(None,),dtype=tf.int32,name='sent_action')
            self.dropout=tf.placeholder(dtype=tf.float32)
            self.intent_y = tf.placeholder(shape=(None, self.intent_num), dtype=tf.int32,name='intent_y')

            self.target_action=tf.placeholder(shape=(None,),dtype=tf.float32,name='target_action')

            sent_action_emb=embedding(self.sent_action,self.intent_num+1,50,'sent_action_emb')


            sent_emb = embedding(self.sent_word, self.word_num, FLAGS.mask_embedding_dim, 'sent_emb')
            sent_emb=tf.nn.dropout(sent_emb,self.dropout)

            label_emb = tf.Variable(tf.random_uniform(shape=(self.intent_num, 300),maxval=1.0,minval=-1.0, dtype=tf.float32), trainable=True)
            sen_enc = sent_encoder(sent_word_emb=sent_emb, hidden_dim=FLAGS.mask_hidden_dim, num=FLAGS.mask_max_len,
                                   sequence_length=self.sent_len, name='sent_enc',dropout=self.dropout)


            sent_attention = self_attention(sen_enc, self.sent_mask)
            stack_sent_enc = tf.stack([tf.concat((ele, sent_attention), 1) for ele in sen_enc], 1,name='stack_sent_enc')

            stack_sent_enc=tf.nn.dropout(stack_sent_enc,self.dropout)

            out = label_sent_attention(stack_sent_enc, label_emb, self.sent_mask)

            out=tf.concat((out,sent_action_emb),1)

            logit = output_layers(out, self.intent_num, name='out_layers', reuse=False)
            self.soft_logit = tf.nn.softmax(logit, 1,name='mask_soft_logit')


            max_logit=tf.reduce_max(tf.multiply(self.soft_logit,tf.cast(self.intent_y,tf.float32)),1)

            # loss=focal_loss(self.soft_logit,tf.cast(self.intent_y,tf.float32))

            # class_y = tf.constant(name='class_y', shape=[self.intent_num, self.intent_num], dtype=tf.float32,
            #                       value=np.identity(self.intent_num), )
            #
            # logit_label = output_layers(label_emb, self.intent_num, name='out_layers', reuse=True)
            # label_loss = tf.losses.softmax_cross_entropy(onehot_labels=class_y, logits=logit_label)
            #
            # loss = tf.losses.softmax_cross_entropy(onehot_labels=self.intent_y, logits=logit)
            #
            self.losses = tf.squared_difference(max_logit, self.target_action)
            self.loss = tf.reduce_mean(self.losses)

            # self.loss = 0.7 * loss + 0.3 * label_loss
            # ss=tf.concat((sent_attention,tf.stack_sent_enc),1)
            # cosin=cosin_com(ss,label_emb,intent_num)

            # loss, soft_logit=loss_function(cosin,intent_y)

            # loss,soft_logit=loss_function(cosin,intent_y)
            # ss=tf.concat((sent_attention,sen_enc[0]),1)
            # logit=tf.layers.dense(ss,intent_num)
            # soft_logit=tf.nn.softmax(logit,1)
            # loss=focal_loss(soft_logit,tf.cast(intent_y,tf.float32))
            # # loss=tf.losses.softmax_cross_entropy(intent_y,logit)
            # #
            self.optimizer = tf.train.AdamOptimizer(FLAGS.mask_learning_rate).minimize(self.loss)



    def update(self,sess,state,action,action_target):


        sent_array, sent_vec,sent_action = [], [] ,[]
        for ele in state:
            sent_array.extend(ele['sent_array'])
            sent_vec.extend(ele['sent_vec'])
            sent_action.extend([ele['sent_action']])
        sent_array = np.array(sent_array)
        sent_vec = np.array(sent_vec)
        sent_action=np.array(sent_action)

        sent_array_mask=get_sent_mask(sent_array,self.entity_id)

        intent_label=np.zeros(shape=(sent_array.shape[0],self.intent_num))

        for index,ele in enumerate(action):
            intent_label[index][ele]=1
        action_target=np.array(action_target)
        soft_logit_, loss_, _ = sess.run([self.soft_logit, self.loss, self.optimizer],
                                         feed_dict={self.sent_word: sent_array,
                                                    self.sent_len:sent_vec ,
                                                    self.intent_y: intent_label,
                                                    self.sent_mask: sent_array_mask,
                                                    self.dropout:FLAGS.mask_keep_dropout,
                                                    self.target_action:action_target,
                                                    self.sent_action:sent_action
                                                    })

        return loss_

    def predict(self,sess,state):
        if isinstance(state,list):
            sent_array,sent_vec,sent_action=[],[],[]
            for ele in state:
                sent_array.extend(ele['sent_array'])
                sent_vec.extend(ele['sent_vec'])
                sent_action.extend([ele['sent_action']])

            sent_array=np.array(sent_array)
            sent_vec=np.array(sent_vec)
            sent_action=np.array(sent_action
                                 )
            sent_mask = get_sent_mask(sent_array, self.entity_id)

            intent_logit = sess.run(self.soft_logit, feed_dict={self.sent_word: sent_array,
                                                                self.sent_len: sent_vec,
                                                                self.sent_mask:sent_mask,
                                                                self.dropout:1.0,
                                                                self.sent_action:sent_action})

            return intent_logit

        elif isinstance(state,dict):
            sent_array=state['sent_array']
            sent_vec=state['sent_vec']
            sent_action=np.array([state['sent_action']])
            sent_mask = get_sent_mask(sent_array, self.entity_id)

            intent_logit = sess.run(self.soft_logit, feed_dict={self.sent_word: sent_array,
                                                                self.sent_len: sent_vec,
                                                                self.sent_mask: sent_mask,
                                                                self.dropout: 1.0,
                                                                self.sent_action:sent_action})
            return intent_logit

    def dev_predict(self,sess):
        '''
        dev 预测结果
        :param sess:
        :return:
        '''
        dev_sent, dev_slot, dev_intent, dev_rel_len, dev_index = self.dd.get_dev()
        dev_sent_mask=get_sent_mask(dev_sent,self.entity_id)

        dev_sent_action=np.ones_like(dev_rel_len)
        dev_sent_action=dev_sent_action*self.intent_num

        intent_logit = sess.run(self.soft_logit, feed_dict={self.sent_word: dev_sent,
                                                            self.sent_len: dev_rel_len,
                                                            self.sent_mask: dev_sent_mask,
                                                            self.dropout: 1.0,
                                                            self.sent_action: dev_sent_action})
        pred=np.argmax(intent_logit,1)
        true=np.argmax(dev_intent,1)
        dev_acc=float(np.equal(pred,true).sum())/float(len(pred))

        train_sent, train_slot, train_intent, train_rel_len, train_index = self.dd.get_train()
        train_sent_mask = get_sent_mask(train_sent, self.entity_id)

        train_sent_action = np.ones_like(train_rel_len)
        train_sent_action = train_sent_action * self.intent_num

        train_intent_logit = sess.run(self.soft_logit, feed_dict={self.sent_word: train_sent,
                                                            self.sent_len: train_rel_len,
                                                            self.sent_mask: train_sent_mask,
                                                            self.dropout: 1.0,
                                                            self.sent_action: train_sent_action})
        pred = np.argmax(train_intent_logit, 1)
        true = np.argmax(train_intent, 1)
        train_acc = float(np.equal(pred, true).sum()) / float(len(pred))
        return dev_acc,train_acc

def make_epsilon_greedy_policy(estimator, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function approximator and epsilon.

    Args:
        estimator: An estimator that returns q values for a given state
        nA: Number of actions in the environment.

    Returns:
        A function that takes the (sess, observation, epsilon) as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.

    """
    def policy_fn(sess, observation, epsilon):
        A = np.ones(nA, dtype=float) * epsilon / nA

        q_values = estimator.predict(sess=sess, state=observation)[0]
        best_action = np.argmax(q_values)
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn


class ENV(object):

    def __init__(self):
        self.dd = Intent_Slot_Data(train_path=base_path + "/corpus_data/train_out_char.txt",
                                   test_path=base_path + "/corpus_data/dev_out_char.txt",
                                   dev_path=base_path + "/corpus_data/dev_out_char.txt",
                                   batch_size=1,
                                   max_length=FLAGS.mask_max_len, flag="train_new",
                                   use_auto_bucket=FLAGS.mask_use_auto_buckets)

        self.id2intent = self.dd.id2intent
        self.intent_num = len(self.id2intent)
        self.word_num = self.dd.vocab_num
        word_vocab = self.dd.vocab
        self.entity_id = []
        for k, v in word_vocab.items():
            if k in entity_dict.keys():
                self.entity_id.append(v)

    def get_init_state(self):
        '''
        获得初始的state
        :return:
        '''
        state=self.dd.next_batch()
        return {'sent_array':state[0],'sent_vec':state[3],'label':state[2],'sent_action':self.intent_num}

    def step(self,action,state,turn,max_turn):
        '''
        根据反馈的action获取下一个state
        :param action:
        :return:
        '''
        done=False
        label=state['label']


        next_state={'sent_array':state['sent_array'],'sent_vec':state['sent_vec'],'sent_action':None,'label':state['label']}
        if turn<max_turn :
            if np.argmax(action)!=np.argmax(label):
                reward=-2
                next_state['sent_action']=np.argmax(action)
            else:
                reward=2
                next_state['sent_action']=np.argmax(action)
            next_state['sent_action']=np.argmax(action)

        elif turn==max_turn:
            done = True
            if np.argmax(action)!=np.argmax(label):
                reward=-5
                # sent_array[0,sent_vec[0]]=np.argmax(action)
                next_state['sent_action']=np.argmax(action)
                # next_state['sent_array']=sent_array
            else:
                reward=5
                # sent_array[0,sent_vec[0]]=np.argmax(action)
                next_state['sent_action']=np.argmax(action)
                # next_state['sent_array']=sent_array

        return next_state,reward,done

    def batch_step(self,batch_action,batch_state,turn,max_turn):

        batch_next_state=[]
        batch_reward=[]
        batch_done=[]
        for action,state in zip(batch_action,batch_state):
            next_state,reward,done=self.step(action,state,turn,max_turn)
            batch_next_state.append(next_state)
            batch_reward.append(reward)
            batch_done.append(done)

        batch_next_state=np.array(batch_next_state)
        batch_reward=np.array(batch_reward)
        batch_done=np.array(batch_done)
        return batch_next_state,batch_reward,batch_done

    def get_train_state(self):
        '''
        获取 train集的 state
        :return:
        '''
        train_sent, train_slot, train_intent, train_rel_len, train_index = self.dd.get_train()
        train_sent_action = np.ones_like(train_rel_len)
        train_sent_action = train_sent_action * self.intent_num
        res=[]
        for i in range(train_sent.shape[0]):
            res.append({'sent_array':train_sent[i],'sent_vec':train_rel_len[i],'label':train_intent[i],'sent_action':train_sent_action[i]})
        return res

    def get_dev_state(self):
        '''
        获取dev的init_state
        :return:
        '''
        dev_sent, dev_slot, dev_intent, dev_rel_len, dev_index = self.dd.get_dev()
        dev_sent_action = np.ones_like(dev_rel_len)
        dev_sent_action = dev_sent_action * self.intent_num

        return {'sent_array': dev_sent, 'sent_vec': dev_rel_len, 'label': dev_intent,
                'sent_action': dev_sent_action}


def copy_model_parameters(sess, estimator1, estimator2):
    """
    Copies the model parameters of one estimator to another.

    Args:
      sess: Tensorflow session instance
      estimator1: Estimator to copy the paramters from
      estimator2: Estimator to copy the parameters to
    """
    e1_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator1.scope)]
    e1_params = sorted(e1_params, key=lambda v: v.name)
    e2_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator2.scope)]
    e2_params = sorted(e2_params, key=lambda v: v.name)

    update_ops = []
    for e1_v, e2_v in zip(e1_params, e2_params):
        op = e2_v.assign(e1_v)
        update_ops.append(op)

    sess.run(update_ops)



def deep_q_learning(sess,
                    env,
                    q_estimator,
                    target_estimator,
                    num_episodes,
                    experiment_dir,
                    replay_memory_size=50,
                    replay_memory_init_size=50,
                    update_target_estimator_every=50,
                    discount_factor=0.5,
                    epsilon_start=0.5,
                    epsilon_end=0.1,
                    epsilon_decay_steps=500,
                    batch_size=32,
                    record_video_every=50,
                    max_turn=4):
    """
    Q-Learning algorithm for off-policy TD control using Function Approximation.
    Finds the optimal greedy policy while following an epsilon-greedy policy.

    Args:
        sess: Tensorflow Session object
        env: OpenAI environment
        q_estimator: Estimator object used for the q values
        target_estimator: Estimator object used for the targets
        state_processor: A StateProcessor object
        num_episodes: Number of episodes to run for
        experiment_dir: Directory to save Tensorflow summaries in
        replay_memory_size: Size of the replay memory
        replay_memory_init_size: Number of random experiences to sampel when initializing
          the reply memory.
        update_target_estimator_every: Copy parameters from the Q estimator to the
          target estimator every N steps
        discount_factor: Gamma discount factor
        epsilon_start: Chance to sample a random action when taking an action.
          Epsilon is decayed over time and this is the start value
        epsilon_end: The final minimum value of epsilon after decaying is done
        epsilon_decay_steps: Number of steps to decay epsilon over
        batch_size: Size of batches to sample from the replay memory
        record_video_every: Record a video every N episodes

    Returns:
        An EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """
    print('开始DQN')
    #有效的action 列表
    VALID_ACTIONS=list(range(q_estimator.intent_num+1))
    stats={}

    # The replay memory
    # 经验池
    replay_memory = []
    # Keeps track of useful statistics
    # stats = plotting.EpisodeStats(
    #     episode_lengths=np.zeros(num_episodes),
    #     episode_rewards=np.zeros(num_episodes))

    # Create directories for checkpoints and summaries
    # checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
    # checkpoint_path = os.path.join(checkpoint_dir, "model")
    # monitor_path = os.path.join(experiment_dir, "monitor")

    # if not os.path.exists(checkpoint_dir):
    #     os.makedirs(checkpoint_dir)
    # if not os.path.exists(monitor_path):
    #     os.makedirs(monitor_path)

    # saver = tf.train.Saver()
    # sess.run(tf.global_variables_initializer())
    # Load a previous checkpoint if we find one
    # latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    # if latest_checkpoint:
    #     print("Loading model checkpoint {}...\n".format(latest_checkpoint))
    #     saver.restore(sess, latest_checkpoint)

    #
    # # The epsilon decay schedule

    sess.run(tf.global_variables_initializer())
    epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)
    print('epsion',epsilons.shape)
    # The policy we're following
    policy = make_epsilon_greedy_policy(
        q_estimator,
        len(VALID_ACTIONS))

    # Populate the replay memory with initial experience
    print("填充 经验池 ...")
    state = env.get_init_state() #获取初始state:{sent_arry,sent_vec,label,sent_action}

    turn=0
    for i in range(replay_memory_init_size):

        action_probs = policy(sess=sess, observation=state, epsilon=epsilons[min(0, epsilon_decay_steps-1)])
        action=np.argmax(action_probs)
        print('action:{}'.format(action))
        # action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
        next_state, reward, done= env.step(action_probs,state,turn,max_turn)
        turn += 1
        replay_memory.append((state, action, reward, next_state, done))
        if done or turn>max_turn:
            state = env.get_init_state()
            turn=0
        else:
            state = next_state
    #
    # # Record videos
    # # Use the gym env Monitor wrapper
    # env = Monitor(env,
    #               directory=monitor_path,
    #               resume=True,
    #               video_callable=lambda count: count % record_video_every ==0)
    #
    total_t = 0
    for i_episode in range(num_episodes):

        # Reset the environment
        state = env.get_init_state()


        # One step in the environment
        turn=0
        for t in range(len(replay_memory)):
            # Epsilon for this time step

            epsilon = epsilons[min(total_t, epsilon_decay_steps-1)]
            print('epsilon',epsilon)

            # Add epsilon to Tensorboard
            # episode_summary = tf.Summary()
            # episode_summary.value.add(simple_value=epsilon, tag="epsilon")
            # q_estimator.summary_writer.add_summary(episode_summary, total_t)
            #
            #Maybe update the target estimator

            if total_t % update_target_estimator_every == 0:
                copy_model_parameters(sess, q_estimator, target_estimator)
                print("\nCopied model parameters to target network.")

            # Print out which step we're on, useful for debugging.

            # sys.stdout.flush()
    #
    #         # Take a step
            action_probs = policy(sess, state, epsilon)
            # action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            action=np.argmax(action_probs)

            print('\n','###### predict',state['label'],action)
            next_state, reward, done = env.step(action_probs,state,turn,max_turn)
            turn+=1
            if turn>max_turn:
                turn=0
                state = env.get_init_state()

            # If our replay memory is full, pop the first element
            if len(replay_memory) == replay_memory_size:
                replay_memory.pop(0)

            # Save transition to replay memory
            replay_memory.append((state, action, reward, next_state, done))

            # Update statistics
            if i_episode in stats:
                w=stats[i_episode]['episode_rewards']
                w+=reward
                stats[i_episode]['episode_rewards']=w
                stats[i_episode]['episode_lengths'] = t
            else:
                stats[i_episode]={"episode_rewards":reward,"episode_lengths":t}

            # Sample a minibatch from the replay memory
            for _ in range(1):
                np.random.shuffle(replay_memory)
                samples=replay_memory[:batch_size]
                # samples = random.sample(replay_memory, batch_size)
                states_batch, action_batch, reward_batch, next_states_batch, done_batch=[],[],[],[],[]
                for ele in samples:
                    states_batch.append(ele[0])
                    action_batch.append(ele[1])
                    reward_batch.append(ele[2])
                    next_states_batch.append(ele[3])
                    done_batch.append(ele[4])

                print('action_batch:{}'.format(action_batch))
                # states_batch, action_batch, reward_batch, next_states_batch, done_batch = map(np.array, zip(*samples))

                # Calculate q values and targets (Double DQN)
                q_values_next = q_estimator.predict(sess, next_states_batch)
                best_actions = np.argmax(q_values_next, axis=1)
                q_values_next_target = target_estimator.predict(sess, next_states_batch)
                targets_batch = reward_batch + np.invert(done_batch).astype(np.float32) * \
                    discount_factor * q_values_next_target[np.arange(batch_size), best_actions]

                # Perform gradient descent update
                states_batch = np.array(states_batch)
                loss = q_estimator.update(sess, states_batch, action_batch, targets_batch)
                print("\rStep {} ({}) @ Episode {}/{}, loss: {}".format(
                    t, total_t, i_episode + 1, num_episodes, loss), end="")
                print('targets_batch:{}'.format(targets_batch),'\n')
                num=float(sum([1 for e in reward_batch if e>=1.0]))/float(len(reward_batch))
                print('success:{}'.format(num))
                print('done:{}'.format(done_batch))
                if done:
                    break

            state = next_state
            total_t += 1

        turn=0
        train_state=env.get_train_state()
        dev_state=env.get_dev_state()
        for i in range(max_turn):
            train_action_probs = policy(sess=sess, observation=train_state, epsilon=epsilons[min(0, epsilon_decay_steps - 1)])
            train_next_state, train_reward, train_done = env.step(train_action_probs, train_state, i, max_turn)

            dev_action_probs = policy(sess=sess, observation=dev_state,
                                       epsilon=epsilons[min(0, epsilon_decay_steps - 1)])
            dev_next_state, dev_reward, dev_done = env.step(dev_action_probs, dev_state, i, max_turn)

            train_state=train_next_state
            dev_state=dev_next_state


        train_logit=q_estimator.predict(sess=sess,state=train_state)
        dev_logit=q_estimator.predict(sess=sess,state=dev_state)

        print('train_logit:{}'.format(train_logit))



        # dev_acc,train_acc=q_estimator.dev_predict(sess)
        # print('#'*10,'dev_acc:{}  train:{}'.format(dev_acc,train_acc))

    #     # Add summaries to tensorboard
    #     episode_summary = tf.Summary()
    #     episode_summary.value.add(simple_value=stats.episode_rewards[i_episode], node_name="episode_reward", tag="episode_reward")
    #     episode_summary.value.add(simple_value=stats.episode_lengths[i_episode], node_name="episode_length", tag="episode_length")
    #     q_estimator.summary_writer.add_summary(episode_summary, total_t)
    #     q_estimator.summary_writer.flush()
    #
    #     yield total_t, plotting.EpisodeStats(
    #         episode_lengths=stats.episode_lengths[:i_episode+1],
    #         episode_rewards=stats.episode_rewards[:i_episode+1])
    #
    # env.monitor.close()
    # return stats
#

def main(_):
    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config) as sess:
        # sess.run(tf.global_variables_initializer())

        env = ENV()
        q_estimator = LstmRl(scope='q_estimator')
        target_estimator = LstmRl(scope='target_estimator')

        deep_q_learning(sess=sess,env=env,q_estimator=q_estimator,
                        target_estimator=target_estimator,
                        num_episodes=100,
                        experiment_dir=None)


if __name__ == '__main__':
    tf.app.run()
