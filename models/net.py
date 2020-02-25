import tensorflow as tf
import numpy as np
import pickle as pickle
import time

import utils.layers as layers
import utils.operations as op

from models.smn_net import smn_model
from models.dam_net import dam_model
from models.ioi_net import ioi_model
from models.esim_net import esim_model
from models.msn_net import msn_model


class Net(object):
    '''Add positional encoding(initializer lambda is 0),
       cross-attention, cnn integrated and grad clip by value.

    Attributes:
        conf: a configuration paramaters dict
        word_embedding_init: a 2-d array with shape [vocab_size+1, emb_size]
    '''
    def __init__(self, conf, is_train=False):
        self._graph = tf.Graph()
        self._conf = conf

        self.is_train = is_train
        self.cr_model = conf["cr_model"]

        if self._conf['word_emb_init'] is not None:
            print('loading word emb init')
            self._word_embedding_init = pickle.load(open(self._conf['word_emb_init'], 'rb'))
        else:
            self._word_embedding_init = None

    def build_graph(self):
        with self._graph.as_default():
            rand_seed = self._conf['rand_seed']
            tf.set_random_seed(rand_seed)

            #word embedding
            if self._word_embedding_init is not None:
                word_embedding_initializer = tf.constant_initializer(self._word_embedding_init)
            else:
                word_embedding_initializer = tf.random_normal_initializer(stddev=0.1)

            self._word_embedding = tf.get_variable(
                name='word_embedding',
                shape=[self._conf['vocab_size']+1, self._conf['emb_size']],
                dtype=tf.float32,
                initializer=word_embedding_initializer, trainable=True)

            batch_size = None
            #define placehloders
            self.turns1 = tf.placeholder(tf.int32, shape=[batch_size, self._conf["max_turn_num"], self._conf["max_turn_len"]], name="turns1")
            self.tt_turns_len1 = tf.placeholder(tf.int32, shape=[batch_size,], name="tt_turns_len1")
            self.every_turn_len1 = tf.placeholder(tf.int32, shape=[batch_size, self._conf["max_turn_num"]], name="every_turn_len1")
            self.response = tf.placeholder(tf.int32, shape=[batch_size, self._conf["max_turn_len"]], name="response")
            self.response_len = tf.placeholder(tf.int32, shape=[batch_size,], name="response_len")
            self.keep_rate = tf.placeholder(tf.float32, [], name="keep_rate")
            self.label = tf.placeholder(tf.float32, shape=[batch_size,])


            # ==================================== Building Model =============================
            print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())), "Starting build Model")

            if self.cr_model=="SMN":
                input_x = self.turns1
                input_y = self.response
                
                with tf.variable_scope('model_cr_smn'):
                    final_info_cr = smn_model(input_x, None, input_y, None, self._word_embedding, self.keep_rate, self._conf, x_len=self.every_turn_len1, y_len=self.response_len)
                    final_info_cr = tf.layers.dense(final_info_cr, 50, kernel_initializer=tf.contrib.layers.xavier_initializer())           

            # DAM
            elif self.cr_model=="DAM":
                input_x = self.turns1
                input_y = self.response
                
                with tf.variable_scope('model_cr_dam'):
                    final_info_cr = dam_model(input_x, None, input_y, None, self._word_embedding, self.keep_rate, self._conf, x_len=self.every_turn_len1, y_len=self.response_len)
                    final_info_cr = tf.layers.dense(final_info_cr, 50, kernel_initializer=tf.contrib.layers.xavier_initializer())           

            # MSN
            elif self.cr_model=="MSN":
                input_x = self.turns1
                input_y = self.response
                
                with tf.variable_scope('model_cr_msn'):
                    final_info_cr, self.final_score  = msn_model(input_x, None, input_y, None, self._word_embedding, self.keep_rate, self._conf, x_len=self.every_turn_len1, y_len=self.response_len)
                    final_info_cr= tf.layers.dense(final_info_cr, 50, kernel_initializer=tf.contrib.layers.xavier_initializer())    

            # ESIM
            elif self.cr_model=="ESIM":
                input_x = tf.reshape(self.turns1, [-1, self._conf["max_turn_num"]*self._conf["max_turn_len"]])
                input_x_mask = tf.sequence_mask(self.every_turn_len1, self._conf["max_turn_len"])
                input_x_mask = tf.reshape(input_x_mask, [-1, self._conf["max_turn_num"]*self._conf["max_turn_len"]])
                input_y = self.response
                input_y_mask = tf.sequence_mask(self.response_len, self._conf["max_turn_len"])
                
                with tf.variable_scope('model_cr_esim'):
                    final_info_cr = esim_model(input_x, input_x_mask, input_y, input_y_mask, self._word_embedding, self.keep_rate)
                    final_info_cr = tf.layers.dense(final_info_cr, 50, kernel_initializer=tf.contrib.layers.xavier_initializer())

            elif self.cr_model=="IOI":
                input_x = tf.reshape(self.turns1, [-1, self._conf["max_turn_num"]*self._conf["max_turn_len"]])
                input_x_mask = tf.sequence_mask(self.every_turn_len1, self._conf["max_turn_len"])
                input_x_mask = tf.reshape(input_x_mask, [-1, self._conf["max_turn_num"]*self._conf["max_turn_len"]])

                input_y = self.response
                input_y_mask = tf.sequence_mask(self.response_len, self._conf["max_turn_len"])

                with tf.variable_scope('model_cr_ioi'):
                    final_info_cr, final_info_cr_ioi = ioi_model(input_x, input_x_mask, input_y, input_y_mask, self._word_embedding, self.keep_rate, self._conf)
                    final_info_cr = tf.layers.dense(final_info_cr, 50, kernel_initializer=tf.contrib.layers.xavier_initializer())


            # ==================================== Calculating Model =============================
            self.trainops = {"cr": dict(),}
            loss_input = final_info_cr

            for loss_type in ["cr", ]:
                with tf.variable_scope('loss_'+loss_type):
                    if self.cr_model=="IOI":
                        loss_list = []
                        logits_list = []
                        for i,j in enumerate(final_info_cr_ioi):
                            with tf.variable_scope("loss"+str(i)): loss_per, logits_per = layers.loss(j, self.label)
                            loss_list.append(loss_per)
                            logits_list.append(logits_per)
                        self.trainops[loss_type]["loss"] = sum([((idx+1)/7.0)*item for idx, item in enumerate(loss_list)])
                        self.trainops[loss_type]["logits"] = sum(logits_list)
                    else:
                        self.trainops[loss_type]["loss"], self.trainops[loss_type]["logits"] = layers.loss(final_info_cr,
                                                                                                           self.label)

                    self.trainops[loss_type]["global_step"] = tf.Variable(0, trainable=False)
                    initial_learning_rate = self._conf['learning_rate']
                    self.trainops[loss_type]["learning_rate"] = tf.train.exponential_decay(
                        initial_learning_rate,
                        global_step=self.trainops[loss_type]["global_step"],
                        decay_steps=self._conf["decay_step"],
                        decay_rate=self._conf["decay_rate"],
                        staircase=True)

                    Optimizer = tf.train.AdamOptimizer(self.trainops[loss_type]["learning_rate"])
                    self.trainops[loss_type]["optimizer"] = Optimizer.minimize(self.trainops[loss_type]["loss"])

                    self.trainops[loss_type]["grads_and_vars"] = Optimizer.compute_gradients(self.trainops[loss_type]["loss"])

                    self.trainops[loss_type]["capped_gvs"] = [(tf.clip_by_value(grad, -5, 5), var) for grad, var in self.trainops[loss_type]["grads_and_vars"] if grad!=None]
                    self.trainops[loss_type]["g_updates"] = Optimizer.apply_gradients(
                         self.trainops[loss_type]["capped_gvs"],
                        global_step=self.trainops[loss_type]["global_step"])


            self.all_variables = tf.global_variables()
            self.init = tf.global_variables_initializer()
            self.saver = tf.train.Saver(max_to_keep = self._conf["max_to_keep"])
            
            self.all_operations = self._graph.get_operations()

        return self._graph

