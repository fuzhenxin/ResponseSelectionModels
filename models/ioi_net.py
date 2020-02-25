

import tensorflow as tf
from models.ioi_layers import *


def ioi_model(input_x, input_x_mask, input_y, input_y_mask, word_emb, keep_rate, conf, max_turn=None):

    embed_dim = conf["emb_size"]
    max_word_len = conf["max_turn_len"]
    max_word_len_a = input_y.shape[-1]
    max_turn = conf["max_turn_num"]
    num_layer = conf["ioi_layer_num"]

    context = input_x
    context_mask = tf.to_float(input_x_mask)
    response = input_y
    response_mask = tf.to_float(input_y_mask)

    expand_response_mask = tf.tile(tf.expand_dims(response_mask, 1), [1, max_turn, 1]) 
    expand_response_mask = tf.reshape(expand_response_mask, [-1, max_word_len_a])  
    parall_context_mask = tf.reshape(context_mask, [-1, max_word_len])  


    context_embeddings = tf.nn.embedding_lookup(word_emb, context)  
    response_embeddings = tf.nn.embedding_lookup(word_emb, response)  
    context_embeddings = tf.layers.dropout(context_embeddings, rate=1.0-keep_rate)
    response_embeddings = tf.layers.dropout(response_embeddings, rate=1.0-keep_rate)
    context_embeddings = tf.multiply(context_embeddings, tf.expand_dims(context_mask, axis=-1))  
    response_embeddings = tf.multiply(response_embeddings, tf.expand_dims(response_mask, axis=-1)) 


    expand_response_embeddings = tf.tile(tf.expand_dims(response_embeddings, 1), [1, max_turn, 1, 1]) 
    expand_response_embeddings = tf.reshape(expand_response_embeddings, [-1, max_word_len_a, embed_dim]) 
    parall_context_embeddings = tf.reshape(context_embeddings, [-1, max_word_len, embed_dim])
    context_rep, response_rep = parall_context_embeddings, expand_response_embeddings
    
    losses_list = []
    y_pred_list = []
    logits_list=[]
    fea_list = []
    for k in range(num_layer):
        inter_feat_collection = []
        with tf.variable_scope('dense_interaction_{}'.format(k)): 
            # get the self rep
            context_self_rep = self_attention(context_rep, context_rep, embed_dim, 
                                                query_masks=parall_context_mask, 
                                                key_masks=parall_context_mask, 
                                                num_blocks=1, num_heads=1, 
                                                dropout_rate=1.0-keep_rate,
                                                use_residual=True, use_feed=True, 
                                                scope='context_self_attention')[1]  # [batch*turn, len_utt, embed_dim, 2]
            response_self_rep = self_attention(response_rep, response_rep, embed_dim, 
                                                query_masks=expand_response_mask, 
                                                key_masks=expand_response_mask, 
                                                num_blocks=1, num_heads=1, 
                                                dropout_rate=1.0-keep_rate, 
                                                use_residual=True, use_feed=True, 
                                                scope='response_self_attention')[1]  # [batch*turn, len_res, embed_dims, 2]

            # get the attended rep
            context_cross_rep = self_attention(context_rep, response_rep, embed_dim, 
                                                query_masks=parall_context_mask, 
                                                key_masks=expand_response_mask, 
                                                num_blocks=1, num_heads=1, 
                                                dropout_rate=1.0-keep_rate, 
                                                use_residual=True, use_feed=True, 
                                                scope='context_cross_attention')[1]  # [batch*turn, len_utt, embed_dim]

            response_cross_rep = self_attention(response_rep, context_rep, embed_dim, 
                                                query_masks=expand_response_mask, 
                                                key_masks=parall_context_mask, 
                                                num_blocks=1, num_heads=1, 
                                                dropout_rate=1.0-keep_rate, 
                                                use_residual=True, use_feed=True, 
                                                scope='response_cross_attention')[1]  # [batch*turn, len_res, embed_dim]


            context_inter_feat_multi = tf.multiply(context_rep, context_cross_rep)
            response_inter_feat_multi = tf.multiply(response_rep, response_cross_rep)


            context_concat_rep = tf.concat([context_rep, context_self_rep, context_cross_rep, context_inter_feat_multi], axis=-1) 
            response_concat_rep = tf.concat([response_rep, response_self_rep, response_cross_rep, response_inter_feat_multi], axis=-1)


            context_concat_dense_rep = tf.layers.dense(context_concat_rep, embed_dim, activation=tf.nn.relu, use_bias=True, 
                                                                name='context_dense1') 
            context_concat_dense_rep = tf.layers.dropout(context_concat_dense_rep, rate=1.0-keep_rate)

            response_concat_dense_rep = tf.layers.dense(response_concat_rep, embed_dim,  activation=tf.nn.relu, use_bias=True, 
                                                                name='response_dense1') 
            response_concat_dense_rep = tf.layers.dropout(response_concat_dense_rep, rate=1.0-keep_rate)


            inter_feat = tf.matmul(context_rep, tf.transpose(response_rep, perm=[0, 2, 1])) / tf.sqrt(tf.to_float(embed_dim))
            inter_feat_self = tf.matmul(context_self_rep, tf.transpose(response_self_rep, perm=[0, 2, 1])) / tf.sqrt(tf.to_float(embed_dim))
            inter_feat_cross = tf.matmul(context_cross_rep, tf.transpose(response_cross_rep, perm=[0, 2, 1])) / tf.sqrt(tf.to_float(embed_dim))


            inter_feat_collection.append(inter_feat)
            inter_feat_collection.append(inter_feat_self)
            inter_feat_collection.append(inter_feat_cross)

            if k==0:
                context_rep = tf.add(context_rep, context_concat_dense_rep)
                response_rep = tf.add(response_rep, response_concat_dense_rep)
            else:
                context_rep = tf.add_n([parall_context_embeddings, context_rep, context_concat_dense_rep])
                response_rep = tf.add_n([expand_response_embeddings, response_rep, response_concat_dense_rep])

            context_rep = normalize(context_rep, scope='layer_context_normalize') 
            response_rep = normalize(response_rep, scope='layer_response_normalize') 

            context_rep = tf.multiply(context_rep, tf.expand_dims(parall_context_mask, axis=-1))
            response_rep = tf.multiply(response_rep, tf.expand_dims(expand_response_mask, axis=-1))

            matching_feat = tf.stack(inter_feat_collection, axis=-1)
            #matrix_trans = tf.reshape(matching_feat, [-1, max_turn, max_word_len, max_word_len, len(inter_feat_collection)])  # embed_dim

        with tf.variable_scope('CRNN_{}'.format(k)): 
            conv1 = tf.layers.conv2d(matching_feat, filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                        activation=tf.nn.relu, name='conv1')
            pool1 = tf.layers.max_pooling2d(conv1, (3, 3), strides=(3, 3), padding='same', name='max_pooling1')

            conv2 = tf.layers.conv2d(pool1, filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                        activation=tf.nn.relu, name='conv2')
            pool2 = tf.layers.max_pooling2d(conv2, (3, 3), strides=(3, 3), padding='same', name='max_pooling2')                    
            flatten = tf.contrib.layers.flatten(pool2)
            flatten = tf.layers.dropout(flatten, rate=1.0-keep_rate)

            matching_vector = tf.layers.dense(flatten, embed_dim,
                                  kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                  activation=tf.tanh, name='dense_feat') 
            matching_vector = tf.reshape(matching_vector, [-1, max_turn, embed_dim]) 

            final_gru_cell = tf.contrib.rnn.GRUCell(embed_dim, kernel_initializer=tf.orthogonal_initializer())
            _, last_hidden = tf.nn.dynamic_rnn(final_gru_cell, matching_vector, dtype=tf.float32, scope='final_GRU')  # TODO: check time_major
            fea_list.append(last_hidden)
            #logits = tf.layers.dense(last_hidden, 2, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='final_v')


    last_hidden = tf.concat(fea_list, axis=-1)
    #tf.layers.dense(last_hidden, 50, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='final_v')
    return last_hidden, fea_list

