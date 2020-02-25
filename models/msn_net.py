

import tensorflow as tf
import utils.layers as layers
import utils.operations as op

def msn_model(input_x, input_x_mask, input_y, input_y_mask, word_emb, keep_rate, conf, x_len=None, y_len=None):


    turn_num = input_x.shape[1]
    sent_len = conf["max_turn_len"]
    emb_dim = conf["emb_size"]
    is_mask = False
    is_layer_norm = False
    # init = None
    init = tf.contrib.layers.xavier_initializer()

    init1 = tf.contrib.layers.xavier_initializer()
    init1 = tf.random_uniform_initializer(0.0, 1.0)

    Hr = tf.nn.embedding_lookup(word_emb, input_y) # bs len emb
    Hu = tf.nn.embedding_lookup(word_emb, input_x) # bs turn len emb

    x_len = tf.reshape(x_len, [-1])
    y_len = tf.tile(tf.expand_dims(y_len, axis=1), [1, turn_num])
    y_len = tf.reshape(y_len, [-1])


    with tf.variable_scope('enc', reuse=tf.AUTO_REUSE):

        # context selector
        context_ = tf.reshape(Hu, [-1, sent_len, emb_dim])


        context_ = layers.block(context_, context_, context_, Q_lengths=x_len, K_lengths=x_len, is_mask=is_mask,is_layer_norm=is_layer_norm, init=init)
        context_ = tf.reshape(context_, [-1, turn_num, sent_len, emb_dim])

        W_word = tf.get_variable(name='w_word', shape=[emb_dim, emb_dim, turn_num], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer()) # 200 200 10
        v = tf.get_variable(name='v', shape=[turn_num, 1], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer()) # 10 1

        ss = []
        for hop_index in [1,2,3]:

            kk = Hu[:, turn_num-hop_index:, :, :]

            kk = tf.reduce_mean(kk, axis=1)
            kk = layers.block(kk, kk, kk, is_mask=False, is_layer_norm=is_layer_norm, init=init)

            # kk context_

            A = tf.einsum("blrm,mdh,bud->blruh", context_, W_word, kk) / tf.sqrt(200.0)
            A = tf.einsum("blruh,hp->blrup", A, v) # bs turn_num sent_len sent_len 1
            A = tf.squeeze(A, [4,]) # bs turn_num sent_len sent_len

            A1 = tf.reduce_max(A, axis=2) # bs turn_num sent_len
            A2 = tf.reduce_max(A, axis=3) # bs turn_num sent_len
            a = tf.concat([A1, A2], axis=-1) # bs turn_num sent_len*2
            a = tf.layers.dense(a, 1, kernel_initializer=init1) # bs turn_num 1

            a = tf.squeeze(a, [2,]) # bs turn_num
            s1 = tf.nn.softmax(a, axis=1)

            # kk context_
            kk1 = tf.reduce_mean(kk, axis=1) # bs emb
            context1 = tf.reduce_mean(context_, axis=2) # bs turn emb
            norm1 = tf.norm(context1, axis=-1)
            norm2 = tf.norm(kk1, axis=-1, keepdims=True)
            # print(context1.shape) # bs 10 200
            # print(kk1.shape) # bs 200
            # print(norm1.shape) # bs 10
            # print(norm2.shape) # bs 1
            # exit()
            s2 = tf.einsum("bud,bd->bu", context1, kk1)/(1e-6 + norm1*norm2) # bs turn
            # print(s1.shape, s2.shape)
            # exit()
            s = 0.5*s1 + 0.5*s2
            ss.append(s)

        #s = tf.expand_dims(s, axis=-1)
        s = tf.stack(ss, axis=-1)

        s = tf.layers.dense(s, 1, kernel_initializer=init1) # bs turn_num 1
        s = tf.squeeze(s, [2,]) # bs turn_num

        if "douban" in conf["data_path"]:
            grmmar_score = 0.3
        else:
            grmmar_score = 0.5

        s_mask1 = tf.nn.sigmoid(s)
        s_mask = tf.math.greater(s_mask1, grmmar_score)
        s_mask = tf.cast(s_mask, tf.float32)
        final_score = [s, s_mask1]
        s = s*s_mask


        Hu = Hu*tf.expand_dims(tf.expand_dims(s, axis=-1), axis=-1)

        Hu = tf.reshape(Hu, [-1, sent_len, emb_dim])

        Hr = tf.tile(tf.expand_dims(Hr, axis=1), [1,turn_num,1,1])
        Hr = tf.reshape(Hr, [-1, sent_len, emb_dim])

        # UR Matching Hu Hr

        def distance(A, B, C, epsilon=1e-6):
            Ma = tf.einsum("bum,md,brd->bur", A, B, C)
            A_norm = tf.norm(A, axis=-1)
            C_norm = tf.norm(C, axis=-1)
            norm_score = tf.einsum("bu,br->bur", A_norm, C_norm) + epsilon
            # norm_score = tf.math.maximum(norm_score, 1.0)
            Mb = tf.einsum("bud,brd->bur", A, C) / norm_score
            return Ma, Mb, norm_score

        v1 = tf.get_variable(name='v1', shape=[emb_dim, emb_dim], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
        M1, M2, norm_score1 = distance(Hu, v1, Hr)

        with tf.variable_scope('enc11', reuse=tf.AUTO_REUSE): Hu1 = layers.block(Hu, Hu, Hu, Q_lengths=x_len, K_lengths=x_len, is_mask=is_mask, is_layer_norm=is_layer_norm, init=init)
        with tf.variable_scope('enc12', reuse=tf.AUTO_REUSE): Hr1 = layers.block(Hr, Hr, Hr, Q_lengths=y_len, K_lengths=y_len, is_mask=is_mask, is_layer_norm=is_layer_norm, init=init)
        v2 = tf.get_variable(name='v2', shape=[emb_dim, emb_dim], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
        M3, M4, norm_score2 = distance(Hu1, v2, Hr1)
        
        with tf.variable_scope('enc21', reuse=tf.AUTO_REUSE): Hu1 = layers.block(Hu, Hr, Hr, Q_lengths=x_len, K_lengths=y_len, is_mask=is_mask, is_layer_norm=is_layer_norm, init=init)
        with tf.variable_scope('enc22', reuse=tf.AUTO_REUSE): Hr1 = layers.block(Hr, Hu, Hu, Q_lengths=y_len, K_lengths=x_len, is_mask=is_mask, is_layer_norm=is_layer_norm, init=init)
        v3 = tf.get_variable(name='v3', shape=[emb_dim, emb_dim], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
        M5, M6, norm_score3 = distance(Hu1, v3, Hr1)

        final_score = [norm_score1, norm_score2, norm_score3]
        final_score = [M2, M4, M6]

        M = tf.stack([M1, M2, M3, M4, M5, M6], axis=1) # bs*turn 6 sent_len sent_len
        
        M = layers.CNN_MSN(M, init=init) # bs*turn 128
        M = tf.layers.dense(M, 300, activation=tf.nn.tanh, kernel_initializer=tf.contrib.layers.xavier_initializer(), name="dense1") # bs turn_num 1

        M = tf.reshape(M, [-1, turn_num, 300])

        gru = tf.contrib.rnn.GRUCell(300)
        M = tf.nn.dynamic_rnn(gru, M, dtype=tf.float32)
        final_info = M[0][:, -1, :]
        final_info = tf.layers.dropout(final_info, rate=1.0-keep_rate)

    return final_info, final_score
