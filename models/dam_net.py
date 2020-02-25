

import tensorflow as tf
import utils.layers as layers
import utils.operations as op

def dam_model(input_x, input_x_mask, input_y, input_y_mask, word_emb, keep_rate, conf, x_len=None, y_len=None):

    Hr = tf.nn.embedding_lookup(word_emb, input_y)

    if conf['is_positional'] and conf['stack_num'] > 0:
        with tf.variable_scope('positional'):
            Hr = op.positional_encoding_vector(Hr, max_timescale=10)
    Hr_stack = [Hr]

    for index in range(conf['stack_num']):
        with tf.variable_scope('self_stack_cr_' + str(index)):
            Hr = layers.block(
                Hr, Hr, Hr, 
                Q_lengths=y_len, K_lengths=y_len)
            Hr_stack.append(Hr)

    #context part
    #a list of length max_turn_num, every element is a tensor with shape [batch, max_turn_len]
    list_turn_t = tf.unstack(input_x, axis=1) 
    list_turn_length = tf.unstack(x_len, axis=1)
    
    sim_turns = []
    #for every turn_t calculate matching vector
    for turn_t, t_turn_length in zip(list_turn_t, list_turn_length):
        Hu = tf.nn.embedding_lookup(word_emb, turn_t) #[batch, max_turn_len, emb_size]

        if conf['is_positional'] and conf['stack_num'] > 0:
            with tf.variable_scope('positional', reuse=True):
                Hu = op.positional_encoding_vector(Hu, max_timescale=10)
        Hu_stack = [Hu]

        for index in range(conf['stack_num']):

            with tf.variable_scope('self_stack_cr_' + str(index), reuse=True):
                Hu = layers.block(
                    Hu, Hu, Hu,
                    Q_lengths=t_turn_length, K_lengths=t_turn_length)

                Hu_stack.append(Hu)


        r_a_t_stack = []
        t_a_r_stack = []
        for index in range(conf['stack_num']+1):

            with tf.variable_scope('t_attend_r_cr_' + str(index)):
                try:
                    t_a_r = layers.block(
                        Hu_stack[index], Hr_stack[index], Hr_stack[index],
                        Q_lengths=t_turn_length, K_lengths=y_len)
                except ValueError:
                    tf.get_variable_scope().reuse_variables()
                    t_a_r = layers.block(
                        Hu_stack[index], Hr_stack[index], Hr_stack[index],
                        Q_lengths=t_turn_length, K_lengths=y_len)


            with tf.variable_scope('r_attend_t_cr_' + str(index)):
                try:
                    r_a_t = layers.block(
                        Hr_stack[index], Hu_stack[index], Hu_stack[index],
                        Q_lengths=y_len, K_lengths=t_turn_length)
                except ValueError:
                    tf.get_variable_scope().reuse_variables()
                    r_a_t = layers.block(
                        Hr_stack[index], Hu_stack[index], Hu_stack[index],
                        Q_lengths=y_len, K_lengths=t_turn_length)

            t_a_r_stack.append(t_a_r)
            r_a_t_stack.append(r_a_t)

        t_a_r_stack.extend(Hu_stack)
        r_a_t_stack.extend(Hr_stack)
        
        t_a_r = tf.stack(t_a_r_stack, axis=-1)
        r_a_t = tf.stack(r_a_t_stack, axis=-1)

                     
        #calculate similarity matrix
        with tf.variable_scope('similarity'):
            # sim shape [batch, max_turn_len, max_turn_len, 2*stack_num+1]
            # divide sqrt(200) to prevent gradient explosion
            sim = tf.einsum('biks,bjks->bijs', t_a_r, r_a_t) / tf.sqrt(200.0)

        sim_turns.append(sim)


    #cnn and aggregation
    sim = tf.stack(sim_turns, axis=1)
    with tf.variable_scope('cnn_aggregation'):
        final_info = layers.CNN_3d(sim, 32, 16)
        #for douban
        #final_info = layers.CNN_3d(sim, 16, 16)

    return final_info
