
import tensorflow as tf

def esim_model(input_x, input_x_mask, input_y, input_y_mask, word_emb, keep_rate):

    inputs = input_x, input_x_mask, input_y, input_y_mask
    inputs = [tf.transpose(i, perm=[1, 0]) for i in inputs]

    x1, x1_mask, x2, x2_mask = inputs
    x1_mask = tf.to_float(x1_mask)
    x2_mask = tf.to_float(x2_mask)

    keep_rate = keep_rate
    hidden_size = 300
    use_cudnn = False

    # embedding: [length, batch, dim]
    emb1 = tf.nn.embedding_lookup(word_emb, x1)
    emb2 = tf.nn.embedding_lookup(word_emb, x2)

    emb1 = tf.nn.dropout(emb1, keep_rate)
    emb2 = tf.nn.dropout(emb2, keep_rate)

    emb1 = emb1 * tf.expand_dims(x1_mask, -1)
    emb2 = emb2 * tf.expand_dims(x2_mask, -1)

    # encode the sentence pair
    with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
        if use_cudnn:
            x1_enc = bilstm_layer_cudnn(emb1, 1, hidden_size)
            x2_enc = bilstm_layer_cudnn(emb2, 1, hidden_size)
        else:
            x1_enc = bilstm_layer(emb1, 1, hidden_size)
            x2_enc = bilstm_layer(emb2, 1, hidden_size)

    x1_enc = x1_enc * tf.expand_dims(x1_mask, -1)
    x2_enc = x2_enc * tf.expand_dims(x2_mask, -1)

    # local inference modeling based on attention mechanism
    x1_dual, x2_dual = local_inference(x1_enc, x1_mask, x2_enc, x2_mask)

    x1_match = tf.concat([x1_enc, x1_dual, x1_enc * x1_dual, x1_enc - x1_dual], 2)
    x2_match = tf.concat([x2_enc, x2_dual, x2_enc * x2_dual, x2_enc - x2_dual], 2)

    # mapping high dimension feature to low dimension
    with tf.variable_scope("projection", reuse=tf.AUTO_REUSE):
        x1_match_mapping = tf.layers.dense(x1_match, hidden_size,
                                           activation=tf.nn.relu,
                                           name="fnn",
                                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
        x2_match_mapping = tf.layers.dense(x2_match, hidden_size,
                                           activation=tf.nn.relu,
                                           name="fnn",
                                           kernel_initializer=tf.truncated_normal_initializer(
                                               stddev=0.02),
                                           reuse=True)

    x1_match_mapping = tf.nn.dropout(x1_match_mapping, keep_rate)
    x2_match_mapping = tf.nn.dropout(x2_match_mapping, keep_rate)

    # inference composition
    with tf.variable_scope("composition", reuse=tf.AUTO_REUSE):
        if use_cudnn:
            x1_cmp = bilstm_layer_cudnn(x1_match_mapping, 1, hidden_size)
            x2_cmp = bilstm_layer_cudnn(x2_match_mapping, 1, hidden_size)
        else:
            x1_cmp = bilstm_layer(x1_match_mapping, 1, hidden_size)
            x2_cmp = bilstm_layer(x2_match_mapping, 1, hidden_size)

    logit_x1_sum = tf.reduce_sum(x1_cmp * tf.expand_dims(x1_mask, -1), 0) / \
        tf.expand_dims(tf.reduce_sum(x1_mask, 0), 1)
    logit_x1_max = tf.reduce_max(x1_cmp * tf.expand_dims(x1_mask, -1), 0)
    logit_x2_sum = tf.reduce_sum(x2_cmp * tf.expand_dims(x2_mask, -1), 0) / \
        tf.expand_dims(tf.reduce_sum(x2_mask, 0), 1)
    logit_x2_max = tf.reduce_max(x2_cmp * tf.expand_dims(x2_mask, -1), 0)

    logit = tf.concat([logit_x1_sum, logit_x1_max, logit_x2_sum, logit_x2_max], 1)

    # final classifier
    with tf.variable_scope("classifier", reuse=tf.AUTO_REUSE):
        logit = tf.nn.dropout(logit, keep_rate)
        logit = tf.layers.dense(logit, hidden_size,
                                activation=tf.nn.tanh,
                                name="fnn1",
                                kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
    return logit





def bilstm_layer_cudnn(input_data, num_layers, rnn_size, keep_prob=1.):
    """Multi-layer BiLSTM cudnn version, faster
    Args:
        input_data: float32 Tensor of shape [seq_length, batch_size, dim].
        num_layers: int64 scalar, number of layers.
        rnn_size: int64 scalar, hidden size for undirectional LSTM.
        keep_prob: float32 scalar, keep probability of dropout between BiLSTM layers 
    Return:
        output: float32 Tensor of shape [seq_length, batch_size, dim * 2]
    """
    with tf.variable_scope("bilstm", reuse=tf.AUTO_REUSE):
        lstm = tf.contrib.cudnn_rnn.CudnnLSTM(
            num_layers=num_layers,
            num_units=rnn_size,
            input_mode="linear_input",
            direction="bidirectional",
            dropout=1 - keep_prob)

        # to do, how to include input_mask
        outputs, output_states = lstm(inputs=input_data)

    return outputs


def bilstm_layer(input_data, num_layers, rnn_size, keep_prob=1.):
    """Multi-layer BiLSTM
    Args:
        input_data: float32 Tensor of shape [seq_length, batch_size, dim].
        num_layers: int64 scalar, number of layers.
        rnn_size: int64 scalar, hidden size for undirectional LSTM.
        keep_prob: float32 scalar, keep probability of dropout between BiLSTM layers 
    Return:
        output: float32 Tensor of shape [seq_length, batch_size, dim * 2]
    """
    input_data = tf.transpose(input_data, [1, 0, 2])

    output = input_data
    for layer in range(num_layers):
        with tf.variable_scope('bilstm_{}'.format(layer), reuse=tf.AUTO_REUSE):

            cell_fw = tf.contrib.rnn.LSTMCell(
                rnn_size, initializer=tf.truncated_normal_initializer(stddev=0.02))
            cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, input_keep_prob=keep_prob)

            cell_bw = tf.contrib.rnn.LSTMCell(
                rnn_size, initializer=tf.truncated_normal_initializer(stddev=0.02))
            cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, input_keep_prob=keep_prob)

            outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw,
                                                              cell_bw,
                                                              output,
                                                              dtype=tf.float32)

            # Concat the forward and backward outputs
            output = tf.concat(outputs, 2)

    output = tf.transpose(output, [1, 0, 2])

    return output



def local_inference(x1, x1_mask, x2, x2_mask):
    """Local inference collected over sequences
    Args:
        x1: float32 Tensor of shape [seq_length1, batch_size, dim].
        x1_mask: float32 Tensor of shape [seq_length1, batch_size].
        x2: float32 Tensor of shape [seq_length2, batch_size, dim].
        x2_mask: float32 Tensor of shape [seq_length2, batch_size].
    Return:
        x1_dual: float32 Tensor of shape [seq_length1, batch_size, dim]
        x2_dual: float32 Tensor of shape [seq_length2, batch_size, dim]
    """

    # x1: [batch_size, seq_length1, dim].
    # x1_mask: [batch_size, seq_length1].
    # x2: [batch_size, seq_length2, dim].
    # x2_mask: [batch_size, seq_length2].
    x1 = tf.transpose(x1, [1, 0, 2])
    x1_mask = tf.transpose(x1_mask, [1, 0])
    x2 = tf.transpose(x2, [1, 0, 2])
    x2_mask = tf.transpose(x2_mask, [1, 0])

    # attention_weight: [batch_size, seq_length1, seq_length2]
    attention_weight = tf.matmul(x1, tf.transpose(x2, [0, 2, 1]))

    # calculate normalized attention weight x1 and x2
    # attention_weight_2: [batch_size, seq_length1, seq_length2]
    attention_weight_2 = tf.exp(
        attention_weight - tf.reduce_max(attention_weight, axis=2, keepdims=True))
    attention_weight_2 = attention_weight_2 * tf.expand_dims(x2_mask, 1)
    # alpha: [batch_size, seq_length1, seq_length2]
    alpha = attention_weight_2 / (tf.reduce_sum(attention_weight_2, -1, keepdims=True) + 1e-8)
    # x1_dual: [batch_size, seq_length1, dim]
    x1_dual = tf.reduce_sum(tf.expand_dims(x2, 1) * tf.expand_dims(alpha, -1), 2)
    # x1_dual: [seq_length1, batch_size, dim]
    x1_dual = tf.transpose(x1_dual, [1, 0, 2])

    # attention_weight_1: [batch_size, seq_length2, seq_length1]
    attention_weight_1 = attention_weight - tf.reduce_max(attention_weight, axis=1, keepdims=True)
    attention_weight_1 = tf.exp(tf.transpose(attention_weight_1, [0, 2, 1]))
    attention_weight_1 = attention_weight_1 * tf.expand_dims(x1_mask, 1)

    # beta: [batch_size, seq_length2, seq_length1]
    beta = attention_weight_1 / \
        (tf.reduce_sum(attention_weight_1, -1, keepdims=True) + 1e-8)
    # x2_dual: [batch_size, seq_length2, dim]
    x2_dual = tf.reduce_sum(tf.expand_dims(x1, 1) * tf.expand_dims(beta, -1), 2)
    # x2_dual: [seq_length2, batch_size, dim]
    x2_dual = tf.transpose(x2_dual, [1, 0, 2])

    return x1_dual, x2_dual

