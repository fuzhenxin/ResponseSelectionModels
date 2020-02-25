import tensorflow as tf
import utils.operations as op

def loss(x, y, num_classes=2, is_clip=True, clip_value=10):
    '''From info x calculate logits as return loss.

    Args:
        x: a tensor with shape [batch, dimension]
        num_classes: a number

    Returns:
        loss: a tensor with shape [1], which is the average loss of one batch
        logits: a tensor with shape [batch, 1]

    Raises:
        AssertionError: if
            num_classes is not a int greater equal than 2.
    TODO:
        num_classes > 2 may be not adapted.
    '''
    assert isinstance(num_classes, int)
    assert num_classes >= 2

    # init = tf.orthogonal_initializer()
    init = tf.contrib.layers.xavier_initializer()

    W = tf.get_variable(
        name='weights',
        shape=[x.shape[-1], num_classes-1],
        initializer=init)
    bias = tf.get_variable(
        name='bias',
        shape=[num_classes-1],
        initializer=tf.zeros_initializer())

    logits = tf.reshape(tf.matmul(x, W) + bias, [-1])
    loss = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.cast(y, tf.float32),
        logits=logits)
    loss = tf.reduce_mean(loss)
    # loss = tf.reduce_mean(tf.clip_by_value(loss, -clip_value, clip_value))

    return loss, logits

def attention(
    Q, K, V, 
    Q_lengths, K_lengths, 
    attention_type='dot', 
    is_mask=True, mask_value=-2**32+1,
    drop_prob=None, q_time=None, k_time=None, use_len=False, init=None):
    '''Add attention layer.
    Args:
        Q: a tensor with shape [batch, Q_time, Q_dimension]
        K: a tensor with shape [batch, time, K_dimension]
        V: a tensor with shape [batch, time, V_dimension]

        Q_length: a tensor with shape [batch]
        K_length: a tensor with shape [batch]

    Returns:
        a tensor with shape [batch, Q_time, V_dimension]

    Raises:
        AssertionError: if
            Q_dimension not equal to K_dimension when attention type is dot.
    '''
    assert attention_type in ('dot', 'bilinear')
    if attention_type == 'dot':
        assert Q.shape[-1] == K.shape[-1]

    Q_time = Q.shape[1]
    K_time = K.shape[1]
    if q_time:
        Q_time = q_time
    if k_time:
        K_time = k_time

    if attention_type == 'dot':
        logits = op.dot_sim(Q, K) #[batch, Q_time, time]
    if attention_type == 'bilinear':
        logits = op.bilinear_sim(Q, K)

    if is_mask:
        if use_len:
            row_mask = tf.expand_dims(Q_lengths, -1)
            col_mask = tf.expand_dims(K_lengths, -1)
            mask = tf.einsum('bik,bjk->bij', row_mask, col_mask)
        else:
            mask = op.mask(Q_lengths, K_lengths, Q_time, K_time) #[batch, Q_time, K_time]
        logits = mask * logits + (1 - mask) * mask_value
    
    attention = tf.nn.softmax(logits)

    if drop_prob is not None:
        print('use attention drop')
        attention = tf.nn.dropout(attention, drop_prob)

    return op.weighted_sum(attention, V)

def FFN(x, out_dimension_0=None, out_dimension_1=None, init=None):
    '''Add two dense connected layer, max(0, x*W0+b0)*W1+b1.

    Args:
        x: a tensor with shape [batch, time, dimension]
        out_dimension: a number which is the output dimension

    Returns:
        a tensor with shape [batch, time, out_dimension]

    Raises:
    '''
    with tf.variable_scope('FFN_1'):
        y = op.dense(x, out_dimension_0, init=init)
        y = tf.nn.relu(y)
    with tf.variable_scope('FFN_2'):
        z = op.dense(y, out_dimension_1, init=init) #, add_bias=False)  #!!!!
    return z

def block(
    Q, K, V, 
    Q_lengths=None, K_lengths=None,
    attention_type='dot', 
    is_layer_norm=True, 
    is_mask=True, mask_value=-2**32+1,
    drop_prob=None, q_time=None, k_time=None, use_len=False, no_mask=False, init=None):
    '''Add a block unit from https://arxiv.org/pdf/1706.03762.pdf.
    Args:
        Q: a tensor with shape [batch, Q_time, Q_dimension]
        K: a tensor with shape [batch, time, K_dimension]
        V: a tensor with shape [batch, time, V_dimension]

        Q_length: a tensor with shape [batch]
        K_length: a tensor with shape [batch]

    Returns:
        a tensor with shape [batch, time, dimension]

    Raises:
    '''
    att = attention(Q, K, V, 
                    Q_lengths, K_lengths, 
                    attention_type='dot', 
                    is_mask=is_mask, mask_value=mask_value,
                    drop_prob=drop_prob, q_time=q_time, k_time=k_time, use_len=use_len, init=init)
    if is_layer_norm:
        with tf.variable_scope('attention_layer_norm'):
            y = op.layer_norm_debug(Q + att)
    else:
        y = Q + att

    z = FFN(y, init=init)
    if is_layer_norm:
        with tf.variable_scope('FFN_layer_norm'):
            w = op.layer_norm_debug(y + z)
    else:
        w = y + z
    return w


def CNN(x, out_channels, filter_size, pooling_size, add_relu=True):
    '''Add a convlution layer with relu and max pooling layer.

    Args:
        x: a tensor with shape [batch, in_height, in_width, in_channels]
        out_channels: a number
        filter_size: a number
        pooling_size: a number

    Returns:
        a flattened tensor with shape [batch, num_features]

    Raises:
    '''
    #calculate the last dimension of return
    #f1 = (tf.shape(x)[1]-filter_size+1)/pooling_size
    #f2 = (tf.shape(x)[2]-filter_size+1)/pooling_size
    #num_features = ((tf.shape(x)[1]-filter_size+1)/pooling_size * 
    #    (tf.shape(x)[2]-filter_size+1)/pooling_size) * out_channels

    in_channels = x.shape[-1]
    weights = tf.get_variable(
        name='filter',
        shape=[filter_size, filter_size, in_channels, out_channels],
        dtype=tf.float32,
        initializer=tf.random_uniform_initializer(-0.01, 0.01))
    bias = tf.get_variable(
        name='bias',
        shape=[out_channels],
        dtype=tf.float32,
        initializer=tf.zeros_initializer())

    conv = tf.nn.conv2d(x, weights, strides=[1, 1, 1, 1], padding="VALID")
    conv = conv + bias

    if add_relu:
        conv = tf.nn.relu(conv)

    pooling = tf.nn.max_pool(
        conv, 
        ksize=[1, pooling_size, pooling_size, 1],
        strides=[1, pooling_size, pooling_size, 1], 
        padding="VALID")

    return tf.contrib.layers.flatten(pooling)



def CNN_MSN(x, init=None):
    if not init:
        init = tf.random_uniform_initializer(-0.01, 0.01)
    #print("======================")
    #print("X: ", x.shape)
    x = tf.contrib.layers.conv2d(
        inputs = x,
        num_outputs= 16,
        kernel_size = (3, 3),
        stride = 1,
        padding = "SAME",
        activation_fn=tf.nn.relu,
        weights_initializer=init,
        biases_initializer=tf.zeros_initializer(),
        #reuse=True,
        trainable=True, 
    )
    #print("X: ", x.shape)
    x = tf.layers.max_pooling2d(
        inputs = x,
        pool_size = (2, 2),
        strides = 2,
        padding = "SAME",
    )
    #print("X: ", x.shape)
    x = tf.contrib.layers.conv2d(
        inputs = x,
        num_outputs= 32,
        kernel_size = (3, 3),
        stride = 1,
        padding = "SAME",
        activation_fn=tf.nn.relu,
        weights_initializer=init,
        biases_initializer=tf.zeros_initializer(),
        #reuse=True,
        trainable=True, 
    )
    #print("X: ", x.shape)
    x = tf.layers.max_pooling2d(
        inputs = x,
        pool_size = (2, 2),
        strides = 2,
        padding = "SAME",
    )
    #print("X: ", x.shape)
    x = tf.contrib.layers.conv2d(
        inputs = x,
        num_outputs= 64,
        kernel_size = (3, 3),
        stride = 1,
        padding = "SAME",
        activation_fn=tf.nn.relu,
        weights_initializer=init,
        biases_initializer=tf.zeros_initializer(),
        #reuse=True,
        trainable=True, 
    )
    #print("X: ", x.shape)
    x = tf.layers.max_pooling2d(
        inputs = x,
        pool_size = (3, 3),
        strides = 3,
        padding = "SAME",
    )
    #print("X: ", x.shape)
    x = tf.reshape(x, [-1, x.get_shape()[1:4].num_elements()])
    return x

def CNN_3d(x, out_channels_0, out_channels_1, add_relu=True):
    '''Add a 3d convlution layer with relu and max pooling layer.

    Args:
        x: a tensor with shape [batch, in_depth, in_height, in_width, in_channels]
        out_channels: a number
        filter_size: a number
        pooling_size: a number

    Returns:
        a flattened tensor with shape [batch, num_features]

    Raises:
    '''
    in_channels = x.shape[-1]
    weights_0 = tf.get_variable(
        name='filter_0',
        shape=[3, 3, 3, in_channels, out_channels_0],
        dtype=tf.float32,
        initializer=tf.random_uniform_initializer(-0.01, 0.01))
    bias_0 = tf.get_variable(
        name='bias_0',
        shape=[out_channels_0],
        dtype=tf.float32,
        initializer=tf.zeros_initializer())

    # print("x shape:", x.shape)
    conv_0 = tf.nn.conv3d(x, weights_0, strides=[1, 1, 1, 1, 1], padding="SAME")
    # print('conv_0 shape: %s' %conv_0.shape)
    conv_0 = conv_0 + bias_0

    if add_relu:
        conv_0 = tf.nn.elu(conv_0)

    pooling_0 = tf.nn.max_pool3d(
        conv_0, 
        ksize=[1, 3, 3, 3, 1],
        strides=[1, 3, 3, 3, 1], 
        padding="SAME")
    # print('pooling_0 shape: %s' %pooling_0.shape)

    #layer_1
    weights_1 = tf.get_variable(
        name='filter_1',
        shape=[3, 3, 3, out_channels_0, out_channels_1],
        dtype=tf.float32,
        initializer=tf.random_uniform_initializer(-0.01, 0.01))
    bias_1 = tf.get_variable(
        name='bias_1',
        shape=[out_channels_1],
        dtype=tf.float32,
        initializer=tf.zeros_initializer())

    conv_1 = tf.nn.conv3d(pooling_0, weights_1, strides=[1, 1, 1, 1, 1], padding="SAME")
    # print('conv_1 shape: %s' %conv_1.shape)
    conv_1 = conv_1 + bias_1

    if add_relu:
        conv_1 = tf.nn.elu(conv_1)

    pooling_1 = tf.nn.max_pool3d(
        conv_1, 
        ksize=[1, 3, 3, 3, 1],
        strides=[1, 3, 3, 3, 1], 
        padding="SAME")
    # print('pooling_1 shape: %s' %pooling_1.shape)

    return tf.contrib.layers.flatten(pooling_1)
