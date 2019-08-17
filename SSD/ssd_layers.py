"""
Layers for SSD
"""

import tensorflow as tf

def conv2d(x,
           filters,
           kernel_size,
           stride=1,
           padding="same",
           dilation_rate=1,
           activation=tf.nn.relu,
           scope="conv2d"):
    '''Conv2d: for stride = 1'''
    kernel_sizes = [kernel_size] * 2
    strides = [stride] * 2
    dilation_rate = [dilation_rate] * 2
    return tf.layers.conv2d(x,
                            filters,
                            kernel_sizes,
                            strides=strides,
                            dilation_rate=dilation_rate,
                            padding=padding,
                            name=scope,
                            activation=activation)

def max_pool2d(x,
               pool_size,
               stride=None,
               scope="max_pool2d"):
    '''max pool2d: default pool_size = stride'''
    pool_sizes = [pool_size] * 2
    strides = [pool_size] * 2 if stride is None else [stride] * 2
    return tf.layers.max_pooling2d(x,
                                   pool_sizes,
                                   strides,
                                   name=scope,
                                   padding="same")

def pad2d(x, pad):
    '''pad2d: for conv2d with stride > 1'''
    return tf.pad(x, paddings=[[0, 0], [pad, pad], [pad, pad], [0, 0]])

def pad2d_single(x, pad):
    '''pad2d: for conv2d with stride > 1'''
    return tf.pad(x, paddings=[[0, 0], [0, pad], [0, pad], [0, 0]])

def dropout(x, rate=0.5, is_training=True):
    return tf.layers.dropout(x, rate=rate, training=is_training)

def l2norm(x, scale, trainable=True, scope="L2Normalization"):
    '''l2norm (not bacth norm, spatial normalization)'''
    return tf.nn.l2_normalize(x, axis=3, epsilon=1e-12)
    '''
    n_channels = x.get_shape().as_list()[-1]
    l2_norm = tf.nn.l2_normalize(x, [3], epsilon=1e-12)

    with tf.variable_scope(scope):
        gamma = tf.get_variable("gamma", shape=[n_channels, ], dtype=tf.float32,
                                initializer=tf.constant_initializer(scale),
                                trainable=trainable)
        return l2_norm * gamma
    '''

def sep_conv2d(x,
               kernel_size=3,
               stride=1,
               padding="same",
               dilation_rate=1,
               depth_multiplier=1,
               scope="sep_conv2d"):

    inputs_shape = x.get_shape().as_list()
    in_channels = inputs_shape[-1]

    kernel_sizes = [kernel_size] * 2
    strides = [stride] * 2
    dilation_rate = [dilation_rate] * 2

    return tf.layers.separable_conv2d(inputs=x,
                                      filters=in_channels,
                                      kernel_size=kernel_sizes,
                                      strides=strides,
                                      padding=padding,
                                      dilation_rate=dilation_rate,
                                      depth_multiplier=depth_multiplier,
                                      name=scope)

def avg_pool2d(x, scope="avg_pool2d"):

    inputs_shape = x.get_shape().as_list()

    pool_sizes = strides = (inputs_shape[1], inputs_shape[2])

    return tf.layers.average_pooling2d(inputs=x,
                                       pool_size=pool_sizes,
                                       strides=strides,
                                       padding='valid',
                                       name=scope)

def batch_norm(x,
               axis=-1,
               momentum=0.99,
               epsilon=1e-05,
               training=True,
               scope="b_norm"):

    return tf.layers.batch_normalization(x,
                                         axis=axis,
                                         momentum=momentum,
                                         epsilon=epsilon,
                                         training=training,
                                         name=scope)

def relu(x):
    return tf.nn.relu(x)

def relu6(x):
    return tf.nn.relu6(x)

def conv_bn_relu(x,
                 filters,
                 kernel_size,
                 stride=1,
                 is_training=True,
                 scope="conv_bn_relu",
                 padding="same"):

    with tf.variable_scope(scope):
        net = conv2d(x, filters, kernel_size, stride, padding=padding, activation=None, scope='conv2d')
        net = batch_norm(net, training=is_training, scope="b_norm")
        return relu6(net)

def project_conv2d(x,
                   filters,
                   is_training=True,
                   scope="project_conv2d"):

    with tf.variable_scope(scope):
        net = conv2d(x, filters, 1, activation=None, scope=scope+"conv2d")
        return batch_norm(net, training=is_training, scope=scope+"b_norm")

def pointwise_conv2d(x,
                     filters,
                     is_training=True,
                     scope="pw_conv2d"):

    with tf.variable_scope(scope):
        net = conv2d(x, filters, 1, activation=None, scope=scope+"conv2d")
        net = batch_norm(net, training=is_training, scope=scope+"b_norm")
        return relu6(net)

def depthwise_conv2d(x,
                     stride=1,
                     is_training=True,
                     scope="dw_conv2d",
                     padding="same"):

    with tf.variable_scope(scope):
        net = sep_conv2d(x, stride=stride, padding=padding, scope=scope+"sep_conv2d")
        net = batch_norm(net, training=is_training, scope=scope+"b_norm_sep")
        return relu6(net)

def depth_point_conv2d(x,
                       filters,
                       stride=1,
                       is_training=True,
                       scope="depth_point_conv2d",
                       padding="same"):
    
    with tf.variable_scope(scope):
        net = depthwise_conv2d(x, stride, is_training, scope+"dw_conv2d", padding=padding)
        net = pointwise_conv2d(net, filters, is_training, scope+"pw_conv2d")
        return net

'''
def depthwise_separable_conv2d(x,
                               filters,
                               stride=1,
                               padding="same",
                               is_training=True,
                               scope="dws_conv2d"):

    with tf.variable_scope(scope):
        net = depthwise_conv2d(x, stride, padding, is_training, scope+"dw_conv2d")
        return conv_bn_relu(net, filters, 1, 1, is_training, scope+"conv_bn_relu")
        #net = sep_conv2d(x, stride=stride, scope=scope+"sep_conv2d")
        #net = batch_norm(net, training=training, scope=scope+"b_norm_sep")
        #net = relu(net)
        #net = conv2d(net, filters, 1, activation=None, scope=scope+"pw_conv2d")
        #net = batch_norm(net, training=is_training, scope=scope+"b_norm_pw")
        #return relu(net)
'''

def res_block(x,
              expansion_ratio,
              output_dim,
              stride=1,
              is_training=True,
              shortcut=True,
              scope="res_block",
              padding="same"):

    with tf.variable_scope(scope):
        bottleneck_dim=round(expansion_ratio*x.get_shape().as_list()[-1])# 中间输出 通道数量随机
        net = pointwise_conv2d(x, bottleneck_dim, is_training, scope+"pw_conv2d")
        net = depthwise_conv2d(net, stride, is_training, scope+"dw_conv2d", padding=padding)
        net = project_conv2d(net, output_dim, is_training, scope+"project_conv2d")

        # element wise add, only for stride==1
        if shortcut and stride == 1: # 需要残差结构
            in_dim = int(x.get_shape().as_list()[-1])
            if in_dim != output_dim:
                ins = conv2d(x, output_dim, 1, activation=None, scope=scope+"conv2d")
                net = net + ins # f(x) + w*x
            else:
                net = net + x # f(x) + x

        return net # 不需要残差结构 直接输出 f(x)

def upsample2d(x,
               size=(2,2)):
    return tf.keras.layers.UpSampling2D(size=size)(x)

def upsample_conv2d(x,
                    filters,
                    size=(2, 2),
                    upsampling=True,
                    scope="upsample_conv2d",):

    with tf.variable_scope(scope):
        net = conv2d(x, filters, 1, scope=scope+"conv2d")

        if upsampling:
            #return tf.keras.layers.UpSampling2D(size=size,
            #                                    interpolation="bilinear")
            return tf.image.resize_bilinear(images=net,
                                            size=size,
                                            align_corners=False,
                                            name=scope+"resize")

        return net

def fpn_block(c,
              p,
              filters,
              is_training=True,
              size=(2, 2),
              padding=False,
              scope="fpn_block"):

    with tf.variable_scope(scope):
        c = conv2d(c, filters, 1, scope=scope+"conv2d_c")
        p = depthwise_conv2d(p, 1, is_training, scope+"dw_conv2d")
        p = conv2d(p, filters, 1, scope=scope+"conv2d_p")
        up_p = upsample2d(p, size)
        if padding:
            up_p = pad2d_single(up_p, 1)

        return c + up_p

def ssd_conv2d(x,
               filters,
               stride=1,
               is_training=True,
               scope="ssd_conv2d",
               padding="valid"):

     with tf.variable_scope(scope):
        net = pointwise_conv2d(x, filters/2, is_training, scope=scope+"pw_conv2d_in")
        net = depthwise_conv2d(net, stride, is_training, scope+"dw_conv2d", padding=padding)
        return pointwise_conv2d(net, filters, is_training, scope=scope+"pw_conv2d_out")

def ssd_multibox_layer(x, num_classes, sizes, ratios, normalization=-1, is_training=True, scope="multibox"):
    '''multibox layer: get class and location predicitions from detection layer'''
    # [-1, size, size,]
    #pre_shape = [-1] + x.get_shape().as_list()[1:-1]
    input_shape = x.get_shape().as_list()
    pre_shape = [-1, input_shape[1]*input_shape[2]] # [-1, size*size]

    # l2 norm
    if normalization > 0:
        x = l2norm(x, normalization)
        #print(x)

    # numbers of anchors
    n_anchors = len(sizes) + len(ratios)
    # class prediction
    net = depthwise_conv2d(x, 1, is_training, scope+"cls_dw_conv2d")
    cls_pred = conv2d(net, n_anchors*num_classes, 1, activation=None, scope=scope+"cls_conv2d")
    cls_pred = tf.reshape(cls_pred, pre_shape + [n_anchors, num_classes])
    # location predictions
    net = depthwise_conv2d(x, 1, is_training, scope+"loc_dw_conv2d")
    loc_pred = conv2d(net, n_anchors*4, 1, activation=None, scope=scope+"loc_conv2d")
    loc_pred = tf.reshape(loc_pred, pre_shape + [n_anchors, 4])

    # cls shape [batch, size*size, n_anchors, num_classes]
    # loc shape [batch, size*size, n_anchors, 4]
    return cls_pred, loc_pred
    '''
    with tf.variable_scope(scope):
        # l2 norm
        if normalization > 0:
            x = l2norm(x, normalization)
            print(x)
        # numbers of anchors
        n_anchors = len(sizes) + len(ratios)
        # location predictions
        loc_pred = conv2d(x, n_anchors*4, 3, activation=None, scope="conv_loc")
        loc_pred = tf.reshape(loc_pred, pre_shape + [n_anchors, 4])
        # class prediction
        cls_pred = conv2d(x, n_anchors*num_classes, 3, activation=None, scope="conv_cls")
        cls_pred = tf.reshape(cls_pred, pre_shape + [n_anchors, num_classes])
        return cls_pred, loc_pred
    '''