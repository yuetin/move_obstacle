import tensorflow as tf
import numpy as np 


def safe_get(name, *args, **kwargs):
    """ Same as tf.get_variable, except flips on reuse_variables automatically """
    try:
        # print('New variable -> {} '.format(name))
        return tf.get_variable(name, *args, **kwargs)
    except ValueError as e :
        # print('ValueError as e  -> ' + str(e))
        print('REuse variable -> {}, scope={} '.format(name, str(tf.get_variable_scope())  ))
        tf.get_variable_scope().reuse_variables()
        return tf.get_variable(name, *args, **kwargs)


def weight_variable(shape, name='w', initializer='xavier'):
    # print('name ={} weight_variable shape = {}'.format(name, shape))
    if initializer == 'xavier':
        xavier_init =  tf.contrib.layers.xavier_initializer(dtype=tf.float32)
        return safe_get(name, shape, initializer=xavier_init, dtype=tf.float32)
    elif initializer == 'truncated_normal':
        truncated_normal_init = tf.truncated_normal_initializer(mean=0.0, stddev=0.1, dtype=tf.float32)
        return safe_get(name, shape, initializer=truncated_normal_init, dtype=tf.float32)
    elif initializer == 'selu':
        muti_exclude_last = np.prod(shape[:-1])  # ex [4, 4, 3, 32] -> 4 * 4*3
        weights = np.random.normal(scale=np.sqrt(1.0/muti_exclude_last), size=shape).astype('f')
        return safe_get(name, list(shape), initializer=tf.constant_initializer(weights), dtype=tf.float32)


def bias_variable(shape, name='b'):
    return safe_get(name, initializer=tf.zeros(shape, dtype=tf.float32))
    # inital = tf.constant(0.1, shape=shape)
    # return tf.Variable(inital, name = name)


def Conv2D(x, kernel_size=3, out_channel=32, in_channel=None, name_prefix='conv',
        w=None, b=None, initializer='xavier', strides=[1,2,2,1]):
    if in_channel is None:
        assert len(x.shape) == 4, 'Conv2D() say the len of input shape is not 4 %s' % name_prefix
        in_channel = int(x.shape[3])

    # w and b
    if w==None:
        w = weight_variable([kernel_size, kernel_size, in_channel, out_channel] , 
                            name= name_prefix + "_w", initializer=initializer) 
    if b==None:
        b = bias_variable([out_channel]  , name= name_prefix + "_b")

    #Combine
    return tf.nn.relu(tf.nn.conv2d(x, w, strides=strides, padding='SAME')+ b, name = name_prefix) #output size 28x28x32


def MaxPool2D(x, pool_size=2):
    return tf.nn.max_pool(x, ksize=[1, pool_size, pool_size, 1],
                        strides=[1, pool_size, pool_size, 1], padding='SAME')


def Flaten(x):
    assert len(x.shape) == 4, 'flat() say the len of input shape is not 4'
    num = int(x.shape[1]) * int(x.shape[2]) * int(x.shape[3])
    # print('flat num = %d' % num)
    return tf.reshape(x, [-1, num])


def FC(x, fc_size=1024, name_prefix='fc', w=None, b=None, initializer='xavier', op='relu'):
    assert len(x.shape) == 2, 'FC() say the len of input shape is not 2'
    num = int(x.shape[1]) if x.shape[1].value is not None else None

    if w == None:
        w = weight_variable([num, fc_size],name= name_prefix + "_w", initializer=initializer) 
    if b == None:
        b = bias_variable([fc_size], name = name_prefix + '_b') 

    if op is None:
        activation = lambda x, name: x
    elif op == 'relu':
        activation = tf.nn.relu
    elif op == 'softmax':
        activation = tf.nn.softmax
    elif op == 'leaky_relu':
        activation = tf.nn.leaky_relu
    else:
        print('error op ==' + op)
        exit()

    return activation(tf.matmul(x, w) + b, name=name_prefix + "_{}".format(op))
