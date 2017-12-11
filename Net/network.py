import tensorflow as tf
import tflearn


def build_flat_net(dim_s, dim_a):
    inputs = tf.placeholder(tf.float32, shape=[None] + dim_s)
    net = tf.contrib.layers.flatten(inputs)

    net = tflearn.fully_connected(net, 300, activation='relu')
    net = tflearn.fully_connected(net, 200, activation='relu')
    net = tflearn.fully_connected(net, 100, activation='linear')

    q_values = tflearn.fully_connected(net, dim_a)

    return inputs, q_values

def build_cnn_pong(dim_s, dim_a):
    inputs = tf.placeholder(tf.float32, shape=[None] + dim_s)

    net = tflearn.conv_2d(inputs, 32, 3, strides=2, activation='relu')
    net = tflearn.max_pool_2d(net, 2)
    net = tflearn.batch_normalization(net)

    net = tflearn.conv_2d(net, 64, 3, strides=2, activation='relu')
    net = tflearn.max_pool_2d(net, 2)
    net = tflearn.batch_normalization(net)

    net = tflearn.fully_connected(net, 64, activation='relu', regularizer='L2')
    net = tflearn.dropout(net, 0.8)

    net = tflearn.fully_connected(net, 128, activation='relu', regularizer='L2')
    net = tflearn.dropout(net, 0.8)

    q_values = tflearn.fully_connected(net, dim_a)

    return inputs, q_values


def build_cnn_bird(dim_s, dim_a):
    inputs = tf.placeholder(tf.float32, shape=[None] + dim_s)

    net = tf.image.resize_images(inputs, size=(80, 80))

    net = tflearn.conv_2d(net, 32, 8, strides=4, activation='relu')
    net = tflearn.max_pool_2d(net, 2)
    net = tflearn.local_response_normalization(net)

    net = tflearn.conv_2d(net, 64, 4, strides=2, activation='relu')
    net = tflearn.max_pool_2d(net, 2)
    net = tflearn.local_response_normalization(net)

    net = tflearn.fully_connected(net, 512, activation='relu')
    net = tflearn.dropout(net, 0.8)

    net = tflearn.fully_connected(net, 512, activation='relu')
    net = tflearn.dropout(net, 0.8)

    q_values = tflearn.fully_connected(net, dim_a)

    return inputs, q_values


def build_simple_cnn(dim_s, dim_a):
    SCREEN_WIDTH = dim_s[0]
    SCREEN_HEIGHT = dim_s[1]
    STATE_FRAMES = dim_s[2]
    ACTIONS_COUNT = dim_a

    # network weights
    convolution_weights_1 = tf.Variable(tf.truncated_normal([8, 8, STATE_FRAMES, 32], stddev=0.01))
    convolution_bias_1 = tf.Variable(tf.constant(0.01, shape=[32]))

    convolution_weights_2 = tf.Variable(tf.truncated_normal([4, 4, 32, 64], stddev=0.01))
    convolution_bias_2 = tf.Variable(tf.constant(0.01, shape=[64]))

    feed_forward_weights_1 = tf.Variable(tf.truncated_normal([256, 256], stddev=0.01))
    feed_forward_bias_1 = tf.Variable(tf.constant(0.01, shape=[256]))

    feed_forward_weights_2 = tf.Variable(tf.truncated_normal([256, ACTIONS_COUNT], stddev=0.01))
    feed_forward_bias_2 = tf.Variable(tf.constant(0.01, shape=[ACTIONS_COUNT]))

    input_layer = tf.placeholder("float", [None, SCREEN_WIDTH, SCREEN_HEIGHT,
                                           STATE_FRAMES])

    hidden_convolutional_layer_1 = tf.nn.relu(
        tf.nn.conv2d(input_layer, convolution_weights_1, strides=[1, 4, 4, 1], padding="SAME") + convolution_bias_1)

    hidden_max_pooling_layer_1 = tf.nn.max_pool(hidden_convolutional_layer_1, ksize=[1, 2, 2, 1],
                                                strides=[1, 2, 2, 1], padding="SAME")

    hidden_convolutional_layer_2 = tf.nn.relu(
        tf.nn.conv2d(hidden_max_pooling_layer_1, convolution_weights_2, strides=[1, 2, 2, 1],
                     padding="SAME") + convolution_bias_2)

    hidden_max_pooling_layer_2 = tf.nn.max_pool(hidden_convolutional_layer_2, ksize=[1, 2, 2, 1],
                                                strides=[1, 2, 2, 1], padding="SAME")

    hidden_convolutional_layer_3_flat = tf.reshape(hidden_max_pooling_layer_2, [-1, 256])

    final_hidden_activations = tf.nn.relu(
        tf.matmul(hidden_convolutional_layer_3_flat, feed_forward_weights_1) + feed_forward_bias_1)

    output_layer = tf.matmul(final_hidden_activations, feed_forward_weights_2) + feed_forward_bias_2

    return input_layer, output_layer
