import math
import tensorflow as tf
import parse_libras
import numpy as np


class Model(object):
    def __init__(self):
        pass

    def train(self):
        dataset = parse_libras.Dataset()
        N, T, D = dataset.get_dimensions()
        num_labels = dataset.get_num_of_labels()
        num_channels = 1

        batch_size = 251
        patch_t_size = 5
        patch_D_size = 1
        depth = 16
        num_hidden = 40
        max_pool_percentage = 0.1
        max_pool_window_size = round(max_pool_percentage * T)
        max_pool_out_size = int(math.ceil(T / max_pool_window_size))

        graph = tf.Graph()

        with graph.as_default():
            # Input data.
            tf_train_dataset = tf.placeholder(
                tf.float32, shape=(batch_size, T, D, num_channels))
            tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
            tf_valid_dataset = tf.constant(dataset.get_validation_set())
            tf_test_dataset = tf.constant(dataset.get_test_set())

            # Variables.
            layer1_weights = tf.Variable(tf.truncated_normal(
                [patch_t_size, patch_D_size, num_channels, depth], stddev=0.1))
            layer1_biases = tf.Variable(tf.zeros([depth]))

            layer2_weights = tf.Variable(tf.truncated_normal(
                [patch_t_size, patch_D_size, depth, depth], stddev=0.1))
            layer2_biases = tf.Variable(tf.zeros([depth]))

            layer3_weights = tf.Variable(tf.truncated_normal(
                [max_pool_out_size * D * depth, num_hidden], stddev=0.1))
            layer3_biases = tf.Variable(tf.zeros([num_hidden]))

            layer4_weights = tf.Variable(tf.truncated_normal(
                [num_hidden, num_labels], stddev=0.1))
            layer4_biases = tf.Variable(tf.zeros([num_labels]))


            # Model.
            def model(data):
                conv = tf.nn.conv2d(data, layer1_weights, [1, 1, 1, 1], padding='SAME')
                hidden = tf.nn.relu(conv + layer1_biases)
                # conv = tf.nn.conv2d(hidden, layer2_weights, [1, 1, 1, 1], padding='SAME')
                # hidden = tf.nn.relu(conv + layer2_biases)
                hidden = tf.nn.max_pool(hidden, ksize=[1, max_pool_window_size, 1, 1], strides=[1, max_pool_window_size, 1, 1], padding='SAME')
                shape = hidden.get_shape().as_list()
                reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])
                hidden = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)
                return tf.matmul(hidden, layer4_weights) + layer4_biases


            def accuracy(predictions, labels):
                return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
                        / predictions.shape[0])

            # Training computation.
            logits = model(tf_train_dataset)
            loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))

            # Optimizer.
            optimizer = tf.train.GradientDescentOptimizer(0.005).minimize(loss)

            # Predictions for the training, validation, and test data.
            train_prediction = tf.nn.softmax(logits)
            valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
            test_prediction = tf.nn.softmax(model(tf_test_dataset))


        num_steps = 20000

        train_dataset = dataset.get_training_set()
        train_labels = dataset.get_train_labels()
        valid_labels = dataset.get_validation_labels()
        test_labels = dataset.get_test_labels()
        with tf.Session(graph=graph) as session:
          tf.global_variables_initializer().run()
          print('Initialized')
          for step in range(num_steps):
            offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
            batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
            batch_labels = train_labels[offset:(offset + batch_size), :]
            feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
            _, l, predictions = session.run(
              [optimizer, loss, train_prediction], feed_dict=feed_dict)
            if (step % 50 == 0):
              print('batch_labels: {}'.format(np.argmax(batch_labels, 1)))
              print('predictions: {}'.format(np.argmax(predictions, 1)))
              print('Minibatch loss at step %d: %f' % (step, l))
              print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
              print('Validation accuracy: %.1f%%' % accuracy(
                valid_prediction.eval(), valid_labels))
          print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))

if __name__ == '__main__':
    libras_model = Model()
    libras_model.train()
