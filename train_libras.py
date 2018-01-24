import math
import tensorflow as tf
import parse_libras
import nearest_neighbor
import numpy as np


class Model(object):
    def __init__(self):
        pass

    def build_model(self):
        self.dataset = parse_libras.Dataset()
        N, T, D = self.dataset.get_dimensions()
        num_labels = self.dataset.get_num_of_labels()
        num_channels = 1

        self.batch_size = 287
        self.patch_t_size = 5
        self.patch_D_size = 1
        depth = 16
        num_hidden = 40
        max_pool_percentage = 0.1
        max_pool_window_size = round(max_pool_percentage * T)
        max_pool_out_size = int(math.ceil(T / max_pool_window_size))
        init_learning_rate = 2*10**-5

        self.graph = tf.Graph()

        with self.graph.as_default():
            # Input data.
            self.tf_train_dataset = tf.placeholder(
                tf.float32, shape=(self.batch_size, T, D, num_channels))
            self.tf_train_labels = tf.placeholder(tf.float32, shape=(self.batch_size, num_labels))
            # tf_valid_dataset = tf.constant(self.dataset.get_validation_set())
            tf_test_dataset = tf.constant(self.dataset.get_test_set())

            # Variables.
            layer1_weights = tf.Variable(tf.truncated_normal(
                [self.patch_t_size, self.patch_D_size, num_channels, depth], stddev=0.1))
            layer1_biases = tf.Variable(tf.zeros([depth]))

            layer2_weights = tf.Variable(tf.truncated_normal(
                [self.patch_t_size, self.patch_D_size, depth, depth], stddev=0.1))
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
                hidden_no_relu = tf.matmul(reshape, layer3_weights) + layer3_biases
                hidden = tf.nn.relu(hidden_no_relu)
                return (tf.matmul(hidden, layer4_weights) + layer4_biases), hidden

            # Training computation.
            # Predictions for the training, validation, and test data.
            logits, self.train_embed_vec = model(self.tf_train_dataset)
            self.train_prediction = tf.nn.softmax(logits)

            # valid_logits, _ = model(tf_valid_dataset)
            # self.valid_prediction = tf.nn.softmax(valid_logits)

            self.test_logits, self.test_embed_vec = model(tf_test_dataset)
            self.test_prediction = tf.nn.softmax(self.test_logits)

            self.loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=self.tf_train_labels, logits=logits))
            # Optimizer.
            # self.optimizer = tf.train.GradientDescentOptimizer(0.005).minimize(self.loss)
            self.optimizer = self.build_optimizer(init_learning_rate)

    def build_optimizer(self, init_learning_rate):
        return tf.train.AdamOptimizer(init_learning_rate).minimize(self.loss)

    def train_model(self):
        num_steps = 20000

        train_dataset = self.dataset.get_train_set()
        train_labels = self.dataset.get_train_labels()
        # valid_labels = self.dataset.get_validation_labels()
        test_labels = self.dataset.get_test_labels()

        with tf.Session(graph=self.graph) as session:
            tf.global_variables_initializer().run()
            print('Initialized')
            for step in range(num_steps):
                offset = (step * self.batch_size) % (train_labels.shape[0] - self.batch_size)
                batch_data = train_dataset[offset:(offset + self.batch_size), :, :, :]
                batch_labels = train_labels[offset:(offset + self.batch_size), :]
                feed_dict = {self.tf_train_dataset : batch_data, self.tf_train_labels : batch_labels}
                _, l, predictions, train_embed_vec = session.run(
                    [self.optimizer, self.loss, self.train_prediction, self.train_embed_vec], feed_dict=feed_dict)
                if (step % 50 == 0):
                    print('batch_labels: {}'.format(np.argmax(batch_labels, 1)))
                    print('predictions: {}'.format(np.argmax(predictions, 1)))
                    print('Minibatch loss at step %d: %f' % (step, l))
                    print('Minibatch accuracy: %.1f%%' % self.accuracy(predictions, batch_labels))
                    # print('Validation accuracy: %.1f%%' % self.accuracy(
                    #     self.valid_prediction.eval(), valid_labels))
            print('Test accuracy: %.1f%%' % self.accuracy(self.test_prediction.eval(), test_labels))
            nn = nearest_neighbor.NearestNeighbor()
            print('Test accuracy for 1NN: %.3f' % nn.compute_one_nearest_neighbor_accuracy
                    (train_embed_vec, train_labels, self.test_embed_vec.eval(), test_labels))

        import collections
        print collections.Counter(tuple(np.argmax(train_labels,1)+1))
        print collections.Counter(tuple(np.argmax(test_labels,1)+1))

    def accuracy(self, predictions, labels):
        return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
                / predictions.shape[0])

if __name__ == '__main__':
    libras_model = Model()
    libras_model.build_model()
    libras_model.train_model()
