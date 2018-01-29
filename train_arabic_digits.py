import math
import tensorflow as tf
import parse_libras
import parse_arabic_digits
import nearest_neighbor
import numpy as np
from tensorflow.python.ops import gen_nn_ops

class Model(object):
    def __init__(self, embed_vec_size = 40):
        self.embed_vec_size = embed_vec_size

    def get_dataset(self, dataset_name = "libras"):
        self.dataset = parse_arabic_digits.Dataset()

    def build_model(self):
        T, D = self.dataset.get_dimensions()
        num_labels = self.dataset.get_num_of_labels()
        num_channels = 1

        self.batch_size = 300
        self.patch_t_size = 5
        self.patch_D_size = 1
        depth = 16
        num_hidden = self.embed_vec_size
        max_pool_percentage = 0.1
        self.max_pool_window_size = round(max_pool_percentage * T)
        max_pool_out_size = int(math.ceil(T / self.max_pool_window_size))
        init_learning_rate = 2e-5


        # Input data.
        self.tf_train_minibatch = tf.placeholder(
            tf.float32, shape=(self.batch_size, T, D, num_channels), name="train_dataset_placeholder")
        self.tf_train_labels = tf.placeholder(tf.float32, shape=(self.batch_size, num_labels),
                                              name="train_labels_placeholder")
        self.tf_train_dataset = tf.constant(self.dataset.get_train_set())
        # tf_valid_dataset = tf.constant(self.dataset.get_validation_set())
        tf_test_dataset = tf.constant(self.dataset.get_test_set())

        self.max_pool_window_size_ph = tf.placeholder(tf.int32, shape=())

        # Variables.
        layer1_weights = tf.Variable(tf.truncated_normal(
            [self.patch_t_size, self.patch_D_size, num_channels, depth], stddev=0.1))
        layer1_biases = tf.Variable(tf.zeros([depth]))

        layer2_weights = tf.Variable(tf.truncated_normal(
            [max_pool_out_size * D * depth, num_hidden], stddev=0.1))
        layer2_biases = tf.Variable(tf.zeros([num_hidden]))

        layer3_weights = tf.Variable(tf.truncated_normal(
            [num_hidden, num_labels], stddev=0.1))
        layer3_biases = tf.Variable(tf.zeros([num_labels]))

        # Model.
        def model(data):
            conv = tf.nn.conv2d(data, layer1_weights, [1, 1, 1, 1], padding='SAME')
            hidden = tf.nn.relu(conv + layer1_biases)
            hidden = gen_nn_ops._max_pool_v2(
                    hidden,
                    ksize=[1, self.max_pool_window_size_ph, 1, 1],
                    strides=[1, self.max_pool_window_size_ph, 1, 1],
                    padding='SAME')
            N = data.get_shape().as_list()[0]
            reshape = tf.reshape(hidden, [N, 10 * D * depth])
            hidden_no_relu = tf.matmul(reshape, layer2_weights) + layer2_biases
            hidden = tf.nn.relu(hidden_no_relu)
            return (tf.matmul(hidden, layer3_weights) + layer3_biases), hidden

        # Training computation.
        # Predictions for the training, validation, and test data.
        logits, _ = model(self.tf_train_minibatch)
        _, self.train_embed_vec = model(self.tf_train_dataset)
        self.train_prediction = tf.nn.softmax(logits)

        # valid_logits, _ = model(tf_valid_dataset)
        # self.valid_prediction = tf.nn.softmax(valid_logits)

        self.test_logits, self.test_embed_vec = model(tf_test_dataset)
        self.test_prediction = tf.nn.softmax(self.test_logits)

        self.loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=self.tf_train_labels, logits=logits))
        # Optimizer.
        self.optimizer = self.build_optimizer(init_learning_rate)


    def train_model(self):
        num_steps = 20000
        self.initial_train_labels = np.copy(self.dataset.get_train_labels())
        # valid_labels = self.dataset.get_validation_labels()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        with self.sess.as_default():
            print('Initialized')
            self.mini_batch_step = 0
            for step in range(num_steps):
                batch_data, batch_labels = self.get_mini_batch()
                feed_dict = {self.tf_train_minibatch : batch_data, self.tf_train_labels : batch_labels,
                        self.max_pool_window_size_ph : self.max_pool_window_size}
                _, l, predictions = self.sess.run(
                    [self.optimizer, self.loss, self.train_prediction], feed_dict=feed_dict)
                if (step % 50 == 0):
                    print('batch_labels: {}'.format(np.argmax(batch_labels, 1)))
                    print('predictions: {}'.format(np.argmax(predictions, 1)))
                    print('Minibatch loss at step %d: %f' % (step, l))
                    print('Minibatch accuracy: %.3f' % self.accuracy(predictions, batch_labels))
                    # print('Validation accuracy: %.1f%%' % self.accuracy(
                    #     self.valid_prediction.eval(), valid_labels))


        import collections
        print collections.Counter(tuple(np.argmax(batch_labels,1)+1))
        print collections.Counter(tuple(np.argmax(self.dataset.test_labels,1)+1))



    def eval_model(self):
        assert (self.initial_train_labels != self.dataset.train_set)
        test_labels = self.dataset.get_test_labels()
        with self.sess.as_default():
            network_acc = self.accuracy(self.test_prediction.eval(feed_dict={self.max_pool_window_size_ph: self.max_pool_window_size}), test_labels)
            print('Test accuracy of the network classifier: %.3f' % network_acc)

            train_embed_vec_res = self.train_embed_vec.eval(feed_dict={self.max_pool_window_size_ph: self.max_pool_window_size})
            test_embed_vec_res = self.test_embed_vec.eval(feed_dict={self.max_pool_window_size_ph: self.max_pool_window_size})

            nn_acc = self.run_baseline(train_embed_vec_res, self.initial_train_labels,
                                       test_embed_vec_res, self.dataset.test_labels)

            print('Test accuracy for 1NN: %.3f' % nn_acc)

        return self.dataset.get_test_set(), test_embed_vec_res, (np.argmax(self.dataset.get_test_labels(), 1) + 1), \
                        network_acc, nn_acc


    def get_mini_batch(self):
        offset = (self.mini_batch_step * self.batch_size)
        if (offset + self.batch_size) > self.dataset.train_labels.shape[0]:
            offset = 0
            self.mini_batch_step = 0
            self.dataset.re_shuffle()

        batch_data = self.dataset.train_set[offset:(offset + self.batch_size), :, :, :]
        batch_labels = self.dataset.train_labels[offset:(offset + self.batch_size), :]
        self.mini_batch_step += 1
        return batch_data, batch_labels


    def accuracy(self, predictions, labels):
        return (1.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
                / predictions.shape[0])

    def run_baseline(self, train_set, train_labels, test_set, test_labels):
        nn = nearest_neighbor.NearestNeighbor()
        return nn.compute_one_nearest_neighbor_accuracy(train_set, train_labels, test_set, test_labels)

    def build_optimizer(self, init_learning_rate):
        # self.optimizer = tf.train.GradientDescentOptimizer(0.005).minimize(self.loss)
        return tf.train.AdamOptimizer(init_learning_rate).minimize(self.loss)


if __name__ == '__main__':
    arabic_model = Model()
    arabic_model.get_dataset()
    # arabic_model.dataset.pca_scatter_plot(arabic_model.dataset.test_set)
    # print('1NN Baseline accuarcy: %.3f' % arabic_model.run_baseline(arabic_model.dataset.train_set,
    #                                                                 arabic_model.dataset.train_labels,
    #                                                                 arabic_model.dataset.test_set,
    #                                                                 arabic_model.dataset.test_labels))
    arabic_model.build_model()
    arabic_model.train_model()
    test_set, test_embed_vec, test_labels, network_acc, nn_acc = arabic_model.eval_model()
    # arabic_model.dataset.pca_scatter_plot(arabic_model.test_embed_vec_result)