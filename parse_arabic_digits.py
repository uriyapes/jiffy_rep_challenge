import os
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import ShuffleSplit

FILENAME_TRAINING_SET = r'datasets/Train_Arabic_Digit.txt'
FILENAME_TESTING_SET = r'datasets/Test_Arabic_Digit.txt'

MAX_T_SIZE = 100
D = 13
TR_SIZE = 6600
TE_SIZE = 2200
MIN_LABEL = 0
MAX_LABEL = 10
NUM_LABELS = MAX_LABEL-MIN_LABEL

class Dataset(object):
    def __init__(self):
        train_set, train_labels, test_set, test_labels = self.read_from_csv(FILENAME_TRAINING_SET, FILENAME_TESTING_SET)

        self.train_set = train_set
        self.test_set = test_set
        self.train_labels = self.encode_one_hot(train_labels)
        self.test_labels = self.encode_one_hot(test_labels)
        self.T = MAX_T_SIZE
        self.D = D

    def get_train_set(self):
        return self.train_set.astype(np.float32)

    def get_test_set(self):
        return self.test_set.astype(np.float32)

    def get_dimensions(self):
        return (self.T, self.D)

    def get_num_of_labels(self):
        return NUM_LABELS

    def get_train_labels(self):
        return self.train_labels

    def get_test_labels(self):
        return self.test_labels

    @classmethod
    def read_from_csv(self, src_path_tr, src_path_te):
        # src_path_tr = os.path.join(data_dir, path_info[0])
        f_tr = open(src_path_tr, 'r')
        X_tr = np.zeros((TR_SIZE,MAX_T_SIZE,D,1))
        cur_ts = []  # used in loop hold cur timeseries in the iteration
        num_samples = 0
        for (i, line) in enumerate(f_tr.readlines()):
            if line[0] != ' ' and (i != 0):
                no_newline_char = line[0:-1]
                cur_ts.append([float(n) for n in no_newline_char.split(" ")])
            if (line[0] == ' ') and (i != 0):  # first line is empty,and no empty line after last ts
                sample = np.expand_dims(np.array(cur_ts), axis=2)
                X_tr[num_samples,:sample.shape[0],:,:] += sample
                cur_ts = []
                num_samples += 1
        sample = np.expand_dims(np.array(cur_ts), axis=2)
        X_tr[num_samples,:sample.shape[0],:,:] += sample

        # src_path_te = os.path.join(data_dir, path_info[1])
        f_tr = open(src_path_te, 'r')
        X_te = np.zeros((TE_SIZE,MAX_T_SIZE,D,1))
        cur_ts = []
        num_samples = 0
        for (i, line) in enumerate(f_tr.readlines()):
            if line[0] != ' ' and (i != 0):
                no_newline_char = line[0:-1]
                cur_ts.append([float(n) for n in no_newline_char.split(" ")])
            if ((line[0] == ' ') and (i != 0)):  # first line is empty,and no empty line after last ts
                sample = np.expand_dims(np.array(cur_ts), axis=2)
                X_te[num_samples,:sample.shape[0],:,:] += sample
                cur_ts = []
                num_samples += 1
        sample = np.expand_dims(np.array(cur_ts), axis=2)
        X_te[num_samples,:sample.shape[0],:,:] += sample

        ###First 660 are digit 0 next 660 are digit 1 etc..
        ###Make training labels
        y_tr = [int(i / 660) for i in range(660 * 10)]

        ###Test data grouped into groups of 220
        ###Make test labels
        y_te = [int(i / 220) for i in range(220 * 10)]

        # return (np.expand_dims(X_tr, axis =3), y_tr, np.expand_dims(X_te, axis = 3), y_te)
        return (X_tr, y_tr, X_te, y_te)

    @classmethod
    def encode_one_hot(self, class_labels):
        labels_one_hot = \
                (np.arange(MIN_LABEL, MAX_LABEL) == \
                    np.array(class_labels)[:,None]).astype(np.float32)
        return labels_one_hot


    def pca_scatter_plot(self, data):
        X = np.reshape(data, (data.shape[0], -1), order='F')
        X_norm = (X - X.min()) / (X.max() - X.min())
        import matplotlib.pyplot as plt
        import pandas as pd
        from sklearn.decomposition import PCA as sklearnPCA
        pca = sklearnPCA(n_components=2, svd_solver='auto')  # 2-dimensional PCA
        transformed = pd.DataFrame(pca.fit_transform(X_norm))

        for i in xrange(10):
            plt.scatter(transformed[0][i*220:(i+1)*220], transformed[1][i*220:(i+1)*220], label='Class{}'.format(i))

        plt.legend()
        plt.show()
