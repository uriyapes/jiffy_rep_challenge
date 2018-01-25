import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import ShuffleSplit

FILENAME = r'datasets/movement_libras.data'
D = 2
train_set_percentage = 0.8
validation_set_percentage = 0.0
test_set_percentage = 0.2
percentages = (train_set_percentage,
        validation_set_percentage,
        test_set_percentage)
num_labels = 15
min_label = 1
max_label = min_label + num_labels - 1

assert train_set_percentage + \
        validation_set_percentage + \
        test_set_percentage == 1.0

class Dataset(object):
    def __init__(self):
        samples, labels = self.read_from_csv(FILENAME)

        N, T, _, _ = samples.shape
        assert (N, T, D, 1) == samples.shape
        assert samples[0,0,0] == 0.79691
        assert samples[1,0,1] == 0.27315
        assert samples[N-1,T-1,1] == 0.49306

        train_set, train_labels, test_set, test_labels = \
                self.generate_balanced_splits(samples, labels)

        train_set_len = train_set.shape[0]
        if validation_set_percentage == 0:
            validation_set_len = 0
        else:
            validation_set_len = validation_set.shape[0]
        test_set_len = test_set.shape[0]
        assert train_set_len + validation_set_len + test_set_len == N

        self.train_set = train_set
        # self.validation_set = validation_set
        self.test_set = test_set
        self.train_labels = self.encode_one_hot(train_labels)
        # self.validation_labels = validation_labels
        self.test_labels = self.encode_one_hot(test_labels)
        self.T = T
        self.D = D

    def get_train_set(self):
        return self.train_set.astype(np.float32)

    def get_validation_set(self):
        return self.validation_set.astype(np.float32)

    def get_test_set(self):
        return self.test_set.astype(np.float32)

    def get_dimensions(self):
        return (self.T, self.D)

    def get_num_of_labels(self):
        return num_labels

    def get_train_labels(self):
        return self.train_labels

    def get_validation_labels(self):
        return self.validation_labels

    def get_test_labels(self):
        return self.test_labels

    @classmethod
    def read_from_csv(self, filename):
        data_from_csv = np.genfromtxt(FILENAME, delimiter=',')

        labels = data_from_csv[:,-1]
        samples_from_csv = data_from_csv[:,:-1]
        # Assumes D==2
        assert D == 2
        samples_d1 = samples_from_csv[:,0::D]
        samples_d2 = samples_from_csv[:,1::D]
        samples = np.stack((samples_d1, samples_d2), axis=-1)
        samples = samples[:,:,:,None]

        return samples, labels

    @classmethod
    def generate_balanced_splits(cls, samples, labels):
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_set_percentage, random_state=0)
        sss.get_n_splits(samples, labels)
        for train_index, test_index in sss.split(samples, labels):
            train_set = samples[train_index]
            train_labels = labels[train_index]
            test_set = samples[test_index]
            test_labels = labels[test_index]

        return train_set, train_labels, test_set, test_labels

    @classmethod
    def encode_one_hot(self, class_labels):
        labels_one_hot = \
                (np.arange(min_label, max_label + 1) == \
                    class_labels[:,None]).astype(np.float32)
        return labels_one_hot

    @classmethod
    def balanced_splits(cls, labels, min_label, max_label, percentages):
        balanced_splits = []
        for ii in range(len(percentages)):
            balanced_splits.append(np.array([], dtype=int))

        for label in range(min_label, max_label + 1):
            label_indices = np.nonzero(labels == label)[0]
            label_indices = cls.shuffle(label_indices)
            label_len = len(label_indices)

            cur_place = 0
            for ii in range(len(percentages)):
                split_len = int(round(percentages[ii]*label_len))
                split_indices = label_indices[cur_place:(cur_place+split_len)]
                balanced_splits[ii] = np.append(balanced_splits[ii], split_indices)
                cur_place += split_len

        assert np.sum([len(bs) for bs in balanced_splits]) == len(labels)

        for ii in range(len(balanced_splits)):
            balanced_splits[ii] = cls.shuffle(balanced_splits[ii])

        return balanced_splits

    @classmethod
    def shuffle(cls, vector):
        permutations = np.random.permutation(len(vector))
        return vector[permutations]

#  if __name__=='__main__':
    #  ds = Dataset()
    #  #  labels = np.concatenate([1 * np.ones(11), 2 * np.ones(11), 3 * np.ones(11)]).astype(int)
    #  labels = np.concatenate([ii*np.ones(24) for ii in range(1,16)]).astype(int)
    #  percentages = [0.7, 0.1, 0.2]
    #  indices = ds.balanced_splits(labels, 1, 15, percentages)
    #  print indices
    #  xx = [labels[indices[ii]] for ii in range(len(percentages))]
    #  for ii in xx:
        #  print ii
    #  import collections
    #  for ii in xx:
        #  print collections.Counter(ii)
