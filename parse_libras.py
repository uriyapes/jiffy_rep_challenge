
import numpy as np

FILENAME = r'datasets/movement_libras.data'
D = 2
training_set_percentage = 0.7
validation_set_percentage = 0.1
test_set_percentage = 0.2
percentages = (training_set_percentage,
        validation_set_percentage,
        test_set_percentage)
num_labels = 15
min_label = 1
max_label = min_label + num_labels - 1

assert training_set_percentage + \
        validation_set_percentage + \
        test_set_percentage == 1.0

class Dataset(object):
    def __init__(self, random_permutation=False):
        with open(FILENAME, 'r') as f:
            lines = f.readlines()

        N = len(lines)
        T = (len(lines[0].split(','))-1) / D

        # Check that T*D == number of measurements in sample
        assert T*D == (len(lines[0].split(','))-1)

        #  samples = np.array(N, T, D, 1)
        data_from_csv = np.genfromtxt(FILENAME, delimiter=',')

        labels = data_from_csv[:,-1]
        labels_one_hot = \
                (np.arange(min_label, max_label + 1) == \
                labels[:,None]).astype(np.float32)

        assert labels_one_hot[359, 14] == 1

        balanced_splits = self.balanced_splits(labels,
                min_label,
                max_label,
                percentages)

        samples = data_from_csv[:,:-1]
        samples_x = samples[:,0::D]
        samples_y = samples[:,1::D]
        dataset = np.stack((samples_x, samples_y), axis=-1)
        dataset = dataset[:,:,:,None]

        assert (N, T, D, 1) == dataset.shape
        assert dataset[0,0,0] == 0.79691
        assert dataset[1,0,1] == 0.27315
        assert dataset[N-1,T-1,1] == 0.49306

        training_set = dataset[balanced_splits[0], :, :, :]
        training_labels = labels_one_hot[balanced_splits[0], :]

        validation_set = dataset[balanced_splits[1], :, :, :]
        validation_labels = labels_one_hot[balanced_splits[1], :]

        test_set = dataset[balanced_splits[2], :, :, :]
        test_labels = labels_one_hot[balanced_splits[2], :]

        training_set_len = training_set.shape[0]
        validation_set_len = validation_set.shape[0]
        test_set_len = test_set.shape[0]
        assert training_set_len + validation_set_len + test_set_len == N

        self.training_set = training_set
        self.validation_set = validation_set
        self.test_set = test_set
        self.training_labels = training_labels
        self.validation_labels = validation_labels
        self.test_labels = test_labels
        self.N = N
        self.T = T
        self.D = D

    def get_training_set(self):
        return self.training_set.astype(np.float32)

    def get_validation_set(self):
        return self.validation_set.astype(np.float32)

    def get_test_set(self):
        return self.test_set.astype(np.float32)

    def get_dimensions(self):
        return (self.N, self.T, self.D)

    def get_num_of_labels(self):
        return num_labels

    def get_train_labels(self):
        return self.training_labels

    def get_validation_labels(self):
        return self.validation_labels

    def get_test_labels(self):
        return self.test_labels

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
