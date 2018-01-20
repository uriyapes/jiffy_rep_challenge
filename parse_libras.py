
import numpy as np

FILENAME = r'datasets/movement_libras.data'
D = 2
training_set_percentage = 0.8
validation_set_percentage = 0.0
test_set_percentage = 0.2
num_labels = 15
min_label = 1
max_label = min_label + num_labels - 1

assert training_set_percentage + \
        validation_set_percentage + \
        test_set_percentage == 1.0

class Dataset(object):
    def __init__(self):
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

        samples = data_from_csv[:,:-1]
        samples_x = samples[:,0::D]
        samples_y = samples[:,1::D]
        dataset = np.stack((samples_x, samples_y), axis=-1)
        dataset = dataset[:,:,:,None]

        assert (N, T, D, 1) == dataset.shape
        assert dataset[0,0,0] == 0.79691
        assert dataset[1,0,1] == 0.27315
        assert dataset[N-1,T-1,1] == 0.49306

        training_set_len = int(round(training_set_percentage*N))
        training_set = dataset[:training_set_len, :, :, :]
        training_labels = labels_one_hot[:training_set_len, :]

        cur_place = training_set_len
        validation_set_len = int(round(validation_set_percentage*N))
        validation_set = dataset[cur_place:(cur_place + validation_set_len), :, :, :]
        validation_labels = labels_one_hot[cur_place:(cur_place + validation_set_len), :]

        cur_place = training_set_len + validation_set_len
        test_set_len = int(round(test_set_percentage*N))
        test_set = dataset[cur_place:(cur_place + test_set_len), :, :, :]
        test_labels = labels_one_hot[cur_place:(cur_place + test_set_len), :]

        assert training_set_len + validation_set_len + test_set_len == N

        self.training_set = training_set
        self.validation_set = validation_set
        self.test_set = test_set
        self.N = N
        self.T = T
        self.D = D

    def get_training_set(self):
        return self.training_set

    def get_validation_set(self):
        return self.validation_set

    def get_test_set(self):
        return self.test_set

    def get_dimensions(self):
        return (self.N, self.T, self.D)

    def get_num_of_labels(self):
        return num_labels
