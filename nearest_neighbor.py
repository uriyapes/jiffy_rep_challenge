import numpy as np

class NearestNeighbor(object):
    def __init__(self):
        pass
    
    def compute_one_nearest_neighbor_accuracy(self, train_dataset, train_labels, test_dataset, test_labels):
        accuracy = 0
        train_dataset_flat = np.reshape(train_dataset, (train_dataset.shape[0], -1), order='F')
        test_dataset_flat = np.reshape(test_dataset, (test_dataset.shape[0], -1), order='F')

        for ii in range(test_dataset.shape[0]): 
            ind = np.argmin(np.sum((train_dataset_flat - test_dataset_flat[ii])**2, axis=1))

            accuracy += np.all(train_labels[ind]==test_labels[ii])

        accuracy = accuracy / float(test_dataset.shape[0])

        return accuracy
