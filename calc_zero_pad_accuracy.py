import numpy as np
import parse_libras
import nearest_neighbor

dataset = parse_libras.Dataset()

train_dataset = dataset.get_training_set()
train_labels = dataset.get_train_labels()
test_dataset = dataset.get_test_set()
test_labels = dataset.get_test_labels()

nn = nearest_neighbor.NearestNeighbor()

accuracy = nn.compute_one_nearest_neighbor_accuracy(train_dataset,
        train_labels,
        test_dataset,
        test_labels)

print '1NN, zero padding, Libras: {}'.format(accuracy)
