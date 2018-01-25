import pickle
import numpy as np
import matplotlib.pyplot as plt

class Plotter(object):
    def __init__(self, samples, embeddings, labels):
        self.samples = samples
        self.embeddings = embeddings
        self.labels = labels

    def time(self, c, d):
        plt.figure()
        for ii in np.nonzero(self.labels == c)[0]:
            plt.plot(self.samples[ii,:,d,:])

    def embedding(self, c):
        plt.figure()
        for ii in np.nonzero(self.labels == c)[0]:
            plt.plot(self.embeddings[ii,:])

if __name__=='__main__':
    test_set, test_embed_vec, test_labels = pickle.load(open('libras_results.pkl'))

    ts = test_set
    te = test_embed_vec
    l = test_labels

    plotter = Plotter(ts, te, l)

    plt.ion()
