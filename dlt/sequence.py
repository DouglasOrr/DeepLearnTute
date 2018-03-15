'''Adaptation of the UJI dataset for the "sequential" version of the problem,
rather than the rasterized Dataset from dlt.data.
'''

import numpy as np
import matplotlib.pyplot as plt
import json


class Dataset:
    def __init__(self, vocab, points, breaks, masks, labels):
        self.vocab = vocab
        self.points = points
        self.breaks = breaks
        self.masks = masks
        self.labels = labels

    def find(self, char):
        label = int(np.where(self.vocab == char)[0])
        return np.where(self.labels == label)[0]

    def show(self, indices=None, limit=64):
        plt.figure(figsize=(16, 16))
        indices = list(range(limit) if indices is None else indices)
        dim = int(np.ceil(np.sqrt(len(indices))))
        for plot_index, index in enumerate(indices):
            plt.subplot(dim, dim, plot_index+1)
            plt.plot(*zip(*self.points[index, self.masks[index]]))
            ends = self.masks[index] & (
                self.breaks[index] | np.roll(self.breaks[index], -1))
            plt.plot(*zip(*self.points[index, ends]), '.')
            plt.title('%d : %s' % (index, self.vocab[self.labels[index]]))
            plt.gca().invert_yaxis()
            plt.gca().set_aspect('equal')
            plt.gca().axis('off')

    @classmethod
    def load(cls, path, max_length=200):
        '''Read the dataset from a JSONlines file.'''
        with open(path) as f:
            data = [json.loads(line) for line in f]

        vocab = np.array(sorted(set(d['target'] for d in data)))
        char_to_index = {ch: n for n, ch in enumerate(vocab)}
        labels = np.array([char_to_index[d['target']] for d in data],
                          dtype=np.int32)

        nsamples = min(max_length, max(
            sum(len(stroke) for stroke in d['strokes']) for d in data))
        points = np.zeros((len(data), nsamples, 2), dtype=np.float32)
        breaks = np.zeros((len(data), nsamples), dtype=np.bool)
        masks = np.zeros((len(data), nsamples), dtype=np.bool)
        for n, d in enumerate(data):
            stroke = np.concatenate(d['strokes'])[:nsamples]
            points[n, :len(stroke)] = stroke
            masks[n, :len(stroke)] = True
            all_breaks = np.cumsum([len(stroke) for stroke in d['strokes']])
            breaks[n, all_breaks[all_breaks < nsamples]] = True

        return cls(vocab=vocab,
                   points=points,
                   breaks=breaks,
                   masks=masks,
                   labels=labels)
