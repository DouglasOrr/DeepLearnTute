'''Individual character handwriting data loading and preprocessing.

Based on https://archive.ics.uci.edu/ml/datasets/UJI+Pen+Characters+(Version+2)
'''

import re
import click
import itertools as it
import random
import json
import numpy as np
import h5py
import matplotlib.pyplot as plt


def parse_uji(lines):
    '''Parse the UJI pen characters dataset text format.

    lines -- a sequence of lines, in the UJIv2 dataset format

    yields -- (character, strokes), where character is a unicode string,
              strokes is a list numpy arrays of shape (npoints, 2).
    '''
    COMMENT = re.compile(r'//.*$')
    WORD = re.compile(r'WORD (.)')
    NUMSTROKES = re.compile(r'NUMSTROKES (\d+)')
    POINTS = re.compile(r'POINTS (\d+) # ((?:[-0-9]+ ?)+)')
    word = None
    numstrokes = None
    strokes = None
    for line in lines:
        line = COMMENT.sub('', line).strip()
        if line == '':
            continue

        m = WORD.match(line)
        if m is not None:
            word = m.group(1)
            continue
        if word is None:
            raise ValueError('Expected WORD...')

        m = NUMSTROKES.match(line)
        if m is not None:
            numstrokes = int(m.group(1))
            strokes = []
            continue
        if numstrokes is None:
            raise ValueError('Expected NUMSTROKES...')

        m = POINTS.match(line)
        if m is not None:
            samples = [int(t) for t in m.group(2).split(' ')]
            points = list(zip(samples[::2], samples[1::2]))
            if len(points) != int(m.group(1)):
                raise ValueError(
                    "Unexpected number of points (expected %d, actual %d)"
                    % (int(m.group(1)), len(points)))
            strokes.append(np.array(points, dtype=np.float32))
            if len(strokes) == numstrokes:
                yield (word, strokes)
                word = numstrokes = strokes = None
            continue

        raise ValueError("Input not matched '%s'" % line)


def _first(x):
    '''Return the first element of an array or tuple.
    '''
    return x[0]


def _dump_jsonlines(file, data):
    '''Dump a dataset of strokes to a file, in the simple JSONlines format.
    '''
    for ch, strokes in data:
        file.write('%s\n' % json.dumps(dict(
            target=ch,
            strokes=[stroke.tolist() for stroke in strokes])))


def _rotate(angle):
    '''Create a 2D rotation matrix for the given angle.
    '''
    cos = np.cos(angle)
    sin = np.sin(angle)
    return np.array([[cos, -sin],
                     [sin, cos]], dtype=np.float32)


def _scale(x, y):
    return np.array([[x, 0],
                     [0, y]], dtype=np.float32)


def _augmentations(strokes,
                   stretches=[1.2],
                   rotations=[0.2, 0.4]):
    '''Generate augmented versions of 'strokes', but applying some
    rotations & stretching.

    strokes -- a multidimensional list [stroke x point x 2], of stroke points
    for a single character

    yields -- a multidimensional list of the same form of `strokes`
    '''
    sum_x = sum(sum(x for x, y in stroke) for stroke in strokes)
    sum_y = sum(sum(y for x, y in stroke) for stroke in strokes)
    n = sum(map(len, strokes))
    center = np.array([sum_x, sum_y], dtype=np.float32) / n

    norm_strokes = [np.array(stroke) - center for stroke in strokes]

    scale_transforms = [_scale(x, 1) for x in ([1] +
                                               stretches +
                                               [1 / s for s in stretches])]
    rotate_transforms = [_rotate(a) for a in ([0] +
                                              rotations +
                                              [-r for r in rotations])]
    for scale in scale_transforms:
        for rotate in rotate_transforms:
            tx = np.dot(scale, rotate)
            yield [np.dot(trace, tx) + center for trace in norm_strokes]


def _rem_lo(x):
    return x - np.floor(x)


def _rem_hi(x):
    return np.ceil(x) - x


def _draw_line(start, end):
    '''Enumerate coordinates of an antialiased line, using Xiaolin Wu's line
    algorithm.

    start -- floating point coordinate pair of line start

    end -- floating point coordinate of line end

    yields -- (x, y, strength) for an antialiased line between start and end,
              where x and y are integer coordinates
    '''
    x0, y0 = start
    x1, y1 = end

    # Simplify case - only draw "shallow" lines
    if abs(x1 - x0) < abs(y1 - y0):
        yield from ((x, y, weight)
                    for y, x, weight in _draw_line((y0, x0), (y1, x1)))
        return

    # Transform so we run low-to-high-x
    if x1 < x0:
        x0, y0, x1, y1 = x1, y1, x0, y0

    # Note: we know dy <= dx, so gradient <= 1
    gradient = 1.0 if x1 == x0 else (y1 - y0) / (x1 - x0)

    # Start of line termination
    xend0 = int(np.round(x0))
    yend0 = y0 + gradient * (xend0 - x0)
    yield (xend0, int(yend0),     _rem_hi(yend0) * _rem_hi(x0 + 0.5))
    yield (xend0, int(yend0) + 1, _rem_lo(yend0) * _rem_hi(x0 + 0.5))

    # End of line termination
    xend1 = int(np.round(x1))
    yend1 = y1 + gradient * (xend1 - x1)
    yield (xend1, int(yend1),     _rem_hi(yend1) * _rem_lo(x1 + 0.5))
    yield (xend1, int(yend1) + 1, _rem_lo(yend1) * _rem_lo(x1 + 0.5))

    # Line drawing loop
    y = yend0 + gradient
    for x in range(xend0 + 1, xend1):
        yield (x, int(y), _rem_hi(y))
        yield (x, int(y) + 1, _rem_lo(y))
        y += gradient


def render(strokes, size):
    '''Render a sequence of strokes to a square numpy array of pixels.

    strokes -- a list of float[N x 2] numpy arrays, arbitrary coordinates

    size -- the side length of the array to render to

    returns -- a float[size x size] containing the image (leading index is y)
    '''
    x_min = min(s[:, 0].min() for s in strokes)
    x_max = max(s[:, 0].max() for s in strokes)
    y_min = min(s[:, 1].min() for s in strokes)
    y_max = max(s[:, 1].max() for s in strokes)
    x_scale = (size - 3) * (1 if x_min == x_max else 1 / (x_max - x_min))
    y_scale = (size - 3) * (1 if y_min == y_max else 1 / (y_max - y_min))
    scale = min(x_scale, y_scale)
    x_off = (size - 1) / (2 * scale) - (x_min + x_max) / 2
    y_off = (size - 1) / (2 * scale) - (y_min + y_max) / 2

    a = np.zeros((size, size), dtype=np.float32)
    for stroke in strokes:
        coords = [(scale * (p[0] + x_off), scale * (p[1] + y_off))
                  for p in stroke]
        for start, end in zip(coords, coords[1:]):
            for x, y, w in _draw_line(start, end):
                a[x, y] = max(a[x, y], w)
    return np.swapaxes(a, 0, 1)


class Dataset:
    '''An in-memory dataset of images & labels.

    dataset.x -- (N x D) array of np.float32, flattened images,
                 where the y-index is major

    dataset.y -- (N) array of np.int labels

    dataset.vocab -- (L) array of characters corresponding to the
                     human-readable labels
    '''
    def __init__(self, x, y, vocab, width, height):
        self.x = x
        self.y = y
        self.vocab = vocab
        self.char_to_index = {ch: i for i, ch in enumerate(vocab)}
        self.width = width
        self.height = height

    def __repr__(self):
        return 'Dataset[%d images, size %s, from %d labels]' % (
            len(self), self.x.shape[-1], len(self.vocab))

    def __len__(self):
        return self.x.shape[0]

    def find_label(self, char):
        return np.where(self.y == self.char_to_index[char])[0]

    def show(self, indices=None, limit=64):
        if indices is None:
            indices = range(limit)
        xs = self.x[indices]
        ys = self.y[indices]
        dim = int(np.ceil(np.sqrt(xs.shape[0])))

        plt.figure(figsize=(16, 16))
        for plot_index, index, x, y in zip(it.count(1), indices, xs, ys):
            plt.subplot(dim, dim, plot_index)
            plt.imshow(x.reshape(self.height, self.width))
            plt.title(r"$y_{%d}$ = %s" % (index, self.vocab[y]), fontsize=14)
            plt.gca().set_xticks([])
            plt.gca().set_yticks([])
            plt.gca().grid(False)


def load_hdf5(path):
    '''Load a Dataset object from an HDF5 file.
    '''
    with h5py.File(path, 'r') as f:
        x = f['x'][...]
        return Dataset(
            x=x.reshape(x.shape[0], -1),
            y=f['y'][...].astype(np.int32),
            vocab=f['vocab'][...],
            height=x.shape[1],
            width=x.shape[2]
        )


@click.command('read')
@click.argument('source', type=click.File('r'))
@click.argument('train', type=click.File('w'))
@click.argument('valid', type=click.File('w'))
@click.argument('test', type=click.File('w'))
@click.option('-f', '--label-filter',
              type=click.STRING, default='.',
              help='Regex to filter allowable labels.')
@click.option('--nvalid', type=click.INT, default=10,
              help='Number of validation examples per label.')
@click.option('--ntest', type=click.INT, default=10,
              help='Number of test examples per label.')
@click.option('--seed', type=click.INT, default=42,
              help='Seed for random number generation.')
@click.option('--augment/--no-augment', default=True,
              help='Should we use data augmentation (rotation & stretching).')
def cli_read(source, train, valid, test, label_filter,
             nvalid, ntest, seed, augment):
    '''Generate a JSONlines dataset from UJI pen characters.
    '''
    random.seed(seed)

    # Load & filter
    data = parse_uji(source)
    label_pattern = re.compile(label_filter)
    data = filter(lambda x: label_pattern.match(x[0]) is not None,
                  data)

    # Partition
    data = it.groupby(sorted(data, key=_first), _first)
    train_data = []
    valid_data = []
    test_data = []
    for char, examples in data:
        shuffled_examples = list(examples)
        random.shuffle(shuffled_examples)
        test_data += shuffled_examples[:ntest]
        valid_data += shuffled_examples[ntest:(ntest + nvalid)]
        train_data += shuffled_examples[(ntest + nvalid):]

    # Augment training data
    if augment:
        train_data = [(ch, ss)
                      for ch, strokes in train_data
                      for ss in _augmentations(strokes)]

    # Shuffle
    random.shuffle(train_data)
    random.shuffle(valid_data)
    random.shuffle(test_data)

    # Save
    _dump_jsonlines(train, train_data)
    _dump_jsonlines(valid, valid_data)
    _dump_jsonlines(test, test_data)


@click.command('render')
@click.argument('input', type=click.File('r'))
@click.argument('output', type=click.Path(dir_okay=False))
@click.option('-n', '--size', default=16, type=click.INT)
def cli_render(input, output, size):
    '''Render a JSONlines dataset to numpy arrays, saved in an HDF5 file.
    '''
    chars = []
    images = []
    for line in input:
        datum = json.loads(line)
        chars.append(datum['target'])
        images.append(render(
            [np.array(s) for s in datum['strokes']],
            size))

    vocab = list(sorted(set(chars)))
    char_to_index = {ch: y for y, ch in enumerate(vocab)}

    with h5py.File(output, 'a') as f:
        str_dt = h5py.special_dtype(vlen=str)
        f.require_dataset(
            'vocab', (len(vocab),), dtype=str_dt
        )[...] = vocab
        f.require_dataset(
            'x', shape=(len(images), size, size), dtype=np.float32
        )[...] = np.array(images)
        f.require_dataset(
            'y', shape=(len(chars),), dtype=np.int
        )[...] = np.array([char_to_index[ch] for ch in chars])


@click.group()
def cli():
    '''Base command for dataset processing.
    '''
    pass


cli.add_command(cli_read)
cli.add_command(cli_render)

if __name__ == '__main__':
    cli()
