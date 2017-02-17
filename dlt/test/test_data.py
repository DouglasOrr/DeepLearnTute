from .. import data
import numpy as np
from nose.tools import eq_
import numpy.testing as npt


def test_parse_uji():
    npt.assert_equal([], list(data.parse_uji([])))

    npt.assert_equal([('a', [[(0, 100), (-200, 300), (456, 777)]]),
                      ('b', [[(1, 2), (3, 4)],
                             [(5, 6), (7, 8), (9, -1)]])],
                     list(data.parse_uji('''
//
// UJI: 100 units per millimetre
//
// ASCII char: a
WORD a some-arbitrary-STRING
  NUMSTROKES 1
    POINTS 3 # 0 100 -200 300 456 777

// more WORD NUMSTROKES POINTS to come...
WORD b some-arbitrary-STRING
  NUMSTROKES 2
    POINTS 2 # 1 2 3 4
    POINTS 3 # 5 6 7 8 9 -1
                     '''.split('\n'))))


def test_augmentations():
    box = [np.array([(0, 0), (1, 0), (1, 1), (0, 1)], dtype=np.float),
           np.array([(0.5, 0.5)], dtype=np.float)]

    augmented = list(data._augmentations(
        box, stretches=[2.0], rotations=[np.pi / 4]
    ))
    eq_(len(augmented), 9)
