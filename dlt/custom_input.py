import string
import random
import os
import weakref
import json
import numpy as np


class CustomInput:
    '''An interactive IPython shape classifier.
    '''

    # Load javascript behaviour code
    with open(os.path.join(os.path.dirname(__file__),
                           'custom_input.js'),
              'r') as f:
        JS = f.read()

    _INSTANCES = weakref.WeakValueDictionary()

    @classmethod
    def _id_classify(cls, payload):
        d = json.loads(payload)
        return cls._INSTANCES[d["id"]].classify(
            np.array(d["data"], dtype=np.float32))

    def __init__(self, classify, width=None, height=None,
                 display_scale=None):
        self._id = '%04x' % random.randint(0, 1 << 16)
        self._INSTANCES[self._id] = self
        self._classify = classify

        if height is None and width is None:
            height = width = 28
        elif height is None:
            height = width
        elif width is None:
            width = height
        self._width = width
        self._height = height

        if display_scale is None:
            display_scale = max(width, 256) / width
        self._display_scale = display_scale

    def classify(self, data):
        return self._classify(data)

    def _repr_html_(self):
        return string.Template('''

        <canvas id="${ID}-canvas" width="${WIDTH}" height="${HEIGHT}"
            style="width: ${DISPLAY_WIDTH}px;
                   height: ${DISPLAY_HEIGHT}px;
                   border-style: solid;">
        </canvas>
        <p>
            <label id="${ID}-label" style="font-size: xx-large;">?</label>
            <button id="${ID}-classify">Classify</button>
            <button id="${ID}-clear">Clear</button>
        </p>
        <script type="text/javascript">
            (function () {
                var custom_input_id = "#${ID}";
                ${JS}
            })()
        </script>

        ''').substitute(ID=self._id,
                        JS=self.JS,
                        WIDTH=self._width,
                        HEIGHT=self._height,
                        DISPLAY_WIDTH=self._display_scale * self._width,
                        DISPLAY_HEIGHT=self._display_scale * self._height)
