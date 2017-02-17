import string
import random
import os
import weakref
import json
import numpy as np
from . import data


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
        x = data.render([np.array(s) for s in d['strokes']], 16).reshape(-1)
        y = cls._INSTANCES[d['id']].classify(x.astype(np.float32))
        # Our JS can't cope with the JSON "'", so replace it with "\u0027"
        return json.dumps(dict(x=x.tolist(), y=y)).replace("'", "\\u0027")

    def __init__(self, classify, width=None, height=None):
        self._id = '%04x' % random.randint(0, 1 << 16)
        self._INSTANCES[self._id] = self
        self._classify = classify

        if height is None and width is None:
            height = width = 16
        elif height is None:
            height = width
        elif width is None:
            width = height
        self._width = width
        self._height = height

    def classify(self, data):
        return self._classify(data)

    def _repr_html_(self):
        return string.Template('''

        <canvas id="${ID}-input-canvas" width="256" height="256"
                style="border-style: solid;">
        </canvas>
        <label id="${ID}-label"
               style="font-size: 12em; width: 1em; text-align: center;"
        >?</label>
        <canvas id="${ID}-output-canvas" width="${WIDTH}" height="${HEIGHT}"
            style="width: 256px; height: 256px;">
        </canvas>
        <p>
            <button id="${ID}-clear"
                style="width: 256px; height: 3em;"
            >Clear (middle click)</button>
        </p>
        <p><pre id="${ID}-error" style="color: #f00;"></pre></p>
        <p><pre id="${ID}-output"></pre></p>
        <script type="text/javascript">
            (function () {
                var custom_input_id = "#${ID}";
                ${JS}
            })()
        </script>

        ''').substitute(ID=self._id,
                        JS=self.JS,
                        WIDTH=self._width,
                        HEIGHT=self._height)
