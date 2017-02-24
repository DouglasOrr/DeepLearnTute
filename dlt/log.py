'''Simple utility to ease plotting of training curves.
'''

import collections
import time
import matplotlib.pyplot as plt
import pandas as pd
import chainer as C


class Log:
    '''A simple training log plotting interface.
    For example:

        log = Log()
        log.add('loss', 'train', 2.3)
        log.add('loss', 'valid', 2.8)
        log.add('accuracy', 'train', 50)
        ...
        log.show()

    '''
    def __init__(self):
        self.clock = 0
        self.events = collections.defaultdict(
            lambda: collections.defaultdict(list))
        self.t0 = time.time()

    @property
    def elapsed(self):
        return time.time() - self.t0

    def add(self, kind, name, value):
        '''Add a logging event. Logging events of the same 'kind' (with
        different 'names') should have the same scale, as they will be plotted
        on a single axis.

        kind -- e.g. 'loss' or 'accuracy' - the type/units of log event

        name -- the name of the event (events with the same kind & name are
        treated as a time series)

        value -- scalar value of the event
        '''
        v = float(value.data if isinstance(value, C.Variable) else value)
        self.events[kind][name].append(dict(
            value=v,
            time=self.elapsed,
            event=self.clock,
        ))
        self.clock += 1

    def show(self, x='event', smooth=None, smooth_n=100):
        '''Display the logging events as a time series.

        x -- either 'event' (logical clock) or 'time' (wall clock)

        smooth - True | False | None - should we smooth the timeseries
                 True -- always smooth
                 False -- never smooth
                 None (defalt) -- auto smooth, when enough datapoints

        smooth_n - number of points to smooth 'towards', smaller numbers
                   (e.g. 10) will result in a very smooth (but possibly
                   misleading) curve, higher numbers (e.g. 1000) are less
                   smooth
        '''
        plt.figure(figsize=(16, 8 * len(self.events)))
        for i, (kind, entries) in enumerate(self.events.items()):
            ax = plt.subplot(len(self.events), 1, i + 1)
            for items in entries.values():
                df = pd.DataFrame.from_dict(items)
                if smooth is True or (smooth is None and smooth_n < len(df)):
                    df['smoothed'] = df['value'].ewm(
                        com=len(df) / smooth_n).mean()
                    df.plot(x=x, y='smoothed', ax=ax)
                else:
                    df.plot(x=x, y='value', ax=ax)
            plt.legend(list(entries.keys()))
            plt.title(kind)
            plt.ylabel(kind)
            if x == 'time':
                plt.xlabel('time /s')
