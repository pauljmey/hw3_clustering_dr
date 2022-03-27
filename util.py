
import json
import pickle

# np_json class from https://stackoverflow.com/questions/30698004/how-can-i-serialize-a-numpy-array-while-preserving-matrix-dimensions
from json import *
import json
import numpy as np
import base64

import pandas
import seaborn



class np_json():
    def to_json(obj):
        if isinstance(obj, (np.ndarray, np.generic)):
            if isinstance(obj, np.ndarray):
                v1 = base64.b64encode(obj.tostring())
                ret = {
                    '__ndarray__': v1,
                    'dtype': obj.dtype.str,
                    'shape': obj.shape,
                }
                return ret
            elif isinstance(obj, (np.bool_, np.number)):
                return {
                    '__npgeneric__': base64.b64encode(obj.tostring()),
                    'dtype': obj.dtype.str,
                }
        if isinstance(obj, set):
            return {'__set__': list(obj)}
        if isinstance(obj, tuple):
            return {'__tuple__': list(obj)}
        if isinstance(obj, complex):
            return {'__complex__': obj.__repr__()}

        # Let the base class default method raise the TypeError
        raise TypeError('Unable to serialise object of type {}'.format(type(obj)))


    def from_json(obj):
        # check for numpy
        if isinstance(obj, dict):
            if '__ndarray__' in obj:
                return np.fromstring(
                    base64.b64decode(obj['__ndarray__']),
                    dtype=np.dtype(obj['dtype'])
                ).reshape(obj['shape'])
            if '__npgeneric__' in obj:
                return np.fromstring(
                    base64.b64decode(obj['__npgeneric__']),
                    dtype=np.dtype(obj['dtype'])
                )[0]
            if '__set__' in obj:
                return set(obj['__set__'])
            if '__tuple__' in obj:
                return tuple(obj['__tuple__'])
            if '__complex__' in obj:
                return complex(obj['__complex__'])

        return obj

    # over-write the load(s)/dump(s) functions
    def load(*args, **kwargs):
        kwargs['object_hook'] = np_json.from_json
        return json.load(*args, **kwargs)


    def loads(*args, **kwargs):
        kwargs['object_hook'] = np_json.from_json
        return json.loads(*args, **kwargs)


    def dump(*args, **kwargs):
        kwargs['default'] = np_json.to_json
        return json.dump(*args, **kwargs)


    def dumps(*args, **kwargs):
        kwargs['default'] = np_json.to_json
        return json.dumps(*args, **kwargs)




def bach_plot():
    '''
    1. Choral ID: corresponding to the file names from (Bach Central)[http://www.bachcentral.com].
   2. Event number: index (starting from 1) of the event inside the chorale.
   3-14. Pitch classes: YES/NO depending on whether a given pitch is present.
      Pitch classes/attribute correspondence is as follows:
        C       -> 3
        C#/Db   -> 4
        D       -> 5
        ...
        B       -> 14
   15. Bass: Pitch class of the bass note
   16. Meter: integers from 1 to 5. Lower numbers denote less accented events,
      higher numbers denote more accented events.
   17. Chord label: Chord resonating during the given event.
    '''
    import numpy as np
    from main import get_plot_fn
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import matplotlib


    columns = [
        'Choral_ID', 'Event_number', 'C',
        'C#/Db', 'D', 'D#/Eb',
        'E', 'F', 'F#/Gb',
        'G', 'G#/Ab', 'A',
        'A#/Bb', 'B', 'Bass',
        'Meter', 'Chord_label'
    ]

    df1 = pd.read_csv('./data/jsbach_chorals_harmony/jsbach_chorals_harmony.data')
    df1.columns = columns
    df1 = df1.drop(columns[0], 1)
    #ret = get_bach()
    #n1 = np.column_stack((ret[0], ret[1]))

    subset1= ['D_M',
        'G_M',  #:   489
        'C_M',  #:   488
        'F_M',  #:   389
        'A_M',  #:   352'
    ]
    filter = [x in subset1 for x in df1['Chord_label']]
    df1_filtered = df1[filter]
    matplotlib.rcParams['font.size'] = 18
    sns.set_context('talk', font_scale=1.2);

    plt.figure(figsize=(20, 20))
    sns.pairplot(df1_filtered, hue='Chord_label', corner=True)
    #plt.margins(10,10)

    plt.savefig(get_plot_fn(tag="bach_ds"))
    hook = True


if __name__ == '__main__':
    from main import get_digits
    from main import get_bach
    #bach_plot()
    digits_plot()
