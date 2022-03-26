
import json
import pickle

# np_json class from https://stackoverflow.com/questions/30698004/how-can-i-serialize-a-numpy-array-while-preserving-matrix-dimensions
from json import *
import json
import numpy as np
import base64

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



