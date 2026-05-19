import os, json, shutil, time, random, string
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    def __init__(self, *args, **kwargs):
        super(NumpyEncoder, self).__init__(*args, **kwargs)
        import torch
        self.torch = torch

    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.complex64, np.complex128)):
            return {'real': obj.real, 'imag': obj.imag}
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        elif isinstance(obj, (np.void,)):
            return None
        elif isinstance(obj, self.torch.dtype):
            return str(obj)
        return str(obj)


def do_save_dicts(save_dicts, save_dir, do_zip=True):
    for key, value in save_dicts.items():
        string_result = json.dumps(value, indent=4, cls=NumpyEncoder)
        path = os.path.join(save_dir, f'{key}.txt')
        with open(path, "w") as f:
            f.write(string_result)
    if do_zip:
        shutil.make_archive(save_dir, 'zip', save_dir)


def timerand_string():
    named_tuple = time.localtime()
    time_string = time.strftime("%Y-%m-%d--%H-%M-%S--", named_tuple)
    characters = string.ascii_letters + string.digits
    random_string = ''.join(random.choice(characters) for i in range(5))
    return time_string + '-' + random_string
