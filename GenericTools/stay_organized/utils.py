import argparse, logging, os, random, time, gc, json, re
from time import strftime, localtime
import importlib.util
from tqdm import tqdm
from itertools import groupby

import numpy as np

# import tensorflow as tf
# from tqdm import tqdm

logger = logging.getLogger('mylogger')


def make_directories(time_string=None):
    experiments_folder = "experiments"
    if not os.path.isdir(experiments_folder):
        os.mkdir(experiments_folder)

    if time_string == None:
        time_string = strftime("%Y-%m-%d-at-%H:%M:%S", localtime())

    experiment_folder = experiments_folder + '/experiment-' + time_string + '/'

    if not os.path.isdir(experiment_folder):
        os.mkdir(experiment_folder)

        # create folder to save new models trained
    model_folder = experiment_folder + '/model/'
    if not os.path.isdir(model_folder):
        os.mkdir(model_folder)

        # create folder to save TensorBoard
    log_folder = experiment_folder + '/log/'
    if not os.path.isdir(log_folder):
        os.mkdir(log_folder)

    return experiment_folder


def plot_softmax_evolution(softmaxes_list, name='softmaxes'):
    import matplotlib.pylab as plt

    f = plt.figure()
    index = range(len(softmaxes_list[0]))
    for softmax in softmaxes_list:
        plt.bar(index, softmax)

    plt.xlabel('Token')
    plt.ylabel('Probability')
    plt.title('softmax evoluti\on during training')
    plt.show()
    f.savefig(name + ".pdf", bbox_inches='tight')


def checkDuringTraining(generator_class, indices_sentences, encoder_model, decoder_model, batch_size, lat_dim):
    # original sentences
    sentences = generator_class.indicesToSentences(indices_sentences)

    # reconstructed sentences
    point = encoder_model.predict(indices_sentences)
    indices_reconstructed, _ = decoder_model.predict(point)

    sentences_reconstructed = generator_class.indicesToSentences(indices_reconstructed)

    # generated sentences
    noise = np.random.rand(batch_size, lat_dim)
    indicess, softmaxes = decoder_model.predict(noise)
    sentences_generated = generator_class.indicesToSentences(indicess)

    from prettytable import PrettyTable

    table = PrettyTable(['original', 'reconstructed', 'generated'])
    for b, a, g in zip(sentences, sentences_reconstructed, sentences_generated):
        table.add_row([b, a, g])
    for column in table.field_names:
        table.align[column] = "l"
    print(table)

    print('')
    print('number unique generated sentences:   ', len(set(sentences_generated)))
    print('')
    print(softmaxes[0][0])
    print('')

    return softmaxes


def get_random_string():
    random_string = ''.join([str(r) for r in np.random.choice(10, 4)])
    # named_tuple = time.localtime()  # get struct_time
    # random_string = str(abs(hash(named_tuple)))[:4]
    return random_string


def timeStructured(random_string=True, seconds=False):
    named_tuple = time.localtime()  # get struct_time
    time_string = time.strftime("%Y-%m-%d--%H-%M-%S-", named_tuple)
    if random_string:
        # random_string = ''.join([str(r) for r in np.random.choice(10, 4)])
        # random_string = str(abs(hash(named_tuple)))[:4]
        time_string += '-' + get_random_string()

    if seconds:
        return time_string, time.time()
    return time_string


def setReproducible(seed=0, disableGpuMemPrealloc=True, prove_seed=True, tensorflow=False, pytorch=False,
                    empty_cuda=False):
    # Fix the seed of all random number generator
    random.seed(seed)
    np.random.seed(seed)

    if pytorch:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.random.manual_seed(seed)

    if prove_seed:
        print(np.random.rand())

    if tensorflow:
        import tensorflow as tf
        tf.random.set_seed(seed)

        config = tf.compat.v1.ConfigProto(
            intra_op_parallelism_threads=1,
            inter_op_parallelism_threads=1,
            device_count={'CPU': 1}
        )

        if disableGpuMemPrealloc:
            config.gpu_options.allow_growth = True

    if empty_cuda:
        gc.collect()

        from numba import cuda
        device = cuda.get_current_device()
        device.reset()
        cuda.select_device(0)
        cuda.close()

        if pytorch:
            torch.cuda.empty_cache()


def Dict2ArgsParser(args_dict):
    parser = argparse.ArgumentParser()
    for k, v in args_dict.items():
        if type(v) == bool and v == True:
            parser.add_argument("--" + k, action="store_true")
        elif type(v) == bool and v == False:
            parser.add_argument("--" + k, action="store_false")
        else:
            parser.add_argument("--" + k, type=type(v), default=v)
    args = parser.parse_args()
    return args


def move_mouse():
    import pyautogui, keyboard

    i = 0
    while True:
        i += 1
        print((-1) ** i * 1, (-1) ** (i + 1) * 1)
        # pyautogui.moveTo((-1)**i*.01, (-1)**(i+1)*.01, duration=1)
        # pyautogui.moveTo(1, 1, duration=1)
        pyautogui.moveRel((-1) ** i * 100, (-1) ** (i + 1) * 100, duration=1)
        if keyboard.is_pressed('q'):  # if key 'q' is pressed
            print('You Pressed A Key!')
            break  # finishing the loop


def str2val(comments, flag, output_type=float, default=None, split_symbol='_', equality_symbol=':', remove_flag=True,
            replace=None, exact_flag=False):
    if replace is None:
        if exact_flag:
            condition = lambda s: s.startswith('{}{}'.format(flag, equality_symbol))
        else:
            condition = lambda s: '{}{}'.format(flag, equality_symbol) in s

        if '{}{}'.format(flag, equality_symbol) in comments:
            flags_detected = [
                s.replace(
                    '{}{}'.format(flag if remove_flag else '', equality_symbol), ''
                )
                for s in comments.split(split_symbol)
                if condition(s)
            ]
            flags_detected = sorted([output_type(f) for f in flags_detected])
            output = flags_detected[0] if len(flags_detected) == 1 else flags_detected
        else:
            output = default
    else:
        splits = [s for s in comments.split(split_symbol) if not flag in s]
        output = split_symbol.join(splits) + split_symbol + flag + equality_symbol + str(replace)
    return output


def rename(newname):
    def decorator(f):
        f.__name__ = newname
        return f

    return decorator


def module_from_file(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class NumpyEncoder(json.JSONEncoder):
    """ Custom encoder for numpy data types """

    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):

            return int(obj)

        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)

        elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
            return {'real': obj.real, 'imag': obj.imag}

        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()

        elif isinstance(obj, (np.bool_)):
            return bool(obj)

        elif isinstance(obj, (np.void)):
            return None

        return json.JSONEncoder.default(self, obj)


def filetail(f, lines=20):
    # original: https://stackoverflow.com/questions/136168/get-last-n-lines-of-a-file-similar-to-tail
    total_lines_wanted = lines

    BLOCK_SIZE = 1024
    f.seek(0, 2)
    block_end_byte = f.tell()
    lines_to_go = total_lines_wanted
    block_number = -1
    blocks = []
    while lines_to_go > 0 and block_end_byte > 0 and f.tell() + block_number * BLOCK_SIZE > 0:
        if (block_end_byte - BLOCK_SIZE > 0):
            # f.seek(block_number * BLOCK_SIZE, 2)
            f.seek(f.tell() + block_number * BLOCK_SIZE, os.SEEK_SET)
            blocks.append(f.read(BLOCK_SIZE))
        else:
            f.seek(0, 0)
            blocks.append(f.read(block_end_byte))
        lines_found = blocks[-1].count('\n')
        lines_to_go -= lines_found
        block_end_byte -= BLOCK_SIZE
        block_number -= 1
    all_read_text = ''.join(reversed(blocks))
    return all_read_text.splitlines()[-total_lines_wanted:]


def summarize_logs(
        containing_folder,
        remove_lines_with=[': I tensorflow', 'WARNING:root:', ' - ETA:', 'Lmod ', 'cuda/11.0'],
        error_keys=['Aborted', 'error', 'Error']
):
    ds = sorted([d for d in os.listdir(containing_folder) if '.out' in d])

    all_lines = []
    errors = []
    error_d = []
    finished_correctly = 0
    completed_tag = 0
    extra_short = 0
    n_lines = 60
    for d in tqdm(ds):
        path = os.path.join(containing_folder, d)

        all_lines.extend(['-' * 50 + '\n'])
        all_lines.extend([d + '\n'])

        # Open the file for reading.
        with open(path, 'r', encoding='utf-8', errors='ignore') as infile:
            i = 0
            double_detection = 0
            while i < n_lines:
                line = infile.readline().rstrip('\r\n')  # Read the contents of the file into memory.
                writeit = all([not remove_line in line for remove_line in remove_lines_with])
                if writeit:
                    i += 1
                    all_lines.extend([line])

                    error_not_found = all([not error_key in line for error_key in error_keys])
                    if not error_not_found:
                        errors.append(line)
                        error_d.append(d)
                        double_detection = 1

        # Return a list of the lines, breaking at line boundaries.
        all_lines.extend(['\n...\n'])

        # with open(path, 'r', encoding="latin1") as infile:
        with open(path, 'r', encoding='utf-8', errors='ignore') as infile:
            try:
                last_lines = filetail(infile, lines=n_lines)
            except Exception as e:
                last_lines = [f'Exception:\n{e}']

        clean_last_lines = []
        for line in last_lines:
            # print(line)
            writeit = all([not remove_line in line for remove_line in remove_lines_with])
            if writeit:
                clean_last_lines.append(line)
                error_not_found = all([not error_key in line for error_key in error_keys])
                if not error_not_found and double_detection == 0:
                    errors.append(line)
                    error_d.append(d)
        all_lines.extend(clean_last_lines)

        if 'All done' in last_lines:
            finished_correctly += 1

        if 'Completed after' in last_lines:
            completed_tag += 1

        if i < 12:
            extra_short += 1

    time_string = timeStructured()
    path = os.path.join(containing_folder, '{}-summary.txt'.format(time_string))

    all_lines = [i[0] for i in groupby(all_lines)]

    with open(path, 'w', encoding="utf-8") as f:
        f.write('\n'.join(all_lines))

    with open(path, 'a') as f:
        f.write('\n' + '-' * 50)
        f.write(f'\nAll done tag:         {finished_correctly} of {len(ds)}')
        f.write(f'\nCompleted after tag:  {completed_tag} of {len(ds)}')
        f.write(f'\nShort codes:          {extra_short} of {len(ds)}')

    # remove digits from errors, to make them easier to consider as a one error
    errors = [re.sub("\d+", "X", e) for e in errors]
    es, cs = np.unique(errors, return_counts=True)

    with open(path, 'a') as f:
        f.write('\n' + '-' * 50)
        f.write(f'\n Errors: {len(errors)}/{len(ds)}\n')
        for e, c in zip(es, cs):
            f.write('\n' + e)
            f.write(f'\n            {c} times')
            index = errors.index(e)
            d = error_d[index]
            f.write(f'\n            e.g. {d}')
            f.write(f'\n')


if __name__ == '__main__':
    # comments = '_thing:23'
    # new_comments = str2val(comments, 'thing', replace=232)
    # print(new_comments)

    errors = ['niceerror', 'bad', 'bad']
    es, c = np.unique(errors, return_counts=True)
    print(es, c)
