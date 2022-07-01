import argparse, logging, os, random, time, gc
from time import strftime, localtime
import importlib.util

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
                    empty_cuda=True):
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
        if pytorch:
            torch.cuda.empty_cache()

        from numba import cuda
        device = cuda.get_current_device()
        device.reset()
        cuda.select_device(0)
        cuda.close()
        # torch.cuda.empty_cache()


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
            replace=None):
    if replace is None:
        if '{}{}'.format(flag, equality_symbol) in comments:
            flags_detected = [s.replace(
                '{}{}'.format(flag if remove_flag else '', equality_symbol), ''
            )
                for s in comments.split(split_symbol)
                if '{}{}'.format(flag, equality_symbol) in s]
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

if __name__ == '__main__':
    comments = '_thing:23'
    new_comments = str2val(comments, 'thing', replace=232)
    print(new_comments)
