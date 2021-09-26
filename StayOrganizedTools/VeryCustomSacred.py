import logging, os, pathlib, shutil, time
import datetime
from tqdm import tqdm
from time import strftime, localtime
import tensorflow as tf

from sacred import Experiment
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds

from GenericTools.StayOrganizedTools.utils import setReproducible, timeStructured


class CustomFileStorageObserver(FileStorageObserver):

    def started_event(self, ex_info, command, host_info, start_time, config, meta_info, _id):
        if _id is None:
            # create your wanted log dir
            time_string = timeStructured(False)
            timestamp = "{}-{}_".format(time_string, ex_info['name'])
            options = '_'.join(meta_info['options']['UPDATE'])
            self.updated_config = meta_info['options']['UPDATE']
            # run_id = timestamp + options
            run_id = timestamp

            # update the basedir of the observer
            self.basedir = os.path.join(self.basedir, run_id)
            self.basedir = os.path.join(ex_info['base_dir'], self.basedir)

            # and again create the basedir
            pathlib.Path(self.basedir).mkdir(exist_ok=True, parents=True)

        # create convenient folders for current experiment
        for relative_path in ['images', 'text', 'other_outputs', 'trained_models']:
            absolute_path = os.path.join(*[self.basedir, relative_path])
            self.__dict__.update(relative_path=absolute_path)
            os.mkdir(absolute_path)

        return super().started_event(ex_info, command, host_info, start_time, config, meta_info, _id)

    def save_comments(self, comments=''):
        if not comments is '':
            comments_txt = os.path.join(*[self.basedir, '1', 'comments.txt'])
            with open(comments_txt, "w") as text_file:
                text_file.write(comments)


def CustomExperiment(experiment_name, base_dir=None, GPU=None, seed=10, ingredients=[]):
    import numpy as np
    random_string = ''.join([str(r) for r in np.random.choice(10, 4)]) + '-'
    ex = Experiment(name=random_string + experiment_name, base_dir=base_dir, ingredients=ingredients,
                    save_git_info=False)
    ex.observers.append(CustomFileStorageObserver("experiments"))

    ex.captured_out_filter = apply_backspaces_and_linefeeds

    # create convenient folders for all experiments
    basic_folders = ['data', 'experiments', 'good_experiments']
    for path in basic_folders:
        complete_path = os.path.join(base_dir, path)
        if not os.path.isdir(complete_path):
            os.mkdir(complete_path)

    gitignore_path = os.path.join(base_dir, '.gitignore')
    if not os.path.isfile(gitignore_path):
        with open(gitignore_path, 'w', encoding="utf-8") as f:
            for folder_name in basic_folders:
                f.write('\n' + folder_name)

    # choose GPU
    if not GPU == None:
        ChooseGPU(GPU)

    # set reproducible
    if not seed is None:
        setReproducible(seed)
    else:
        import numpy as np
        print(np.random.rand())
    return ex


def ChooseGPU(GPU=None, memory_growth=True):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, memory_growth)
        except RuntimeError as e:
            print(e)

    if not GPU is None:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU)
        config = tf.compat.v1.ConfigProto()  # tf.ConfigProto()
        config.gpu_options.allow_growth = True

        sess = tf.compat.v1.Session(config=config)  # tf.Session(config=config)


def remove_folder(folder_path):
    time.sleep(2)
    shutil.rmtree(folder_path, ignore_errors=True)


def summarize(CDIR, track_params=[]):
    # CDIR = os.path.dirname(os.path.realpath(__file__))
    # CDIR = os.path.abspath(os.path.join(*[CDIR, '..', ]))
    EXPERIMENTS = os.path.join(*[CDIR, 'experiments'])

    time_string = timeStructured()

    summary_file = os.path.join(EXPERIMENTS, time_string + '_summary_slurms.txt')

    ds = [d for d in os.listdir(CDIR) if 'slurm' in d]
    ds = sorted(ds)
    print(ds)

    for d in tqdm(ds):
        update = False
        d_path = os.path.join(*[CDIR, d])
        with open(d_path, "r", encoding="cp1252", errors='ignore') as file:
            last_line = 'empty'
            previous_line = 'empty'
            add_lines = ''
            epoch_line = ''
            time_per_epoch = ''
            for line in file:
                update += line == '}\n'
                if update:
                    for w in track_params:
                        if w in line:
                            add_lines += line
                if 'Epoch' in line:
                    epoch_line = line

                if 'experiment: ' in line:
                    add_lines += line

                if 'ETA' in line:
                    idx_ETA = line.index('ETA')
                    time = line[idx_ETA + 4:]
                    idx_hyphen = time.index('-')
                    time = time[:idx_hyphen].replace(' ', '')
                    if 's' in time:
                        time = '0:' + time.replace('s', '')
                    if time.count(':') == 1: time = '0:{}'.format(time)
                    try:
                        time = datetime.datetime.strptime(time, "%H:%M:%S") - datetime.datetime(1900, 1, 1)
                        time = time.total_seconds()
                    except:
                        time = int(time.split(':')[0]) * 24 * 60 + int(time.split(':')[1]) * 60 + int(
                            time.split(':')[2])
                    time_per_epoch += ' {}'.format(time)
                previous_line_2 = previous_line
                previous_line = last_line
                last_line = line.replace('^H', '')
            # time per epoch
            time_per_epoch = time_per_epoch.replace(':', '.')
            times = [float(t) for t in time_per_epoch.split(' ') if not t == '']
            try:
                max_t = max(times)
                max_t = datetime.timedelta(seconds=max_t)
            except:
                max_t = 'max t = ?'

        f = open(summary_file, "a")
        f.write('\n       ' + d + '\n')
        f.write(add_lines)
        f.write(epoch_line.replace('\n', '') + ' -> {}s/e\n'.format(max_t))
        f.write(previous_line_2)
        f.write(previous_line)
        f.write(last_line)
        f.write('\n\n')
        f.close()
