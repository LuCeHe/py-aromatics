import os, itertools


def run_experiments(experiments, init_command='python language_main.py with ',
                    run_string='sbatch run_tf2.sh ', is_argparse=False):
    if not experiments is None:
        ds = dict2iter(experiments)
    else:
        ds = ['']
    print('Number jobs: {}'.format(len(ds)))
    for d in ds:
        if not experiments is None:
            a = '--' if is_argparse else ''
            config_update = ''.join([a + '{}={} '.format(k, v) for k, v in d.items()])
            command = init_command + config_update
        else:
            command = init_command

        command = run_string + "'{}'".format(command)
        command = command.replace('  ', ' ')
        print(command)
        os.system(command)
    print('Number jobs: {}'.format(len(ds)))


def dict2iter(experiments):
    full_ds = []
    for experiment in experiments:
        c = list(itertools.product(*experiment.values()))
        ds = [{k: v for k, v in zip(experiment.keys(), i)} for i in c]
        full_ds.extend(ds)
    return full_ds
