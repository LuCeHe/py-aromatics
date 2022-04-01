import os, itertools, time, socket
from datetime import datetime, timedelta


def run_experiments(
        experiments=None, init_command='python language_main.py with ',
        run_string='sbatch run_tf2.sh ', is_argparse=False, sh_location='', py_location='', account='',
        duration={'days': 0, 'hours': 12, 'minutes': 0, 'prestop_training_hours': 0}
):
    delta = timedelta(days=duration['days'], hours=duration['hours'], minutes=duration['minutes'])

    # stop training 2 hours before the total allocated time, to run tests
    stop_training = delta.total_seconds() - duration['prestop_training_hours'] * 3600

    hours, remainder = divmod(delta.total_seconds(), 3600)
    minutes, seconds = divmod(remainder, 60)
    sh_duration = "{}:{}:00".format(str(int(hours)).zfill(2), str(int(minutes)).zfill(2))

    if run_string is None:
        sh_name = create_sbatch_sh(sh_duration, sh_location, py_location, account)
        run_string = 'sbatch ' + sh_name

    print()
    stop_training = '' if duration['prestop_training_hours'] == 0 else ' stop_time={} '.format(int(stop_training))
    if not experiments is None:
        ds = dict2iter(experiments)
    else:
        ds = ['']

    print('Number jobs: {}'.format(len(ds)))
    for i, d in enumerate(ds):
        if not experiments is None:
            a = '--' if is_argparse else ''
            config_update = ''.join([a + '{}={} '.format(k, v) for k, v in d.items()])
            command = init_command + config_update
        else:
            command = init_command

        command = "{} '{}'".format(run_string, command + stop_training)
        command = command.replace('  ', ' ')
        print('{}/{}'.format(i + 1, len(ds)), command)
        # os.system(command)
    print('Number jobs: {}'.format(len(ds)))


def dict2iter(experiments):
    full_ds = []
    for experiment in experiments:
        c = list(itertools.product(*experiment.values()))
        ds = [{k: v for k, v in zip(experiment.keys(), i)} for i in c]
        full_ds.extend(ds)
    return full_ds


def create_sbatch_sh(duration, sh_location, py_location, account):
    sh_name = '{0:010x}'.format(int(time.time() * 256))[:15] + '.sh'
    sh_path = os.path.join(sh_location, sh_name)
    with open(sh_path, 'w') as f:
        f.write(sh_base(duration, account, py_location))
    return sh_path


def sh_base(time, account, py_location):
    env_location = '~/scratch/denv2/bin/activate'
    if 'cedar' in socket.gethostname():
        env_location = '~/project/lucacehe/denv2/bin/activate'
    if 'gra' == socket.gethostname()[:3]:
        env_location = '~/projects/def-jrouat/lucacehe/denv2/bin/activate'
    return """#!/bin/bash
#SBATCH --time={}
#SBATCH --account={}
#SBATCH --mem 32G
#SBATCH --cpus-per-task 4
#SBATCH --gres=gpu:1

module load StdEnv/2020  gcc/9.3.0  cuda/11.0 arrow/1.0.0 python/3.6 scipy-stack
source {}
cd {}
$1
""".format(time, account, env_location, py_location)
