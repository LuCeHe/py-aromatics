import os, itertools, time, socket
from datetime import datetime, timedelta


def run_experiments(
        experiments=None, subset=[0, None], init_command='python language_main.py with ',
        run_string='sbatch run_tf2.sh ', is_argparse=False, sh_location='', py_location='', account='',
        duration={'days': 0, 'hours': 12, 'minutes': 0, 'prestop_training_hours': -1},
        env_name='denv2', n_gpus=0, id='', mem='32G', cpus_per_task=4, mock_send=False
):
    delta = timedelta(days=duration['days'], hours=duration['hours'], minutes=duration['minutes'])

    # stop training 2 hours before the total allocated time, to run tests
    stop_training = int(delta.total_seconds() - duration['prestop_training_hours'] * 3600)

    hours, remainder = divmod(delta.total_seconds(), 3600)
    minutes, seconds = divmod(remainder, 60)
    sh_duration = "{}:{}:00".format(str(int(hours)).zfill(2), str(int(minutes)).zfill(2))

    if run_string is None:
        sh_name = create_sbatch_sh(sh_duration, sh_location, py_location, account, env_name, n_gpus, id, mem=mem,
                                   cpus_per_task=cpus_per_task)
        run_string = 'sbatch ' + sh_name

    print()
    stop_training = '' if duration['prestop_training_hours'] == -1 else ' stop_time={} '.format(int(stop_training))
    if is_argparse:
        stop_training = stop_training.replace('stop_time', '--stop_time')
    if not experiments is None:
        ds = dict2iter(experiments)
    else:
        ds = ['']

    ods = ds
    ds = ds[subset[0]:subset[1]]

    if len(ds) > 0:
        print(f'Number jobs: {len(ds)}/{len(ods)}')
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
            if not mock_send:
                os.system(command)
        print(f'Number jobs: {len(ds)}/{len(ods)}')

    print(subset)
    print(socket.gethostname())


def dict2iter(experiments):
    full_ds = []
    for experiment in experiments:
        c = list(itertools.product(*experiment.values()))
        ds = [{k: v for k, v in zip(experiment.keys(), i)} for i in c]
        full_ds.extend(ds)
    return full_ds


def create_sbatch_sh(duration, sh_location, py_location, account, env_name, n_gpus, id, mem='32G', cpus_per_task=4):
    sh_name = '{0:010x}'.format(int(time.time() * 256))[-7:] + f'-{id}.sh'
    sh_path = os.path.join(sh_location, sh_name)
    with open(sh_path, 'w') as f:
        f.write(sh_base(duration, account, py_location, env_name, n_gpus, mem, cpus_per_task=cpus_per_task))
    return sh_path


def sh_base(time, account, py_location, env_name, n_gpus, mem='32G', cpus_per_task=4):
    env_location = f'~/scratch/{env_name}/bin/activate'
    if 'cedar' in socket.gethostname():
        env_location = f'~/project/lucacehe/{env_name}/bin/activate'
    if 'gra' == socket.gethostname()[:3]:
        env_location = f'~/projects/def-jrouat/lucacehe/{env_name}/bin/activate'
    gpus_line = '' if n_gpus == 0 else f'#SBATCH --gres=gpu:{n_gpus}'
    return f"""#!/bin/bash
#SBATCH --time={time}
#SBATCH --account={account}
#SBATCH --mem {mem}
#SBATCH --cpus-per-task {cpus_per_task}
{gpus_line}

module load gcc/9.3.0 arrow cuda/11.0 python/3.8 scipy-stack StdEnv/2020 
source {env_location}
cd {py_location}
$1
"""

# SBATCH --mem 32G
# SBATCH --cpus-per-task 4
# SBATCH --gres=gpu:1

# salloc  --time 17:0:0 --cpus-per-task 8 --mem 32G
# module load StdEnv/2020 gcc/10 cuda/11.0 arrow/1.0.0 python/3.8 scipy-stack
