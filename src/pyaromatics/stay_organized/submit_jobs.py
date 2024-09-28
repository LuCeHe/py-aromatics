import os, itertools, time, socket, random
from datetime import datetime, timedelta
import numpy as np
from CCsubmit.helpers import get_subset


def run_experiments(
        experiments=None, subset=[0, None], init_command='python language_main.py with ',
        run_string='sbatch run_tf2.sh ', is_argparse=False, sh_location='', py_location='', account='',
        duration={'days': 0, 'hours': 12, 'minutes': 0, 'prestop_training_hours': -1},
        env_location='denv2', n_gpus=0, id='', mem='32G', cpus_per_task=4, mock_send=False,
        load_modules='module load gcc arrow cuda/11.1 python/3.9 scipy-stack StdEnv/2020'
):
    delta = timedelta(days=duration['days'], hours=duration['hours'], minutes=duration['minutes'])

    # stop training 2 hours before the total allocated time, to run tests
    stop_training = int(delta.total_seconds() - duration['prestop_training_hours'] * 3600)

    hours, remainder = divmod(delta.total_seconds(), 3600)
    minutes, seconds = divmod(remainder, 60)
    sh_duration = "{}:{}:00".format(str(int(hours)).zfill(2), str(int(minutes)).zfill(2))

    if run_string is None:
        sh_name = create_sbatch_sh(sh_duration, sh_location, py_location, account, env_location, n_gpus, id, mem=mem,
                                   cpus_per_task=cpus_per_task, load_modules=load_modules)
        run_string = 'sbatch ' + sh_name

    stop_training = '' if duration['prestop_training_hours'] < 0 else ' stop_time={} '.format(int(stop_training))
    if is_argparse:
        stop_training = stop_training.replace('stop_time', '--stop_time')

    if not experiments is None and not isinstance(experiments, int):
        ds = dict2iter(experiments)

    elif isinstance(experiments, int):
        ds = ['' for _ in range(experiments)]
    else:
        ds = ['']

    if subset == True:
        subset, _ = get_subset(ds)

    elif 'DESKTOP' in socket.gethostname():
        subset = [0, None]

    elif isinstance(subset, dict):
        servers = [k for k, v in subset.items()]
        probs = [v for k, v in subset.items()]
        cumprobs = np.cumsum(probs)

        amount = len(ds)
        server_found = False
        for i, server in enumerate(servers):
            if server in socket.gethostname():
                server_found = True
                if not probs[i] == 0:
                    cp = cumprobs[i]
                    cp_1 = cumprobs[i - 1] if i > 0 else 0
                    from_ = int(cp_1 * amount)
                    to_ = int(cp * amount)

                    if cp == 1:
                        to_ = None

                    if cp_1 == 1:
                        from_ = None
                    subset = [from_, to_]
                else:
                    subset = [0, 0]
                break

        if not server_found:
            subset = [0, 0]

    random.seed(0)
    random.shuffle(ds)

    ods = ds
    ds = ds[subset[0]:subset[1]]

    if len(ds) > 0:
        print(f'Number jobs: {len(ds)}/{len(ods)}')
        for i, d in enumerate(ds):
            if not experiments is None and not isinstance(experiments, int):
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


def dict2iter(experiments, to_list=False):
    full_ds = []
    for experiment in experiments:
        c = list(itertools.product(*experiment.values()))
        ds = [{k: v if not to_list else [v] for k, v in zip(experiment.keys(), i)} for i in c]
        full_ds.extend(ds)
    return full_ds


def create_sbatch_sh(
        duration, sh_location, py_location, account, env_location, n_gpus, id, mem='32G', cpus_per_task=4,
                     load_modules=''):
    import numpy as np
    named_tuple = time.localtime()  # get struct_time
    time_string = time.strftime("%Y-%m-%d--%H-%M-%S--", named_tuple)
    random_string = ''.join([str(r) for r in np.random.choice(10, 4)])

    sh_name = f'{id}--' + time_string + random_string + '.sh'
    sh_path = os.path.join(sh_location, sh_name)
    with open(sh_path, 'w') as f:
        f.write(
            sh_base(duration, account, py_location, env_location, n_gpus, mem, cpus_per_task=cpus_per_task,
                        load_modules=load_modules)
        )
    return sh_path


def sh_base(
        time, account, py_location, env_location, n_gpus, mem='32G', cpus_per_task=4,
        load_modules='module load gcc/9.3.0 arrow cuda/11.1 python/3.9 scipy-stack StdEnv/2020'
):
    gpus_line = '' if n_gpus == 0 else f'#SBATCH --gres=gpu:{n_gpus}'
    return f"""#!/bin/bash
#SBATCH --time={time}
#SBATCH --account={account}
#SBATCH --mem {mem}
#SBATCH --cpus-per-task {cpus_per_task}
{gpus_line}

{load_modules}
source {env_location}
cd {py_location}
$1
"""
