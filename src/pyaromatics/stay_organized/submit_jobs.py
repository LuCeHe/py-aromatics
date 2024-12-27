import os, itertools, time, socket, random
from datetime import timedelta
import numpy as np
from CCsubmit.helpers import get_subset


def run_experiments(
        experiments=None, subset=None, init_command='python language_main.py with ',
        run_string='sbatch run_tf2.sh ', is_argparse=False, sh_location='', py_location='',
        env_location='denv2', id='', mock_send=False,
        load_modules='module load gcc arrow cuda/11.1 python/3.9 scipy-stack StdEnv/2020',
        randomize_seed=None, prevent=[], sbatch_args={}):
    if isinstance(randomize_seed, int):
        random.seed(randomize_seed)
        np.random.seed(randomize_seed)


    if run_string is None:
        sh_name = create_sbatch_sh(
            sh_location, py_location, env_location, id,
            load_modules=load_modules, sbatch_arg=sbatch_args
        )
        run_string = 'sbatch ' + sh_name

    if not experiments is None and not isinstance(experiments, int):
        ds = dict2iter(experiments)
        print('len(ds)', len(ds))
        new_ds = []
        for d in ds:
            passed = True
            for pd in prevent:
                n_trues = 0
                for k, v in pd.items():
                    n_trues += v in d[k]
                if n_trues > 1:
                    passed = False
            if passed:
                new_ds.append(d)
        ds = new_ds
        print('len(ds)', len(ds))

    elif isinstance(experiments, int):
        ds = ['' for _ in range(experiments)]
    else:
        ds = ['']

    if subset is None:
        subset = [0, None]

    elif subset == True:
        subset, _ = get_subset(ds)

    elif 'DESKTOP' in socket.gethostname():
        subset = [0, None]

    elif isinstance(subset, dict):
        servers = [k for k, v in subset.items()]
        probs = [v for k, v in subset.items()]
        cumprobs = np.cumsum(probs)
        current_server = socket.gethostname()
        # print('here?')
        # current_server = 'graham'

        amount = len(ds)
        server_found = False
        for i, server in enumerate(servers):
            if server in current_server:
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

    if isinstance(randomize_seed, int):
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

            command = "{} '{}'".format(run_string, command)
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
        sh_location, py_location, env_location, id,
        load_modules='', sbatch_args={}
):
    import numpy as np
    named_tuple = time.localtime()  # get struct_time
    time_string = time.strftime("%Y-%m-%d--%H-%M-%S--", named_tuple)
    random_string = ''.join([str(r) for r in np.random.choice(10, 4)])

    sh_name = f'{id}--' + time_string + random_string + '.sh'
    sh_path = os.path.join(sh_location, sh_name)
    with open(sh_path, 'w') as f:
        f.write(
            sh_base(py_location, env_location,
                    load_modules=load_modules, sbatch_args=sbatch_args)
        )
    return sh_path


def sh_base(
        py_location, env_location,
        load_modules='module load gcc/9.3.0 arrow cuda/11.1 python/3.9 scipy-stack StdEnv/2020',
        sbatch_args={}
):
    sbatch_args_line = ''.join([f'#SBATCH --{k}={v}\n' for k, v in sbatch_args.items()])
    return f"""#!/bin/bash
{sbatch_args_line}
{load_modules}
source {env_location}
cd {py_location}
$1
"""
