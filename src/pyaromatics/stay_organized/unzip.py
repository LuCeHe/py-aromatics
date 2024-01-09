import os, shutil
from zipfile import ZipFile
import numpy as np
from tqdm import tqdm

CDIR = os.path.dirname(os.path.realpath(__file__))


def unzip_good_exps(exp_origin, exp_destination, exp_identifiers=[''], except_folders=[], except_files=[],
                    unzip_what=None):
    r"""
    Unzips files collected in a folder, possibly from experiments, and unzips only desired files in temporary folder.
    By keeping the unzipped files in a temporary folder, it is faster to eliminate them when the analysis is done,
    and they are no longer necessary.

        Args:
            exp_origin: location of the folder with the zipped files
            exp_destination: folder with the temporary unzipped files
            exp_identifiers: list of strings in folders to unzip
            except_folders: list of strings present in folders to exclude from unzipping
            except_files: list of strings present in files to exclude from unzipping
            unzip_what: list of strings in files to unzip

        Returns:
            list of locations of unzipped folders

        Examples::
            >>> EXPERIMENTS = os.path.join(CDIR, 'results')
            >>> TEMPORARY = os.path.join(CDIR, 'tmp')
            >>> ds = unzip_good_exps(EXPERIMENTS, TEMPORARY, unzip_what=['.txt', '.ckpt'])
            >>> for d in ds:
            >>>     ...
    """

    if not isinstance(exp_origin, list):
        exp_origin = [exp_origin]

    tmp_ds = []
    for eo in exp_origin:
        ttds = [os.path.join(*[eo, e]) for e in os.listdir(eo) if 'zip' in e]
        tmp_ds.extend(ttds)

    os.makedirs(exp_destination, exist_ok=True)

    ds = []
    for d in tmp_ds:
        _, tail = os.path.split(d)
        for t_in in exp_identifiers:
            if t_in in tail:
                any_exception = False
                for t_out in except_folders:
                    if t_out in d:
                        any_exception = True

                if not any_exception:
                    ds += [d]

    ds = sorted(ds)
    destinations = []
    for d in tqdm(ds, desc='Unzipping...'):
        try:
            # Create a ZipFile Object and load sample.zip in it
            with ZipFile(d, 'r') as zipObj:
                tail = d.split('\\')[-1].replace('.zip', '')
                destination = os.path.join(*[exp_destination, tail])
                destinations.append(destination)
                if not os.path.isdir(destination):
                    os.mkdir(destination)

                    # Extract all the contents of zip file in different directory
                    for z in zipObj.infolist():
                        do_unzip = not np.any([e in z.filename for e in except_files])
                        if do_unzip:
                            try:
                                if 'other_outputs' in z.filename:
                                    zipObj.extract(z, destination)
                                elif 'config' in z.filename:
                                    zipObj.extract(z, destination)

                                if not unzip_what is None:
                                    if isinstance(unzip_what, list):
                                        for string in unzip_what:
                                            if string in z.filename:
                                                zipObj.extract(z, destination)
                                    elif unzip_what == 'all':
                                        zipObj.extract(z, destination)
                                    else:
                                        raise NotImplementedError
                            except Exception as e:
                                print(e)
        except Exception as e:
            print(e)
            print(d)
    return destinations



if __name__ == '__main__':
    GEXPERIMENTS = os.path.join(CDIR, 'good_experiments')
    EXPERIMENTS = os.path.join(CDIR, 'experiments')
    TENSORBOARDS = os.path.join(EXPERIMENTS, 'other_outputs')

    for d in [EXPERIMENTS, TENSORBOARDS]:
        if not os.path.isdir(d):
            os.mkdir(d)

    exp_identifiers = [
        '',
        # '2020-09-09--11',
        # '2020-10-09--',
        # '2020-09-23--',
        # 'freeze',
    ]

    except_identifiers = [
        # '2020-09-09--11',
        # 'freeze',
        # 'time_ae'
        # 'TwoWays'
        # 'experiment-',
        # 'time_ae_merge'
    ]
    unzip_good_exps(GEXPERIMENTS, exp_identifiers=exp_identifiers, except_identifiers=except_identifiers)
