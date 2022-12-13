import json, os, glob
import pandas as pd
from GenericTools.keras_tools.plot_tools import history_pick
from tqdm import tqdm

from GenericTools.stay_organized.unzip import unzip_good_exps


def simplify_col_names(df):
    for co, cd in [
        ('firing_rate_ma_lsnn', 'fr'),
        ('mode_', 'm'),
        ('accuracy', 'acc'),
        ('perplexity', 'ppl'),
        ('crossentropy', 'xnt'),
        ('sparse_', ''),
        ('categorical_', ''),
    ]:
        df.columns = df.columns.str.replace(co, cd)

    return df


def zips_to_pandas(h5path, zips_folder, unzips_folder, extension_of_interest=['.txt', '.json', '.csv'],
                          experiments_identifier=[], exclude_files=['']):
    if isinstance(experiments_identifier, str):
        experiments_identifier = [experiments_identifier]

    if not os.path.exists(h5path):

        ds = unzip_good_exps(
            zips_folder, unzips_folder,
            exp_identifiers=experiments_identifier, except_folders=[],
            unzip_what=extension_of_interest
        )

        list_results = []
        for d in tqdm(ds, desc='Creating pandas'):
            print(d)

            results = {}
            filepaths = []
            for ext in extension_of_interest:
                fps = glob.glob(os.path.join(d, f'**/*{ext}'), recursive=True)
                filepaths.extend(fps)

            for e in exclude_files:
                filepaths = [fp for fp in filepaths if not e in fp]

            for fp in filepaths:
                try:
                    if os.path.exists(fp):
                        if fp.endswith('checkpoint') or fp.endswith('.csv'):
                            history_df = pd.read_csv(fp)
                            if ' "env_id": "stocks-v1"}' in history_df.columns:
                                history_df = pd.DataFrame(history_df.iloc[:, 1].values[1:],
                                                          columns=['total_profit']).astype(float)

                            res = {k: history_df[k].tolist() for k in history_df.columns.tolist()}
                        else:
                            with open(fp) as f:
                                res = json.load(f)

                        results.update(h for k, v in res.items() for h in history_pick(k, v))
                except Exception as e:
                    print(e)
            results.update(path=d)
            list_results.append(results)

        df = pd.DataFrame.from_records(list_results)

        df.to_hdf(h5path, key='df', mode='w')
    else:
        df = pd.read_hdf(h5path, 'df')
    return df


def experiments_to_pandas(h5path, zips_folder, unzips_folder, extension_of_interest=['.txt', '.json', '.csv'],
                          experiments_identifier=[], exclude_files=[''], check_for_new=False):
    df = zips_to_pandas(h5path, zips_folder, unzips_folder, extension_of_interest=extension_of_interest,
                          experiments_identifier=experiments_identifier, exclude_files=exclude_files)

    if check_for_new:
        new = []
        old = [os.path.split(p)[1] for p in df['path'].values]

        for gpath in zips_folder:
            new.extend([p.replace('.zip', '') for p in os.listdir(gpath) if 'zip' in p])
        missing = [p for p in new if not p in old and experiments_identifier in p]

        newh5path = h5path.replace('.h5', '_missing.h5')

        if len(missing) > 0:
            ndf = zips_to_pandas(
                h5path=newh5path, zips_folder=zips_folder, unzips_folder=unzips_folder, experiments_identifier=missing,
                exclude_files=['cout.txt']
            )
            bigdf = pd.concat([df, ndf])
            bigdf.to_hdf(h5path, key='df', mode='w')
            df = bigdf

        if os.path.exists(newh5path):
            os.remove(newh5path)


    return df
