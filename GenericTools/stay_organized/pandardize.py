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


def experiments_to_pandas(h5path, zips_folder, unzips_folder, extension_of_interest=['.txt', '.json'],
                          experiments_identifier='', exclude_files=['']):
    if not os.path.exists(h5path):

        ds = unzip_good_exps(
            zips_folder, unzips_folder,
            exp_identifiers=[experiments_identifier], except_folders=[],
            unzip_what=extension_of_interest
        )

        list_results = []
        for d in tqdm(ds, desc='Creating pandas'):
            # print('-'*30)
            results = {}
            filepaths = []
            for ext in extension_of_interest:
                fps = glob.glob(os.path.join(d, f'**/*{ext}'), recursive=True)
                filepaths.extend(fps)

            for e in exclude_files:
                filepaths = [fp for fp in filepaths if not e in fp]

            for fp in filepaths:
                # print(fp)
                json_path = os.path.join(d, fp)
                if os.path.exists(json_path):
                    if json_path.endswith('checkpoint'):
                        history_df = pd.read_csv(json_path)
                        res = {k: history_df[k].tolist() for k in history_df.columns.tolist()}
                    else:
                        with open(json_path) as f:
                            res = json.load(f)

                    results.update(h for k, v in res.items() for h in history_pick(k, v))
            results.update(path=d)
            # print(results)
            list_results.append(results)

        df = pd.DataFrame.from_records(list_results)

        df.to_hdf(h5path, key='df', mode='w')
    else:
        df = pd.read_hdf(h5path, 'df')  # load it

    return df
