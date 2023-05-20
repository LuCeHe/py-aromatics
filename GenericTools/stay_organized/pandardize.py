import json, os, glob
import pandas as pd
from GenericTools.keras_tools.plot_tools import history_pick
from tqdm import tqdm

from GenericTools.stay_organized.unzip import unzip_good_exps
from scipy.stats import pearsonr


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
                   experiments_identifier=[], exclude_files=[''], exclude_columns=[],
                   force_keep_column=[]):
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

            results = {}
            filepaths = []
            for ext in extension_of_interest:
                fps = glob.glob(os.path.join(d, f'**/*{ext}'), recursive=True)
                filepaths.extend(fps)

            for e in exclude_files:
                filepaths = [fp for fp in filepaths if not e in fp]

            for fp in filepaths:
                file_stats = os.stat(fp)
                if not file_stats.st_size == 0:
                    # try:
                    if os.path.exists(fp):
                        if fp.endswith('checkpoint') or fp.endswith('.csv'):
                            history_df = pd.read_csv(fp)
                            if ' "env_id": "stocks-v1"}' in history_df.columns:
                                history_df = pd.DataFrame(history_df.iloc[:, 1].values[1:],
                                                          columns=['total_profit']).astype(float)

                            res = {k: history_df[k].tolist() for k in history_df.columns.tolist()}
                        elif fp.endswith('.json') or fp.endswith('.txt'):
                            with open(fp) as f:
                                res = json.load(f)
                        else:
                            res = {}


                        # flatten dictionaries inside res
                        res2 = {}
                        for k, v in res.items():
                            if isinstance(v, dict):
                                for k2, v2 in v.items():
                                    res2[f'{k}_{k2}'] = v2
                                # del res[k]
                            else:
                                res2[k] = v

                        # print(res2.keys())
                        res2 = {k: v for k, v in res2.items()
                               if not any([e in k for e in exclude_columns]) or
                               any([e in k for e in force_keep_column])}
                        # print(res2.keys())

                        results.update(
                            h
                            for k, v in res2.items()
                            for h in history_pick(k, v)
                        )
                        # results.update({k: v for k, v in aux_results.items()
                        #                 if (not any([e in k for e in exclude_columns])
                        #                     or k in force_keep_column)})
                # except Exception as e:
                #     print('\n')
                #     print(fp)
                #     print('    ', e)
            results.update(path=d)
            list_results.append(results)

        df = pd.DataFrame.from_records(list_results)
        # print(list(df.columns))
        df.to_hdf(h5path, key='df', mode='w')
    else:
        df = pd.read_hdf(h5path, 'df')
    return df


def experiments_to_pandas(h5path, zips_folder, unzips_folder, extension_of_interest=['.txt', '.json', '.csv'],
                          experiments_identifier=[], exclude_files=[''], exclude_columns=[],
                          force_keep_column=[], check_for_new=False):
    df = zips_to_pandas(h5path, zips_folder, unzips_folder, extension_of_interest=extension_of_interest,
                        experiments_identifier=experiments_identifier, exclude_files=exclude_files,
                        exclude_columns=exclude_columns, force_keep_column=force_keep_column)

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
                exclude_files=['cout.txt'], exclude_columns=exclude_columns,
                extension_of_interest=extension_of_interest,
            )
            bigdf = pd.concat([df, ndf])
            # print(bigdf.to_string())
            bigdf.to_hdf(h5path, key='df', mode='w')
            df = bigdf

        if os.path.exists(newh5path):
            os.remove(newh5path)

    return df


def complete_missing_exps(sdf, exps, coi):
    if not isinstance(exps, pd.DataFrame):
        data = {k: [] for k in coi}
        for d in exps:
            for k in data.keys():
                insertion = d[k]
                data[k].append(insertion)

        all_exps = pd.DataFrame.from_dict(data)
    else:
        all_exps = exps
    # print(all_exps.to_string())
    all_exps = all_exps.drop_duplicates()
    sdf = sdf.drop_duplicates()

    # remove the experiments that were run successfully
    # df = pd.concat([sdf, all_exps])
    # df = df.drop_duplicates(keep=False)
    #
    # keys = list(all_exps.columns.values)
    # i1 = all_exps.set_index(keys).index
    # i2 = df.set_index(keys).index
    # df = df[~i2.isin(i1)]



    keys = list(all_exps.columns.values)
    i1 = all_exps.set_index(keys).index
    i2 = sdf.set_index(keys).index
    sdf = sdf[i2.isin(i1)]

    df = pd.merge(all_exps, sdf, indicator=True, how='left').query("_merge == 'left_only'")
    df.drop(columns='_merge', inplace=True)

    # df = df.drop_duplicates()

    # print('left')
    # print(df.to_string())

    experiments = []
    for index, row in df.iterrows():
        experiment = {k: [row[k]] for k in df.columns}
        experiments.append(experiment)

    # print(experiments)
    print('left, done, all: ', df.shape, sdf.shape, all_exps.shape)
    return df, experiments




def calculate_pvalues(df, wrt=None):
    # original by toto_tico
    # https://stackoverflow.com/questions/25571882/pandas-columns-correlation-with-statistical-significance
    dfcols = pd.DataFrame(columns=df.columns)
    pvalues = dfcols.transpose().join(dfcols, how='outer')
    wrt = wrt if isinstance(wrt, list) else df.columns
    print(wrt)
    for r in df.columns:
        for c in wrt:
            tmp = df[df[r].notnull() & df[c].notnull()]
            pvalues[r][c] = pearsonr(tmp[r], tmp[c])[1].round(4)
    return pvalues