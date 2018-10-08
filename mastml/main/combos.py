__init__ = ['generate_features']

from os.path import join

import pandas as pd
import logging

log = logging.getLogger('mastml')


def generate_features(df, X_noinput, y, generators, outdir):
    log.info("Doing feature generation...")
    dataframes = [instance.fit_transform(df, y) for _, instance in generators]
    dataframe = pd.concat(dataframes, 1)
    log.info("Saving generated data to csv...")
    log.debug(f'generated cols: {dataframe.columns}')
    filename = join(outdir, "generated_features.csv")
    pd.concat([dataframe, X_noinput, y], 1).to_csv(filename, index=False)

    return dataframe
