__all__ = ['datcln']

from .. import data_cleaner


def datcln(dc, df, X, X_noinput, X_grouped):
    '''
    Perform data cleaning here
    '''

    if 'cleaning_method' not in dc.keys():
        log.warning(
                    "You have chosen not to specify a method of data_cleaning "
                    "in the input file. By default, any feature entries "
                    "containing NaN will result in removal of the feature and "
                    "any target data entries containing NaN will "
                    "result in removal of that target data point."
                    )

        dc['cleaning_method'] = 'remove'

    if dc['cleaning_method'] == 'remove':
        df = data_cleaner.remove(df, axis=1)
        X = data_cleaner.remove(X, axis=1)
        X_noinput = data_cleaner.remove(X_noinput, axis=1)
        X_grouped = data_cleaner.remove(X_grouped, axis=1)
        # TODO: have method to first remove rows of missing
        # target data, then do columns for features
        # y = data_cleaner.remove(y, axis=0)

    elif dc['cleaning_method'] == 'imputation':
        log.warning(
                    "You have selected data cleaning with Imputation. Note "
                    "that imputation will not resolve missing target data. "
                    "It is recommended to remove missing target data"
                    )

        if 'imputation_strategy' not in dc.keys():
            log.warning(
                        "You have chosen to perform data imputation but "
                        "have not selected an imputation strategy. By default"
                        ", the mean will be used as the imputation strategy"
                        )

            dc['imputation_strategy'] = 'mean'

        df = data_cleaner.imputation(
                                     df,
                                     dc['imputation_strategy'],
                                     X_noinput.columns
                                     )

        X = data_cleaner.imputation(
                                    X,
                                    dc['imputation_strategy']
                                    )

    elif dc['cleaning_method'] == 'ppca':
        log.warning(
                    "You have selected data cleaning with PPCA. Note that "
                    "PPCA will not work to estimate missing target values, "
                    "at least a 2D matrix is needed. It is recommended you "
                    "remove missing target data"
                    )

        df = data_cleaner.ppca(df, X_noinput.columns)
        X = data_cleaner.ppca(X)

    else:
        log.error(
                  "You have specified an invalid data cleaning method. "
                  "Choose from: remove, imputation, or ppca"
                  )

        exit()

    return dc, df, X, X_noinput, X_grouped
