__all__ = ['valcolpar']

from functools import reduce

import pandas as pd
import numpy as np


def valcolpar(df, conf, is_validation, _exclude_validation, X, y, X_grouped):
    '''
    Get parameters out for 'validation_column'
    '''

    if is_validation:
        if type(conf['GeneralSetup']['validation_columns']) is list:
            templist = [conf['GeneralSetup']['validation_columns']]
            validation_column_names = templist

        elif type(conf['GeneralSetup']['validation_columns']) is str:
            templist = [conf['GeneralSetup']['validation_columns'][:]]
            validation_column_names = templist

        validation_columns = {}

        for name in validation_column_names:
            validation_columns[name] = df[name]

        validation_columns = pd.DataFrame(validation_columns)

        validation_X = []
        validation_y = []

        # TODO make this block its own function
        for name in validation_column_names:
            # X_, y_ = _exclude_validation(
            # X, validation_columns[validation_column_name]),
            # _exclude_validation(y,
            # validation_columns[validation_column_name])
            tempexclude = _exclude_validation(
                                              X,
                                              validation_columns[name]
                                              )

            validation_X.append(pd.DataFrame(tempexclude))

            tempexclude = _exclude_validation(
                                              y,
                                              validation_columns[name]
                                              )

            validation_y.append(pd.DataFrame(tempexclude))

        idxy_list = []
        for i, _ in enumerate(validation_y):
            idxy_list.append(validation_y[i].index)

        # Get intersection of indices between all prediction columns
        intersection = reduce(np.intersect1d, (i for i in idxy_list))
        X_novalidation = X.iloc[intersection]
        y_novalidation = y.iloc[intersection]
        X_grouped_novalidation = X_grouped.iloc[intersection]

    else:
        X_novalidation = X
        y_novalidation = y
        X_grouped_novalidation = X_grouped

    return (
            validation_columns,
            validation_column_names,
            X_novalidation,
            y_novalidation,
            X_grouped_novalidation
            )
