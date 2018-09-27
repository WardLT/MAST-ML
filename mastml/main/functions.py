__all__ = [
           '_exclude_validation',
           '_extract_grouping_column_names',
           '_snatch_models',
           '_snatch_splitters',
           '_remove_constant_features',
           '_only_validation',
           '_write_stats',
           '_save_all_runs'
           ]

from collections import OrderedDict
from os.path import join

import pandas as pd
import logging
import os

log = logging.getLogger('mastml')


def _grouping_column_to_group_number(X_grouped):
    group_list = X_grouped.values.reshape((1, -1))
    unique_groups = np.unique(group_list).tolist()
    group_dict = dict()
    group_list_asnumber = list()

    for i, group in enumerate(unique_groups):
        group_dict[group] = i+1

    for i, group in enumerate(group_list.tolist()[0]):
        group_list_asnumber.append(group_dict[group])

    X_grouped_asnumber = np.asarray(group_list_asnumber)

    return X_grouped_asnumber


def _snatch_models(models, conf_feature_selection):
    log.debug(f'models, pre-snatching: \n{models}')

    for selector_name, (_, args_dict) in conf_feature_selection.items():
        if 'estimator' in args_dict:
            model_name = args_dict['estimator']
            try:
                args_dict['estimator'] = models[model_name]
                del models[model_name]

            except KeyError:
                raise utils.MastError(
                                      f"The selector {selector_name}"
                                      f"specified model {model_name},"
                                      f"which was not found in the [Models]"
                                      f"section"
                                      )

    log.debug(f'models, post-snatching: \n{models}')


def _snatch_splitters(splitters, conf_feature_selection):
    log.debug(f'cv, pre-snatching: \n{splitters}')
    for selector_name, (_, args_dict) in conf_feature_selection.items():
        # Here: add snatch to cv object for feature selection with RFECV
        if 'cv' in args_dict:
            cv_name = args_dict['cv']
            try:
                args_dict['cv'] = splitters[cv_name]
                del splitters[cv_name]
            except KeyError:
                raise utils.MastError(
                                      f"The selector {selector_name}"
                                      f"specified cv splitter {cv_name},"
                                      f"which was not found in the"
                                      f"[DataSplits] section"
                                      )

    log.debug(f'cv, post-snatching: \n{splitters}')


def _extract_grouping_column_names(splitter_to_kwargs):
    splitter_to_group_names = dict()
    for splitter_name, name_and_kwargs in splitter_to_kwargs.items():
        _, kwargs = name_and_kwargs
        if 'grouping_column' in kwargs:
            column_name = kwargs['grouping_column']
            # because the splitter doesn't actually take this
            del kwargs['grouping_column']
            splitter_to_group_names[splitter_name] = column_name

    return splitter_to_group_names


def _remove_constant_features(df):
    log.info("Removing constant features, regardless of feature selectors.")
    before = set(df.columns)
    df = df.loc[:, (df != df.iloc[0]).any()]
    removed = list(before - set(df.columns))
    if removed != []:
        log.warning(f'Removed {len(removed)}/{len(before)} constant columns.')
        log.debug("Removed the following constant columns: " + str(removed))

    return df


def _save_all_runs(runs, outdir):
    """
    Produces a giant html table of all stats for all runs
    """
    table = []
    for run in runs:
        od = OrderedDict()
        for name, value in run.items():
            if name == 'train_metrics':
                for k, v in run['train_metrics'].items():
                    od['train_'+k] = v
            elif name == 'test_metrics':
                for k, v in run['test_metrics'].items():
                    od['test_'+k] = v
            else:
                od[name] = value
        table.append(od)
    pd.DataFrame(table).to_html(join(outdir, 'all_runs_table.html'))


def _write_stats(
                 train_metrics,
                 test_metrics,
                 outdir,
                 prediction_metrics=None,
                 prediction_names=None
                 ):

    with open(join(outdir, 'stats.txt'), 'w') as f:
        f.write("TRAIN:\n")

        for name, score in train_metrics.items():
            f.write(f"{name}: {'%.3f'%float(score)}\n")

        f.write("TEST:\n")

        for name, score in test_metrics.items():
                f.write(f"{name}: {'%.3f'%float(score)}\n")

        if prediction_metrics:
            # prediction metrics now list of dicts
            # for predicting multiple values
            for prediction_metric, prediction_name in zip(
                                                          prediction_metrics,
                                                          prediction_names
                                                          ):

                f.write("PREDICTION for "+str(prediction_name)+":\n")

                for name, score in prediction_metric.items():
                    f.write(f"{name}: {'%.3f'%float(score)}\n")


def _exclude_validation(df, validation_column):
    return df.loc[validation_column != 1]


def _only_validation(df, validation_column):
    return df.loc[validation_column == 1]
