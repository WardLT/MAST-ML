__author__ = 'Ryan Jacobs'

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from DataOperations import DataframeUtilities

class FeatureIO(object):
    """Class to selectively filter (add/remove) features from a dataframe
    """
    def __init__(self, dataframe):
        self.dataframe = dataframe

    @property
    def get_original_dataframe(self):
        return self.dataframe

    def remove_duplicate_features_by_name(self):
        # Only removes features that have the same name, not features containing the same data vector
        dataframe = self.dataframe.drop_duplicates()
        return dataframe

    def remove_custom_features(self, features_to_remove):
        dataframe = self.dataframe
        for feature in features_to_remove:
            del dataframe[feature]
        return dataframe

    def keep_custom_features(self, features_to_keep, y_feature):
        dataframe_dict = {}
        for feature in features_to_keep:
            dataframe_dict[feature] = self.dataframe[feature]
        dataframe_dict[y_feature] = self.dataframe[y_feature]
        dataframe = pd.DataFrame(dataframe_dict)
        return dataframe

    def add_custom_features(self, features_to_add, data_to_add):
        dataframe = self.dataframe
        for feature in features_to_add:
            dataframe[feature] = pd.Series(data=data_to_add, index=(self.dataframe).index)
        return dataframe

    def custom_feature_filter(self, feature, operator, threshold):
        # Searches values in feature that meet the condition. If it does, that entire row of data is removed from the dataframe
        rows_to_remove = []
        for i in range(len(self.dataframe[feature])):
            fdata = self.dataframe[feature].iloc[i]
            try:
                fdata = float(fdata)
            except ValueError:
                fdata = fdata
            if operator == '<':
                if fdata < threshold:
                    rows_to_remove.append(i)
            if operator == '>':
                if fdata > threshold:
                    rows_to_remove.append(i)
            if operator == '=':
                if fdata == threshold:
                    rows_to_remove.append(i)
            if operator == '<=':
                if fdata <= threshold:
                    rows_to_remove.append(i)
            if operator == '>=':
                if fdata >= threshold:
                    rows_to_remove.append(i)
            if operator == '<>':
                if not(fdata == threshold):
                    rows_to_remove.append(i)
        dataframe = self.dataframe.drop(self.dataframe.index[rows_to_remove])
        return dataframe

class FeatureNormalization(object):
    """This class is used to normalize and unnormalize features in a dataframe.
    """
    def __init__(self, dataframe):
        self.dataframe = dataframe

    @property
    def get_original_dataframe(self):
        return self.dataframe

    def normalize_features(self, x_features, y_feature, to_csv=True):
        # First remove features containing strings before doing feature normalization
        x_features, dataframe = MiscOperations().remove_features_containing_strings(dataframe=self.dataframe,
                                                                                    x_features=x_features)
        scaler = StandardScaler().fit(X=dataframe[x_features])
        array_normalized = scaler.fit_transform(X=dataframe[x_features], y=self.dataframe[y_feature])
        array_normalized = DataframeUtilities()._concatenate_arrays(X_array=array_normalized, y_array=np.asarray(self.dataframe[y_feature]).reshape([-1, 1]))
        dataframe_normalized = DataframeUtilities()._array_to_dataframe(array=array_normalized)
        dataframe_normalized = DataframeUtilities()._assign_columns_as_features(dataframe=dataframe_normalized, x_features=x_features, y_feature=y_feature, remove_first_row=False)
        if to_csv == True:
            dataframe_normalized.to_csv('input_data_normalized.csv')
        return dataframe_normalized, scaler

    def minmax_scale_single_feature(self, featurename, smin=None, smax=None):
        feature = self.dataframe[featurename]
        if smin is None:
            smin = np.min(feature)
        if smax is None:
            smax = np.max(feature)
        scaled_feature = (feature - smin) / (smax - smin)
        return scaled_feature

    def unnormalize_features(self, x_features, y_feature, scaler):
        array_unnormalized = scaler.inverse_transform(X=self.dataframe[x_features])
        array_unnormalized = DataframeUtilities()._concatenate_arrays(X_array=array_unnormalized, y_array=np.asarray(self.dataframe[y_feature]).reshape([-1, 1]))
        dataframe_unnormalized = DataframeUtilities()._array_to_dataframe(array=array_unnormalized)
        dataframe_unnormalized = DataframeUtilities()._assign_columns_as_features(dataframe=dataframe_unnormalized, x_features=x_features, y_feature=y_feature, remove_first_row=False)
        return dataframe_unnormalized, scaler

    def normalize_and_merge_with_original_dataframe(self, x_features, y_feature):
        dataframe_normalized, scaler = self.normalize_features(x_features=x_features, y_feature=y_feature)
        dataframe = DataframeUtilities()._merge_dataframe_columns(dataframe1=self.dataframe, dataframe2=dataframe_normalized)
        return dataframe

class MiscOperations():

    @classmethod
    def remove_features_containing_strings(cls, dataframe, x_features):
        x_features_pruned = []
        x_features_to_remove = []
        for x_feature in x_features:
            is_str = False
            for entry in dataframe[x_feature]:
                if type(entry) is str:
                    #print('found a string')
                    is_str = True
            if is_str == True:
                x_features_to_remove.append(x_feature)

        for x_feature in x_features:
            if x_feature not in x_features_to_remove:
                x_features_pruned.append(x_feature)

        dataframe = FeatureIO(dataframe=dataframe).remove_custom_features(features_to_remove=x_features_to_remove)
        return x_features_pruned, dataframe
