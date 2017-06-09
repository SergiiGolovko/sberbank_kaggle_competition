"""
__file__

    feature_classification.py

__description__

    Create all features.

__author__

    Sergii Golovko < sergii.golovko@gmail.com >

"""

import numpy as np
import os
import pandas as pd
import logging
import re

from globals import CONFIG
from pickle_utils import dump_features, check_if_exists

# Global directories.
BASE_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
DATA_DIR = os.path.join(BASE_DIR, CONFIG['DATA_DIR'])
PICKLE_DIR = os.path.join(BASE_DIR, CONFIG['PICKLED_DIR'])

# Global files.
TRAIN_FILE = os.path.join(DATA_DIR, 'train.csv')
TEST_FILE = os.path.join(DATA_DIR, 'test.csv')
MACRO_FILE = os.path.join(DATA_DIR, 'macro.csv')

# Number of rows to read from files.
TEST_NROWS = CONFIG['TEST_NROWS']
TRAIN_NROWS = CONFIG['TRAIN_NROWS']


def feature_engineering():

    logging.info('FEATURE ENGINEERING')
    create_basic_features()
    create_standard_features()
    logging.info('FINISHED FEATURE ENGINEERING')


def create_standard_features():

    logging.info('Creating standard features')
    feature_class = 'standard'
    if check_if_exists(feature_class):
        logging.info('Standard features already created')
        return

    X_train = pd.read_csv(TRAIN_FILE, nrows=TRAIN_NROWS)
    X_test = pd.read_csv(TEST_FILE, nrows=TEST_NROWS)
    macro = pd.read_csv(MACRO_FILE)

    wrong_format_cols = ['child_on_acc_pre_school', 'modern_education_share',
                         'old_education_build_share']

    def convert_str_to_float(str_):
        try:
            return float(str_)
        except:
            return np.nan

    for col in wrong_format_cols:
        inds = ~pd.isnull(macro[col])
        macro[col] = macro[col].apply(lambda x: convert_str_to_float(x))

    X = pd.concat([X_train, X_test])
    X = pd.merge(X, macro, on='timestamp')

    drop_cols = ['id', 'timestamp', 'price_doc']
    cat_fatures = ['product_type', 'sub_area', 'ecology']
    X.drop(drop_cols, axis=1, inplace=True)

    X = pd.get_dummies(X, columns=cat_fatures)

    bool_features = ['culture_objects_top_25', 'thermal_power_plant_raion',
                     'incineration_raion', 'oil_chemistry_raion',
                     'radiation_raion', 'railroad_terminal_raion',
                     'big_market_raion', 'nuclear_reactor_raion',
                     'detention_facility_raion', 'water_1line',
                     'big_road1_1line', 'railroad_1line']

    for f in bool_features:
        X[f] = X[f] == 'yes'

    X.fillna(-99, inplace=True)

    dump_features(feature_class, X)
    logging.info('Standard features are created and saved to pickle file.')


def create_basic_features():

    logging.info('Creating basic features')
    feature_class = 'basic'
    if check_if_exists(feature_class):
        logging.info('Basic features already created')
        return

    X_train = pd.read_csv(TRAIN_FILE, nrows=TRAIN_NROWS)
    X_test = pd.read_csv(TEST_FILE, nrows=TEST_NROWS)
#    macro = pd.read_csv(MACRO_FILE)
#
#    wrong_format_cols = ['child_on_acc_pre_school', 'modern_education_share',
#                         'old_education_build_share']
#
#    def convert_str_to_float(str_):
#        try:
#            return float(str_)
#        except:
#            return np.nan
#
#    for col in wrong_format_cols:
#        inds = ~pd.isnull(macro[col])
#        macro[col] = macro[col].apply(lambda x: convert_str_to_float(x))
#
    X = pd.concat([X_train, X_test])

    drop_cols = ['id', 'timestamp', 'price_doc']
    cat_fatures = ['product_type', 'sub_area', 'ecology']
    X.drop(drop_cols, axis=1, inplace=True)

    X = pd.get_dummies(X, columns=cat_fatures)

    bool_features = ['culture_objects_top_25', 'thermal_power_plant_raion',
                     'incineration_raion', 'oil_chemistry_raion',
                     'radiation_raion', 'railroad_terminal_raion',
                     'big_market_raion', 'nuclear_reactor_raion',
                     'detention_facility_raion', 'water_1line',
                     'big_road1_1line', 'railroad_1line']

    for f in bool_features:
        X[f] = X[f] == 'yes'

    X.fillna(-99, inplace=True)

    dump_features(feature_class, X)
    logging.info('Basic features are created and saved to pickle file.')


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s: %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p')
    feature_engineering()
