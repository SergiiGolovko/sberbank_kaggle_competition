"""
__file__

    modelling.py

__description__

    Generate predictions.

__author__

    Sergii Golovko < sergii.golovko@gmail.com >

"""

import numpy as np
import logging
import os
import math
import pandas as pd
from xgboost import XGBClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.linear_model import LogisticRegression

from globals import CONFIG
# from feature_classification import get_features, get_feature_classes
# from pickle_utils import load_X
from cv_utils import get_cv
from model_utils import tune_parameters, fit_and_predict, cross_validation
from model_utils import get_regressors, get_param_grids

# Global directories.
BASE_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
DATA_DIR = os.path.join(BASE_DIR, CONFIG['DATA_DIR'])
OUTPUT_DIR = os.path.join(BASE_DIR, CONFIG['OUTPUT_DIR'])
PICKLE_DIR = os.path.join(BASE_DIR, CONFIG['PICKLED_DIR'])
PRED_DIR = os.path.join(OUTPUT_DIR, CONFIG['PRED_DIR'])

# Global files.
TRAIN_FILE = os.path.join(DATA_DIR, 'train.csv')
TEST_FILE = os.path.join(DATA_DIR, 'test.csv')
MACRO_FILE = os.path.join(DATA_DIR, 'macro.csv')

# Number of rows to read from files.
TEST_NROWS = CONFIG['TEST_NROWS']
TRAIN_NROWS = CONFIG['TRAIN_NROWS']

# Turn off/on parameters tuning and cross validation.
TUNE_PARAMETERS = True
DO_CROSS_VALIDATION = True


# TODO: this needs to be moved to utils type of file.
def save_predictions(preds, name):

    if not os.path.exists(PRED_DIR):
        os.makedirs(PRED_DIR)

    path = os.path.join(PRED_DIR, name + '.csv')
    preds['price_doc'] = preds['price_doc'].apply(lambda x: math.exp(x) - 1)
    preds.to_csv(path, index=False)


def generate_predictions(estimators, names, par_grids, test_ids, X_train,
                         y_train, X_test):

    preds = test_ids.copy()

    # CV for parameter tuning. To speed up the process I am doing stratisfied
    # 2 iteration CV when tuning the parameters.
    cv1 = get_cv(y_train)

    # CV for cross validation. Must be KFold to create metafeatures.
    cv2 = get_cv(y_train, n_folds=5, type='kfold')

    for (estimator, par_grid, name) in zip(estimators, par_grids, names):
        filename = name

        if TUNE_PARAMETERS:
            logging.info('Doing parameter tuning for %s model' % name)
            best_params, best_score = tune_parameters(estimator, name,
                                                      par_grid, X_train.values,
                                                      y_train, cv1)
            estimator.set_params(**best_params)
            logging.info('Finished parameter tuning for %s model' % name)

        if DO_CROSS_VALIDATION:
            logging.info('Doing cross validation for %s model' % name)
            cross_validation(estimator, X_train.values, y_train, cv2,
                             filename=None)
            logging.info('Finished cross validation for %s model' % name)

        logging.info('Fitting %s model' % name)
        preds['price_doc'] = (
            fit_and_predict(estimator, X_train, y_train, X_test))
        save_predictions(preds, filename)
        logging.info('Finished fitting %s model' % name)


def modelling():

    logging.info('MODELLING')

    test_ids = pd.read_csv(TEST_FILE, usecols=['id'], nrows=TEST_NROWS)
    y_train = pd.read_csv(TRAIN_FILE, usecols=['price_doc'],
                          nrows=TRAIN_NROWS)
    y_train['price_doc'] = y_train['price_doc'].apply(lambda x: math.log(x + 1))
    y_train = y_train['price_doc'].values

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

    X_train = X[:len(X_train)]
    X_test = X[len(X_train):]

    names = ['XGBRegressor']
    estimators = get_regressors(names)
    par_grids = get_param_grids(names)

    generate_predictions(estimators, names, par_grids, test_ids, X_train,
                         y_train, X_test)

    logging.info('FINISHED MODELLING.')

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s: %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p')
    modelling()
