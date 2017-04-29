import logging
import numpy as np
import math
from sklearn.grid_search import ParameterGrid
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from xgboost import XGBClassifier, XGBRegressor

# from pickle_utils import dump_metafeatures
from globals import CONFIG

# Parameters for Xgboost
EARLY_STOPPING_ROUNDS = 100
VERBOSE = 50

# Number of positive cases in train set and public leaderboard.
POS_TRAIN = 0.37
POS_PUBLIC_LB = 0.165

# Parameters for rescaling predictions.
A = POS_PUBLIC_LB / POS_TRAIN
B = (1 - POS_PUBLIC_LB) / (1 - POS_TRAIN)


def score(y_true, y_pred, eps=1.e-15):
    return mean_squared_error(y_true, y_pred) ** 0.5


def cross_validation(estimator, X, y, cv, use_watch_list=False, filename=None):

    logging.info('Doing cross validation.')

    mean_score = None
    opt_n_estimators = None
    if filename is not None:
        metafeatures = np.zeros((len(y), 1))

    for i, (train_ind, test_ind) in enumerate(cv):
        # logging.info(np.array(train_ind))
        # logging.info(X.tail())
        # logging.info(y)
        # Split model into training and validation sets.
        X_train, y_train = X[np.array(train_ind)], y[np.array(train_ind)]
        X_test, y_test = X[np.array(test_ind)], y[np.array(test_ind)]

        if isinstance(estimator, XGBRegressor) and use_watch_list:
            # Fit and monitor the progress on test set.
            estimator.fit(X_train, y_train,
                          eval_set=[(X_train, y_train), (X_test, y_test)],
                          eval_metric='logloss',
                          early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                          verbose=VERBOSE)
            # TODO: check if there is an easier way to find out n estimators
            n_estimators = len(estimator.evals_result()['validation_1']['logloss'])
            if opt_n_estimators is None:
                opt_n_estimators = n_estimators
            else:
                opt_n_estimators = max(n_estimators, opt_n_estimators)
        else:
            # Fit the model on training set.
            estimator.fit(X_train, y_train)

        # Make a prediction for test and train sets.
        # y_train_pred = rescale_preds(estimator.predict_proba(X_train)[:, 1])
        # y_test_pred = rescale_preds(estimator.predict_proba(X_test)[:, 1])

        y_train_pred = estimator.predict(X_train)
        y_test_pred = estimator.predict(X_test)

        # Metafeatures!
        if filename is not None:
            metafeatures[test_ind, :] = np.reshape(y_test_pred, (len(test_ind), 1))

        # Calculate scores.
        train_score = score(y_train, y_train_pred)
        test_score = score(y_test, y_test_pred)

        if mean_score is None:
            mean_score = test_score
        else:
            mean_score += test_score

        logging.info('Fold %d, train score: %.5f, test score: %.5f' % (
                      i, train_score, test_score))

    # if filename is not None:
    #    dump_metafeatures(metafeatures, filename)

    mean_score /= len(cv)
    logging.info('Cross validation is done.')

    return mean_score, opt_n_estimators


def fit_and_predict(estimator, X_train, y_train, X_test):

    estimator.fit(X_train, y_train)
    return estimator.predict(X_test)


def tune_parameters(estimator, name, param_grid, X, y, cv):

    logging.info('Tuning parameters for %s model' % name)
    grid_iterable = ParameterGrid(param_grid)

    logging.info('Fitting {0} folds for each of {1} candidates, totalling '
                 '{2} fits'.format(len(cv), len(grid_iterable),
                                   len(cv) * len(grid_iterable)))

    best_score, best_params = None, None
    for grid in grid_iterable:
        estimator.set_params(**grid)
        logging.info('Params: %s' % grid)
        mean_score, opt_n_estimators = cross_validation(estimator, X, y, cv,
                                                        use_watch_list=True)

        if isinstance(estimator, xgb.XGBRegressor):
            grid['n_estimators'] = opt_n_estimators
        if (best_score is None) or (best_score > mean_score):
            best_score, best_params = mean_score, grid

    logging.info('Best parameters: %s, best score: %.5f' % (best_params,
                                                            best_score))
    logging.info('Parameters are tuned for %s model' % name)

    return best_params, best_score


def get_classifiers(names):

    classifiers = []
    for name in names:
        if name == 'LogisticRegression':
            clf = LogisticRegression(penalty='l1', C=0.007,
                                     random_state=CONFIG['RANDOM_SEED'])
        elif name == 'XGBClassifier':
            clf = XGBClassifier(base_score=0.5,
                                colsample_bylevel=1,
                                colsample_bytree=0.9,
                                gamma=0.7,
                                learning_rate=0.1,
                                max_delta_step=0,
                                max_depth=6,
                                min_child_weight=9.0,
                                missing=None,
                                n_estimators=1500,
                                nthread=-1,
                                objective='binary:logistic',
                                reg_alpha=0,
                                reg_lambda=1,
                                scale_pos_weight=1,
                                seed=CONFIG['RANDOM_SEED'],
                                silent=True,
                                subsample=0.9)
        elif name == 'ExtraTreesClassifier':
            clf = ExtraTreesClassifier(n_estimators=50,
                                       max_depth=None,
                                       min_samples_split=10,
                                       min_samples_leaf=5,
                                       max_features='auto',
                                       n_jobs=-1,
                                       random_state=CONFIG['RANDOM_SEED'])
        else:
            raise ValueError('Unknown classifier name.')

        classifiers.append(clf)

    return classifiers


def get_regressors(names):

    regressors = []
    for name in names:
        if name == 'XGBRegressor':
            rgr = XGBRegressor(base_score=0.5,
                               colsample_bylevel=1,
                               colsample_bytree=0.9,
                               gamma=0.7,
                               learning_rate=0.01,
                               max_delta_step=0,
                               max_depth=6,
                               min_child_weight=9.0,
                               missing=None,
                               n_estimators=1000,
                               nthread=-1,
                               reg_alpha=0,
                               reg_lambda=1,
                               scale_pos_weight=1,
                               seed=CONFIG['RANDOM_SEED'],
                               silent=True,
                               subsample=0.9)
        elif name == 'Lasso':
            rgr = Lasso(alpha=1)
        else:
            raise ValueError('Unknown classifier name.')

        regressors.append(rgr)

    return regressors


def get_param_grids(names):

    param_grids = []
    for name in names:
        if name == 'LogisticRegression':
            param_grid = {'C': [0.05, 0.07, 0.08],
                          'penalty': ['l1', 'l2']}
        elif name == 'XGBClassifier':
            param_grid = {'max_depth': [6, 9]}
        elif name == 'XGBRegressor':
            param_grid = {'max_depth': [6, 9]}
        elif name == 'Lasso':
            param_grid = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100]}
        elif name == 'ExtraTreesClassifier':
            param_grid = {'max_features': ['auto', 0.5, 0.9, 1.0]}
        else:
            raise ValueError('Unknown classifier name.')

        param_grids.append(param_grid)

    return param_grids


def get_stacking_param_grids(names):

    param_grids = []
    for name in names:
        if name == 'LogisticRegression':
            param_grid = {'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000],
                          'penalty': ['l1', 'l2'],
                          'fit_intercept': [True]}
        elif name == 'XGBClassifier':
            param_grid = {'max_depth': [6, 9]}
        elif name == 'ExtraTreesClassifier':
            param_grid = {'max_features': ['auto', 0.5, 0.9, 1.0]}
        else:
            raise ValueError('Unknown classifier name.')

        param_grids.append(param_grid)

    return param_grids
