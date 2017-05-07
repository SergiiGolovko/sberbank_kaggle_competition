import logging
import numpy as np
import os
import pandas as pd
import pickle
import scipy.sparse as sp
from sklearn.preprocessing import normalize, StandardScaler

from globals import CONFIG

BASE_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
PICKLE_DIR = os.path.join(BASE_DIR, CONFIG['PICKLED_DIR'])
METAFEATURES_DIR = os.path.join(PICKLE_DIR, CONFIG['METAFEATURES_DIR'])
MODELS_DIR = os.path.join(PICKLE_DIR, CONFIG['MODELS_DIR'])


def load_X(feature_classes, train_size, sparse=False, norm=True):

    logging.info('Loading features %s' % feature_classes)
    data = [load_features(feature_class) for feature_class in feature_classes]

    if sparse:
        data = [sp.csr_matrix(features) for features in data]
        res = sp.hstack(data, format='csr')
        if norm:
            scaler = StandardScaler(with_mean=False)
            res = scaler.fit_transform(res)
    else:
        for df in data:
            df.reset_index(inplace=True, drop=True)
        res = pd.concat(data, axis=1)
        # res = np.concatenate([df.values for df in data], axis=1)

    logging.info('Features are loaded.')
    return res[:train_size], res[train_size:]


def load_features(feature_class, dir=PICKLE_DIR):

    path = os.path.join(dir, '%s_features.pickle' % feature_class)
    with open(path, 'rb') as file:
        data = pickle.load(file)

    return data


def dump_features(feature_class, data, dir=PICKLE_DIR):

    path = os.path.join(dir, '%s_features.pickle' % feature_class)
    with open(path, 'wb') as file:
        pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)


def dump_metafeatures(metafeatures, filename, dir=METAFEATURES_DIR):

    if not os.path.exists(dir):
        os.makedirs(dir)

    with open(os.path.join(dir, filename + '.pickle'), 'wb') as file:
        pickle.dump(metafeatures, file, pickle.HIGHEST_PROTOCOL)


def dump_model(model, filename, dir=MODELS_DIR):

    if not os.path.exists(dir):
        os.makedirs(dir)

    with open(os.path.join(dir, filename + '.pickle'), 'wb') as file:
        pickle.dump(model, file, pickle.HIGHEST_PROTOCOL)


def check_if_exists(feature_class, dir=PICKLE_DIR):

    path = os.path.join(dir, '%s_features.pickle' % feature_class)
    return bool(os.path.exists(path))
