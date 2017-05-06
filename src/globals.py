# Define different run configurations.
CONFIG_normal = {'CONFIG': 'config_normal',
                 'OUTPUT_DIR': 'output',
                 'PICKLED_DIR': 'pickled',
                 'DATA_DIR': 'data',
                 'TEST_NROWS': None,
                 'TRAIN_NROWS': None
                 }

CONFIG_test = {'CONFIG': 'config_test',
               'OUTPUT_DIR': 'testoutput',
               'PICKLED_DIR': 'testpickled',
               'DATA_DIR': 'data',
               'TEST_NROWS': 100,
               'TRAIN_NROWS': 10000
               }

# Select a configuration to run.
CONFIG = CONFIG_normal
# In order to run 'normal', just uncomment the following line, do NOT delete it
# CONFIG = CONFIG_test

CONFIG['RANDOM_SEED'] = 2016
CONFIG['MAX_RANDOM_SEED'] = 2014
CONFIG['FOLDS_NUM'] = 2
CONFIG['CV_TYPE'] = 'STRATIFIED'
CONFIG['REPETITION_NUM'] = 1
CONFIG['PRED_DIR'] = 'preds'
CONFIG['METAFEATURES_DIR'] = 'metafeatures'
CONFIG['PREPROCESS_DIR'] = 'preprocessed'
CONFIG['STACKING_PRED_DIR'] = 'stacking_preds'
