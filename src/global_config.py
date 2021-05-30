from os.path import dirname, abspath
import numpy as np

HOSTS = ['wally113', 'wally117', 'wally122', 'wally123', 'wally124']

WINDOW_SIZES = [30, 60]
WINDOW_STEPS = [5, 10, 20]

Ls = [5, 10, 20, 50, 100]
THRESHOLDS = [0.6, 0.7, 0.8]
REWARD_KINDS = ['top', 'threshold']
SEQ = [True, False]

DATA_DIR = '%s/data' % (dirname(dirname(abspath(__file__))))
REWARDS_DIR = '%s/processed/rewards' % DATA_DIR
EXPERIMENT_CONFIG_DIR = '%s/interim/experiment_configs' % DATA_DIR


START_TRACES_SEQUENTIAL = np.datetime64('2019-11-19 17:38:39')
END_TRACES_SEQUENTIAL = np.datetime64('2019-11-20 01:30:00')

START_TRACES_CONCURRENT = np.datetime64('2019-11-19 15:12:13')
END_TRACES_CONCURRENT = np.datetime64('2019-11-20 19:45:00')
