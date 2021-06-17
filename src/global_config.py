from os.path import dirname, abspath
import numpy as np
from itertools import product

HOSTS = ['wally113', 'wally117', 'wally122', 'wally123', 'wally124']

WINDOW_SIZES = [60]
WINDOW_STEPS = [1]

Ls = [21,22,23,24,26,27,28,29,31,32,33,34]
THRESHOLDS = [0.7]
REWARD_KINDS = ['continous', 'threshold']
BINARY_REWARD_KINDS = ['top', 'threshold']
SEQ = [True]

DATA_DIR = '%s/data' % (dirname(dirname(abspath(__file__))))
REWARDS_DIR = '%s/processed/rewards' % DATA_DIR
EXPERIMENT_CONFIG_DIR = '%s/interim/experiment_configs' % DATA_DIR
EXPERIMENT_SERIALIZATION_DIR = '%s/processed/experiment_results/' % DATA_DIR
EXPERIMENT_SERIALIZATION_ADDED_DIR = '%s/processed/experiment_results/added' % DATA_DIR


START_TRACES_SEQUENTIAL = np.datetime64('2019-11-19 17:38:39')
END_TRACES_SEQUENTIAL = np.datetime64('2019-11-20 01:30:00')

START_TRACES_CONCURRENT = np.datetime64('2019-11-25 15:12:13')
END_TRACES_CONCURRENT = np.datetime64('2019-11-25 19:45:00')

SLIDING_WINDOW_SIZES = [None, 500]

GRAPH_DOMAIN_KNOWLEDGES = [
    None
]

GRAPH_DOMAIN_KNOWLEDGES.extend([
    {'name' : 'correct', 'weight': weight} for weight in [0.8,1.0]
])

GRAPH_DOMAIN_KNOWLEDGES.extend([
    {'name' : 'unify', 'weight': weight, 'n_affected' : n_affected} for weight, n_affected in product([0.8,1.0], [1,2,5,10])
])

GRAPH_DOMAIN_KNOWLEDGES.extend([
    {'name' : 'add', 'weight': weight, 'n_affected' : n_affected} for weight, n_affected in product([0.2,0.5,0.8,1.0], [1,2,5,10,15])
])

GRAPH_DOMAIN_KNOWLEDGES.extend([
    {'name' : 'remove', 'weight': weight, 'n_affected' : n_affected} for weight, n_affected in product([0.8,1.0], [1,5,10,20,50,100,200,300,420])
])


