from os.path import dirname, abspath
import numpy as np
from itertools import product

HOSTS = ['wally113', 'wally117', 'wally122', 'wally123', 'wally124']

WINDOW_SIZES = [15, 30, 60]
WINDOW_STEPS = [5, 10, 15]

Ls = [1,5,10,20,50,100]
THRESHOLDS = [0.6,0.7,0.8]
REWARD_KINDS = ['continous', 'top', 'threshold']
SEQ = [True, False]

DATA_DIR = '%s/data' % (dirname(dirname(abspath(__file__))))
REWARDS_DIR = '%s/processed/rewards' % DATA_DIR
EXPERIMENT_CONFIG_DIR = '%s/interim/experiment_configs' % DATA_DIR
EXPERIMENT_SERIALIZATION_DIR = '%s/processed/experiment_results/' % DATA_DIR


START_TRACES_SEQUENTIAL = np.datetime64('2019-11-19 17:38:39')
END_TRACES_SEQUENTIAL = np.datetime64('2019-11-20 01:30:00')

START_TRACES_CONCURRENT = np.datetime64('2019-11-25 15:12:13')
END_TRACES_CONCURRENT = np.datetime64('2019-11-25 19:45:00')

SLIDING_WINDOW_SIZES = [None, 100, 250, 500]

GRAPH_DOMAIN_KNOWLEDGES = [
    None
]

GRAPH_DOMAIN_KNOWLEDGES.extend([
    {'name' : 'correct', 'weight': weight} for weight in [0.2,0.5,0.8,1.0]
])

GRAPH_DOMAIN_KNOWLEDGES.extend([
    {'name' : 'flip', 'weight': weight, 'n_affected' : n_affected} for weight, n_affected in product([0.2,0.5,0.8,1.0], [10,100,1000])
])

GRAPH_DOMAIN_KNOWLEDGES.extend([
    {'name' : 'unify', 'weight': weight, 'n_affected' : n_affected} for weight, n_affected in product([0.2,0.5,0.8,1.0], [1,2,5,10])
])

GRAPH_DOMAIN_KNOWLEDGES.extend([
    {'name' : 'add', 'weight': weight, 'n_affected' : n_affected} for weight, n_affected in product([0.2,0.5,0.8,1.0], [1,2,5,10])
])

GRAPH_DOMAIN_KNOWLEDGES.extend([
    {'name' : 'remove', 'weight': weight, 'n_affected' : n_affected} for weight, n_affected in product([0.2,0.5,0.8,1.0], [1,5,10,20,50])
])


