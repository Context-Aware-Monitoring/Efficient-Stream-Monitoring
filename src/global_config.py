from os.path import dirname, abspath
import numpy as np
from itertools import product


HOSTS = ['wally113', 'wally117', 'wally122', 'wally123', 'wally124']

WINDOW_SIZES = [60]
WINDOW_STEPS = [1]

Ls = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35 , 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100]

THRESHOLDS = [0.6, 0.7]
REWARD_KINDS = ['continous', 'threshold']
BINARY_REWARD_KINDS = ['top', 'threshold']
SEQ = [False]

DATA_DIR = '%s/data' % (dirname(dirname(abspath(__file__))))
CONTEXT_DIR = '%s/processed/context' % DATA_DIR
REWARDS_DIR = '%s/processed/rewards' % DATA_DIR
EXPERIMENT_CONFIG_DIR = '%s/interim/experiment_configs' % DATA_DIR
EXPERIMENT_SERIALIZATION_DIR = '%s/processed/experiment_results/' % DATA_DIR
EXPERIMENT_SERIALIZATION_ADDED_DIR = '%s/processed/experiment_results/added' % DATA_DIR

PLOT_DATA_DIR = '/home/tom/Documents/Efficient-Stream-Monitoring_optimized/latex/images/plots/data'

START_TRACES_SEQUENTIAL = np.datetime64('2019-11-19 17:38:39')
END_TRACES_SEQUENTIAL = np.datetime64('2019-11-20 01:30:00')

START_TRACES_CONCURRENT = np.datetime64('2019-11-25 15:12:13')
END_TRACES_CONCURRENT = np.datetime64('2019-11-25 19:45:00')

SLIDING_WINDOW_SIZES = [500, 1000]

GRAPH_DOMAIN_KNOWLEDGES = [
    None
]

GRAPH_DOMAIN_KNOWLEDGES.extend([
    {'name' : 'correct', 'weight': weight, 'only_push_arms_that_were_not_picked' : opatwnp} for weight, opatwnp in product([0.8,1.0], [True, False])
])

GRAPH_DOMAIN_KNOWLEDGES.extend([
    {'name' : 'unify', 'weight': weight, 'n_affected' : n_affected, 'only_push_arms_that_were_not_picked': opatwnp} for weight, n_affected, opatwnp in product([0.8,1.0], [1,2,5,10], [True, False])
])

GRAPH_DOMAIN_KNOWLEDGES.extend([
    {'name' : 'add', 'weight': weight, 'n_affected' : n_affected, 'only_push_arms_that_were_not_picked': opatwnp} for weight, n_affected, opatwnp in product([0.2,0.5,0.8,1.0], [1,2,5,10,15], [True, False])
])

GRAPH_DOMAIN_KNOWLEDGES.extend([
    {'name' : 'remove', 'weight': weight, 'n_affected' : n_affected, 'only_push_arms_that_were_not_picked': opatwnp} for weight, n_affected, opatwnp in product([0.8,1.0], [1,5,10,20,50,100,200,300,420], [True, False])
])


