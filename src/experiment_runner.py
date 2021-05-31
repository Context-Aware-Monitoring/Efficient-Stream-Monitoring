"""Runs the experiment in using multiple processes.
"""

import multiprocessing as mp
import os
from os.path import dirname, abspath
import argparse
import logging
import numpy as np
from experiment import Experiment

DATA_DIR = '%s/data' % dirname(dirname(abspath(__file__)))
EXPERIMENT_CONFIG_DIR = '%s/interim/experiment_configs' % DATA_DIR

logging.basicConfig(filename='experiments.log',
                    encoding='utf-8', level=logging.INFO)

def perform_experiment_for_config_files(experiment_config_paths):
    """Target for the process to run the experiments.

    Args:
      experiment_config_paths (string[]): Path to the .yaml config files of the
      experiments.
    """
    for i,current_path in enumerate(experiment_config_paths):
        logging.info('Start experiment %d/%d on %d' % (i, len(experiment_config_paths), os.getpid()))
        experiment = Experiment(current_path)
        experiment.run()

        del experiment


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Runs experiments using multiple processes"
    )

    parser.add_argument('number_processes',
                        help="The number of processes used")

    args = parser.parse_args()

    no_processes = int(args.number_processes)

    files = os.listdir(EXPERIMENT_CONFIG_DIR)
    filepaths = np.array(list(map(lambda current_file: '%s/%s' %
                         (EXPERIMENT_CONFIG_DIR, current_file), files)))

    batch_size = int(np.ceil(len(files) / no_processes))

    processes = []
    for i in range(no_processes):
        proc = mp.Process(
            target=perform_experiment_for_config_files,
            args=(filepaths[i * batch_size: (i + 1) * batch_size],))
        processes.append(proc)
        proc.start()

    for p in processes:
        p.join()
