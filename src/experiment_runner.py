"""Runs the experiment in using multiple processes.
"""

import multiprocessing as mp
import os
from os.path import dirname, abspath
import argparse
import numpy as np
from experiment import Experiment
import global_config


def perform_experiment_for_config_files(queue: mp.Queue):
    """Target for the process to run the experiments.

    Args:
      experiment_config_paths (string[]): Path to the .yaml config files of the
      experiments.
    """
    
    while True:
        try:
            filepath = queue.get()
            experiment = Experiment(filepath)
            experiment.run()

            del experiment
        except mp.queues.Empty:
            return
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Runs experiments using multiple processes"
    )

    parser.add_argument('number_processes',
                        help="The number of processes used")

    args = parser.parse_args()

    no_processes = int(args.number_processes)

    files = os.listdir(global_config.EXPERIMENT_CONFIG_DIR)
    filepaths = np.array(list(map(lambda current_file: '%s/%s' %
                         (global_config.EXPERIMENT_CONFIG_DIR, current_file), files)))

    config_files_queue = mp.Queue(len(filepaths))
    for filepath in filepaths:
        config_files_queue.put(filepath)

    processes = []
    for i in range(no_processes):
        proc = mp.Process(
            target=perform_experiment_for_config_files,
            args=(config_files_queue,))
        processes.append(proc)
        proc.start()

    for p in processes:
        p.join()
