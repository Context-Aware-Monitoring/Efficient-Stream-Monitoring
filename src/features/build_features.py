"""Generates context.csv files that are used by contextual bandits.
"""
from os.path import dirname, abspath
import sys
import csv
import itertools
from . import context
import global_config

DATA_DIR = '%s/data' % dirname(dirname(dirname(abspath(__file__))))

WINDOW_SIZES = [30,60]
WINDOW_STEPS = [5,10,20]

def _get_headers_from_csv_file(filepath):
    headers = None
    with open(filepath) as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader)

    return headers


if __name__ == '__main__':
    for seq, window_size, window_step in itertools.product(
            global_config.SEQ,
            global_config.WINDOW_SIZES,
            global_config.WINDOW_STEPS
    ):
        context.generate_context_csv(
            context.HostExtractor(),
            [
                '%s/raw/sequential_data/traces/boot_delete/' % DATA_DIR,
                '%s/raw/sequential_data/traces/image_create_delete/' % DATA_DIR,
                '%s/raw/sequential_data/traces/network_create_delete/' % DATA_DIR
            ],
            seq=seq,
            window_size=window_size,
            step=window_step,
            outdir='%s/processed/context/' % DATA_DIR,
            columns=['wally113', 'wally117', 'wally122', 'wally123', 'wally124']
        )
        
        context_filepath = context.generate_context_csv(
            context.WorkloadExtractor(),
            [
                '%s/raw/sequential_data/traces/boot_delete/' % DATA_DIR,
                '%s/raw/sequential_data/traces/image_create_delete/' % DATA_DIR,
                '%s/raw/sequential_data/traces/network_create_delete/' % DATA_DIR
            ],
            seq=seq,
            window_size=window_size,
            step=window_step,
            outdir='%s/processed/context/' % DATA_DIR
        )
