"""Generates context.csv files that are used by contextual bandits.
"""
from os.path import dirname, abspath
import sys
import csv
import itertools
import context

DATA_DIR = '%s/data' % dirname(dirname(dirname(abspath(__file__))))


def _get_headers_from_csv_file(filepath):
    headers = None
    with open(filepath) as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader)

    return headers


if __name__ == '__main__':
    args = sys.argv
    if len(args) == 1:
        print("Possible arguments are --context --all")
    args = args[1:]
    if "--all" in args:
        args = ['--context']

    for arg in args:
        if arg == '--context':
            for seq, window_size, window_step in itertools.product(
                [True, False],
                [10, 30, 60],
                    [1, 5, 10]):
                context.generate_context_csv(
                    context.HostExtractor(),
                    ['%s/raw/sequential_data/traces/boot_delete/' % DATA_DIR,
                     '%s/raw/sequential_data/traces/image_create_delete/' %
                     DATA_DIR,
                     '%s/raw/sequential_data/traces/network_create_delete/' %
                     DATA_DIR],
                    seq=seq, window_size=window_size, step=window_step,
                    outdir='%s/processed/context/' % DATA_DIR,
                    columns=['wally113', 'wally117', 'wally122', 'wally123',
                             'wally124'])
                context_filepath = context.generate_context_csv(
                    context.WorkloadExtractor(),
                    ['%s/raw/sequential_data/traces/boot_delete/' % DATA_DIR,
                     '%s/raw/sequential_data/traces/image_create_delete/' %
                     DATA_DIR,
                     '%s/raw/sequential_data/traces/network_create_delete/' %
                     DATA_DIR],
                    seq=seq, window_size=window_size, step=window_step,
                    outdir='%s/processed/context/' % DATA_DIR)
        else:
            sys.exit('Invalid argument %s found' % arg)
