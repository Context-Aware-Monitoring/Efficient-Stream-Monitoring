"""Generates context.csv files that are used by contextual bandits.
"""
from os.path import dirname, abspath
import sys
import csv
import itertools
from . import context
import global_config


for seq, window_size, window_step in itertools.product(
        global_config.SEQ,
        global_config.WINDOW_SIZES,
        global_config.WINDOW_STEPS
):
    seq_or_con = 'sequential_data' if seq else 'concurrent_data'
    context.generate_context_csv(
        context.HostExtractor(),
        [
            '%s/raw/%s/traces/boot_delete/' % (
                global_config.DATA_DIR, seq_or_con),
            '%s/raw/%s/traces/image_create_delete/' % (
                global_config.DATA_DIR, seq_or_con),
            '%s/raw/%s/traces/network_create_delete/' % (
                global_config.DATA_DIR, seq_or_con)
        ],
        seq=seq,
        window_size=window_size,
        step=window_step,
        outdir='%s/processed/context/' % global_config.DATA_DIR,
        columns=['wally113', 'wally117', 'wally122', 'wally123', 'wally124']
    )

    context_filepath = context.generate_context_csv(
        context.WorkloadExtractor(),
        [
            '%s/raw/%s/traces/boot_delete/' % (
                global_config.DATA_DIR, seq_or_con),
            '%s/raw/%s/traces/image_create_delete/' % (
                global_config.DATA_DIR, seq_or_con),
            '%s/raw/%s/traces/network_create_delete/' % (
                global_config.DATA_DIR, seq_or_con)
        ],
        seq=seq,
        window_size=window_size,
        step=window_step,
        outdir='%s/processed/context/' % global_config.DATA_DIR
    )
