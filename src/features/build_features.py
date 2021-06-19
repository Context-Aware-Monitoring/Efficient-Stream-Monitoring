"""Generates context.csv files that are used by contextual bandits.
"""
from os.path import dirname, abspath
import sys
import csv
import itertools
from . import context
import global_config
import pandas as pd
import numpy as np

print('Generate similiarity context')
context.generate_context_csv(
        context.HostExtractor(),
        [
            '%s/raw/sequential_data/traces/boot_delete/' % global_config.DATA_DIR,
            '%s/raw/sequential_data/traces/image_create_delete/' % global_config.DATA_DIR, 
            '%s/raw/sequential_data/traces/network_create_delete/' % global_config.DATA_DIR
        ],
        seq=True,
        window_size=1,
        step=1,
        outdir='%s/processed/context/' % global_config.DATA_DIR,
        columns=['wally113', 'wally117', 'wally122', 'wally123', 'wally124']
    )

df = pd.read_csv('%s/seq_context_host-traces_w1_s1.csv' % global_config.CONTEXT_DIR)
df['Unnamed: 0'] = pd.to_datetime(df['Unnamed: 0'])
df = df.set_index('Unnamed: 0')
min_1 = df.rolling(60, 0).sum().loc[df.index >= '2019-11-19 17:39:38']
min_5 = df.rolling(300, 0).sum().loc[df.index >= '2019-11-19 17:39:38']
min_15 = df.rolling(900, 0).sum().loc[df.index >= '2019-11-19 17:39:38']

cs = []
for i in range(len(global_config.HOSTS)):
    for j in np.arange(i+1,5):
        h1 = global_config.HOSTS[i]
        h2 = global_config.HOSTS[j]        
        for minute in [1,5,15]:
            cs.append('%s-%s.%dmin.sim' % (hosts[i], hosts[j], minute))
sim_df = pd.DataFrame(index=min_15.index.values, columns=cs)

for i in range(len(global_config.HOSTS)):
    for j in np.arange(i+1,5):
        h1 = global_config.HOSTS[i]
        h2 = global_config.HOSTS[j]

        if h1 == h2:
            continue
        for minute,tdf in zip([1,5,15],[min_1, min_5, min_15]):
            target_col = '%s-%s.%dmin.sim' % (h1, h2, minute)
            sim_df[target_col] = np.abs(tdf[h1].values - tdf[h2].values)
sim_df.to_csv('/seq_context_sim_w60_s1.csv')

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
    
