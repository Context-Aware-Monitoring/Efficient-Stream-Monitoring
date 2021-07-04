import pandas as pd
import numpy as np
import glob
import os
import global_config
import yaml
import time
import sys

import collections

def flatten(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def collect_yaml_config(config):
    df = pd.DataFrame()
    keys = list(filter(lambda key: key not in ['policies', 'seed'], config.keys()))
    global_setting = {key:config[key] for key in keys}

    baseline_regret = 0
    for pol in config['policies']:
        if pol.get('identifier') == 'baseline':
            baseline_regret = pol['regret']

    for pol in config['policies']:
        if pol.get('identifier') == 'baseline':
            continue
        else:
            flat_dict = flatten(pol)
            regret = flat_dict['regret']
            del flat_dict['regret']

            if regret != 0.0:
                flat_dict['improvement'] = baseline_regret / regret
            else:
                flat_dict['improvement'] = 0
            df = df.append(flat_dict | global_setting, ignore_index=True)

    return df

if __name__ == "__main__":
    filestart = sys.argv[1]

    os.chdir(global_config.EXPERIMENT_SERIALIZATION_DIR)
    experiment_files = glob.glob("*.yml")

    df =pd.DataFrame()
    i = 0 
    for ef in experiment_files:
        print('%d/%d' % (i, len(experiment_files)))
        i += 1

        with open(ef, 'r') as ymlfile:
            if filestart == ef[:len(filestart)]:
                experiment_data = yaml.safe_load(ymlfile)
                df = df.append(collect_yaml_config(experiment_data), ignore_index=True)

    df.to_csv('collected_%s.csv' % filestart, index=False)
