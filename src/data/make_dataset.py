"""Creates the required data for the experiments."""

from datetime import datetime
import copy
import itertools
import sys
import os
from os.path import dirname, abspath
import pandas as pd
import yaml
import reward

DATA_DIR = '%s/data' % dirname(dirname(dirname(abspath(__file__))))
REWARDS_DIR = '%s/processed/rewards' % DATA_DIR
EXPERIMENT_CONFIG_DIR = '%s/interim/experiment_configs' % DATA_DIR

seed = 0


def clean_metrics_data(metrics_dir, start, end, normalize=True):
    """Cleans the metrics csv files by removing the rows that don't lie within
    the time window specified by start and end. Further linear interpolation is
    used to fill missing values. Writes them to the data/interim directory.

    Args:
      metrics_dir (string): Directory where the metrics csv files are located.
      start (datetime): Start of the time window
      end (datetime): End of the time window
      normalize (bool): If true normalizes the different metrics column wise
    """
    metrics_file_paths = []
    metrics_file_paths.extend(
        list(map(lambda x: metrics_dir + x, os.listdir(metrics_dir))))

    for current_path in metrics_file_paths:
        metrics_df = pd.read_csv(current_path)
        metrics_df['now'] = pd.to_datetime(metrics_df['now'])
        metrics_df = metrics_df.loc[(
            metrics_df.now >= start) & (metrics_df.now <= end), ]

        metrics_df_single_timestamp = pd.pivot_table(
            metrics_df, index=['now'], aggfunc='mean')

        df_without_missing_timestamps = pd.DataFrame(
            index=pd.date_range(start=start, end=end, freq='1S')
        )
        df_without_missing_timestamps = df_without_missing_timestamps.merge(
            metrics_df_single_timestamp,
            how='left',
            left_index=True,
            right_index=True
        )
        df_without_missing_timestamps.interpolate(
            method='linear', axis=0, inplace=True)

        normalized_df = df_without_missing_timestamps

        if normalize:
            normalized_df = (df_without_missing_timestamps -
                             df_without_missing_timestamps.min()) / (
                df_without_missing_timestamps.max() -
                df_without_missing_timestamps.min())
            normalized_df = normalized_df.fillna(1.0)

        new_filepath = current_path.replace('raw', 'interim')
        new_file_dir = metrics_dir.replace('raw', 'interim')
        if not os.path.exists(new_file_dir):
            os.makedirs(new_file_dir)
        normalized_df.to_csv(new_filepath, index=True, index_label='now')


def _clean_metrics():
    print("Cleaning metrics data")
    clean_metrics_data(
        '%s/raw/sequential_data/metrics/' % DATA_DIR,
        pd.to_datetime('2019-11-19 18:38:39 CEST'),
        pd.to_datetime('2019-11-20 01:30:00 CEST')
    )
    clean_metrics_data(
        '%s/raw/concurrent_data/metrics/' % DATA_DIR,
        pd.to_datetime('2019-11-25 15:12:13 CEST'),
        pd.to_datetime('2019-11-25 19:45:00 CEST')
    )


def _generate_rewards():
    """Computes and writes the reward files for the bandit algorithms."""
    print("Generate rewards")
    hosts = ['wally113', 'wally117', 'wally122', 'wally123', 'wally124']

    for seq, window_size, window_step in itertools.product(
        [True, False],
        [10, 30, 60],
            [1, 5, 10]):
        reward.generate_reward_csv(
            hosts,
            sequential=seq,
            window_size=window_size,
            step=window_step,
            outdir='%s/continous/' % REWARDS_DIR
        )
        for L in [5, 10, 20, 50, 100]:
            reward.generate_reward_csv(
                hosts,
                sequential=seq,
                window_size=window_size,
                step=window_step,
                kind='top',
                outdir='%s/top/' % REWARDS_DIR,
                L=L
            )
        for threshold in [0.6, 0.7, 0.8]:
            reward.generate_reward_csv(
                hosts,
                sequential=seq,
                window_size=window_size,
                step=window_step,
                kind='threshold',
                outdir='%s/threshold/' % REWARDS_DIR,
                threshold=threshold
            )


def get_reward_path(kind, seq, window_size, window_step, **kwargs):
    """For a given configuration of an experiment this method returns the path
    where of the respective reward function.

    Args:
      kind (string): One of 'threshold', 'top' or 'continous'. Defines what
      kind of reward function was used.
      seq (bool): Whether the experiment is for the sequential or concurrent
      data.
      window_size (int): Size of the sliding window
      window_step (int): Step of the sliding window
      kwargs (dict): For the kind 'threshold' this dict should contain
      'threshold', for the kind 'top' it should contain 'L'. These parameters
      are needed to identify the right reward file.

    Returns:
      string: Path of the reward file
    """
    seq_or_con = 'seq' if seq else 'con'
    if kind == 'continous':
        return '%s/continous/%s_rewards_w%d_s%d_pearson_continous.csv' % (
            REWARDS_DIR, seq_or_con, window_size, window_step)

    if kind == 'threshold':
        return '%s/threshold/%s_rewards_w%d_s%d_pearson_threshold_%.1f.csv' % (
            REWARDS_DIR, seq_or_con, window_size, window_step,
            kwargs['threshold'])

    if kind == 'top':
        return '%s/top/%s_rewards_w%d_s%d_pearson_top_%d.csv' % (
            REWARDS_DIR, seq_or_con, window_size, window_step, kwargs['L'])

    return ''


def get_cross_validated_policies(config_for_policy, params):
    """Creates policies with different combinations of parameters.
    The config_for_policy is completed with the different combinations of
    parameters in the params dictionary.

    Args:
      config_for_policy (dict): Config of a policy. E.g: {'name' : 'egreedy'}
      params (dict): Keys in the dict correspond to parameter names that will
      be filled into the config. Values correspond to arrays that contain
      different values. E.g.: {'epsilon': [0,0.5]}

    Returns:
      dict[]: All possible configurations of policies that get generated by using
      different combinations of parameters in the params dict.
      Here: [{'name': 'egreedy', 'epsilon': 0},
             {'name': 'egreedy', 'epsilon': 0.5}]
    """
    policies = []
    if len(params.keys()) == 1:
        for k, values in params.items():
            for current_value in values:
                policies.append(config_for_policy | {k: current_value})
    else:
        k1 = list(params.keys())[0]
        v1 = params[k1]
        updated_params = dict(params)
        del updated_params[k1]
        for current_value in v1:
            new_config = config_for_policy | {k1: current_value}
            policies.extend(get_cross_validated_policies(
                new_config, updated_params))

    return policies


def set_window_size_and_step_in_context_path(
        config, seq, window_size, window_step):
    """Fills the placeholders in the context_path property of a policy with the
    parameters of the experiment.

    Args:
      config (dict): Config of an experiment
      seq (bool): Whether the experiment is for the sequential or concurrent
      data of an experiment.
      window_size (int): Size of the sliding window
      window_step (int): Step of the sliding window

    Returns:
      dict: The configuration of the experiment with the filled in context
      paths.
    """
    config = copy.deepcopy(config)
    seq_or_con = 'seq' if seq else 'con'
    for i in range(len(config['policies'])):
        if config['policies'][i].get('context_path') is not None \
           and '%s' in config['policies'][i]['context_path']:
            config['policies'][i]['context_path'] %= (
                seq_or_con, window_size, window_step)

    return config


experiment_id = 0


def _write_experiment_config(config, name):
    """Write the config to the data/interim/experimental_configs directory.

    Args:
      config (dict): Configuration of the experiment
    """
    global experiment_id
    experiment_id += 1
    with open('%s/%s_experiment_config%d.yml' % (EXPERIMENT_CONFIG_DIR, name, experiment_id), 'w') \
            as yml_file:
        yaml.dump(config, yml_file, default_flow_style=False)


def _generate_experiment_configs():
    """Generates the yaml files that contain the configs of the experiments."""
    print('Generating experiment configs')

    _generate_baseline_experiment_configs()
    _generate_dkgreedy_parameter_optimization_configs()
    _generate_push_mpts_parameter_optimization_experiment_configs()
    _generate_cdkegreedy_parameter_optimization_experiment_configs()
    _generate_cpush_mpts_parameter_optimization_experiment_configs()
    _generate_dkegreedy_wrong_domainknowledge_configs()
    _generate_push_mpts_wrong_domainknowledge_configs()


def _write_configs_for_policies(policies, name=''):
    global seed
    Ls = [5, 10, 20, 50, 100]
    window_size = [10, 30, 60]
    window_step = [1, 5, 10]
    seq = [True, False]

    reward_kinds = ['top', 'threshold']  # , 'continous']

    for L, wsi, ws, s, rk in itertools.product(
            Ls, window_size, window_step, seq, reward_kinds):
        config_for_this_round = set_window_size_and_step_in_context_path(
            {'seed': seed, 'L': L, 'policies': policies}, s, wsi, ws)
        seed += 1
        if rk == 'top':
            reward_path = get_reward_path('top', s, wsi, ws, L=L)
            _write_experiment_config(config_for_this_round | {
                                     'reward_path': reward_path}, name)
        elif rk == 'continous':
            reward_path = get_reward_path('continous', s, wsi, ws)
            _write_experiment_config(config_for_this_round | {
                                     'reward_path': reward_path}, name)
        elif rk == 'threshold':
            for th in [0.6, 0.7, 0.8]:
                reward_path = get_reward_path(
                    'threshold', s, wsi, ws, threshold=th)
                _write_experiment_config(config_for_this_round | {
                                         'reward_path': reward_path}, name)


def _generate_baseline_experiment_configs():
    policies = [
        {'name': 'mpts'},
        {
            'name': 'cb-egreedy',
            'batch_size': 50,
            'epsilon': 0.1,
            'identifier': 'empty-context-0.1-egreedy'
        },
        {
            'name': 'cb-egreedy',
            'batch_size': 50,
            'epsilon': 0.1,
            'identifier': 'workload-context-0.1-egreedy',
            'max_iter': 2500,
            'context_path': DATA_DIR
            + '/processed/context/%s_context_workload-extractor_w%d_s%d.csv'
        }
    ]

    policies.extend(get_cross_validated_policies(
        {'name': 'egreedy'}, {'epsilon': [0.0, 0.01, 0.05, 0.1, 0.2]}))

    _write_configs_for_policies(policies, name='baselines')


def _generate_dkgreedy_parameter_optimization_configs():
    policies = [
        {'name': 'egreedy', 'epsilon': 0.1},
        {'name': 'egreedy'},
    ]

    policies.extend(get_cross_validated_policies(
        {'name': 'dkgreedy'}, {
            'epsilon': [0, 0.01, 0.05, 0.1, 0.2],
            'init_ev_likely_arms': [0.8, 0.9, 1.0],
            'init_ev_unlikely_arms': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        }))

    _write_configs_for_policies(
        policies, name='parameter_optimization_dkgreedy')


def _generate_push_mpts_parameter_optimization_experiment_configs():
    policies = [
        {'name': 'mpts'}
    ]

    policies.extend(get_cross_validated_policies(
        {'name': 'push-mpts'}, {
            'push_likely_arms': [0.1, 0.5, 1, 2, 5, 10],
            'push_unlikely_arms': [0.1, 0.5, 1, 2, 5, 10]
        }))

    _write_configs_for_policies(
        policies, name='parameter_optimization_push_mpts')


def _generate_cdkegreedy_parameter_optimization_experiment_configs():
    policies = [
        {'name': 'egreedy', 'epsilon': 0.1},
        {'name': 'greedy'}
    ]

    policies.extend(
        get_cross_validated_policies(
            {'name': 'cdkegreedy',
             'context_path':
             DATA_DIR + '/processed/context/%s_context_host-traces_w%d_s%d.csv',
             'context_kind': 'plus'},
            {'epsilon': [0, 0.01, 0.05, 0.1, 0.2],
             'init_ev_likely_arms': [0.7, 0.8, 0.9],
             'init_ev_unlikely_arms': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
             'push': [0.01, 0.05, 0.1, 0.2, 0.3, 0.5],
             'max_number_pushes': [5, 10, 20, 100]}))

    _write_configs_for_policies(
        policies, name='parameter_optimization_cdkegreedy')


def _generate_dkegreedy_wrong_domainknowledge_configs():
    policies = [
        {'name': 'egreedy', 'epsilon': 0.1},
        {'name': 'greedy'}
    ]

    policies.extend(
        get_cross_validated_policies(
            {'name': 'dkgreedy'}, {
                'epsilon': [0, 0.01, 0.05, 0.1, 0.2],
                'init_ev_likely_arms': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                'init_ev_unlikely_arms': [0.8, 0.9, 1.0]
            }
        )
    )

    _write_configs_for_policies(policies, name="dkegreedy_wrong_dk")


def _generate_push_mpts_wrong_domainknowledge_configs():
    policies = [
        {'name': 'mpts'}
    ]

    policies.extend(get_cross_validated_policies(
        {'name': 'inverted-push-mpts'}, {
            'push_likely_arms': [0.1, 0.5, 1, 2, 5, 10],
            'push_unlikely_arms': [0.1, 0.5, 1, 2, 5, 10]
        }))

    _write_configs_for_policies(
        policies, name='push_mpts_wrong_dk')


def _generate_cpush_mpts_parameter_optimization_experiment_configs():
    policies = [{'name': 'mpts'}]

    policies.extend(
        get_cross_validated_policies(
            {'name': 'cpush-mpts',
             'context_path':
             DATA_DIR + '/processed/context/%s_context_host-traces_w%d_s%d.csv'},
            {'push': [0.1, 0.5, 1, 5, 10],
             'cpush': [0.1, 0.5, 1, 5, 10],
             'q': [5, 10, 20, 100]}))

    _write_configs_for_policies(
        policies, name='parameter_optimization_cpush_mpts')


if __name__ == '__main__':
    args = sys.argv
    if len(args) == 1:
        print("Possible arguments are --clean, --rewards, --experiments")
    args = args[1:]

    for arg in args:
        stime = datetime.now()
        if arg == '--clean':
            _clean_metrics()
        elif arg == '--rewards':
            _generate_rewards()
        elif arg == '--experiments':
            _generate_experiment_configs()
        elif arg == '--all':
            _clean_metrics()
            _generate_rewards()
            _generate_experiment_configs()
        else:
            sys.exit('Invalid argument %s found' % arg)
        print('Finished, took %d seconds' %
              ((datetime.now() - stime).seconds))
