"""Creates the required data for the experiments."""

from datetime import datetime
import copy
import itertools
import os
import argparse
import pandas as pd
import yaml
from time import time
import global_config
from . import reward


def clean_metrics_data(metrics_dir: str, start: str, end: str, normalize=True):
    """Cleans the metrics csv files by removing the rows that don't lie within
    the time window specified by start and end. Further linear interpolation is
    used to fill missing values. Writes them to the data/interim directory.

    Args:
      metrics_dir (string): Directory where the metrics csv files are located.
      start (str): Start of the time window
      end (str): End of the time window
      normalize (bool): If true normalizes the different metrics column wise
    """
    metrics_file_paths = []
    metrics_file_paths.extend(
        list(map(lambda x: metrics_dir + x, os.listdir(metrics_dir))))

    for current_path in metrics_file_paths:
        _clean_metrics_data_for_csv_file(
            current_path, metrics_dir, start, end, normalize)


def _clean_metrics_data_for_csv_file(current_path: str, metrics_dir: str,
                                     start: str, end: str, normalize: bool):
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

    if normalize == False:
        _write_cleaned_df_to_csv(current_path, metrics_dir,
                                 df_without_missing_timestamps)
    else:
        _write_cleaned_df_to_csv(current_path, metrics_dir,
                                 _normalize_df(df_without_missing_timestamps))


def _normalize_df(metrics_df: pd.DataFrame) -> pd.DataFrame:
    normalized_df = (metrics_df - metrics_df.min())\
        / (metrics_df.max() - metrics_df.min())
    normalized_df = normalized_df.fillna(1.0)

    return normalized_df


def _write_cleaned_df_to_csv(current_path: str, metrics_dir: str,
                             metrics_df: pd.DataFrame):
    new_filepath = current_path.replace('raw', 'interim')
    new_file_dir = metrics_dir.replace('raw', 'interim')

    if not os.path.exists(new_file_dir):
        os.makedirs(new_file_dir)

    metrics_df.to_csv(new_filepath, index=True, index_label='now')


def _clean_metrics():
    print('Clean metrics data')
    clean_metrics_data(
        '%s/raw/sequential_data/metrics/' % global_config.DATA_DIR,
        '2019-11-19 18:38:39 CEST',
        '2019-11-20 02:30:00 CEST'
    )
    clean_metrics_data(
        '%s/raw/concurrent_data/metrics/' % global_config.DATA_DIR,
        '2019-11-25 16:12:13 CEST',
        '2019-11-25 20:45:00 CEST'
    )


def _generate_rewards():
    """Computes and writes the reward files for the bandit algorithms."""
    print('Generate rewards')
    for seq, window_size, window_step in itertools.product(
            global_config.SEQ,
            global_config.WINDOW_SIZES,
            global_config.WINDOW_STEPS
    ):
        _generate_reward_for_config(seq, window_size, window_step)


def _generate_reward_for_config(seq: bool, window_size: int, window_step: int):
    reward.generate_reward_csv(
        global_config.HOSTS,
        sequential=seq,
        window_size=window_size,
        step=window_step,
        outdir='%s/continous/' % global_config.REWARDS_DIR
    )
    for L in global_config.Ls:
        reward.generate_reward_csv(
            global_config.HOSTS,
            sequential=seq,
            window_size=window_size,
            step=window_step,
            kind='top',
            outdir='%s/top/' % global_config.REWARDS_DIR,
            L=L
        )
    for threshold in global_config.THRESHOLDS:
        reward.generate_reward_csv(
            global_config.HOSTS,
            sequential=seq,
            window_size=window_size,
            step=window_step,
            kind='threshold',
            outdir='%s/threshold/' % global_config.REWARDS_DIR,
            threshold=threshold
        )


def get_reward_path(kind: str, seq: bool, window_size: int, window_step: int,
                    **kwargs) -> str:
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
        return '%s/continous/%s_rewards_w%d_s%d_pearson_continous.csv' \
            % (global_config.REWARDS_DIR, seq_or_con, window_size, window_step)

    if kind == 'threshold':
        return '%s/threshold/%s_rewards_w%d_s%d_pearson_threshold_%.1f.csv'\
            % (global_config.REWARDS_DIR, seq_or_con, window_size, window_step,
               kwargs['threshold'])

    if kind == 'top':
        return '%s/top/%s_rewards_w%d_s%d_pearson_top_%d.csv'\
            % (global_config.REWARDS_DIR, seq_or_con, window_size, window_step,
               kwargs['L'])

    return ''


def get_cross_validated_policies(config_for_policy: dict, params):
    """Creates policies with different combinations of parameters.
    The config_for_policy is completed with the different combinations of
    parameters in the params dictionary.

    Args:
      config_for_policy (dict): Config of a policy. E.g: {'name' : 'egreedy'}
      params (dict): Keys in the dict correspond to parameter names that will
      be filled into the global_config. Values correspond to arrays that contain
      different values. E.g.: {'epsilon': [0,0.5]}

    Returns:
      dict[]: All possible configurations of policies that get generated by using
      different combinations of parameters in the params dict.
      Here: [{'name': 'egreedy', 'epsilon': 0},
             {'name': 'egreedy', 'epsilon': 0.5}]
    """
    policies = []

    for permutation in itertools.product(*(params.values())):
        permutation_config = dict((zip(params.keys(), permutation)))

        policies.append(config_for_policy | permutation_config)

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


def _write_experiment_config(config: dict, name: str):
    """Write the config to the data/interim/experimental_configs directory.

    Args:
      config (dict): Configuration of the experiment
      name (str): Includes this name into the filename of the yaml file, to
      make the experiment identifiable.
    """
    with open('%s/%s.yml' % (global_config.EXPERIMENT_CONFIG_DIR, name), 'w') as yml_file:
        yaml.dump(config, yml_file, default_flow_style=False)

def _generate_mpts_parameter_optimization():
    policies = []
    policies.extend(get_cross_validated_policies(
        {'name': 'push-mpts'},
        {
            'push_likely_arms': [0,5,10],
            'push_temporal_correlated_arms': [0,5,10],
            'push_unlikely_arms': [0,5,10]
        }))

    _write_configs_for_policies(policies, name='mpts_parameter_tuning')

def _generate_egreedy_parameter_optimization():
    policies = []

    policies.extend(get_cross_validated_policies(
        {'name': 'cdkegreedy'},
        {
            'epsilon' : [0.0,0.1],
            'init_ev_likely_arms': [0.8,1.0],
            'init_ev_temporal_correlated_arms': [0.8,1.0],
            'init_ev_unlikely_arms': [0.0,0.5],
        }))

    _write_configs_for_policies(policies, name='mpts_parameter_tuning')

def _generate_mpts():
    # policies = [{'name': 'mpts', 'graph_knowledge': {'name' : 'add', 'n_affected' : 15}}]

    # policies.extend(
    #     get_cross_validated_policies(
    #         {'name': 'push-mpts', 'push_likely_arms': 0.0,
    #             'push_unlikely_arms': 10},
    #         {
    #             'push_temporal_correlated_arms': [0,1,5],
    #             'sliding_window_size': global_config.SLIDING_WINDOW_SIZES,
    #             'graph_knowledge': global_config.GRAPH_DOMAIN_KNOWLEDGES
    #         }
    #     )
    # )
    policies = [{'name' : 'mpts'}]

    _write_configs_for_policies(policies, name='mpts')


def _generate_egreedy():
    policies = []

    policies.extend(get_cross_validated_policies(
        {'name' : 'egreedy'},
        {'epsilon' : [0,0.1]}
    ))
    
    policies.extend(get_cross_validated_policies(
        {'name': 'dkegreedy'},
        {
            'epsilon' : [0.0,0.1],
            'init_ev_likely_arms': [0.8,1.0],
            'init_ev_temporal_correlated_arms': [0.8,1.0],
            'init_ev_unlikely_arms': [0.0,0.5],
            'graph_knowledge': global_config.GRAPH_DOMAIN_KNOWLEDGES,
            'sliding_window_size': global_config.SLIDING_WINDOW_SIZES
        }))
    
    _write_configs_for_policies(policies, name='egreedy')

def _generate_cb():
    policies = []

    policies.extend(
        get_cross_validated_policies(
            {'name': 'cb-full-model',
             'context_path': global_config.DATA_DIR
             + '/processed/context/%s_context_workload-extractor_w%d_s%d.csv',
             'context_identifier': 'workload'
             },
            {'base_algorithm_name': ['logistic_regression', 'ridge', 'ard_regression',
                                     'lin_svc', 'ridge_classifier'], 'algorithm_name': ['egreedy', 'bootstrapped_ucb']}
        )
    )

    _write_configs_for_policies(policies, name='cb', binary_rewards_only=True)


def _generate_awcdkegreedy():
    policies = []

    policies.extend(
        get_cross_validated_policies(
            {
                'name': 'awcdkegreedy',
                'context_path':
                global_config.DATA_DIR + '/processed/context/%s_context_host-traces_w%d_s%d.csv',
                'push_kind': 'plus',
                'init_ev_likely_arms': 0.8,
                'init_ev_temporal_correlated_arms': 1.0,
                'max_number_pushes': 100,
                'push_kind': 'multiply',
                'one_active_host_sufficient_for_push': True,
                'mean_diviation': 1000
            },
            {
                'epsilon': [0, 0.1],
                'push': [1.0, 1.2],
                'graph_knowledge': [{'name': 'correct', 'weight': 1.0}, {'name': 'correct', 'weight': 0.8}, None]
            }
        )
    )

    _write_configs_for_policies(policies, name='awcdkegreedy')


def _generate_cdkegreedy():
    policies = []

    policies.extend(
        get_cross_validated_policies(
            {
                'name': 'cdkegreedy',
                'context_path':
                global_config.DATA_DIR + '/processed/context/%s_context_host-traces_w%d_s%d.csv',
                'push_kind': 'plus',
                'init_ev_likely_arms': 0.8,
                'init_ev_temporal_correlated_arms': 1.0,
                'init_ev_unlikely_arms': 0.5,
                'push_kind': 'multiply'
            },
            {
                'epsilon': [0, 0.1],
                'one_active_host_sufficient_for_push': [True, False],
                'push': [1.0, 1.1, 1.2],
                'max_number_pushes': [10,100,1000]
                # 'sliding_window_size': global_config.SLIDING_WINDOW_SIZES,
                # 'graph_knowledge': [None, {'name': 'correct', 'weight': 1.0}, {'name': 'correct', 'weight': 0.8}]
            }
        )
    )

    _write_configs_for_policies(policies, name='cdkegreedy')

def _generate_sim_cdkegreedy():
    policies = []

    policies.extend(
        get_cross_validated_policies(
            {
                'name': 'cdkegreedy',
                'context_path':
                global_config.DATA_DIR + '/processed/context/%s_context_sim_w%d_s%d.csv',
                'push_kind': 'plus',
                'init_ev_likely_arms': 0.8,
                'init_ev_temporal_correlated_arms': 1.0,
                'kind_knowledge': 'sim'
            },
            {
                'epsilon': [0, 0.1],
                'push': [0.01,0.05,0.1,0.2],
                'threshold': [100,1000,2000,5000],
                'max_number_pushes': [10,100,1000]
            }
        )
    )

    _write_configs_for_policies(policies, name='sim_cdkegreedy')    


def _generate_awcpush_mpts():
    policies = []

    policies.extend(
        get_cross_validated_policies(
            {
                'name': 'awcpush-mpts',
                'context_path':
                global_config.DATA_DIR + '/processed/context/%s_context_host-traces_w%d_s%d.csv',
                'push_likely_arms': 0,
                'push_unlikely_arms': 10,
                'push_temporal_correlated_arms': 1.0,
                'q': 100,
                'mean_diviation': 1000
            },
            {
                'one_active_host_sufficient_for_push': [True, False],
                'cpush': [0, 1],
                'graph_knowledge': [None, {'name': 'correct', 'weight': 1.0}, {'name': 'correct', 'weight': 0.8}]
            }
        )
    )

    _write_configs_for_policies(policies, name='awcpush_mpts')


def _generate_cpush_mpts():
    policies = []

    policies.extend(
        get_cross_validated_policies(
            {
                'name': 'cpush-mpts',
                'context_path':
                global_config.DATA_DIR + '/processed/context/%s_context_host-traces_w%d_s%d.csv',
                'push_likely_arms': 0,
                'push_unlikely_arms': 10,
                'push_temporal_correlated_arms': 1.0
            },
            {
                'q': [10,100,1000],
                'one_active_host_sufficient_for_push': [True, False],
                'cpush': [1, 5, 10]
                # 'sliding_window_size': global_config.SLIDING_WINDOW_SIZES,
                # 'graph_knowledge': [None, {'name': 'correct', 'weight': 1.0}, {'name': 'correct', 'weight': 0.8}]
            }
        )
    )

    _write_configs_for_policies(policies, name='cpush_mpts')

def _generate_sim_cpush_mpts():
    policies = []

    policies.extend(
        get_cross_validated_policies(
            {
                'name': 'cpush-mpts',
                'context_path':
                global_config.DATA_DIR + '/processed/context/%s_context_sim_w%d_s%d.csv',
                'push_likely_arms': 0,
                'push_unlikely_arms': 5,
                'push_temporal_correlated_arms': 5,
                'kind_knowledge': 'sim'
            },
            {
                'q': [10,100,1000],
                'cpush': [1,5,10],
                'threshold': [100,1000,2000,5000]
            }
        )
    )

    _write_configs_for_policies(policies, name='sim_cpush_mpts')    


def _generate_experiment_configs():
    """Generates the yaml files that contain the configs of the experiments."""
    print('Generate experiment configs')
    # _generate_mpts_parameter_optimization()
    # _generate_mpts()
    # _generate_egreedy()
    _generate_sim_cpush_mpts()
    _generate_sim_cdkegreedy()
    # _generate_cb()
    _generate_cdkegreedy()
    _generate_cpush_mpts()
    # _generate_awcdkegreedy()
    # _generate_awcpush_mpts()


def _write_config_for_params(
        seed: int,
        L: int,
        wsi: int,
        ws: int,
        s: bool,
        rk: str,
        policies: dict,
        name: str
):
    name = "%s_%s_L_%d_wsi_%d_ws_%d" % (
        name, 'seq' if s else 'con', L, wsi, ws)

    config_for_this_round = set_window_size_and_step_in_context_path(
        {'seed': seed, 'L': L, 'policies': policies}, s, wsi, ws)

    if rk == 'top':
        reward_path = get_reward_path('top', s, wsi, ws, L=L)
        _write_experiment_config(config_for_this_round | {
            'reward_path': reward_path}, '%s_r_top' % name)
    elif rk == 'continous':
        reward_path = get_reward_path('continous', s, wsi, ws)
        _write_experiment_config(config_for_this_round | {
            'reward_path': reward_path}, '%s_r_continous' % name)
    elif rk == 'threshold':
        for th in global_config.THRESHOLDS:
            reward_path = get_reward_path(
                'threshold', s, wsi, ws, threshold=th)
            _write_experiment_config(config_for_this_round | {
                'reward_path': reward_path}, '%s_r_threshold_%.1f' % (name, th))


def _write_configs_for_policies(policies, name='', binary_rewards_only=False):
    rewards = global_config.BINARY_REWARD_KINDS if binary_rewards_only else global_config.REWARD_KINDS
    for seed, params in enumerate(
            itertools.product(
                global_config.Ls,
                global_config.WINDOW_SIZES,
                global_config.WINDOW_STEPS,
                global_config.SEQ,
                rewards
            )
    ):
        _write_config_for_params(
            seed + int(time()),
            params[0],
            params[1],
            params[2],
            params[3],
            params[4],
            policies,
            name
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-t',
                        type=str,
                        choices=['clean', 'experiments', 'rewards', 'all'],
                        required=True,
                        help='type of data that will be generated, one of clean, rewards, experiments, all'
                        )

    args = parser.parse_args()

    stime = datetime.now()
    if args.t == 'all':
        _clean_metrics()
        _generate_rewards()
        _generate_experiment_configs()
    elif args.t == 'clean':
        _clean_metrics()
    elif args.t == 'rewards':
        _generate_rewards()
    elif args.t == 'experiments':
        _generate_experiment_configs()

    print('Finished, took %d seconds' % ((datetime.now() - stime).seconds))
