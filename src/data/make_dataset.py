"""Creates the required data for the experiments."""

from datetime import datetime
import copy
import itertools
import os
import argparse
from joblib import Parallel, delayed
import pandas as pd
import yaml
import global_config
from . import reward
experiment_id = 0
n_jobs = 1


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

    Parallel(n_jobs=min(n_jobs, len(metrics_file_paths)))(
        delayed(_clean_metrics_data_for_csv_file)(
            current_path, metrics_dir, start, end, normalize)
        for current_path in metrics_file_paths
    )


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
    Parallel(n_jobs=n_jobs)(
        delayed(_generate_reward_for_config)(seq, window_size, window_step)
        for seq, window_size, window_step in itertools.product(
            [True, False],
            global_config.WINDOW_SIZES,
            global_config.WINDOW_STEPS
        )
    )


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


def _write_experiment_config(config: dict, name: str):
    """Write the config to the data/interim/experimental_configs directory.

    Args:
      config (dict): Configuration of the experiment
      name (str): Includes this name into the filename of the yaml file, to
      make the experiment identifiable.
    """
    global experiment_id
    experiment_id += 1
    with open('%s/%s_experiment_config%d.yml'
              % (global_config.EXPERIMENT_CONFIG_DIR, name, experiment_id), 'w') as yml_file:
        yaml.dump(config, yml_file, default_flow_style=False)


def _generate_experiment_configs():
    """Generates the yaml files that contain the configs of the experiments."""
    # _generate_baseline_experiment_configs()
    # _generate_dkgreedy_parameter_optimization_configs()
    # _generate_push_mpts_parameter_optimization_experiment_configs()
    _generate_cdkegreedy_parameter_optimization_experiment_configs()
    # _generate_cpush_mpts_parameter_optimization_experiment_configs()
    # _generate_dkegreedy_wrong_domainknowledge_configs()
    # _generate_push_mpts_wrong_domainknowledge_configs()
    # _generate_static_network_mpts_configs()
    # _generate_dynamic_network_mpts_configs()
    # _generate_random_network_mpts_configs()


def _write_config_for_params(
        seed: int,
        L: int,
        wsi: int,
        ws: int,
        s: bool,
        rk: str, policies:
        dict, name: str
):
    config_for_this_round = set_window_size_and_step_in_context_path(
        {'seed': seed, 'L': L, 'policies': policies}, s, wsi, ws)

    if rk == 'top':
        reward_path = get_reward_path('top', s, wsi, ws, L=L)
        _write_experiment_config(config_for_this_round | {
            'reward_path': reward_path}, name)
    elif rk == 'continous':
        reward_path = get_reward_path('continous', s, wsi, ws)
        _write_experiment_config(config_for_this_round | {
            'reward_path': reward_path}, name)
    elif rk == 'threshold':
        for th in global_config.THRESHOLDS:
            reward_path = get_reward_path(
                'threshold', s, wsi, ws, threshold=th)
            _write_experiment_config(config_for_this_round | {
                'reward_path': reward_path}, name)


def _write_configs_for_policies(policies, name=''):
    Parallel(n_jobs=n_jobs)(
        delayed(_write_config_for_params)(
            seed + 4500,
            params[0],
            params[1],
            params[2],
            params[3],
            params[4],
            policies,
            name
        ) for seed, params in enumerate(
            itertools.product(
                global_config.Ls,
                global_config.WINDOW_SIZES,
                global_config.WINDOW_STEPS,
                global_config.SEQ,
                global_config.REWARD_KINDS
            )
        )
    )

def _generate_baseline_experiment_configs():
    policies = [
        {'name': 'mpts'}
    ]

    policies.extend(get_cross_validated_policies(
        {'name': 'egreedy'}, {'epsilon': [0.0, 0.01, 0.05, 0.1, 0.2]}))

    solvers = ['liblinear', 'lbfgs', 'newton-cg', 'sag', 'saga']
    policies.extend(get_cross_validated_policies(
        {
            'name': 'cb-egreedy',
            'batch_size': 50,
            'epsilon': 0.1,
            'max_iter': 2500,
            'context_path': global_config.DATA_DIR
            + '/processed/context/%s_context_workload-extractor_w%d_s%d.csv',
            'context_identifier': 'workload'
        }, {'solver': solvers}))

    policies.extend(get_cross_validated_policies(
        {
            'name': 'cb-egreedy',
            'batch_size': 50,
            'epsilon': 0.1,
            'context_identifier': 'empty'
        }, {'solver': solvers}))

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

    policies.extend(
        get_cross_validated_policies(
            {'name': 'push-mpts'},
            {
                'push_likely_arms': [0.1, 0.5, 1, 2, 5, 10],
                'push_unlikely_arms': [0.1, 0.5, 1, 2, 5, 10]
            }
        )
    )

    _write_configs_for_policies(
        policies, name='parameter_optimization_push_mpts'
    )


def _generate_cdkegreedy_parameter_optimization_experiment_configs():
    policies = [
        {'name': 'egreedy', 'epsilon': 0.1},
        {'name': 'greedy'}
    ]

    policies.extend(
        get_cross_validated_policies(
            {
                'name': 'cdkegreedy',
                'context_path':
                global_config.DATA_DIR + '/processed/context/%s_context_host-traces_w%d_s%d.csv',
                'push_kind': 'plus'
            },
            {
                'epsilon': [0, 0.01, 0.05, 0.1, 0.2],
                'init_ev_likely_arms': [0.7, 0.8, 0.9],
                'init_ev_unlikely_arms': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                'init_ev_temporal_correlated_arms': [0.8, 0.9, 1.0],
                'push': [0.01, 0.05, 0.1, 0.2, 0.3, 0.5],
                'max_number_pushes': [5, 10, 20, 100],
                'one_active_host_sufficient_for_push': [True, False]
            }
        )
    )

    policies.extend(
        get_cross_validated_policies(
            {
                'name': 'cdkegreedy',
                'context_path':
                global_config.DATA_DIR + '/processed/context/%s_context_host-traces_w%d_s%d.csv',
                'push_kind': 'multiply'
            },
            {
                'epsilon': [0, 0.01, 0.05, 0.1, 0.2],
                'init_ev_likely_arms': [0.7, 0.8, 0.9],
                'init_ev_unlikely_arms': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                'init_ev_temporal_correlated_arms': [0.8, 0.9, 1.0],
                'push': [1.05, 1.1, 1.2, 1.5],
                'max_number_pushes': [5, 10, 20, 100],
                'one_active_host_sufficient_for_push': [True, False]
            }
        )
    )

    _write_configs_for_policies(
        policies, name='parameter_optimization_cdkegreedy')


def _generate_dkegreedy_wrong_domainknowledge_configs():
    policies = [
        {'name': 'egreedy', 'epsilon': 0.1},
        {'name': 'greedy'}
    ]

    policies.extend(
        get_cross_validated_policies(
            {'name': 'dkgreedy'},
            {
                'epsilon': [0, 0.01, 0.05, 0.1, 0.2],
                'init_ev_likely_arms': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                'init_ev_unlikely_arms': [0.8, 0.9, 1.0],
                'init_ev_temporal_correlated_arms': [0.0, 0.2, 0.4],
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
            'push_unlikely_arms': [0.1, 0.5, 1, 2, 5, 10],
            'push_temporal_correlated_arms': [0, 0.2, 0.4],
        }))

    _write_configs_for_policies(policies, name='push_mpts_wrong_dk')


def _generate_static_network_mpts_configs():
    policies = [
        {'name': 'mpts'}
    ]

    policies.extend(
        get_cross_validated_policies(
            {'name': 'static-network-mpts'},
            {'weight': [0.1, 0.25, 0.5, 0.75, 1.0]}
        )
    )

    _write_configs_for_policies(policies, name='static_network_mpts')


def _generate_dynamic_network_mpts_configs():
    policies = [
        {'name': 'mpts'}
    ]

    policies.extend(
        get_cross_validated_policies(
            {
                'name': 'dynamic-network-mpts',
                'context_path':
                global_config.DATA_DIR + '/processed/context/%s_context_host-traces_w%d_s%d.csv'
            }, {'weight': [0.1, 0.25, 0.5, 0.75, 1.0]}
        )
    )

    _write_configs_for_policies(policies, name='dynamic-network_mpts')


def _generate_random_network_mpts_configs():
    policies = [
        {'name': 'mpts'},
        {'name': 'random-network-mpts'}
    ]

    _write_configs_for_policies(policies, name='random_network_mpts')


def _generate_cpush_mpts_parameter_optimization_experiment_configs():
    policies = [{'name': 'mpts'}]

    policies.extend(
        get_cross_validated_policies(
            {
                'name': 'cpush-mpts',
                'context_path': global_config.DATA_DIR
                + '/processed/context/%s_context_host-traces_w%d_s%d.csv'
            },
            {
                'push_likely_arms': [0.1, 0.5, 1, 2, 5, 10],
                'push_unlikely_arms': [0.1, 0.5, 1, 2, 5, 10],
                'cpush': [0.1, 0.5, 1, 5, 10],
                'q': [5, 10, 20, 100],
                'push_temporal_correlated_arms': [0.8, 0.9, 1.0],
                'one_active_host_sufficient_for_push': [True, False]
            }
        )
    )

    _write_configs_for_policies(policies,
                                name='parameter_optimization_cpush_mpts')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-t',
                        type=str,
                        choices=['clean', 'experiments', 'rewards', 'all'],
                        required=True,
                        help='type of data that will be generated, one of clean, rewards, experiments, all'
                        )

    parser.add_argument('-p',
                        type=int,
                        default=1,
                        help='number of cores to utilize for generation'
                        )

    args = parser.parse_args()

    n_jobs = args.p

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
