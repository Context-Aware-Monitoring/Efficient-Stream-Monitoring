"""Creates the required data for the experiments."""

from datetime import datetime
import copy
import itertools
import os
import argparse
import pandas as pd
import numpy as np
from time import time
import yaml
import global_config
from . import reward
from itertools import product

groups = np.repeat(np.arange(10) + 1, 10)
rnd = np.random.RandomState(10)
arms = 100
seed = int(time())

def _generate_synthetic_experiments_for_gk():
    for c, T, kind in product([0, 0.01,0.05,0.1, 0.2], [10, 100,500,1000], ['bern', 'norm-sigma-0.1', 'norm-sigma-0.25']):
        reward = np.zeros(shape=(T,arms))
        unique_groups = np.unique(groups)
        ps = rnd.random(unique_groups.shape[0])

        for p, cg in zip(ps, unique_groups):
            no_members = (groups == cg).sum()
            deviation = rnd.uniform(c, -c, no_members)
            ps_for_group = np.minimum(np.maximum(p + deviation, 0), 1)

            if kind == 'bern':
                reward[:, groups == cg] = rnd.binomial(1, ps_for_group, (T, no_members))
            else:
                if kind == 'norm-sigma-0.1':
                    sigma_max = 0.1
                elif kind == 'norm-sigma-0.25':
                    sigma_max = 0.25
                sigmas = rnd.uniform(0, sigma_max, no_members)
                reward[:, groups == cg] = np.maximum(0, rnd.normal(ps_for_group, sigmas, (T, no_members)))

        reward_df_name = '%s/synthetic/synthetic_%s_reward_df_T_%d_arms_%d_c_%.2f_groups_%d.csv' % (
            global_config.REWARDS_DIR, kind, T, arms, c, unique_groups.shape[0])
        pd.DataFrame(data = reward).to_csv(reward_df_name)
        for weight in [0.2,0.5,0.8,1.0]:
            for L in range(1,51):
                policies = [{'name' : 'mpts', 'identifier': 'baseline'}]

                policies.extend([
                    {
                            'name': 'mpts',
                            'graph_knowledge': {
                                'name' : 'synthetic',
                                'weight' : weight,
                                'groups': groups.tolist(),
                                'only_push_arms_that_were_not_picked': opatwnp
                            }
                    } for opatwnp in [True, False]
                ])
                
                error_gk_policies = []
                for error_kind, perc_affected, opatwnp in product(['remove', 'random'], [0.01,0.05,0.1,0.25,0.5,0.75,1.0], [True, False]):
                    error_gk_policies.append(
                        {
                            'name': 'mpts',
                            'graph_knowledge': {
                                'name' : 'wrong-synthetic',
                                'weight' : weight,
                                'groups': groups.tolist(),
                                'percentage_affected' : perc_affected,
                                'error_kind' : error_kind,
                                'only_push_arms_that_were_not_picked': opatwnp
                            }
                        }
                    )
                policies.extend(error_gk_policies)
                config = {
                    'policies': policies,
                    'reward_path': reward_df_name,
                    'seed': rnd.randint(10000),
                    'L': L,
                    'T': T,
                    'c': c,
                    'dist': kind
                }

                with open('%s/synthetic_%s_gk_dk_w_%.1f_L_%d_groups_%d_T_%d_c_%.2f.yml' % (global_config.EXPERIMENT_CONFIG_DIR, kind, weight, L, unique_groups.shape[0], T, c), 'w') as outfile:
                    yaml.dump(config, outfile, default_flow_style=False)
                    

def _generate_synthetic_experiments_for_static_push():
    for T, dist in product([10,100,500,1000], ['bern', 'norm-sigma-0.1', 'norm-sigma-0.25']):        
        reward = np.zeros(shape=(T,arms))

        mus = -np.sort(-rnd.uniform(0, 1, arms))

        if dist == 'bern':
            reward = rnd.binomial(1, mus, (T, arms))
        else:
            if dist == 'norm-sigma-0.1':
                sigmas = rnd.uniform(0, 0.1, arms)
            elif dist == 'norm-sigma-0.25':
                sigmas = rnd.uniform(0, 0.25, arms)
            reward = np.minimum(np.maximum(0, rnd.normal(mus, sigmas, (T, arms))), 1)


        reward_path = '%s/synthetic/synthetic_static_push_dist_%s_arms_%d_T_%d.csv' % (
            global_config.REWARDS_DIR, dist, arms, T)

        pd.DataFrame(data = reward).to_csv(reward_path)

        for L in range(1,51):
            policies = [{'name' : 'mpts', 'identifier': 'baseline'}]

            policies.extend(get_cross_validated_policies(
                {'name': 'push-mpts', 'arm_knowledge':{'name':'synthetic-static-push'}},
                {'push_likely_arms' : [0,1,3,5], 'push_unlikely_arms': [0,1,3,5]}
            ))
            for kind in ['remove', 'random']:
                for paff in [0.01,0.05,0.1,0.25,0.5,0.75,1.0]:
                    policies.extend(get_cross_validated_policies(
                        {'name': 'push-mpts', 'arm_knowledge':{'name':'synthetic-static-push-wrong', 'kind':kind, 'percentage_affected' : paff}},
                        {
                            'push_likely_arms' : [0,1,3,5],
                            'push_unlikely_arms' : [0,1,3,5]
                        }
                    ))

            config = {
                'policies': policies,
                'reward_path': reward_path,
                'seed': rnd.randint(10000),
                'L': L,
                'T': T,
                'dist' : dist
            }

            with open('%s/synthetic_static_push_dist_%s_arms_%d_T_%d_L_%d.yml' % (
                global_config.EXPERIMENT_CONFIG_DIR, dist, arms, T, L), 'w') as outfile:
                yaml.dump(config, outfile, default_flow_style=False)

def _generate_synthetic_experiments_for_push():
    for c, T, dist in product([0.1,0.2,0.3,0.5], [10,100,500,1000], ['bern', 'norm-sigma-0.1', 'norm-sigma-0.25']):
        reward = np.zeros(shape=(T,arms))

        mus = -np.sort(-rnd.uniform(0, 1-c, arms))

        if dist == 'bern':
            reward = rnd.binomial(1, mus, (T, arms))
            reward_pushed = rnd.binomial(1, mus + c, (T, arms))
        else:
            if dist == 'norm-sigma-0.1':
                sigmas = rnd.uniform(0, 0.1, arms)
            elif dist == 'norm-sigma-0.25':
                sigmas = rnd.uniform(0, 0.25, arms)
            reward = np.minimum(np.maximum(0, rnd.normal(mus, sigmas, (T, arms))), 1)
            reward_pushed = np.minimum(np.maximum(0, rnd.normal(mus + c, sigmas, (T, arms))), 1)

        for pushes_perc in [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]:
            context = np.zeros(T * arms, dtype=bool)
            total_num_pushes = int(np.floor(T * arms * pushes_perc))

            context[rnd.choice(T * arms, total_num_pushes, replace=False)] = True
            context = context.reshape(T, arms)
            num_pushes_for_arm = context.sum(axis=1)
            for i in range(arms):
                reward[context[:,i], i] = reward_pushed[context[:,i], i]
            reward_path = '%s/synthetic/synthetic_push_c_%.2f_pc_%.2f_dist_%s_arms_%d_T_%d.csv' % (
                global_config.REWARDS_DIR, c, pushes_perc, dist, arms, T)
            context_path = '%s/context_synthetic_push_c_%.2f_pc_%.2f_dist_%s_arms_%d_T_%d.csv' % (
                global_config.CONTEXT_DIR, c, pushes_perc, dist, arms, T)

            pd.DataFrame(data = reward).to_csv(reward_path)
            pd.DataFrame(data = context).to_csv(context_path)

            for L in range(1,51):
                policies = [{'name' : 'mpts', 'identifier': 'baseline'}]
                for cpush in [1.01,1.03,1.05,1.1,1.2,1.25, 1.5,1.75,2.0,2.5,3,5]:
                    policies.append(
                        {
                            'name': 'cpush-mpts',
                            'context_path' : context_path,
                            'arm_knowledge' : {'name': 'synthetic-dynamic-push'},
                            'push_kind' : 'multiply',
                            'cpush': cpush,
                            'learn_pushed' : True
                        })
                    policies.append(
                        {
                            'name': 'cpush-mpts',
                            'context_path' : context_path,
                            'arm_knowledge' : {'name': 'synthetic-dynamic-push'},
                            'push_kind' : 'multiply',
                            'cpush': cpush,
                            'learn_pushed' : False
                        })

                for cpush in [1,2,3,4,5]:
                    policies.append({
                            'name': 'cpush-mpts',
                            'context_path' : context_path,
                            'arm_knowledge' : {'name': 'synthetic-dynamic-push'},
                            'push_kind' : 'plus',
                            'learn_pushed' : True,
                            'cpush': cpush})

                    policies.append({
                            'name': 'cpush-mpts',
                            'context_path' : context_path,
                            'arm_knowledge' : {'name': 'synthetic-dynamic-push'},
                            'push_kind' : 'plus',
                            'learn_pushed' : False,
                            'cpush': cpush})
                    
                config = {
                    'policies': policies,
                    'reward_path': reward_path,
                    'seed': rnd.randint(10000),
                    'L': L,
                    'c': c,
                    'pc': pushes_perc,
                    'T': T,
                    'dist' : dist
                }

                with open('%s/synthetic_push_dist_%s_c_%.2f_pc_%.2f_arms_%d_T_%d_L_%d.yml' % (
                    global_config.EXPERIMENT_CONFIG_DIR, dist, c, pushes_perc, arms, T, L), 'w') as outfile:
                    yaml.dump(config, outfile, default_flow_style=False)                    

def clean_metrics_data(metrics_dir: str, start: str, end: str, normalize:bool =True):
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
        _write_cleaned_metrics_df_to_csv(current_path, metrics_dir,
                                 df_without_missing_timestamps)
    else:
        _write_cleaned_metrics_df_to_csv(current_path, metrics_dir,
                                 _normalize_df(df_without_missing_timestamps))


def _normalize_df(metrics_df: pd.DataFrame) -> pd.DataFrame:
    normalized_df = (metrics_df - metrics_df.min())\
        / (metrics_df.max() - metrics_df.min())
    normalized_df = normalized_df.fillna(1.0)

    return normalized_df


def _write_cleaned_metrics_df_to_csv(current_path: str, metrics_dir: str,
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
        '2019-11-20 02:30:00 CEST',
    )
    # clean_metrics_data(
    #     '%s/raw/concurrent_data/metrics/' % global_config.DATA_DIR,
    #     '2019-11-25 16:12:13 CEST',
    #     '2019-11-25 20:45:00 CEST'
    # )


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
    of the csv file for the respective reward function.

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

def _write_experiment_config_to_disk(config: dict, name: str):
    """Write the config to the data/interim/experimental_configs directory.

    Args:
      config (dict): Configuration of the experiment
      name (str): Includes this name into the filename of the yaml file, to
      make the experiment identifiable.
    """
    with open('%s/%s.yml' % (global_config.EXPERIMENT_CONFIG_DIR, name), 'w') as yml_file:
        yaml.dump(config, yml_file, default_flow_style=False)

def _generate_push_mpts():
    policies = [{'name': 'mpts', 'identifier' : 'baseline'}]

    policies.extend(get_cross_validated_policies(
        {'name': 'push-mpts', 'arm_knowledge' : {'name' : 'correct'}},
        {
            'push_unlikely_arms' : [1,3,5,10],
            # 'push_likely_arms': [1,3,5,10],
            # 'push_temporal_correlated_arms': [1,3,5,10]
        }))

    _write_configs_for_policies(policies, name='push_mpts')

def _generate_mpts():
    policies = [{'name': 'mpts', 'identifier' : 'baseline'}]

    policies.extend(
        get_cross_validated_policies(
            {'name': 'mpts'},
            {
                'graph_knowledge' : global_config.GRAPH_DOMAIN_KNOWLEDGES
            }
        )
    )
    
    # policies.extend(
    #     get_cross_validated_policies(
    #         {'name': 'mpts'},
    #         {
    #             'graph_knowledge' : global_config.GRAPH_DOMAIN_KNOWLEDGES,
    #             'sliding_window_size': global_config.SLIDING_WINDOW_SIZES,
    #             'sliding_window_type' : ['all', 'some']
    #         }
    #     )
    # )
    
    _write_configs_for_policies(policies, name='mpts')

def _generate_cb():
    policies = [{'name' : 'mpts', 'identifier' : 'baseline'}, {'name': 'random'}]

    policies.append(
            {
                'name': 'cb-streaming-model',
                'context_path': global_config.DATA_DIR
                + '/processed/context/%s_context_workload-extractor_w%d_s%d.csv',
                'context_identifier': 'workload',
                'max_iter' : 1000,
                'scaler_sample_size' : 5000,
                'batch_size' : 100,
                'base_algorithm_name' : 'linear_regression',
                'algorithm_name' : 'bootstrapped_ucb'
    })

    _write_configs_for_policies(policies, name='cb', binary_rewards_only=True)

def _generate_awcpush_mpts():
    policies = [{'name': 'mpts', 'identifier' : 'baseline'}]

    policies.extend(
        get_cross_validated_policies(
            {
                'name': 'awcpush-mpts',
                'context_path':
                global_config.DATA_DIR + '/processed/context/%s_context_host-traces_w%d_s%d.csv',
                'push_likely_arms': 0,
                'push_unlikely_arms': 10,
                'push_temporal_correlated_arms': 1.0,
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
    policies = [{'name' : 'mpts', 'identifier': 'baseline'}]

    policies.extend(
        get_cross_validated_policies(
            {
                'name': 'cpush-mpts',
                'context_path':
                global_config.DATA_DIR + '/processed/context/%s_context_workload-extractor_w%d_s%d.csv',
                'push_kind' : 'plus',

                'arm_knowledge' : {'name':'push'}
            },
            {
                'learn_pushed': [True, False],
                'cpush': [1, 3, 5],
                'q': [10,100,1000]
            }
        )
    )

    policies.extend(
        get_cross_validated_policies(
            {
                'name': 'cpush-mpts',
                'context_path':
                global_config.DATA_DIR + '/processed/context/%s_context_host-traces_w%d_s%d.csv',
                'push_kind' : 'multiply',

                'arm_knowledge' : {'name':'push'}
            },
            {
                'learn_pushed': [True, False],
                'cpush': [1.1, 1.05, 1.25, 1.5,1.75, 2,3],
                'q': [None, 10,100,1000]                
            }
        )
    )

    _write_configs_for_policies(policies, name='cpush_con')

def _generate_sim_cpush_mpts():
    policies = [{'name': 'mpts', 'identifier' : 'baseline'}]

    policies.extend(
        get_cross_validated_policies(
            {
                'name': 'cpush-mpts',
                'context_path':
                global_config.DATA_DIR + '/processed/context/%s_context_sim_w%d_s%d.csv',
                'push_likely_arms': 0,
                'push_temporal_correlated_arms': 0,
                'kind_knowledge': 'sim',
                'push_kind' : 'plus',
                'learn_pushed' : True,
                'arm_knowledge' : {'name': 'sim'}
            },
            {
                'threshold' : [1000, 100, 10, 50],
                'cpush': [None, 1,3,5]
            }
        )
    )

    _write_configs_for_policies(policies, name='sim_cpush_plus')    


def _generate_exclude_certain_arms_test():
    policies = [{'name': 'mpts', 'identifier' : 'baseline'}]

    policies.append(
        {
            'name' : 'push-mpts',
            'arm_knowledge' : {'name': 'correct'}
        }
    )
    _write_configs_for_policies(policies, name='cpush_posterior')

def _generate_experiment_configs():
    """Generates the yaml files that contain the configs of the experiments."""
    print('Generate experiment configs')

    # _generate_mpts()
    # _generate_sim_cpush_mpts()
    # _generate_cpush_mpts()
    _generate_cb()
    # _generate_synthetic_experiments_for_gk()
    # _generate_synthetic_experiments_for_static_push()
    # _generate_synthetic_experiments_for_push()
    # _generate_push_mpts()


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
        _write_experiment_config_to_disk(config_for_this_round | {
            'reward_path': reward_path}, '%s_r_top' % name)
    elif rk == 'continous':
        reward_path = get_reward_path('continous', s, wsi, ws)
        _write_experiment_config_to_disk(config_for_this_round | {
            'reward_path': reward_path}, '%s_r_continous' % name)
    elif rk == 'threshold':
        for th in global_config.THRESHOLDS:
            reward_path = get_reward_path(
                'threshold', s, wsi, ws, threshold=th)
            _write_experiment_config_to_disk(config_for_this_round | {
                'reward_path': reward_path}, '%s_r_threshold_%.1f' % (name, th))


def _write_configs_for_policies(policies, name='', binary_rewards_only=False):
    global seed
    rewards = global_config.BINARY_REWARD_KINDS if binary_rewards_only else global_config.REWARD_KINDS
    for params in itertools.product(
                global_config.Ls,
                global_config.WINDOW_SIZES,
                global_config.WINDOW_STEPS,
                global_config.SEQ,
                rewards
    ):
        _write_config_for_params(
            seed,
            params[0],
            params[1],
            params[2],
            params[3],
            params[4],
            policies,
            name
        )
        seed += 1

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
