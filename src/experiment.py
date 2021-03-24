import sys
import yaml
import pdb
import pandas as pd
from policy import RandomPolicy, MPTS, ContextualBandit, MostFrequentMapper, KMeansMapper, PushMPTS, AbstractContextualBandit, EGreedy, DKEGreedy
from matplotlib import pyplot as plt
from datetime import datetime
import re
import time
import numpy as np

LOWER_BOUND_SAMPLE_SIZE_KMEANS = 10
class Experiment:
    """Performs an experiment where multiple different policies are run and
    can be compared against each other. The configuration for the experiment is
    read from a yaml file.

    Attributes:
    ___________
    _reward_df (DataFrame): DataFrame containing the reward. This is read from
    a reward csv file provided in the config.yml.
    _K (int): Number of total arms
    _L (int): Number of arms to pick each iteration
    _T (int): Total number of iterations
    _policies (AbstractBandit[]): Different policies that are executed
    _context (float[][]): Contains the context for each of the policies. Context
    is a vector of floats.

    Methods:
    ________
    run():
        Runs the experiment
    get_policies():
        Returns _policies
    get_top_correlated_arms(t, names):
        Returns either the names or the indicies of the top L correlated arms
        for iteration t.
    """
    
    def __init__(self, config_path,additional_config={}):
        """Constructs the Experiment instance from the passed yaml file.
        Further configurations can be direcly passed too.

        Args:
          config_path (string): Path to a yaml config file
          additional_config (dict): Additional configuration
        """
        with open(config_path) as yml_file:
            config = yaml.safe_load(yml_file) | additional_config

        if config.get('existing_files') is None or config['existing_files'].get('reward_path') is None:
            sys.exit('Config file does not containg reward csv file')

        filepath_rewards = config['existing_files']['reward_path']
        self._reward_df = pd.read_csv(filepath_rewards)
        self._K = len(self._reward_df.columns)
        self._T = len(self._reward_df.index)
        self._L = config['L']
        
        self._policies = []
        self._context = []
        
        self._create_policies(config)

    def _create_policies(self, config):
        """Creates the policies based on the configuration and adds them to
        _policies.

        Args:
          config (string): Path to a yaml config file
        """
        pattern_for_kmeans = re.compile('scb-[0-9]+means')
        pattern_for_push_mpts = re.compile('[0-9]+[.]{0,1}[0-9]+-push-mpts')
        pattern_for_egreedy = re.compile('[0]+[.]{0,1}[0-9]+-greedy')
        pattern_for_edkgreedy = re.compile('[0]+[.]{0,1}[0-9]+-dkgreedy')
        for pol, config_for_policy in config['policies'].items():
            if pol == 'random':
                self._policies.append(RandomPolicy(self._L, self._K))
                self._context.append(None)
            elif pol == 'mpts':
                self._policies.append(MPTS(self._L, self._K, int(time.time())))
                self._context.append(None)
            elif pol=='greedy':
                self._policies.append(EGreedy(self._L, self._K, int(time.time()), 0))
                self._context.append(None)
            elif pol == 'dkgreedy':
                self._policies.append(DKEGreedy(self._L, self._K, int(time.time()), 0, self._reward_df.columns.values))
                self._context.append(None)
            elif pattern_for_egreedy.match(pol):
                e = float(pol.split('-greedy')[0])
                self._policies.append(EGreedy(self._L, self._K, int(time.time()), e))
                self._context.append(None)
            elif pattern_for_edkgreedy.match(pol):
                e = float(pol.split('-dkgreedy')[0])
                self._policies.append(DKEGreedy(self._L, self._K, int(time.time()), e, self._reward_df.columns.values))
                self._context.append(None)
            elif pattern_for_push_mpts.match(pol):
                push = float(pol.split('-push-mpts')[0])
                self._policies.append(PushMPTS(self._L, self._K, int(time.time()), self._reward_df.columns.values, push))
                self._context.append(None)
            else:
                if 'context_path' not in config_for_policy:
                    filepath_context = generate_context.generate_context_csv(
                        config_for_policy['context']['kind'],
                        config_for_policy['context']['paths'],
                        self.window_start,
                        self.window_end,
                        self.window_size,
                        self.window_step
                    )
                    config_for_policy['context_path'] = filepath_context
                self._context.append(pd.read_csv(config_for_policy['context_path']))
                if pol == 'scb-mpts':
                    mapper = MostFrequentMapper(self._context[-1].columns.values)
                    policies = [MPTS(self._L, self._K, int(time.time())) for _ in range(mapper.get_size_of_mapping())]
                    self._policies.append(ContextualBandit(self._L, self._K, policies, mapper))
                elif pattern_for_kmeans.match(pol):
                    k = int(pol.split('scb-')[1][:-5])
                    sample_size = int(config_for_policy['sample_size_percentage'] *  len(self._context[-1].index))
                    sample_size = max(sample_size, LOWER_BOUND_SAMPLE_SIZE_KMEANS)
                    mapper = KMeansMapper(self._context[-1], k, sample_size)
                    policies = [MPTS(self._L, self._K, int(time.time())) for _ in range(k)]
                    self._policies.append(ContextualBandit(self._L, self._K, policies, mapper))


        
    def run(self):
        """Performs the experiment. In each of T iterations the policies pick
        L arms and afterwards receive the reward for their choices and the
        maximaliy obtainable reward.
        """
        for i,pol in enumerate(self._policies):
            regret_over_time = [0] * self._T
            total_regret = 0

            for t in range(self._T):
                if isinstance(pol, AbstractContextualBandit):
                    pol.pick_arms(self._context[i].loc[t])
                else:
                    pol.pick_arms()

                picked_arms = pol.get_picked_arms()
                max_reward = sum(sorted(self._reward_df.loc[t])[-self._L:])

                reward_for_arms = self._reward_df.loc[t][self._reward_df.columns.values[picked_arms]]
                pol.learn(reward_for_arms, max_reward)


    def get_policies(self):
        """Getter for _policies
        """
        return self._policies

    def get_top_correlated_arms(self, t, names=False):
        """Returns either the names or index of the top L correlated arms for
        iteration t.

        Args:
          t (int): Iteration
          names (bool): If true, return names of the arms, otherwise indicies.
        """
        if names:
            return self._reward_df.columns.values[np.argsort(self._reward_df.loc[t])[-self._L:]]
        else:
            return np.argsort(self._reward_df.loc[t])[-self._L:]
    
if __name__ == '__main__':
    # random_regret = Experiment('./experiment_configs/random.yml').run()
    # mpts_regret = Experiment('./experiment_configs/mpts.yml').run()
    scb_regret = Experiment('./experiment_configs/scb-mpts.yml').run()

    # fig = plt.figure()
    # ax = plt.axes()

    # ax.plot(range(len(random_regret)),random_regret,label='random')
    # ax.plot(range(len(mpts_regret)), mpts_regret, label='mpts')
    # ax.plot(range(len(scb_regret)), scb_regret, label='scb')
