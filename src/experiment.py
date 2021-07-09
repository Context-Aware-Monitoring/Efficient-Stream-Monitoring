"""Groups multiple policies and runs them on the same reward csv file and L.
"""
import sys
import os
from os.path import dirname, abspath
import logging
import time
import argparse
from random import seed, randint
from datetime import datetime
import yaml
import numpy as np
import pandas as pd
from models.domain_knowledge import GraphArmKnowledge, RandomGraphKnowledge, WrongGraphArmknowledge, SyntheticGraphArmKnowledge, WrongSyntheticGraphArmKnowledge, ArmKnowledge, SyntheticPushArmKnowledge, PushArmKnowledge, SimiliarPushArmKnowledge, SyntheticStaticPushKnowledge, WrongSyntheticStaticPushKnowledge
from models.policy import RandomPolicy, MPTS, PushMPTS, CPushMpts, CBFullModel, CBStreamingModel, AWCPushMpts, CBMPTS

DATA_DIR = '%s/data' % dirname(dirname(abspath(__file__)))
SERIALIZATION_DIR = '%s/processed/experiment_results/' % DATA_DIR

logging.basicConfig(filename='experiments.log',
                    encoding='utf-8', level=logging.INFO)


def _get_now_as_string():
    now = datetime.now()

    return now.strftime("%d.%m.%Y %H:%M:%S")


def _read_context_from_config(config):
    """Checks if the current configuration of a policy contains a the path
    to a context csv file. If so it gets appended to self._context,
    otherwise an error gets thrown and the program gets terminated.
    """
    if 'context_path' not in config:
        return None
    return pd.read_csv(config['context_path'], index_col=0)


def _get_dict_of_policy_params_(config: dict):
    policy_param_keys = set(
        config) - set(['name', 'context_path', 'graph_knowledge'])

    return {key: config[key] for key in policy_param_keys}


def print_information_about_experiment(experiment):
    """Prints textual information about the experiment.

    Args:
      experiment (Experiment): Run experiment containing the results
    """
    no_policies = len(experiment.average_regret.keys())
    print('Experiment: Pick %d arms out of %d' %
          (experiment.L, experiment.K))
    print('A total of %d different policies were evaluated' % no_policies)
    i = 1

    for pol_name in experiment.average_cum_regret.keys():
        total_average_cum_regret = experiment.average_cum_regret[pol_name][-1]
        print('%d. %s, total regret: %f' %
              (i, pol_name, total_average_cum_regret))
        i += 1


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
    _seed (int): Seed for random generator
    _number_of_runs (int): Number of runs for each policy
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

    def __init__(
            self, config_path=None, additional_config={}, runs=3):
        """Constructs the Experiment instance from the passed yaml file.
        Further configurations can be direcly passed too.

        Args:
          config_path (string): Path to a yaml config file
          additional_config (dict): Additional configuration
        """
        self._experiment_name = None
        self._config_path = None
        if config_path is not None and additional_config == {}:
            self._config_path = config_path
            self._experiment_name = os.path.splitext(
                os.path.basename(config_path))[0]

        config = additional_config
        if config_path is not None:
            with open(config_path) as yml_file:
                config = config | yaml.safe_load(yml_file)

        self._L = config['L']
        self._number_of_runs = runs

        if config.get('reward_path') is None:
            sys.exit('Config file does not containg reward csv file')

        filepath_rewards = config['reward_path']
        self._reward_df = pd.read_csv(filepath_rewards, index_col=0)

        self._K = len(self._reward_df.columns)
        self._T = len(self._reward_df.index)

        self._seed = config.get('seed', 0)

        self._config = config

        self._average_regret = {}
        self._average_cum_regret = {}

    def _read_arm_knowledge_from_config(self, config, context_df: pd.DataFrame):
        if 'arm_knowledge' not in config or config['arm_knowledge'] is None:
            return None

        name = config['arm_knowledge'].get('name')
        if name is None:
            return None

        arms = self._reward_df.columns.values
        if name == 'correct':
            return ArmKnowledge(arms, 'wally113')
        elif name == 'push':
            one_active_host_sufficient_for_push = config['arm_knowledge'].get('one_active_host_sufficient_for_push', True)
            return PushArmKnowledge(arms, context_df.columns.values, one_active_host_sufficient_for_push, 'wally113')
        elif name == 'sim':
            threshold = config['arm_knowledge'].get('threshold', 1000)
            return SimiliarPushArmKnowledge(arms, threshold, context_df.columns.values, 'wally113')
        elif name == 'synthetic-dynamic-push':
            return SyntheticPushArmKnowledge(arms)
        elif name == 'synthetic-static-push':
            return SyntheticStaticPushKnowledge()
        elif name == 'synthetic-static-push-wrong':
            random_seed = self._seed
            self._seed += 1
            return WrongSyntheticStaticPushKnowledge(config['arm_knowledge']['kind'], config['arm_knowledge']['percentage_affected'], random_seed)

        return None
        
    def _read_graph_knowledge_from_config(self, config):
        if 'graph_knowledge' not in config or config['graph_knowledge'] is None:
            return None

        name = config['graph_knowledge'].get('name')

        if name is None:
            return None

        if name == 'correct':
            return GraphArmKnowledge(self._reward_df.columns.values, **_get_dict_of_policy_params_(config['graph_knowledge']))
        elif name in ['flip', 'remove', 'add', 'unify']:
            config['graph_knowledge']['random_seed'] = self._seed
            self._seed += 1
            return WrongGraphArmknowledge(self._reward_df.columns.values, name, **_get_dict_of_policy_params_(config['graph_knowledge']))
        elif name in('synthetic', 'wrong-synthetic'):
            weight = config['graph_knowledge'].get('weight', 1.0)
            groups = config['graph_knowledge']['groups']
            if name == 'synthetic':
                return SyntheticGraphArmKnowledge(self._reward_df.columns.values, groups, weight)
            else:
                percentage_affected = config['graph_knowledge']['percentage_affected']
                error_kind = config['graph_knowledge']['error_kind']
                random_seed = self._seed
                self._seed +=1
                
                return WrongSyntheticGraphArmKnowledge(self._reward_df.columns.values, groups, error_kind, percentage_affected, weight, random_seed)

        return None

    def _create_policy(self, config_for_policy):
        """Creates the policy based on the configuration and returns it.
        If the configuration is not valid returns None.

        Args:
          config (string): Path to a yaml config file
        """
        name = config_for_policy['name']
        context = _read_context_from_config(config_for_policy)
        graph_knowledge = self._read_graph_knowledge_from_config(
            config_for_policy)
        arm_knowledge = self._read_arm_knowledge_from_config(
            config_for_policy, context)
        config_for_policy = _get_dict_of_policy_params_(
            config_for_policy) | {'graph_knowledge': graph_knowledge, 'arm_knowledge' : arm_knowledge}

        pol = None

        if name == 'random':
            pol = RandomPolicy(self._L, self._reward_df,
                               self._seed, **config_for_policy)
        elif name == 'mpts':
            pol = MPTS(self._L, self._reward_df,
                       self._seed, **config_for_policy)
        elif name == 'push-mpts':
            pol = PushMPTS(self._L, self._reward_df,
                           self._seed, **config_for_policy)
        elif name == 'cpush-mpts':
            pol = CPushMpts(self._L, self._reward_df,
                            self._seed, context, **config_for_policy)
        elif name == 'awcpush-mpts':
            pol = AWCPushMpts(self._L, self._reward_df,
                              self._seed, context, **config_for_policy)
        elif name == 'cb-full-model':
            pol = CBFullModel(self._L, self._reward_df,
                              self._seed, context, **config_for_policy)
        elif name == 'cb-streaming-model':
            pol = CBStreamingModel(
                self._L, self._reward_df, self._seed, context, **config_for_policy)
        elif name == 'cbmpts':
            pol = CBMPTS(
                self._L, self._reward_df, self._seed, context, **config_for_policy)
        self._seed += 1

        return pol

    def serialize_results(self):
        """Write the results of the experiment into the config and stores the
        regret of the policies in csv files. Writes both to the
        SERIALIZATION_DIR.
        """
        yaml_file = None
        if self._experiment_name is not None:
            experiment_id = self._experiment_name
            yaml_file = '%s%s_results.yml' % (
                SERIALIZATION_DIR, self._experiment_name)
        else:
            experiment_id = str(time.time())
            yaml_file = '%sexperiment%s_config.yml' % (
                SERIALIZATION_DIR, experiment_id)

        with open(yaml_file, 'w') as outfile:
            yaml.dump(self._config, outfile, default_flow_style=False)

    def run(self):
        """Performs the experiment. Creates each policy and runs it 
        number_of_runs times. The policy picks L arms in each of the T
        iterations. Afterwards serializes the results.
        """
        if self._experiment_name is not None:
            logging.info(
                '%s: Started experiment %s on pid %d' %
                (_get_now_as_string(), self._experiment_name, os.getpid()))

        for pol_config in self._config['policies']:
            T = pol_config.get('T', self._T)
            regret_each_run = np.zeros(shape=(T, self._number_of_runs))
            cum_regret_each_run = np.zeros(
                shape=(T, self._number_of_runs))

            for current_run in range(self._number_of_runs):
                pol = self._create_policy(pol_config)

                if pol is None:
                    break

                pol.run()

                regret_each_run[:, current_run] = pol.regret
                cum_regret_each_run[:, current_run] = pol.cum_regret
                del pol

            pol_config['regret']= float(cum_regret_each_run.mean(axis=1)[-1])

        self.serialize_results()

        if self._config_path is not None:
            os.remove(self._config_path)

        if self._experiment_name is not None:
            logging.info(
                '%s: Finished experiment %s' %
                (_get_now_as_string(), self._experiment_name))

    def get_top_correlated_arms(self, t, names=False):
        """Returns either the names or index of the top L correlated arms for
        iteration t.

        Args:
          t (int): Iteration
          names (bool): If true, return names of the arms, otherwise indicies.
        """
        if names:
            return self._reward_df.columns.values[np.argsort(
                self._reward_df.loc
                [t])[-self._L:]]

        return np.argsort(self._reward_df.loc[t])[-self._L:]

    @property
    def L(self):
        """Number of arms to pick each iteration

        Returns:
          int
        """
        return self._L

    @property
    def K(self):
        """Number of total arms

        Returns:
          int
        """
        return self._K

    @property
    def T(self):
        """Number of iterations

        Returns:
          int
        """
        return self._T

    @property
    def average_regret(self):
        """Average regret over the runs for the policies.

        Returns:
          dict (string->float[])
        """
        return self._average_regret

    @property
    def average_cum_regret(self):
        """Average cummulated regret over the runs for the policies.

        Returns:
          dict (string->float[])
        """
        return self._average_cum_regret


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Runs experiments"
    )

    parser.add_argument('filepath', help='The path of the config yaml file')

    args = parser.parse_args()

    e = Experiment(args.filepath)
    e.run()
