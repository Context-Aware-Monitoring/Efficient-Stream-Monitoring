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
from models.policy import RandomPolicy, EGreedy, MPTS, DKEGreedy, PushMPTS, CPushMpts, CDKEGreedy, EGreedyCB, InvertedPushMPTS, StaticNetworkMPTS, RandomNetworkMPTS, DynamicNetworkMPTS

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
    policy_param_keys = set(config) - set(['name', 'context_path'])

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

        if config.get('kind') is not None and config['kind'] == 'top':
            self._reward_df.loc[:] = self._reward_df.values >= np.sort(
                self._reward_df.values, axis=1)[:, -self._L].reshape(-1, 1)
        elif config.get('kind') is not None and config['kind'] == 'threshold':
            self._reward_df.loc[:] = self._reward_df.values >= config['threshold']

        self._policies = [[] for _ in range(runs)]

        self._seed = config.get('seed')
        if self._seed is not None:
            seed(self._seed)

        self._number_policies = len(config['policies'])
        self._create_policies(config)
        self._config = config

        self._average_regret = {}
        self._average_cum_regret = {}

    def _create_policies(self, config):
        """Creates the policies based on the configuration and adds them to
        _policies.

        Args:
          config (string): Path to a yaml config file
        """
        for config_for_policy in config['policies']:
            name = config_for_policy['name']
            context = _read_context_from_config(config_for_policy)
            config_for_policy = _get_dict_of_policy_params_(config_for_policy)
            for current_run in range(self._number_of_runs):
                if name == 'random':
                    self._policies[current_run].append(RandomPolicy(
                        self._L, self._reward_df, self._seed, **config_for_policy))
                elif name == 'mpts':
                    self._policies[current_run].append(
                        MPTS(self._L, self._reward_df, self._seed, **config_for_policy))
                elif name in ('egreedy', 'greedy'):
                    self._policies[current_run].append(
                        EGreedy(self._L, self._reward_df, self._seed, **config_for_policy))
                elif name == 'push-mpts':
                    self._policies[current_run].append(
                        PushMPTS(self._L, self._reward_df, self._seed, **config_for_policy))
                elif name == 'static-network-mpts':
                    self._policies[current_run].append(StaticNetworkMPTS(
                        self._L, self._reward_df, self._seed, **config_for_policy))
                elif name == 'dynamic-network-mpts':
                    self._policies[current_run].append(DynamicNetworkMPTS(
                        self._L, self._reward_df, self._seed, context, **config_for_policy))
                elif name == 'random-network-mpts':
                    self._policies[current_run].append(RandomNetworkMPTS(
                        self._L, self._reward_df, self._seed, **config_for_policy))
                elif name == 'inverted-push-mpts':
                    self._policies[current_run].append(InvertedPushMPTS(
                        self._L, self._reward_df, self._seed, **config_for_policy))
                elif name == 'dkgreedy':
                    self._policies[current_run].append(
                        DKEGreedy(self._L, self._reward_df, self._seed, **config_for_policy))
                elif name == 'cdkegreedy':
                    self._policies[current_run].append(CDKEGreedy(
                        self._L, self._reward_df, self._seed, context, **config_for_policy))
                elif name == 'cpush-mpts':
                    self._policies[current_run].append(
                        CPushMpts(self._L, self._reward_df, self._seed, context, **config_for_policy))
                elif name == 'cb-egreedy':
                    self._policies[current_run].append(
                        EGreedyCB(self._L, self._reward_df, self._seed, context, **config_for_policy))

                self._seed += 1

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

        cum_regret_csv_file = '%scum_regret_experiment_%s.csv' % (
            SERIALIZATION_DIR,
            experiment_id)
        average_regret_csv_file = '%saverage_regret_experiment_%s.csv' % (
            SERIALIZATION_DIR,
            experiment_id)

        cum_regret_df = pd.DataFrame(data=self._average_cum_regret)
        average_regret_df = pd.DataFrame(data=self._average_regret)

        cum_regret_df.to_csv(cum_regret_csv_file)
        average_regret_df.to_csv(average_regret_csv_file)
        with open(yaml_file, 'w') as outfile:
            self._config |= {
                'results':
                {
                    policy_name: float("{:.2f}".format(
                        average_cum_regret_for_policy[-1]))
                    for policy_name, average_cum_regret_for_policy in self._average_cum_regret.items()
                },
                'cum_regret_csv_file': cum_regret_csv_file,
                'average_regret_csv_file': average_regret_csv_file}
            yaml.dump(self._config, outfile, default_flow_style=False)

    def run(self):
        """Performs the experiment. In each of T iterations the policies pick
        L arms and afterwards receive the reward for their choices and the
        maximaliy obtainable reward.
        """
        if self._experiment_name is not None:
            logging.info(
                '%s: Started experiment %s on pid %d' %
                (_get_now_as_string(), self._experiment_name, os.getpid()))
        for current_run in range(self._number_of_runs):
            for i, pol in enumerate(self._policies[current_run]):
                pol.run()

        for i, pol in enumerate(self._policies[0]):
            total_regret = pol.regret
            total_cum_regret = pol.cum_regret

            if self._number_of_runs >= 2:
                for j in range(self._number_of_runs-1):
                    total_regret = [
                        x + y for x,
                        y in zip(
                            total_regret, self._policies[j + 1][i].regret)]
                    total_cum_regret = [
                        x+y for x, y in zip(total_cum_regret, self._policies[j+1][i].cum_regret)]

            self._average_regret[pol.name] = [
                x / self._number_of_runs for x in total_regret]
            self._average_cum_regret[pol.name] = [
                x / self._number_of_runs for x in total_cum_regret]

        ordered_policies = sorted(
            self._average_cum_regret.keys(),
            key=lambda pol_name: self._average_cum_regret[pol_name][-1])

        ordered_average_regret = {}
        ordered_average_cum_regret = {}

        for pol_name in ordered_policies:
            ordered_average_regret[pol_name] = self._average_regret[pol_name]
            ordered_average_cum_regret[pol_name] = self._average_cum_regret[pol_name]

        self._average_regret = ordered_average_regret
        self._average_cum_regret = ordered_average_cum_regret

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
