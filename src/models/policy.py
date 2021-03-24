"""This class contains different policies on how to select pairs of metrics.
It contains both contextual and non-contextual policies.
"""

from random import seed, random, sample, randrange
from abc import ABC, abstractmethod
import numpy as np
from sklearn.cluster import KMeans

class AbstractBandit(ABC):
    """Provides functionality for a basic non-contextual bandit.

    Attributes:
      _K (int): Total number of available arms
      _L (int): Number of arms to pick each iteration
      _picked_arms (int[]): Indicies of lastest selected arms # maybe use array instead
      _iteration (int): Current iteration
      _regret (float[]): Regret for each of the earlier rounds
    """

    def __init__(self, L, K):
        self._L = L
        self._K = K
        self._picked_arms = None
        self._iteration = 0
        self._regret = []

    @abstractmethod
    def pick_arms(self):
        """Picks arms for the current round and saves them to self._picked_arms
        """
        return

    def get_picked_arms(self):
        """Returns self._picked_arms

        Returns:
          int[]: Indicies of picked arms in the most recent iteration
        """
        return self._picked_arms

    def learn(self, reward, max_reward):
        """Improve the policy by adjusting the picking strategy

        Args:
          reward (float[]): Contains the reward for each of the picked arms.
          reward[i] contains the reward for self._picked_arms[i].
          max_reward (float): The reward an optimal picking strategy would have
          obtained this round.
        """
        self._regret.append(max_reward - sum(reward))
        self._iteration += 1

    def get_regret(self):
        """Returns self._regret

        Returns:
          float[]: The regret for each round.
        """
        return self._regret

    @abstractmethod
    def get_name(self):
        """Returns the name of the policy.

        Returns:
          string: Name of the policy
        """
        return


class AbstractContextualBandit(AbstractBandit):
    """Provides functionality for a basic contextual bandit.
    """
    @abstractmethod
    def pick_arms(self, context):
        return


class RandomPolicy(AbstractBandit):
    """Policy that picks arms randomly.
    """
    def __init__(self, L, K, random_seed):
        """Constructs the random policy.

        Args:
          L (int): Number of arms to pick each round
          K (int): Number of total arms
          random_seed (int): Seed for pseudo random generator
        """
        super().__init__(L,K)
        seed(random_seed)
        
    def get_name(self):
        return 'random'

    def pick_arms(self):
        self._picked_arms = sample(range(self._K), self._L)


class EGreedy(AbstractBandit):
    """E-Greedy bandit algorithm that picks the greedily arms with highest
    expected reward with probability 1-e (exploitation) and random arms with
    probablity e (exploration).
    """

    def __init__(self, L, K, seed, epsilon):
        """Constrcuts the E-Greedy bandit algorithm.

        Args:
          L (int): Number of arms to pick each round
          K (int): Number of total arms
          seed (int): Seed for the pseudo random generator
          epsilon (float): Epsilon parameter of the algorithm. The Algorithm
          performs an exploitation step with probability 1-epsilon and an
          exploration step with probability epsilon.
        """
        super().__init__(L, K)
        self.epsilon = epsilon
        self.expected_values = [1] * self._K
        self.num_plays = [0] * self._K

    def get_name(self):
        if self.epsilon != 0:
            return str(self.epsilon) + '-greedy'
        return 'greedy'

    def pick_arms(self):
        if self.epsilon > random():
            self._picked_arms = sample(range(self._K), self._L)
        else:
            self._picked_arms = np.argsort(self.expected_values)[-self._L:]

    def learn(self, reward, max_reward):
        for posrew, iarm in enumerate(self._picked_arms):
            self.num_plays[iarm] += 1
            self.expected_values[iarm] = ((self.num_plays[iarm] - 1) *
                             self.expected_values[iarm] + reward[posrew]) / self.num_plays[iarm]

        self._iteration += 1
        self._regret.append(max_reward - sum(reward))


class DKEGreedy(EGreedy):
    """E-Greedy bandit algorithm using domain knowledge to set the initial
    expected reward.
    """

    def __init__(self, L, K, seed, epsilon, arms):
        """Constructs the Domain Knowledge Epsilon-Greedy Algorithm.

        Args:
          L (int): Number of arms to pick each round
          K (int): Number of total arms
          seed (int): Seed for the random generator
          epsilon (float): Epsilon parameter of the algorithm. The Algorithm
          performs an exploitation step with probability 1-epsilon and an
          exploration step with probability epsilon.
          arms (string[]): Name of the arms. Used to initialize the expected
          value.
        """
        super().__init__(L, K, seed, epsilon)
        self._init_ev(arms)

    def _init_ev(self, arms):
        """Intializes the expected value for the arms. An arm is a pair of
        metrics. If the metrics are on the same host or are a pair between a
        computing and the control host the expected value is set 1, otherwise
        to 0.
        """
        for i, arm in enumerate(arms):
            if 'load.cpucore' in arm:
                self.expected_values[i] = 0

            host_metrics = arm.split('-')
            host1, host2 = host_metrics[0][0:8], host_metrics[1][0:8]

            if host1 not in ('wally113', host2):
                self.expected_values[i] = 0.0

    def get_name(self):
        if self.epsilon != 0:
            return str(self.epsilon) + '-dkgreedy'
        return 'dkgreedy'


class MPTS(AbstractBandit):
    """Multi-Play Thompson-Sampling bandit algorithm. This baysian bandit
    algorithm maintains a beta distribution for each of the arms. Arms get
    picked if they are likely to yield high rewards.
    """

    def __init__(self, L, K, seed):
        """Constructs the random policy.

        Args:
          L (int): Number of arms to pick each round
          K (int): Number of total arms
          seed (int): Seed for pseudo random generator
        """
        super().__init__(L, K)
        self.rnd = np.random.RandomState(seed)
        self.alpha = [0] * self._K
        self.beta = [0] * self._K
        self.num_plays = [0] * self._K
        self.sum_plays = [0] * self._K

    def get_name(self):
        return 'mpts'

    def pick_arms(self):
        theta = [0] * self._K
        for i in range(self._K):
            theta[i] = self.rnd.beta(self.alpha[i] + 1, self.beta[i] + 1)

        self._picked_arms = np.argsort(theta)[-self._L:]

    def learn(self, reward, max_reward):
        for posrew, iarm in enumerate(self._picked_arms):
            reward_for_arm = reward[posrew]
            self.alpha[iarm] = self.alpha[iarm] + reward_for_arm
            self.beta[iarm] = self.beta[iarm] + (1-reward_for_arm)
            self.num_plays[iarm] += 1
            self.sum_plays[iarm] += reward_for_arm

        self._iteration += 1
        self._regret.append(max_reward - sum(reward))


class PushMPTS(AbstractBandit):
    """Variant of the MPTS where domain knowledge is used to set the initial
    parameters for beta distributions.
    """

    def __init__(self, L, K, seed, arms, push):
        super().__init__(L, K)
        self.rnd = np.random.RandomState(seed)
        self.alpha = self._compute_init_prior(arms, push)
        self.beta = self._compute_init_posterior(arms, push)
        self.beta = [0] * self._K
        self.num_plays = [0] * self._K
        self.sum_plays = [0] * self._K
        self.push = push
        self.context = []

    def _compute_init_prior(self, arms, push):
        init_prior = len(arms) * [0]
        for i, arm in enumerate(arms):
            host_metrics = arm.split('-')
            host1, host2 = host_metrics[0][0:8], host_metrics[1][0:8]

            if host1 == host2:
                init_prior[i] += push

        return init_prior

    def _compute_init_posterior(self, arms, push):
        init_posterior = len(arms) * [0]
        for i, arm in enumerate(arms):
            host_metrics = arm.split('-')
            host1, host2 = host_metrics[0][0:8], host_metrics[1][0:8]

            if host1 not in ('wally113', host2):
                init_posterior[i] = push

            if 'load.cpucore' in host_metrics:
                init_posterior[i] = push

        return init_posterior

    def get_name(self):
        return 'push-' + str(self.push) + 'mpts'

    def pick_arms(self):
        theta = [0] * self._K
        for i in range(self._K):
            theta[i] = self.rnd.beta(self.alpha[i] + 1, self.beta[i] + 1)

        self._picked_arms = np.argsort(theta)[-self._L:]

    # def pick_arms(self,context):
    #     self.context.append(context)
    #     theta = [0] * self._K
    #     for i in range(self._K):
    #         theta[i] = self.rnd.beta(self.alpha[i] + 1, self.beta[i] + 1)

    #     self._picked_arms = np.argsort(theta)[-self._L:]

    def learn(self, reward, max_reward):
        for posrew, iarm in enumerate(self._picked_arms):
            reward_for_arm = reward[posrew]
            self.alpha[iarm] = self.alpha[iarm] + reward_for_arm
            self.beta[iarm] = self.beta[iarm] + (1-reward_for_arm)
            self.num_plays[iarm] += 1
            self.sum_plays[iarm] += reward_for_arm

        self._iteration += 1
        self._regret.append(max_reward - sum(reward))


class ContextualBandit(AbstractContextualBandit):
    """Provides functionality for a contextual bandit algorithm. A contextual
    bandit algorithm receives a context each round that is used, to pick arms.
    The contextual bandit maintains a set of policies on how to pick arms.
    Further it maintains a mapper that maps the context to a policy, that is
    used to pick arms.

    Attributes:
      _L (int): Number of arms to select in each round
      _K (int): Number of total arms
      _policies (AbstractBandit[]): Non-contextual policies that pick arms
      _mapper (AbstractMapper): Mapper to map context to policy
      _context (float[][]): Contexts received in each iteration
      _selected_policy_index_over_time (int[]): Indicies of the policies used
      to pick arms each round.
    """

    def __init__(self, L, K, policies, mapper):
        """Intializes a ContextualBandit policy.

        Args:
          L (int): Arms to select each round
          K (int): Total number of arms
          policies (AbstractBandit[]): Non-contextual policies
          mapper ()
        """
        super().__init__(L, K)
        self._policies = policies
        self._selected_policy_index_over_time = []
        self._mapper = mapper
        self._context = []

    def pick_arms(self, context):
        """Picks arms based on the context

        Args:
          context (float[]): Vector of features
        """
        self._context.append(context)
        if len(self._context) > 1:
            policy_index = self._mapper.get_mapping(
                self._context[-2] - self._context[-1])
        else:
            policy_index = randrange(len(self._policies))
        self._selected_policy_index_over_time.append(policy_index)
        self._policies[policy_index].pick_arms()
        self._picked_arms = self._policies[policy_index].get_picked_arms()

    def learn(self, reward, max_reward):
        self._regret.append(max_reward - sum(reward))
        self._iteration += 1
        self._policies[self._selected_policy_index_over_time[-1]
                      ].learn(reward, max_reward)

    def get_name(self):
        return 'scb-' + self._mapper.get_name()

    def get_selected_policy_index(self):
        """Returns
        """
        return self._selected_policy_index_over_time


class AbstractMapper(ABC):
    """Base class for mappers that map context to policies.

    Methods:
    -------
    get_mapping(context)
        Returns the index of the policy for the current context.
    """
    @abstractmethod
    def get_mapping(self, context):
        """Returns the index of the policy that shall be used to pick arms
        based on the current context.

        Args:
          context (float[]): Vector describing the context

        Returns
          int: Index of the policy that will be used for picking arms
        """
        return

    @abstractmethod
    def get_name(self):
        """Returns name of the mapper

        Returns:
          string: Name of the mapper
        """

class MostFrequentMapper(AbstractMapper):
    """For each possible feature this mapper returns the most frequent occuring
    in the context. Therefore the contextual bandit needs as many policies as
    features.
    """

    def __init__(self, number_of_features):
        """Constructs a MostFrequentMapper

        Args:
          number_of_features (int): Number of features the context has
        """
        self.mapping = list(range(len(number_of_features)))

    def get_mapping(self, context):
        return self.mapping[np.argmax(list(context.values))]

    def get_size_of_mapping(self):
        """Returns the number of policies the mapper maps to.
        """
        return len(self.mapping)

    def get_name(self):
        return 'most-frequent-mapper'


class KMeansMapper(AbstractMapper):
    """Uses a clustering of a sample of contexts. A newly arriving context is
    clustered and mapped to the policy associated with the cluster.
    """

    def __init__(self, context_df, k, sample_size, rnd_state=0):
        """Constructs a KMeansMapper

        Args:
          context_df (DataFrame): DataFrame containing all the contexts
          k (int): Parameter K for the KMeans algorithm
          sample_size (flaot): Percentage of contexts in DataFrame that will be
          used to build the cluster.
          rnd_state (int): Seed for random generator
        """
        features_sample = self._get_feature_samples(context_df, sample_size)
        self.k = k
        self.kmeans = KMeans(
            n_clusters=k, random_state=rnd_state).fit(features_sample)

    def get_mapping(self, context):
        return self.kmeans.predict(context.values.reshape(1, -1))[0]

    def get_name(self):
        return str(self.k) + 'means-mapper'

    def _get_feature_samples(self, context_df, sample_size):
        """Extracts the samples from the context DataFrame by subtracting two
        consecutive contexts.

        Args:
          context_df (DataFrame): DataFrame containing all the contexts
          sample_size (float): Percentage of contexts that will be used from
          the DataFrame

        Returns:
          float[][]: Array of contexts that will be used for clustering.
        """
        samples = []
        sample_indicies = sample(range(1, len(context_df.index) - 1), sample_size)

        for current_index in sample_indicies:
            samples.append(context_df.loc[current_index + 1] - context_df.loc[current_index])

        return samples
