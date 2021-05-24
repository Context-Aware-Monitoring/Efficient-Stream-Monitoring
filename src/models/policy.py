"""This class contains different policies on how to select pairs of metrics.
It contains both contextual and non-contextual policies.
"""
from abc import ABC, abstractmethod
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from contextualbandits.online import EpsilonGreedy
import itertools
import pdb

def repeat_entry_L_times(X, L):
    return np.tile(X, L).reshape(-1, X.shape[1])


def extract_hosts_from_arm(arm):
    """Arm names have the following format:
    host1.metric1-host2.metric2

    Extracts and returns the name of the two hosts from the arm them. The names
    of the host might be the same.

    Args:
      arm (string): Name of the arm in format host1.metrics1-host2.metrics2

    Returns:
      (string, string): Names of the hosts
    """
    host_metrics = arm.split('-')
    host1, host2 = host_metrics[0][0:8], host_metrics[1][0:8]

    return host1, host2

def extract_metrics_from_arm(arm):
    """Arm names have to following format:
    host1.metric1-host2.metric2

    Extracts and returns the name of the two metrics from the arm.

    Args:
      arm (string): Name of the arm in format host1.metrics1-host2.metric2

    Returns:
      (string, string): Names of the hosts
    """
    host_metrics = arm.split('-')
    metrics1, metrics2 = host_metrics[0][9:], host_metrics[1][9:]

    return metrics1, metrics2

def _get_other_arms_replace_single_host(arm, excluded_host, compute_hosts=['wally117','wally122','wally123','wally124']):
    other_compute_hosts = list(filter(lambda host: host != excluded_host, compute_hosts))

    return [arm.replace(excluded_host, och) for och in other_compute_hosts]

def _get_other_arms_replace_both_hosts(arm, compute_hosts=['wally117','wally122','wally123','wally124']):
    metric1, metric2 = extract_metrics_from_arm(arm)

    arms = []
    for h1, h2 in itertools.product(compute_hosts, compute_hosts):
        arms.append('%s.%s-%s.%s' % (h1,metric1,h2,metric2))
        if metric1 != metric2:
            arms.append('%s.%s-%s.%s' % (h1,metric2,h2,metric1))


    arms.remove(arm)

    return arms

class AbstractBandit(ABC):
    """Provides functionality for a basic non-contextual bandit.

    Attributes:
      L (int): Number of arms picked each iteration
      K (int): Total number of arms
      T (int): Number of iterations
      iteration (int): Current iteration
      overall_regret (float): Total regret of the bandit
      regret (float[]): Regret for each round
      cum_regret (float[]): Cumulated regret for each round
      average_regret (float): Average regret per round
      name (string): Name of the bandit

    Methods:
      run: Runs the bandit. Picks each round L out of K arms and computes the
      regret.
    """

    def __init__(self, L, reward_df, identifier=None):
        """Constructs the bandit

        Args:
          L (int): Number of arms to pick
          reward_df (DataFrame): This data frame contains the reward for each
          arm (columns) and iteration (rows).
          identifier (string): Id for the bandit. If an id is provided the name
          property returns this id, otherwise the bandit computes a name by
          depending on its parameters.
        """
        self._L = L
        self._K = len(reward_df.columns)
        self._reward_df = reward_df
        self._arms = reward_df.columns.values
        self._T = len(reward_df.index)

        self._picked_arms_indicies = None
        self._iteration = 0
        self._regret = np.zeros(self._T)
        self._identifier = identifier

    @property
    def L(self):
        """Number of arms to pick each iteration

        Returns:
          int
        """
        return self._L

    @property
    def K(self):
        """Total number of arms

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
    def iteration(self):
        """Current iteration

        Returns:
          int
        """
        return self._iteration

    @property
    def overall_regret(self):
        """Total regret of the policy

        Returns:
          float
        """
        return np.sum(self._regret)

    @property
    def regret(self):
        """Regret for each round

        Returns:
          float[]
        """
        return self._regret

    @property
    def cum_regret(self):
        """Cumulated regret over the rounds

        Returns:
          float[]
        """
        return np.cumsum(self._regret)

    @property
    def average_regret(self):
        """Average regret over the rounds

        Returns:
          float
        """
        return np.mean(self._regret)

    def run(self):
        """Picks each iteration L of K arms. Receives the reward for the picked
        arms. Adjusts the picking strategy.
        """
        for _ in range(0, self._T):
            self._pick_arms()
            self._learn()
            self._regret[self._iteration] = np.sort(self._reward_df.values[self._iteration, :])[-self._L:].sum() \
                - (self._reward_df.values[self._iteration, self._picked_arms_indicies]).sum()
            self._iteration += 1

    def _set_information_about_hosts(self, control_host):
        self._hosts_for_arm = np.zeros(2 * self._K, dtype=object)

        for i, arm in enumerate(self._arms):
            self._hosts_for_arm[2 * i], self._hosts_for_arm[2 *
                                                            i + 1] = extract_hosts_from_arm(arm)

        self._hosts_for_arm = self._hosts_for_arm.reshape(-1, 2)

        self._arm_lays_on_same_host = self._hosts_for_arm[:,
                                                          0] == self._hosts_for_arm[:, 1]
        self._arm_lays_on_control_host = np.logical_or(
            self._hosts_for_arm[:, 0] == control_host, self._hosts_for_arm
            [:, 1] == control_host)

        self._arm_lays_on_different_compute_hosts = np.logical_and(
            np.logical_not(self._arm_lays_on_same_host),
            np.logical_not(self._arm_lays_on_control_host)
        )

        self._arm_lays_on_same_compute_host = np.logical_and(
            self._arm_lays_on_same_host,
            np.logical_not(self._arm_lays_on_control_host)
        )
        
        self._indicies_of_arms_that_will_not_be_explored = np.arange(
            self._K)[list(map(lambda arm: 'load.cpucore' in arm, self._arms))]

    @abstractmethod
    def _pick_arms(self):
        """Picks arms for the current round and saves them to self._picked_arms"""
        return

    @abstractmethod
    def _learn(self):
        """Improve the policy by adjusting the picking strategy"""
        return

    @property
    @abstractmethod
    def name(self):
        """Returns the name of the bandit

        Returns:
          string
        """
        return


class RandomPolicy(AbstractBandit):
    """Policy that picks arms randomly."""

    def __init__(self, L, reward_df, random_seed, identifier=None):
        """Constructs the random policy.

        Args:
          random_seed (int): Seed for the PRNG
        """
        super().__init__(L, reward_df, identifier)
        self._rnd = np.random.RandomState(random_seed)

    @property
    def name(self):
        if self._identifier is not None:
            return self._identifier

        return 'random'

    def _pick_arms(self):
        self._picked_arms_indicies = self._rnd.choice(
            self._K, self._L, replace=False)


class EGreedy(AbstractBandit):
    """E-Greedy bandit algorithm that picks the arms with highest expected
    reward greedily with probability 1-e (exploitation) and random arms with
    probablity e (exploration).

    Attributes:
      epsilon (float): Probability of exploration
    """

    def __init__(self, L, reward_df, random_seed, epsilon, identifier=None):
        """Constrcuts the E-Greedy bandit algorithm.

        Args:
          random_seed (int): Seed for the PRNG
          epsilon (float): Probability of exploration
        """
        super().__init__(L, reward_df, identifier)
        self._epsilon = epsilon
        self._expected_values = np.ones(self._K)
        self._num_plays = np.zeros(self._K)
        self._rnd = np.random.RandomState(random_seed)

    @property
    def epsilon(self):
        return self._epsilon

    @property
    def name(self):
        if self._identifier is not None:
            return self._identifier

        if self._epsilon != 0:
            return str(self._epsilon) + '-greedy'

        return 'greedy'

    def _pick_arms(self):
        """Picks arms. Performs an exploration step with probability epsilon
        and an exploitation step with probability 1-epsilon.
        """
        if self._epsilon > self._rnd.rand():
            self.explore_arms()
        else:
            self._picked_arms_indicies = np.argsort(
                self._expected_values)[-self._L:]

    def explore_arms(self):
        self._picked_arms_indicies = self._rnd.choice(
            self._K, self._L, replace=False)

    def _learn(self):
        """Learns from the reward for the picked arms. Updates the empirical
        expected value for the arms.
        """
        self._expected_values[self._picked_arms_indicies] = (
            self._expected_values[self._picked_arms_indicies]
            * self._num_plays[self._picked_arms_indicies]
            + self._reward_df.values[self._iteration, self._picked_arms_indicies]
        ) / (self._num_plays[self._picked_arms_indicies] + 1)
        self._num_plays[self._picked_arms_indicies] += 1


class DKEGreedy(EGreedy):
    """E-Greedy bandit algorithm using domain knowledge to set the initial
    expected reward and exclude arms from exploration. Initial reward of arms
    that lay on the same host or between the control host and a compute host
    will receive a higher initial expected value than the other arms.
    """

    def __init__(
            self, L, reward_df, random_seed, epsilon, init_ev_likely_arms,
            init_ev_unlikely_arms, control_host, identifier=None):
        """Constructs the Domain Knowledge Epsilon-Greedy Algorithm.

        Args:
          control_host (string): Name of the control host
          init_ev_likely_arms (float): During initialization of the expected
          values, arms on the same host and arms between control host and
          compute hosts receive this value as initial expected value.
          init_ev_unlikely_arms (float): The other arms will receive this value
          as intial expected value.
        """
        super().__init__(L, reward_df, random_seed, epsilon, identifier)
        super()._set_information_about_hosts(control_host)
        self._init_ev_likely_arms = init_ev_likely_arms
        self._init_ev_unlikely_arms = init_ev_unlikely_arms
        self._init_ev()
        self._indicies_of_arms_that_will_be_explored = np.delete(
            np.arange(self._K), self._indicies_of_arms_that_will_not_be_explored)

    def _init_ev(self):
        """Intializes the expected value for the arms. An arm is a pair of
        metrics. If the metrics are on the same host or are a pair between a
        computing and the control host the expected value is set to
        self._init_ev_likely_arms, otherwise to self._init_ev_unlikely_arms.
        The ev of arms that will not be explored is set to 0.
        """
        expected_values = np.where(
            np.logical_or(
                self._arm_lays_on_same_host, self._arm_lays_on_control_host),
            self._init_ev_likely_arms, self._init_ev_unlikely_arms)
        expected_values[self._indicies_of_arms_that_will_not_be_explored] = 0.0
        self._expected_values = expected_values

    def explore_arms(self):
        """Performs an exploration step but never exploration arms that we know
        are uninteresting (arms for the metric load.cpucore because this has a
        static value).
        """
        explore_arm_indicies = self._rnd.choice(
            self._indicies_of_arms_that_will_be_explored, self._L, replace=False)
        self._picked_arms_indicies = explore_arm_indicies

    @property
    def name(self):
        if self._identifier is not None:
            return self._identifier

        if self._epsilon != 0:
            return '%.1f/%.1f-dk-%.1f-greedy' % (self._init_ev_likely_arms,
                                                 self._init_ev_unlikely_arms,
                                                 self._epsilon)

        return '%.1f/%.1f-dkgreedy' % (
            self._init_ev_likely_arms, self._init_ev_unlikely_arms)


class CDKEGreedy(DKEGreedy):
    """E-Greedy bandit algorithm using domain knowledge to initially push more
    likely arms and using dynamic pushes based on the context to further
    improve picking of arms. The context contains the number of events per
    host.
    """

    def __init__(
            self, L, reward_df, random_seed, epsilon, init_ev_likely_arms,
            init_ev_unlikely_arms, control_host, context_df, push,
            max_number_pushes, push_kind='plus', identifier=None):
        """Constructs the Domain Knowledge Epsilon-Greedy Algorithm.

        Args:
          context_df (DataFrame): DataFrame containing the number of events per
          host (columns) and iterations (rows).
          push (float): Push that gets performed for active arms.
          max_number_pushes (int): Number of times an arm will be pushed.
          push_kind (string): One of 'plus' or 'multiply'
        """
        super().__init__(L, reward_df, random_seed, epsilon, init_ev_likely_arms,
                         init_ev_unlikely_arms, control_host, identifier=identifier)
        self._context_df = context_df
        self._no_pushed = np.zeros(self._K)
        self._push = push
        self._max_number_pushes = max_number_pushes
        self._push_kind = push_kind

    def pick_arms(self):
        """Pushes the arms with the context. Uses the underlying DKEgreedy
        algorithm to pick the arms.
        """
        if self._epsilon > self._rnd.rand():
            self.explore_arms()
        else:
            self._push_and_pick_arms()

    def _push_and_pick_arms(self):
        current_context = self._context_df.values[self._iteration, :]

        first_host_active = current_context[self._hosts_for_arm[:, 0]] > 0
        second_host_active = current_context[self._hosts_for_arm[:, 1]] > 0

        arm_gets_pushed = np.logical_and(
            np.logical_or(
                first_host_active.values, second_host_active.values),
            self._no_pushed < self._max_number_pushes)

        if self._push_kind == 'plus':
            pushed_expected_values = self._expected_values + \
                (self._push * arm_gets_pushed)
        else:
            factors = np.where(arm_gets_pushed, self._push, 1)
            pushed_expected_values *= factors

        self._picked_arms_indicies = np.argsort(
            pushed_expected_values)[-self._L:]

        self._no_pushed[self._picked_arms_indicies] += arm_gets_pushed[self._picked_arms_indicies]

    @property
    def name(self):
        if self._identifier is not None:
            return self._identifier

        return '%s-c%.1f/%d_%s' % (
            self._push_kind, self._push, self._max_number_pushes, super().name)


class MPTS(AbstractBandit):
    """Multi-Play Thompson-Sampling bandit algorithm. This baysian bandit
    algorithm maintains a beta distribution for each of the arms. Arms get
    picked if they are likely to yield high rewards.
    """

    def __init__(self, L, reward_df, random_seed, identifier=None):
        """Constructs the MPTS policy.

        Args:
          random_seed (int): Seed for PRNG
        """
        super().__init__(L, reward_df, identifier)
        self._rnd = np.random.RandomState(random_seed)
        self._alpha = np.zeros(self._K)
        self._beta = np.zeros(self._K)
        self._num_plays = np.zeros(self._K)
        self._sum_plays = np.zeros(self._K)

    @property
    def name(self):
        if self._identifier is not None:
            return self._identifier

        return 'mpts'

    def _pick_arms(self):
        """For each arm a random value gets drawn according to its beta
        distribution. The arms that have the highest L random values get
        picked. Some arms never get explored.
        """
        theta = self._rnd.beta(self._alpha + 1, self._beta + 1)
        self._picked_arms_indicies = np.argsort(theta)[-self._L:]

    def _learn(self):
        """Beta distribution gets updated based on the reward. If reward is
        good alpha gets incremented, if reward is bad beta gets incremented.
        """
        reward_this_round = self._reward_df.values[self._iteration,
                                                   self._picked_arms_indicies]
        self._alpha[self._picked_arms_indicies] += reward_this_round
        self._beta[self._picked_arms_indicies] += 1 - reward_this_round
        self._num_plays[self._picked_arms_indicies] += 1
        self._sum_plays[self._picked_arms_indicies] = + reward_this_round


class PushMPTS(MPTS):
    """Variant of the MPTS where domain knowledge is used to set the initial
    parameters for beta distributions.
    Arms that lie within the same host or lie between a computing and the
    control host get a push in the prior (making them more likely to be
    picked). Other arms get a push in the posterior (making them less likely to
    be picked). Additionally arms containing the 'load.cpucore' metrics will
    never be picked.
    """

    def __init__(self, L, reward_df, random_seed, push_likely_arms,
                 push_unlikely_arms, control_host, identifier=None):
        """Constructs the Push Mpts algorithm.

        Args:
          random_seed (int): Seed for the PRNG
          push_likely_arms (float): Likely arms get this value as push in the
          initial prior distribution.
          push_unlikely_arms (float): Unlikely arms get this value as push in
          the initial posterior distribution.
          control_host (string): Name of the control host
        """
        super().__init__(L, reward_df, random_seed, identifier)
        super()._set_information_about_hosts(control_host)

        self._push_likely_arms = push_likely_arms
        self._push_unlikely_arms = push_unlikely_arms

        self._alpha = self._compute_init_prior()
        self._beta = self._compute_init_posterior()

    def _compute_init_prior(self):
        """Computes the init prior distribution (alpha) for the arms.
        Arms that lie on the same host or between the control host and
        computing hosts get a puhs.

        Returns:
          float[]: Prior distribution
        """
        return np.repeat(float(self._push_likely_arms),
                         self._K) * np.logical_or(self._arm_lays_on_same_host,
                                                  self._arm_lays_on_control_host)

    def _compute_init_posterior(self):
        """Computes the posterior distribution (beta) for the arms.
        Arms that don't get a push in the prior get a push in the posterior.

        Returns:
          float[]: Posterior distribution
        """
        return np.repeat(float(self._push_unlikely_arms),
                         self._K) * (self._compute_init_prior() == 0)

    @property
    def name(self):
        if self._identifier is not None:
            return self._identifier

        return '%.1f-%.1f-push-mpts' % (
            self._push_likely_arms, self._push_unlikely_arms)

    def _pick_arms(self):
        """For each arm a random value gets drawn according to its beta
        distribution. The arms that have the highest L random values get
        picked.
        """
        theta = self._rnd.beta(self._alpha + 1, self._beta + 1)
        theta[self._indicies_of_arms_that_will_not_be_explored] = 0.0
        self._picked_arms_indicies = np.argsort(theta)[-self._L:]

class CPushMpts(PushMPTS):
    """MPTS bandit algorithm using domain knowledge to initially push more
    likely arms and using dynamic pushes based on the context to further
    improve picking of arms. The expected context csv file contains a boolean
    for each arms that says whether or not to push the arm.
    """

    def __init__(self, L, reward_df, random_seed, push_likely_arms,
                 push_unlikely_arms, control_host, context_df, cpush,
                 max_number_pushes, identifier=None):
        """Constructs the contextual push MPTS algorithm.

        Args:
          context_df (DataFrame): Context that contains the number of events
          per host (columns) and iterations (rows).
          cpush (float): Push for active arms
          max_number_pushes (int): Number of times an arm gets pushed
        """
        super().__init__(L, reward_df, random_seed, push_likely_arms,
                         push_unlikely_arms, control_host, identifier)
        self._context_df = context_df
        self._cpush = cpush
        self._max_number_pushes = max_number_pushes
        self._no_pushed = np.zeros(self._K)

    def _pick_arms(self):
        """Pushes the arms with the context. Uses the underlying PushMPTS
        algorithm to pick the arms.
        """
        current_context = self._context_df.loc[self._context_df.index
                                               [self._iteration]]

        first_host_active = current_context[self._hosts_for_arm[:, 0]] > 0
        second_host_active = current_context[self._hosts_for_arm[:, 1]] > 0

        arm_gets_pushed = np.logical_and(
            np.logical_or(
                first_host_active.values, second_host_active.values),
            self._no_pushed < self._max_number_pushes)

        alpha_pushed = self._alpha + arm_gets_pushed * self._cpush

        theta = self._rnd.beta(alpha_pushed + 1, self._beta + 1)
        theta[self._indicies_of_arms_that_will_not_be_explored] = 0.0
        self._picked_arms_indicies = np.argsort(theta)[-self._L:]

        self._no_pushed[self._picked_arms_indicies] += arm_gets_pushed[self._picked_arms_indicies]

    @property
    def name(self):
        """Returns the name of the bandit algorithm.

        Returns:
          string: The name
        """
        if self._identifier is not None:
            return self._identifier

        return 'c%1.f/%d_%s' % (self._cpush, self._max_number_pushes,
                                super().name)


class EGreedyCB(AbstractBandit):
    """Epsilon Greedy Contextual Bandit baseline algorithm from the
    contextualbandits package.
    """

    def __init__(
            self, L, reward_df, context_df, random_seed, epsilon,
            batch_size=20, identifier='', max_iter=100):
        super().__init__(L, reward_df)
        self._identifier = identifier
        self._context_df = context_df
        base_algorithm = LogisticRegression(
            solver='lbfgs', warm_start=True, max_iter=max_iter)
        self._rnd = np.random.RandomState(random_seed)
        self._epsilon_greedy = EpsilonGreedy(
            base_algorithm, nchoices=self._K, random_state=random_seed,
            explore_prob=epsilon)
        self._scaler = StandardScaler()
        self._batch_size = batch_size
        self._picked_arms = np.zeros(batch_size * L).reshape(-1, L)
        self._received_rewards = np.zeros(batch_size * L).reshape(-1, L)

    def _pick_arms(self):
        # Picks for the next batch size
        if self._iteration <= self._batch_size:  # model not fitted yet, pick random arms
            random_arm_indicies = self._rnd.choice(
                self._K, size=self._L, replace=False)
            self._picked_arms_indicies = random_arm_indicies
        else:
            self._picked_arms_indicies = (
                self._epsilon_greedy.topN(
                    self._context_df.values[self._iteration, :],
                    self._L)[0])

        self._picked_arms[self._iteration %
                          self._batch_size, :] = self._picked_arms_indicies
        self._received_rewards[self._iteration % self._batch_size,
                               :] = self._reward_df.values[self._iteration, self._picked_arms_indicies]

    def _learn(self):
        # (re)fit model
        if self._iteration > 0 and self._iteration % self._batch_size == 0:
            if self._iteration == self._batch_size:
                self._scaler.fit(self._context_df.values[:self._batch_size, :])
                self._epsilon_greedy.fit(
                    repeat_entry_L_times(
                        self._scaler.transform(
                            self._context_df.values
                            [: self._batch_size, :]),
                        self._L),
                    self._picked_arms.flatten(),
                    self._received_rewards.flatten())
            else:
                self._epsilon_greedy.fit(
                    repeat_entry_L_times(
                        self._scaler.transform(
                            self._context_df.values
                            [self._iteration - self._batch_size: self.
                             _iteration, :]),
                        self._L),
                    self._picked_arms.flatten(),
                    self._received_rewards.flatten(),
                    warm_start=True)

            self._picked_arms = np.zeros(
                self._batch_size * self._L).reshape(-1, self._L)
            self._received_rewards = np.zeros(
                self._batch_size * self._L).reshape(-1, self._L)

    @property
    def name(self):
        if self._identifier != '':
            return self._identifier

        if self._epsilon_greedy.explore_prob == 0.0:
            return 'cb-greedy-%d' % self._batch_size

        return 'cb-%f-greedy-%d' % (
            self._epsilon_greedy.explore_prob, self._batch_size)


class InvertedPushMPTS(PushMPTS):
    """Policy that simulates wrong domain knowledge by inverting the pushes.
    Likely arms get a push in the posterior, while unlikely arms get a push in
    the prior.
    """

    def __init__(self, L, reward_df, random_seed, push_likely_arms,
             push_unlikely_arms, control_host, identifier=None):
        """Constructs the Inverted Push Mpts algorithm.

        Args:
          random_seed (int): Seed for the PRNG
          push_likely_arms (float): Likely arms get this value as push in the
          initial prior distribution.
          push_unlikely_arms (float): Unlikely arms get this value as push in
          the initial posterior distribution.
          control_host (string): Name of the control host
        """
        super().__init__(L, reward_df, random_seed, push_likely_arms,
                         push_unlikely_arms, control_host, identifier)

        self._alpha = self._compute_init_posterior()
        self._beta = self._compute_init_prior()

    @property
    def name(self):
        if self._identifier is not None:
            return self._identifier

        return '%.1f-%.1f-inverted-push-mpts' % (
            self._push_likely_arms, self._push_unlikely_arms)

    
class NetworkMPTS(MPTS):
    """Policy where relationships between arms exist. Arms move in groups, and
    one arms might be related to other arms. Therefore when an arm receives a
    reward we can derive some knowledge for other arms as well. A graph exists
    where nodes are the arms and weighted edges (w in [0,1]) indicate the
    strength of the relationship. If an arm receives a reward r we will push
    the prior/ posterior of the related arms by w * r.
    """

    def __init__(self, L, reward_df, random_seed, weight=0.5, control_host='wally113', compute_hosts=['wally117', 'wally122', 'wally123', 'wally124'], identifier=None):
        """Constructs the NetworkMPTS policy.
        """
        super().__init__(L, reward_df, random_seed, identifier)
        super()._set_information_about_hosts(control_host)

        self._weight = weight
        self._control_host = control_host
        self._compute_hosts = compute_hosts
        self._nodes = reward_df.columns.values
        self._edges = self._get_edges()

    def _get_edges(self):
        edges = []
        for arm in self._nodes:
            host1,host2 = extract_hosts_from_arm(arm)
            if host1 == host2:
                if host1 == self._control_host:
                    edges.extend(np.zeros(self._nodes.shape[0], dtype=bool))
                    continue
                else:
                    relevant_arms = _get_other_arms_replace_single_host(arm, host1)
            else:
                if host1 == self._control_host:
                    relevant_arms = _get_other_arms_replace_single_host(arm, host2)
                elif host2 == self._control_host:
                    relevant_arms = _get_other_arms_replace_single_host(arm, host1)
                else:
                    relevant_arms = _get_other_arms_replace_both_hosts(arm)
            edges.extend(np.isin(self._nodes, relevant_arms))

        return np.array(edges).reshape(self._nodes.shape[0],self._nodes.shape[0])


    def _learn(self):
        """Update beta distribution of picked arms based on the reward. Updates
        beta distribution of related arms as well if they didn't get picked.
        Does this by increasing the prior/ posterior by the weighted reward.
        """
        super()._learn()
        number_of_arms = self._nodes.shape[0]
        
        neighbors_played_arms = self._edges[self._picked_arms_indicies]
        arm_number_of_updates = neighbors_played_arms.sum(axis=0)
        arm_gets_update = arm_number_of_updates > 0

        reshaped_rewards = np.repeat(self._reward_df.values[self._iteration,self._picked_arms_indicies], number_of_arms).reshape(-1, number_of_arms)

        update = (neighbors_played_arms * reshaped_rewards * self._weight).sum(axis=0)[arm_gets_update]

        self._alpha[arm_gets_update] += update
        self._beta[arm_gets_update] += arm_number_of_updates[arm_gets_update] * self._weight - update
        
    @property
    def name(self):
        return '%.1f-network-mpts' % self._weight

    
class RandomNetworkMPTS(NetworkMPTS):
    def __init__(self, L, reward_df, random_seed, weight=0.5, control_host='wally113', compute_hosts=['wally117', 'wally122', 'wally123', 'wally124'], probability_neighbors=[0.4,0.15,0.15,0.15,0.15], identifier=None):
        self._probability_neighbors = probability_neighbors
        super().__init__(L, reward_df, random_seed, weight, control_host, compute_hosts, identifier)

    
    def _get_edges(self):
        number_arms = self._nodes.shape[0]
        edges = []
        for i,_ in enumerate(self._nodes):
            number_neighbors = self._rnd.choice(len(self._probability_neighbors), 1, p=self._probability_neighbors)
            neighbors = np.zeros(number_arms, dtype=bool)
            neighbors[self._rnd.choice(number_arms, number_neighbors)] = True
            neighbors[i] = False
            edges.extend(neighbors)

        return np.array(edges).reshape(number_arms,number_arms)

    @property
    def name(self):
        if self._identifier is not None:
            return self._identifier

        return '%.1f-%s-random-network-mpts' % (self._weight, str(self._probability_neighbors))
