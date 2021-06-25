"""This class contains different policies on how to select pairs of metrics.
It contains both contextual and non-contextual policies.
"""
from abc import ABC, abstractmethod
import sys
import numpy as np
from sklearn.linear_model import SGDClassifier, LogisticRegression, Ridge, ARDRegression, RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from contextualbandits.linreg import LinearRegression
from contextualbandits.online import EpsilonGreedy, BootstrappedUCB
import itertools
import pandas as pd
import typing
from typing import List, Tuple
from .domain_knowledge import ArmKnowledge, PushArmKnowledge, GraphArmKnowledge, RandomGraphKnowledge, Knowledge, SimiliarPushArmKnowledge, SyntheticPushArmKnowledge

np.seterr(invalid='ignore')
np.seterr(divide='ignore')


def repeat_entry_L_times(X: np.ndarray, L: int) -> np.ndarray:
    return np.tile(X, L).reshape(-1, X.shape[1])


class AbstractBandit(ABC):

    L: int
    K: int
    T: int
    iteration: int
    overall_regret: float
    regret: List[float]
    cum_regret: List[float]
    average_regret: List[float]

    def __init__(self,
                 L: int,
                 reward_df: pd.DataFrame,
                 sliding_window_size: int = None,
                 sliding_window_type: str = None,
                 graph_knowledge: Knowledge = None,
                 identifier: typing.Optional[str] = None
                 ):
        self._L = L
        self._K = len(reward_df.columns)
        self._reward_df = reward_df
        self._arms = reward_df.columns.values
        self._T = len(reward_df.index)

        self._picked_arms_indicies = None
        self._iteration = 0
        self._regret = np.zeros(self._T)

        self._graph_knowledge = graph_knowledge

        self._sliding_window_size = sliding_window_size
        if sliding_window_size is not None:
            self._sliding_window_type = sliding_window_type
            self._sliding_window_index = 0

        self._identifier = identifier

    @property
    def L(self) -> int:
        """Number of arms to pick each iteration"""
        return self._L

    @property
    def K(self) -> int:
        """Total number of arms"""
        return self._K

    @property
    def T(self) -> int:
        """Number of iterations"""
        return self._T

    @property
    def iteration(self) -> int:
        """Current iteration"""
        return self._iteration

    @property
    def overall_regret(self) -> float:
        """Total regret of the policy"""
        return np.sum(self._regret)

    @property
    def regret(self) -> List[float]:
        """Regret for each round"""
        return self._regret

    @property
    def cum_regret(self) -> List[float]:
        """Cumulated regret over the rounds"""
        return np.cumsum(self._regret)

    @property
    def average_regret(self) -> List[float]:
        """Average regret over the rounds"""
        return np.mean(self._regret)

    def run(self):
        """Picks each iteration L of K arms. Receives the reward for the picked
        arms. Adjusts the picking strategy.
        """
        for _ in range(0, self._T):
            self.perform_iteration()

    def perform_iteration(self):
        self._picked_arms_indicies = self._pick_arms()
        self._learn()

        if self._graph_knowledge is not None:
            self._dynamically_update_neighborhood()
            self._propagate_reward_to_neighborhood()

        if self._sliding_window_size is not None:
            self._update_sliding_window()
            self._sliding_window_index += 1
            self._sliding_window_index %= self._sliding_window_size

        max_reward_this_round = np.sort(
            self._reward_df.values[self._iteration, :])[-self._L:].sum()
        received_reward_this_round = (
            self._reward_df.values[self._iteration, self._picked_arms_indicies]).sum()
        self._regret[self._iteration] = max_reward_this_round - \
            received_reward_this_round
        self._iteration += 1

    def _init_sliding_window(self):
        """Initializes the required data structures if a sliding window is
        used."""
        pass

    def _dynamically_update_neighborhood(self):
        pass

    def _propagate_reward_to_neighborhood(self):
        neighbors_played_arms = self._graph_knowledge.edges[self._picked_arms_indicies]
        arm_number_of_updates = neighbors_played_arms.sum(
            axis=0)
        arm_gets_update = arm_number_of_updates > 0

        if self._graph_knowledge.only_push_arms_that_were_not_picked:
            arm_gets_update[self._picked_arms_indicies] = False

        reshaped_rewards = np.repeat(
            self._reward_df.values[self._iteration, self._picked_arms_indicies], self._K).reshape(-1, self._K)

        update = (neighbors_played_arms *
                  reshaped_rewards).sum(axis=0)[arm_gets_update]

        self._update_parameters_for_neighbored_arms(
            update, arm_gets_update, arm_number_of_updates[arm_gets_update])

    @abstractmethod
    def _update_parameters_for_neighbored_arms(self, update, arm_gets_update, number_updates):
        pass

    @abstractmethod
    def _pick_arms(self):
        """Picks arms for the current round and saves them to self._picked_arms"""
        return

    @abstractmethod
    def _learn(self):
        """Improve the policy by adjusting the picking strategy"""
        return

    def _update_sliding_window(self):
        """Updates the data structures of the sliding window"""
        pass


class RandomPolicy(AbstractBandit):
    """Policy that picks arms randomly."""

    def __init__(self,
                 L: int,
                 reward_df: pd.DataFrame,
                 random_seed: int,
                 sliding_window_size: int = None,                 
                 graph_knowledge: Knowledge = None,
                 identifier: typing.Optional[str] = None
                 ):
        super().__init__(L, reward_df, sliding_window_size=sliding_window_size, graph_knowledge=graph_knowledge, identifier=identifier)
        self._rnd = np.random.RandomState(random_seed)

    def _pick_arms(self):
        return self._rnd.choice(self._K, self._L, replace=False)

    def _learn(self):
        pass

    def _update_parameters_for_neighbored_arms(self, update, arm_gets_update, number_updates):
        pass

class MPTS(AbstractBandit):
    """Multi-Play Thompson-Sampling bandit algorithm. This baysian bandit
    algorithm maintains a beta distribution for each of the arms. Arms get
    picked if they are likely to yield high rewards.
    """

    def __init__(self,
                 L: int,
                 reward_df: pd.DataFrame,
                 random_seed: int,
                 sliding_window_size: int = None,
                 sliding_window_type: str = None,
                 graph_knowledge: Knowledge = None,
                 identifier: typing.Optional[str] = None
                 ):
        """Constructs the MPTS policy.

        Args:
          random_seed (int): Seed for PRNG
        """
        super().__init__(
            L,
            reward_df,
            sliding_window_size=sliding_window_size,
            sliding_window_type=sliding_window_type,
            graph_knowledge=graph_knowledge,
            identifier=identifier
        )

        self._rnd = np.random.RandomState(random_seed)
        self._alpha = np.zeros(self._K)
        self._beta = np.zeros(self._K)

        MPTS._init_sliding_window(self)

    def _init_sliding_window(self):
        if self._sliding_window_size is None:
            return

        if self._sliding_window_type == 'all' or self._sliding_window_type is None:
            self._sliding_window_relevant = np.ones(self._K, dtype=bool)
        else:
            self._sliding_window_relevant = ArmKnowledge(self._arms).arm_relevant_for_sliding_window
        
        self._sliding_alpha = np.zeros((self._sliding_window_size, self._K))
        self._sliding_beta = np.zeros((self._sliding_window_size, self._K))

    def _update_parameters_for_neighbored_arms(self, update, arm_gets_update, number_updates):
        beta_update = number_updates - update
        self._alpha[arm_gets_update] += update
        self._beta[arm_gets_update] += beta_update

        if self._graph_knowledge is not None:
            self._alpha_update_through_neighbors = np.zeros(self._K)
            self._beta_update_through_neighbors = np.zeros(self._K)
            self._alpha_update_through_neighbors[arm_gets_update] = update
            self._beta_update_through_neighbors[arm_gets_update] = beta_update

    def _pick_arms(self):
        """For each arm a random value gets drawn according to its beta
        distribution. The arms that have the highest L random values get
        picked. Some arms never get explored.
        """
        theta = self._rnd.beta(np.maximum(
            1.0, self._alpha + 1), np.maximum(1.0, self._beta + 1))

        return np.argsort(theta)[-self._L:]

    def _learn(self):
        """Beta distribution gets updated based on the reward. If reward is
        good alpha gets incremented, if reward is bad beta gets incremented.
        """
        reward_this_round = self._reward_df.values[self._iteration,
                                                   self._picked_arms_indicies]
        self._alpha[self._picked_arms_indicies] += reward_this_round
        self._beta[self._picked_arms_indicies] += 1 - reward_this_round

    def _update_sliding_window(self):
        reward_this_round = self._reward_df.values[self._iteration,
                                                   self._picked_arms_indicies]

        self._alpha -= np.where(self._sliding_window_relevant, self._sliding_alpha[self._sliding_window_index, :], 0)
        self._beta -= np.where(self._sliding_window_relevant, self._sliding_beta[self._sliding_window_index, :], 0)

        self._sliding_alpha[self._sliding_window_index, :] = np.zeros(self._K)
        self._sliding_beta[self._sliding_window_index, :] = np.zeros(self._K)
        self._sliding_alpha[self._sliding_window_index,
                            self._picked_arms_indicies] = reward_this_round
        self._sliding_beta[self._sliding_window_index,
                           self._picked_arms_indicies] = 1 - reward_this_round

        if self._graph_knowledge is not None:
            self._sliding_alpha[self._sliding_window_index] += self._alpha_update_through_neighbors
            self._sliding_beta[self._sliding_window_index] += self._beta_update_through_neighbors

    def reset_policy(self):
        self._alpha = np.zeros(self._K)
        self._beta = np.zeros(self._K)


class PushMPTS(MPTS):
    """Variant of the MPTS where domain knowledge is used to set the initial
    parameters for beta distributions.
    Arms that lie within the same host or lie between a computing and the
    control host get a push in the prior (making them more likely to be
    picked). Other arms get a push in the posterior (making them less likely to
    be picked). Additionally arms containing the 'load.cpucore' metrics will
    never be picked.
    """

    def __init__(self,
                 L: int,
                 reward_df: pd.DataFrame,
                 random_seed: int,
                 push_likely_arms: float = 0.0,
                 push_unlikely_arms: float = 0.0,
                 push_temporal_correlated_arms: float = 0.0,
                 control_host: str = 'wally113',
                 sliding_window_size: int = None,
                 graph_knowledge: Knowledge = None,
                 identifier: typing.Optional[str] = None
                 ):
        """Constructs the Push Mpts algorithm.

        Args:
          random_seed (int): Seed for the PRNG
          push_likely_arms (float): Likely arms get this value as push in the
          initial prior distribution.
          push_unlikely_arms (float): Unlikely arms get this value as push in
          the initial posterior distribution.
          control_host (string): Name of the control host
        """
        super().__init__(L, reward_df, random_seed,
                         sliding_window_size=sliding_window_size,
                         graph_knowledge=graph_knowledge, identifier=identifier)

        self._push_temporal_correlated_arms = 0.0
        self._push_likely_arms = 0.0
        self._push_unlikely_arms = 0.0
        
        if push_likely_arms > 0.0 or push_unlikely_arms > 0.0 or push_temporal_correlated_arms > 0.0:
        
            self._arm_knowledge = ArmKnowledge(self._arms, control_host)

            self._push_temporal_correlated_arms = push_temporal_correlated_arms
            self._push_likely_arms = push_likely_arms
            self._push_unlikely_arms = push_unlikely_arms

            self._alpha = self._compute_init_prior()
            self._beta = self._compute_init_posterior()

            self._alpha[self._arm_knowledge.arm_has_temporal_correlation] = push_temporal_correlated_arms

    def _compute_init_prior(self) -> np.ndarray:
        """Computes the init prior distribution (alpha) for the arms.
        Arms that lie on the same host or between the control host and
        computing hosts get a puhs.

        Returns:
          float[]: Prior distribution
        """
        return np.repeat(float(self._push_likely_arms), self._K)\
            * np.logical_or(
                self._arm_knowledge.arm_lays_on_same_host,
                self._arm_knowledge.arm_lays_on_control_host
        )

    def _compute_init_posterior(self) -> np.ndarray:
        """Computes the posterior distribution (beta) for the arms.
        Arms that don't get a push in the prior get a push in the posterior.

        Returns:
          float[]: Posterior distribution
        """
        return np.repeat(float(self._push_unlikely_arms), self._K)\
            * (self._compute_init_prior() == 0)

    def _pick_arms(self):
        """For each arm a random value gets drawn according to its beta
        distribution. The arms that have the highest L random values get
        picked.
        """
        theta = self._rnd.beta(np.maximum(
            1.0, self._alpha + 1), np.maximum(1.0, self._beta + 1))
        # theta[self._arm_knowledge.indicies_of_arms_that_will_not_be_explored] = 0.0

        return np.argsort(theta)[-self._L:]

    def reset_policy(self):
        self._alpha = self._compute_init_prior()
        self._beta = self._compute_init_posterior()


class CPushMpts(PushMPTS):
    """MPTS bandit algorithm using domain knowledge to initially push more
    likely arms and using dynamic pushes based on the context to further
    improve picking of arms. The expected context csv file contains a boolean
    for each arms that says whether or not to push the arm.
    """

    def __init__(self,
                 L: int,
                 reward_df: pd.DataFrame,
                 random_seed: int,
                 context_df: pd.DataFrame,
                 push_likely_arms: float = 0.0,
                 push_unlikely_arms: float = 0.0,
                 push_temporal_correlated_arms: float = 0.0,
                 control_host: str = 'wally113',
                 cpush: float = 1.0,
                 one_active_host_sufficient_for_push: bool = True,
                 sliding_window_size: int = None,
                 graph_knowledge: Knowledge = None,
                 kind_knowledge = 'push',
                 identifier: typing.Optional[str] = None,
                 **kwargs
                 ):
        """Constructs the contextual push MPTS algorithm.

        Args:
          context_df (DataFrame): Context that contains the number of events
          per host (columns) and iterations (rows).
          cpush (float): Push for active arms
          max_number_pushes (int): Number of times an arm gets pushed
        """

        super().__init__(L, reward_df, random_seed, push_likely_arms,
                         push_unlikely_arms, push_temporal_correlated_arms,
                         control_host, sliding_window_size=sliding_window_size,
                         graph_knowledge=graph_knowledge, identifier=identifier)
        assert reward_df.values.shape[0] == context_df.values.shape[0]

        self._alpha = np.ones(self._K)
        self._beta = np.ones(self._K)
        self._context_df = context_df
        self._kind_knowledge = kind_knowledge
        if kind_knowledge == 'push':
            self._arm_knowledge = PushArmKnowledge(
                self._arms, one_active_host_sufficient_for_push, control_host)
        elif kind_knowledge == 'sim':
            self._arm_knowledge = SimiliarPushArmKnowledge(self._arms, kwargs['threshold'], self._context_df.columns.values, control_host)
        elif kind_knowledge == 'syn':
            self._arm_knowledge = SyntheticPushArmKnowledge(self._arms)
            
        self._one_active_host_sufficient_for_push = one_active_host_sufficient_for_push
        self._cpush = cpush
        self._push_received_this_iteration = np.zeros(self._K, dtype=bool)
        
        CPushMpts._init_sliding_window(self)

    def _init_sliding_window(self):
        if self._sliding_window_size is not None:
            self._sliding_push_received = np.zeros(
                shape=(self._sliding_window_size, self._K), dtype=bool)

    def _pick_arms(self):
        """Pushes the arms with the context. Uses the underlying PushMPTS
        algorithm to pick the arms.
        """
        current_context = self._context_df.values[self._iteration, :]

        if isinstance(self._arm_knowledge, SimiliarPushArmKnowledge):
            self._arm_knowledge._compute_arms_eligible_for_push(current_context)
        elif isinstance(self._arm_knowledge, PushArmKnowledge):
            active_hosts = self._context_df.columns.values[current_context > 0]
            self._arm_knowledge.update_active_hosts(active_hosts)
        elif isinstance(self._arm_knowledge, SyntheticPushArmKnowledge):
            self._arm_knowledge.compute_arms_eligible_for_push(current_context)

        arm_gets_pushed = self._arm_knowledge.arms_eligible_for_push

        self._push_received_this_iteration = np.zeros(self._K, dtype=bool)        
        self._push_received_this_iteration[arm_gets_pushed] = True

        alpha_pushed = np.where(arm_gets_pushed, self._alpha * self._cpush, self._alpha)
        theta = self._rnd.beta(np.maximum(
            1.0, alpha_pushed + 1), np.maximum(1.0, self._beta + 1))

        return np.argsort(theta)[-self._L:]

    def _dynamically_update_neighborhood(self):
        if self._kind_knowledge == 'push':
            current_context = self._context_df.values[self._iteration, :]
            active_hosts = self._context_df.columns.values[current_context > 0]

            self._graph_knowledge.update_active_hosts(active_hosts)
            self._edges = self._graph_knowledge.edges

    def _learn(self):
        reward_this_round = self._reward_df.values[self._iteration, :]
        picked_arm_receives_update = self._push_received_this_iteration[self._picked_arms_indicies] == False
        
        self._alpha[self._picked_arms_indicies] += np.where(
            picked_arm_receives_update,
            reward_this_round[self._picked_arms_indicies],
            0
        )
        self._beta[self._picked_arms_indicies] += np.where(
            picked_arm_receives_update,
            1 - reward_this_round[self._picked_arms_indicies],
            0
        )

class AWCPushMpts(CPushMpts):
    def __init__(self,
                 L: int,
                 reward_df: pd.DataFrame,
                 random_seed: int,
                 context_df: pd.DataFrame,
                 push_likely_arms: float = 1.0,
                 push_unlikely_arms: float = 1.0,
                 push_temporal_correlated_arms: float = 1.0,
                 control_host: str = 'wally113',
                 cpush: float = 1.0,
                 one_active_host_sufficient_for_push: bool = True,
                 mean_diviation=200,
                 sliding_window_size: int = None,
                 graph_knowledge: Knowledge = None,
                 identifier: typing.Optional[str] = None
                 ):
        assert sliding_window_size is None

        super().__init__(L, reward_df, random_seed, context_df, push_likely_arms, push_unlikely_arms, push_temporal_correlated_arms,
              control_host, cpush, q, one_active_host_sufficient_for_push, sliding_window_size, graph_knowledge, identifier)

        self._mean_diviation = mean_diviation
        self._context_mean = None
        self._size_aw = 0

    def reset_policy(self):
        super().reset_policy()
        self._context_mean = None
        self._size_aw = 0

    def pick_arms(self):
        current_context = self._context_df.values[self._iteration, :]
        if self._context_mean is None:
            self._context_mean = np.copy(current_context)
        else:
            diff = np.lingalg.norm(self._context_mean - current_context)
            if diff > self._mean_diviation:
                self.reset_policy()
                self._context_mean = np.copy(current_context)
            else:
                self._context_mean = (
                    (self._context_mean) * self._size_aw + current_context) / (self._size_aw + 1)

        self._size_aw += 1
        super().pick_arms()

class CBAbstractBandit(AbstractBandit):

    def __init__(
            self,
            L: int,
            reward_df: pd.DataFrame,
            random_seed: int,
            context_df: pd.DataFrame,
            base_algorithm_name: str,
            algorithm_name: str,
            batch_size: int,
            scaler_sample_size: int,
            context_identifier: str = '',
            identifier: typing.Optional[str] = None,
    ):
        super().__init__(L, reward_df, identifier=identifier)

        self._context_identifier = context_identifier

        if context_df is not None:
            self._context_df = context_df
        else:
            self._context_df = pd.DataFrame(data={0: np.zeros(self._T)})

        self._base_algorithm_name = base_algorithm_name
        self._algorithm_name = algorithm_name

        self._rnd = np.random.RandomState(random_seed)

        self._batch_size = batch_size

        self._scaler = StandardScaler()
        self._scaler.fit(
            self._context_df.values[np.random.choice(self._T, self._K, replace=False), :])

    def _create_algorithm(self, base_algorithm, batch_train: bool, **kwargs):
        if self._algorithm_name == 'egreedy':
            self._algorithm = EpsilonGreedy(
                base_algorithm,
                nchoices=self._K,
                random_state=self._rnd.randint(1000),
                explore_prob=kwargs.get('epsilon', 0.1),
                batch_train=batch_train
            )
        elif self._algorithm_name == 'bootstrapped_ucb':
            self._algorithm = BootstrappedUCB(
                base_algorithm,
                nchoices=self._K,
                random_state=self._rnd.randint(1000),
                batch_train=batch_train
            )
        else:
            sys.exit("no such algorithm: %s" % algorithm)

    def _update_parameters_for_neighbored_arms(self):
        pass


class CBFullModel(CBAbstractBandit):
    """Epsilon Greedy Contextual Bandit baseline algorithm from the
    contextualbandits package.
    """

    def __init__(
            self,
            L: int,
            reward_df: pd.DataFrame,
            random_seed: int,
            context_df: pd.DataFrame,
            base_algorithm_name: str = 'logistic_regression',
            algorithm_name: str = 'egreedy',
            batch_size: int = 20,
            scaler_sample_size: int = 300,
            context_identifier: str = '',
            identifier: typing.Optional[str] = None,
            **kwargs
    ):
        super().__init__(L, reward_df, random_seed, context_df, base_algorithm_name,
                         algorithm_name, batch_size, scaler_sample_size, context_identifier, identifier)

        self._picked_arms = np.zeros((self._T, self._L))
        self._received_rewards = np.zeros((self._T, self._L))

        if self._base_algorithm_name == 'logistic_regression':
            base_algorithm = LogisticRegression(solver=kwargs.get(
                'solver', 'lbfgs'), warm_start=True, max_iter=kwargs.get('max_iter', 100))
        elif self._base_algorithm_name == 'ridge':
            base_algorithm = Ridge()
        elif self._base_algorithm_name == 'ard_regression':
            base_algorithm = ARDRegression()
        elif self._base_algorithm_name == 'lin_svc':
            base_algorithm = LinearSVC()
        elif self._base_algorithm_name == 'ridge_classifier':
            base_algorithm = RidgeClassifier()
        else:
            sys.exit("no such a base algorithm: %s " %
                     self._base_algorithm_name)

        self._create_algorithm(base_algorithm, False, **kwargs)

    def _pick_arms(self):
        # Picks for the next batch size
        if self._iteration <= self._batch_size:  # model not fitted yet, pick random arms
            random_arm_indicies = self._rnd.choice(
                self._K, size=self._L, replace=False)
            picked_arms_indicies = random_arm_indicies
        else:
            picked_arms_indicies = (self._algorithm.topN(
                self._context_df.values[self._iteration, :], self._L)[0])

        self._picked_arms[self._iteration] = picked_arms_indicies
        self._received_rewards[self._iteration,
                               :] = self._reward_df.values[self._iteration, picked_arms_indicies]

        return picked_arms_indicies

    def _learn(self):
        # (re)fit model
        if self._iteration > 0 and self._iteration % self._batch_size == 0:
            if self._iteration == self._batch_size:
                self._algorithm.fit(
                    repeat_entry_L_times(
                        self._scaler.transform(
                            self._context_df.values[: self._batch_size, :]),
                        self._L
                    ),
                    self._picked_arms[: self._batch_size, :].flatten(),
                    self._received_rewards[: self._batch_size, :].flatten()
                )
            else:
                self._algorithm.fit(
                    repeat_entry_L_times(
                        self._scaler.transform(
                            self._context_df.values[:self._iteration, :]),
                        self._L
                    ),
                    self._picked_arms[: self._iteration, :].flatten(),
                    self._received_rewards[: self._iteration, :].flatten(),
                    warm_start=True
                )

class CBStreamingModel(CBAbstractBandit):
    """Contextual bandit using batches to refit models periodically."""

    def __init__(
            self,
            L: int,
            reward_df: pd.DataFrame,
            random_seed: int,
            context_df: pd.DataFrame,
            base_algorithm_name: str = 'linear_regression',
            algorithm_name: str = 'egreedy',
            batch_size: int = 20,
            scaler_sample_size: int = 1000,
            context_identifier: str = '',
            identifier: typing.Optional[str] = None,
            **kwargs
    ):
        super().__init__(L, reward_df, random_seed, context_df, base_algorithm_name,
                         algorithm_name, batch_size, scaler_sample_size, context_identifier,
                         identifier)

        self._picked_arms = np.zeros((batch_size, L))
        self._received_rewards = np.zeros((batch_size, L))

        if self._base_algorithm_name == 'linear_regression':
            base_algorithm = LinearRegression(
                lambda_=10., fit_intercept=True, method='sm')
        elif self._base_algorithm_name == 'sgd_classifier':
            base_algorithm = SGDClassifier(
                random_state=self._rnd.randint(1000), loss='log', warm_start=False)
        else:
            sys.exit("no such base algorithm: %s" % self._base_algorithm_name)

        self._create_algorithm(base_algorithm, True, **kwargs)

    def _pick_arms(self):
        # Picks for the next batch size
        if self._iteration <= self._batch_size:  # model not fitted yet, pick random arms
            random_arm_indicies = self._rnd.choice(
                self._K, size=self._L, replace=False)
            picked_arms_indicies = random_arm_indicies
        else:
            picked_arms_indicies = (
                self._algorithm.topN(
                    self._context_df.values[self._iteration, :],
                    self._L)[0])

        self._picked_arms[self._iteration %
                          self._batch_size, :] = picked_arms_indicies
        self._received_rewards[self._iteration % self._batch_size,
                               :] = self._reward_df.values[self._iteration, picked_arms_indicies]

        return picked_arms_indicies

    def _learn(self):
        # (re)fit model
        if self._iteration > 0 and self._iteration % self._batch_size == 0:
            breakpoint()
            if self._iteration == self._batch_size:
                self._algorithm.fit(
                    repeat_entry_L_times(
                        self._scaler.transform(
                            self._context_df.values[: self._batch_size, :]),
                        self._L
                    ),
                    self._picked_arms.flatten(),
                    self._received_rewards.flatten()
                )
            else:
                self._algorithm.partial_fit(
                    repeat_entry_L_times(
                        self._scaler.transform(
                            self._context_df.values[self._iteration -
                                                    self._batch_size: self._iteration, :]
                        ),
                        self._L
                    ),
                    self._picked_arms.flatten(),
                    self._received_rewards.flatten()
                )

            self._picked_arms = np.zeros(
                self._batch_size * self._L).reshape(-1, self._L)
            self._received_rewards = np.zeros(
                self._batch_size * self._L).reshape(-1, self._L)
