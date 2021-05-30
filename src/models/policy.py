"""This class contains different policies on how to select pairs of metrics.
It contains both contextual and non-contextual policies.
"""
from abc import ABC, abstractmethod
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
from contextualbandits.online import EpsilonGreedy
import itertools
import pandas as pd
import typing
from typing import List, Tuple
from .domain_knowledge import ArmKnowledge, PushArmKnowledge, GraphArmKnowledge

def repeat_entry_L_times(X: np.ndarray, L: int) -> np.ndarray:
    return np.tile(X, L).reshape(-1, X.shape[1])

class AbstractBandit(ABC):
    """Provides functionality for a basic non-contextual bandit."""

    L : int
    K : int
    T : int
    iteration : int
    overall_regret : float
    regret : List[float]
    cum_regret: List[float]
    average_regret: List[float]
    name : str
    
    def __init__(self,
                 L: int,
                 reward_df: pd.DataFrame,
                 identifier: typing.Optional[str]=None
    ):
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
    def iteration(self)->int:
        """Current iteration"""
        return self._iteration

    @property
    def overall_regret(self)->float:
        """Total regret of the policy"""
        return np.sum(self._regret)

    @property
    def regret(self) -> List[float]:
        """Regret for each round"""
        return self._regret

    @property
    def cum_regret(self)-> List[float]:
        """Cumulated regret over the rounds"""
        return np.cumsum(self._regret)

    @property
    def average_regret(self)-> List[float]:
        """Average regret over the rounds"""
        return np.mean(self._regret)

    def run(self):
        """Picks each iteration L of K arms. Receives the reward for the picked
        arms. Adjusts the picking strategy.
        """
        for _ in range(0, self._T):
            self._pick_arms()
            self._learn()
            
            max_reward_this_round = np.sort(self._reward_df.values[self._iteration, :])[-self._L:].sum()
            received_reward_this_round = (self._reward_df.values[self._iteration, self._picked_arms_indicies]).sum()
            self._regret[self._iteration] =  max_reward_this_round - received_reward_this_round
            self._iteration += 1

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
    def name(self)->str:
        """Returns the name of the bandit

        Returns:
          string
        """
        return


class RandomPolicy(AbstractBandit):
    """Policy that picks arms randomly."""

    def __init__(self,
                 L: int,
                 reward_df: pd.DataFrame,
                 random_seed: int,
                 identifier: typing.Optional[str]=None
    ):
        super().__init__(L, reward_df, identifier)
        self._rnd = np.random.RandomState(random_seed)

    @property
    def name(self):
        if self._identifier is not None:
            return self._identifier

        return 'random'

    def _pick_arms(self):
        self._picked_arms_indicies = self._rnd.choice(self._K, self._L, replace=False)
    
class EGreedy(AbstractBandit):
    """E-Greedy bandit algorithm that picks the arms with highest expected
    reward greedily with probability 1-e (exploitation) and random arms with
    probablity e (exploration).
    """

    epsilon : float
    
    def __init__(
            self, L : int,
            reward_df : pd.DataFrame,
            random_seed : int,
            epsilon : float=0.1,
            identifier : typing.Optional[str]=None
    ):
        super().__init__(L, reward_df, identifier)
        self._epsilon = epsilon
        self._expected_values = np.ones(self._K)
        self._num_plays = np.zeros(self._K)
        self._rnd = np.random.RandomState(random_seed)

    @property
    def epsilon(self) -> float:
        """Probability of exploration"""
        return self._epsilon

    @property
    def name(self) -> str:
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
            self._picked_arms_indicies = np.argsort(self._expected_values)[-self._L:]

    def explore_arms(self):
        self._picked_arms_indicies = self._rnd.choice(self._K, self._L, replace=False)

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
            self,
            L: int,
            reward_df: pd.DataFrame,
            random_seed: int,
            epsilon: float=0.1,
            init_ev_likely_arms: float=0.95,
            init_ev_unlikely_arms: float=0.75,
            init_ev_temporal_correlated_arms: float=1.0,
            control_host: str='wally113',
            identifier: typing.Optional[str]=None
    ):
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
        self._arm_knowledge = ArmKnowledge(self._arms, control_host)

        self._init_ev_likely_arms = init_ev_likely_arms
        self._init_ev_unlikely_arms = init_ev_unlikely_arms
        self._expected_values[self._arm_knowledge.arm_has_temporal_correlation] = init_ev_temporal_correlated_arms
        self._init_ev()


    def _init_ev(self):
        """Intializes the expected value for the arms. An arm is a pair of
        metrics. If the metrics are on the same host or are a pair between a
        computing and the control host the expected value is set to
        self._init_ev_likely_arms, otherwise to self._init_ev_unlikely_arms.
        The ev of arms that will not be explored is set to 0.
        """
        expected_values = np.where(
            np.logical_or(
                self._arm_knowledge.arm_lays_on_same_host,
                self._arm_knowledge.arm_lays_on_control_host
            ),
            self._init_ev_likely_arms,
            self._init_ev_unlikely_arms
        )
        expected_values[self._arm_knowledge.indicies_of_arms_that_will_not_be_explored] = 0.0
        self._expected_values = expected_values

    def explore_arms(self):
        """Performs an exploration step but never exploration arms that we know
        are uninteresting (arms for the metric load.cpucore because this has a
        static value).
        """
        explore_arm_indicies = self._rnd.choice(
            self._arm_knowledge.indicies_of_arms_that_will_be_explored,
            self._L,
            replace=False
        )
        self._picked_arms_indicies = explore_arm_indicies

    @property
    def name(self) -> str:
        if self._identifier is not None:
            return self._identifier

        if self._epsilon != 0:
            return '%.1f/%.1f-dk-%.1f-greedy' % (self._init_ev_likely_arms,
                                                 self._init_ev_unlikely_arms,
                                                 self._epsilon)
        
        return '%.1f/%.1f-dkgreedy' % (self._init_ev_likely_arms, self._init_ev_unlikely_arms)


class CDKEGreedy(DKEGreedy):
    """E-Greedy bandit algorithm using domain knowledge to initially push more
    likely arms and using dynamic pushes based on the context to further
    improve picking of arms. The context contains the number of events per
    host.
    """

    def __init__(
            self,
            L : int,
            reward_df :pd.DataFrame,
            random_seed : int,
            context_df : pd.DataFrame,
            epsilon : float=0.1,
            init_ev_likely_arms : float=0.95,
            init_ev_unlikely_arms: float=0.75,
            init_ev_temporal_correlated_arms: float=1.0,
            control_host: str='wally113',
            push: float=1.0,
            max_number_pushes: int=10,
            push_kind:str='plus',
            one_active_host_sufficient_for_push:bool=True,
            identifier:typing.Optional[str]=None
    ):
        """
        Args:
          context_df (DataFrame): DataFrame containing the number of events per
          host (columns) and iterations (rows).
          push (float): Push that gets performed for active arms.
          max_number_pushes (int): Number of times an arm will be pushed.
          push_kind (string): One of 'plus' or 'multiply'
        """
        super().__init__(L, reward_df, random_seed, epsilon, init_ev_likely_arms,
                         init_ev_unlikely_arms, init_ev_temporal_correlated_arms,
                         control_host,identifier=identifier)
        self._arm_knowledge = PushArmKnowledge(self._arms, one_active_host_sufficient_for_push, control_host)
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
        active_hosts = self._context_df.columns.values[current_context > 0]

        self._arm_knowledge.update_active_hosts(active_hosts)

        arm_gets_pushed = np.logical_and(
            self._arm_knowledge.arms_eligible_for_push,
            self._no_pushed < self._max_number_pushes
        )
        
        if self._push_kind == 'plus':
            pushed_expected_values = self._expected_values + \
                (self._push * arm_gets_pushed)
        else:
            factors = np.where(arm_gets_pushed, self._push, 1.0)
            pushed_expected_values *= factors

        self._picked_arms_indicies = np.argsort(pushed_expected_values)[-self._L:]

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

    def __init__(self,
                 L: int,
                 reward_df: pd.DataFrame,
                 random_seed: int,
                 identifier:typing.Optional[str]=None
    ):
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

    def __init__(self,
                 L: int,
                 reward_df: pd.DataFrame,
                 random_seed: int,
                 push_likely_arms:float=1.0,
                 push_unlikely_arms:float=1.0,
                 push_temporal_correlated_arms: float=1.0,
                 control_host: str='wally113',
                 identifier:typing.Optional[str]=None
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
        super().__init__(L, reward_df, random_seed, identifier)
        self._arm_knowledge = ArmKnowledge(self._arms, control_host)

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

    def _compute_init_posterior(self)->np.ndarray:
        """Computes the posterior distribution (beta) for the arms.
        Arms that don't get a push in the prior get a push in the posterior.

        Returns:
          float[]: Posterior distribution
        """
        return np.repeat(float(self._push_unlikely_arms), self._K)\
            * (self._compute_init_prior() == 0)

    @property
    def name(self)->str:
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
        theta[self._arm_knowledge.indicies_of_arms_that_will_not_be_explored] = 0.0
        self._picked_arms_indicies = np.argsort(theta)[-self._L:]


class CPushMpts(PushMPTS):
    """MPTS bandit algorithm using domain knowledge to initially push more
    likely arms and using dynamic pushes based on the context to further
    improve picking of arms. The expected context csv file contains a boolean
    for each arms that says whether or not to push the arm.
    """

    def __init__(self,
                 L:int,
                 reward_df: pd.DataFrame,
                 random_seed: int,
                 context_df:pd.DataFrame,
                 push_likely_arms:float=1.0,
                 push_unlikely_arms:float=1.0,
                 push_temporal_correlated_arms:float=1.0,
                 control_host:str='wally113',
                 cpush:float=1.0,
                 q:int=10,
                 one_active_host_sufficient_for_push:bool=True,
                 identifier:typing.Optional[str]=None
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
                         control_host, identifier)
        self._arm_knowledge = PushArmKnowledge(self._arms, one_active_host_sufficient_for_push, control_host)        
        self._context_df = context_df
        self._cpush = cpush
        self._max_number_pushes = q
        self._no_pushed = np.zeros(self._K)

    def _pick_arms(self):
        """Pushes the arms with the context. Uses the underlying PushMPTS
        algorithm to pick the arms.
        """
        current_context = self._context_df.values[self._iteration, :]
        active_hosts = self._context_df.columns.values[current_context > 0]        

        self._arm_knowledge.update_active_hosts(active_hosts)        

        alpha_pushed = self._alpha + self._arm_knowledge.arms_eligible_for_push * self._cpush

        theta = self._rnd.beta(alpha_pushed + 1, self._beta + 1)
        theta[self._arm_knowledge.indicies_of_arms_that_will_not_be_explored] = 0.0
        
        self._picked_arms_indicies = np.argsort(theta)[-self._L:]

        self._no_pushed[self._picked_arms_indicies] += self._arm_knowledge.arms_eligible_for_push[self._picked_arms_indicies]

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
            self,
            L:int,
            reward_df: pd.DataFrame,
            random_seed:int,
            context_df:pd.DataFrame,
            epsilon: float=0.1,
            batch_size:int=20,
            max_iter:int=100,
            solver:str='lbfgs',
            context_identifier:str='',
            identifier:typing.Optional[str]=None
    ):
        super().__init__(L, reward_df,identifier)
        self._context_df = context_df
        self._solver = solver
        self._context_identifier = context_identifier
        self._identifier = identifier

        base_algorithm = LogisticRegression(
            solver=solver, warm_start=True, max_iter=max_iter)
        self._rnd = np.random.RandomState(random_seed)
        self._epsilon_greedy = EpsilonGreedy(
            base_algorithm, nchoices=self._K, random_state=random_seed,
            explore_prob=epsilon)
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
                self._epsilon_greedy.fit(
                    repeat_entry_L_times(
                        normalize(self._context_df.values[: self._batch_size, :], axis=1),
                        self._L
                    ),
                    self._picked_arms.flatten(),
                    self._received_rewards.flatten()
                )
            else:
                self._epsilon_greedy.fit(
                    repeat_entry_L_times(
                        normalize(
                            self._context_df.values[self._iteration - self._batch_size: self._iteration, :],
                            axis=1
                        ),
                        self._L
                    ),
                    self._picked_arms.flatten(),
                    self._received_rewards.flatten(),
                    warm_start=True
                )

            self._picked_arms = np.zeros(
                self._batch_size * self._L).reshape(-1, self._L)
            self._received_rewards = np.zeros(
                self._batch_size * self._L).reshape(-1, self._L)

    @property
    def name(self):
        if self._identifier is not None:
            return self._identifier

        return 'cb-%s-%.2f-greedy-%d-%s' % (self._context_identifier, self._epsilon_greedy.explore_prob, self._batch_size, self._solver)


class InvertedPushMPTS(PushMPTS):
    """Policy that simulates wrong domain knowledge by inverting the pushes.
    Likely arms get a push in the posterior, while unlikely arms get a push in
    the prior.
    """

    def __init__(self,
                 L :int,
                 reward_df: pd.DataFrame,
                 random_seed: int,
                 push_likely_arms: float=1.0,
                 push_unlikely_arms: float=1.0,
                 push_temporal_correlated_arms: float=1.0,
                 control_host: str='wally113',
                 identifier: typing.Optional[str]=None
    ):
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
                         push_unlikely_arms, push_temporal_correlated_arms,
                         control_host, identifier)

        self._alpha = self._compute_init_posterior()
        self._beta = self._compute_init_prior()
        self._alpha[self._arm_knowledge.arm_has_temporal_correlation] = push_temporal_correlated_arms

    @property
    def name(self):
        if self._identifier is not None:
            return self._identifier

        return '%.1f-%.1f-inverted-push-mpts' % (
            self._push_likely_arms, self._push_unlikely_arms)
    
class StaticNetworkMPTS(MPTS):
    """Policy where relationships between arms exist. Arms move in groups, and
    one arms might be related to other arms. Therefore when an arm receives a
    reward we can derive some knowledge for other arms as well. A graph exists
    where nodes are the arms and weighted edges (w in [0,1]) indicate the
    strength of the relationship. If an arm receives a reward r we will push
    the prior/ posterior of the related arms by w * r.
    """

    weight: int
    control_host: str
    compute_hosts: List[str]
    
    def __init__(
            self,
            L: int,
            reward_df: pd.DataFrame,
            random_seed: int,
            weight:float =0.5,
            control_host:str='wally113',
            compute_hosts:List[str]=['wally117', 'wally122', 'wally123', 'wally124'],
            identifier:typing.Optional[str]=None
    ):
        super().__init__(L, reward_df, random_seed, identifier)
        self._arm_knowledge = GraphArmKnowledge(self._arms, control_host)

        self._weight = weight
        self._control_host = control_host
        self._compute_hosts = compute_hosts


    def _learn(self):
        """Update beta distribution of picked arms based on the reward. Updates
        beta distribution of related arms as well if they didn't get picked.
        Does this by increasing the prior/ posterior by the weighted reward.
        """
        super()._learn()

        neighbors_played_arms = self._arm_knowledge.edges[self._picked_arms_indicies]
        arm_number_of_updates = neighbors_played_arms.sum(axis=0)
        arm_gets_update = arm_number_of_updates > 0

        reshaped_rewards = np.repeat(self._reward_df.values[self._iteration, self._picked_arms_indicies], self._K).reshape(-1, self._K)

        update = (self._weight * neighbors_played_arms * reshaped_rewards).sum(axis=0)[arm_gets_update]

        self._alpha[arm_gets_update] += update
        self._beta[arm_gets_update] += (arm_number_of_updates[arm_gets_update] * self._weight - update)
        
    @property
    def name(self):
        return '%.1f-network-mpts' % self._weight



class DynamicNetworkMPTS(StaticNetworkMPTS):

    def __init__(
            self,
            L: int,
            reward_df: pd.DataFrame,
            random_seed: int,
            context_df: pd.DataFrame,            
            weight:float=0.5,
            control_host:str='wally113',
            compute_hosts:List[str]=['wally117', 'wally122', 'wally123', 'wally124'],
            identifier:typing.Optional[str]=None
    ):
        super().__init__(L, reward_df, random_seed, weight, control_host, compute_hosts)
        self._context_df = context_df
    

    def _pick_arms(self):
        super()._pick_arms()

        current_context = self._context_df.values[self._iteration, :]
        active_hosts = self._context_df.columns.values[current_context > 0]

        self._arm_knowledge.update_active_hosts(active_hosts)
        self._edges = self._arm_knowledge.edges


    @property
    def name(self):
        return '%.1f-dynamic-network-mpts' % self._weight    

class RandomNetworkMPTS(StaticNetworkMPTS):
    def __init__(self,
                 L: int,
                 reward_df: pd.DataFrame,
                 random_seed: int,
                 weight: float=0.5,
                 control_host: str='wally113',
                 compute_hosts: List[str]=['wally117', 'wally122', 'wally123', 'wally124'],
                 probability_neighbors: List[float]=[0.4, 0.15, 0.15, 0.15, 0.15],
                 identifier=None
    ):
        self._probability_neighbors = probability_neighbors
        super().__init__(L, reward_df, random_seed, weight, control_host, compute_hosts, identifier)

    def _get_edges(self):
        number_arms = self._K
        edges = []
        for i, _ in enumerate(self._nodes):
            number_neighbors = self._rnd.choice(
                len(self._probability_neighbors),
                1,
                p=self._probability_neighbors
            )
            neighbors = np.zeros(number_arms, dtype=bool)
            neighbors[self._rnd.choice(number_arms, number_neighbors)] = True
            neighbors[i] = False
            edges.extend(np.where(
                neighbors,
                self._weight,
                0.0
            )
            )

        return np.array(edges).reshape(number_arms, number_arms)

    @property
    def name(self):
        if self._identifier is not None:
            return self._identifier

        return '%.1f-%s-random-network-mpts' % (self._weight, str(self._probability_neighbors))
