import typing
from typing import List, Tuple
import numpy as np
import re


def extract_hosts_from_arm(arm: str) -> Tuple[str, str]:
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


def extract_metrics_from_arm(arm: str) -> Tuple[str, str]:
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


def get_adjacency_matrix_from_groups(groups, weight):
    indicies_of_cliques = list()
    current_clique_number = None
    for clique_number, idx in zip(sorted(groups), np.argsort(groups)):
        if clique_number == 0:
            continue
        if current_clique_number is None or clique_number != current_clique_number:
            current_clique_number = clique_number
            indicies_of_cliques.append(list([idx]))
        else:
            indicies_of_cliques[-1].append(idx)

    edges = np.zeros(shape=(len(groups), len(groups)))

    for clique_idx in indicies_of_cliques:
        adjacency = np.zeros(len(groups))
        adjacency[clique_idx] = weight
        edges[clique_idx, :] = adjacency

    np.fill_diagonal(edges, 0.0)
    assert (edges == edges.T).all()

    return edges


class Knowledge:
    pass

class GraphKnowledge:
    weight: float
    edges: np.ndarray
    only_push_arms_that_were_not_picked: bool

    @property
    def weight(self) -> float:
        return self._weight
    
    @property
    def edges(self) -> np.ndarray:
        """Returns an adjaceny matrix of size self._K * self._K. Arms i and j
        are neighbors if self._edges[i,j] is True. The matrix is symmetric.
        """
        return self._edges
    
    @property
    def only_push_arms_that_were_not_picked(self) -> np.ndarray:
        """If set only the unpicked arms infer knowledge within the group."""
        return self._only_push_arms_that_were_not_picked

class StaticPushArmKnowledge:
    arm_has_temporal_correlation: np.ndarray
    arm_likely: np.ndarray
    arm_unlikely: np.ndarray
    indicies_of_arms_that_will_not_be_explored: np.ndarray
    indicies_of_arms_that_will_be_explored: np.ndarray
    
    @property
    def arm_has_temporal_correlation(self) -> np.ndarray:
        return self._arm_has_temporal_correlation

    @property
    def arm_likely(self) -> np.ndarray:
        return self._arm_likely

    @property
    def arm_unlikely(self) -> np.ndarray:
        return np.logical_not(self._arm_likely)

    @property
    def indicies_of_arms_that_will_not_be_explored(self) -> np.ndarray:
        """Indicies of uninteresting arms, that contain constant metrics.
        """
        return self._indicies_of_arms_that_will_not_be_explored

    @property
    def indicies_of_arms_that_will_be_explored(self) -> np.ndarray:
        """Indicies of arms without the indicies of arms that will not be
        explored"""
        return self._indicies_of_arms_that_will_be_explored


class ArmKnowledge(Knowledge, StaticPushArmKnowledge):
    """Contains information about metrics and hosts for the arms."""

    arm_relevant_for_sliding_window: np.ndarray
    
    def __init__(self, arms: np.ndarray, control_host: str = 'wally113'):
        self._K = len(arms)
        self._arms = arms
        self._hosts_for_arm = np.zeros(2 * self._K, dtype=object)
        self._metrics_for_arm = np.zeros(2 * self._K, dtype=object)

        for i, arm in enumerate(self._arms):
            host1, host2 = extract_hosts_from_arm(arm)
            self._hosts_for_arm[2 * i] = host1
            self._hosts_for_arm[2 * i + 1] = host2

            metric1, metric2 = extract_metrics_from_arm(arm)
            self._metrics_for_arm[2 * i] = metric1
            self._metrics_for_arm[2 * i + 1] = metric2

        self._hosts_for_arm = self._hosts_for_arm.reshape(-1, 2)
        self._hosts_active_for_arm = np.ones((self._K, 2), dtype=bool)
        self._is_control_host_for_arm = self._hosts_for_arm == control_host

        self._metrics_for_arm = self._metrics_for_arm.reshape(-1, 2)

        self._arm_lays_on_same_host = self._hosts_for_arm[:,
                                                          0] == self._hosts_for_arm[:, 1]

        self._arm_lays_on_control_host = np.logical_or(
            self._hosts_for_arm[:,
                                0] == control_host, self._hosts_for_arm[:, 1] == control_host
        )

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
        self._indicies_of_arms_that_will_be_explored = np.delete(
            np.arange(self._K), self._indicies_of_arms_that_will_not_be_explored)

        self._arm_has_temporal_correlation = np.logical_and(
            np.isin(self._metrics_for_arm, 'load.min').all(axis=1),
            self._arm_lays_on_same_host
        )

        self._arm_relevant_for_sliding_window = np.isin(self._metrics_for_arm, 'load.min').all(axis=1)

        self._arm_likely = np.logical_or(self._arm_lays_on_same_host,self._arm_lays_on_control_host)

    @property
    def arm_relevant_for_sliding_window(self):
        return self._arm_relevant_for_sliding_window    

class ActiveHostKnowledge(ArmKnowledge):
    """Base class for domain knowledge that can change dynamically."""

    def __init__(self, arms: np.ndarray, control_host: str = 'wally113'):
        super().__init__(arms, control_host)
        self._active_hosts = set([])

    def recompute_properties(self):
        """Add recomputation for properties that change dynamically in child
        class."""
        pass

    def update_active_hosts(self, context_row):
        breakpoint()
        if self._active_hosts != set(active_hosts):
            self._hosts_active_for_arm = np.isin(
                self._hosts_for_arm, active_hosts
            )
            self.recompute_properties()
            self._active_hosts = set(active_hosts)


class DynamicPushKnowledge:
    arms_eligible_for_push: np.ndarray

    def compute_arms_eligible_for_push(self, context):
        pass
    
    @property
    def arms_eligible_for_push(self) -> np.ndarray:
        """Returns for each arm whether or not it is eligible for a push."""
        return self._arms_eligible_for_push
            
class PushArmKnowledge(ActiveHostKnowledge, DynamicPushKnowledge):
    """Arms are eligible for a push if either one or both of its hosts are
    active.
    """
    def __init__(self, arms: np.ndarray, context_columns,
                 one_active_host_sufficient_for_push: bool = True,
                 control_host: str = 'wally113'
                 ):
        super().__init__(arms, control_host)
        self._context_columns = context_columns
        self._one_active_host_sufficient_for_push = one_active_host_sufficient_for_push
        self._interesting_metrics = np.isin(self._metrics_for_arm, ['cpu.user', 'mem.used', 'load.min1', 'load.min5', 'load.min15']).all(axis=1)
        self._arms_eligible_for_push = np.zeros(self._K, dtype=bool)

    def compute_arms_eligible_for_push(self, context):
        if context[0] > 3000 and (context[1,2,3,4] == 0).all():
            return np.logical_and(
                (self._hosts_for_arm == 'wally113').all(axis=1),
                self._interesting_metrics
            )
        elif (context[1,2] > 10).all() and (context[3,4] == 0).all():
            return np.logical_and(
                np.isin(self._hosts_for_arm, ['wally113', 'wally122', 'wally124']),
                self._interesting_metrics
            )
        
        return np.zeros(self._K, dtype=bool)
        # self.update_active_hosts(self._context_columns[context > 0])
        # if self._one_active_host_sufficient_for_push:
        #     return np.logical_not(np.logical_and(self._hosts_active_for_arm.any(axis=1), self._interesting_metrics))
        # else:
        #     return np.logical_not(np.logical_and(self._hosts_active_for_arm.all(axis=1), self._interesting_metrics))                                           

    @property
    def arm_likely(self) -> np.ndarray:
        return
    
class SimiliarPushArmKnowledge(ActiveHostKnowledge, DynamicPushKnowledge):
    threshold : int
    
    def __init__(self, arms, threshold, columns, control_host='wally113'):
        super().__init__(arms, control_host)
        self._columns = columns
        self._threshold = threshold
        self._compute_push_matrix()
        
        self._arms_eligible_for_push = np.zeros(self._K, dtype=bool)

    def compute_arms_eligible_for_push(self, similiary):
        self._arms_eligible_for_push = self._matrix[:, similiary < self._threshold].any(axis=1)

    def _compute_push_matrix(self):
        self._matrix = np.zeros(shape=(self._K, self._columns.shape[0]), dtype=bool)
        for i, c in enumerate(self._columns):
            h1, h2 = c.split('.')[0].split('-')

            minute = int(re.findall('\d+min', c)[0].split('min')[0])


            if minute == 15:
                self._matrix[:, i] = np.logical_and(
                    np.isin(self._hosts_for_arm, [h1,h2]).all(axis=1),
                    np.isin(self._metrics_for_arm, ['load.min15']).all(axis=1)
                )
            elif minute == 5:
                self._matrix[:, i] = np.logical_and(
                    np.isin(self._hosts_for_arm, [h1,h2]).all(axis=1),
                    np.isin(self._metrics_for_arm, ['load.min15', 'load.min5']).all(axis=1)
                )
            elif minute == 1:
                self._matrix[:, i] = np.logical_and(
                    np.isin(self._hosts_for_arm, [h1,h2]).all(axis=1),
                    np.isin(self._metrics_for_arm, ['load.min1', 'load.min5', 'load.min15']).all(axis=1)
                )                    

            
    def compute_arms_eligible_for_push(self, similiarity: List[float]):
        self._arms_eligible_for_push = self._matrix[:, similiarity < self._threshold].any(axis=1)

    @property
    def threshold(self) -> int:
        return self._threshold
        
class SyntheticPushArmKnowledge(DynamicPushKnowledge, StaticPushArmKnowledge):
    def __init__(self, arms: np.ndarray):
        self._arms_eligible_for_push = np.zeros(len(arms), dtype=bool)
        self._indicies_of_arms_that_will_not_be_explored = np.array([], dtype=int)

    def compute_arms_eligible_for_push(self, context : np.ndarray):
        self._arms_eligible_for_push = context > 0
        
class GraphArmKnowledge(ActiveHostKnowledge, GraphKnowledge):
    """Contains the neighbors for each arm. Arms are neighbors if they are
    in the same group. Groups are defined based on the type of host (control,
    compute), the metric and the state of the host.

    E.g. if all hosts are active the arm wally113.cpu.user-wally117.mem.used
    has three neighbors:
      - wally113.cpu.user-wally122.mem.used
      - wally113.cpu.user-wally123.mem.used
      - wally113.cpu.user-wally124.mem.used

    If only hosts wally113, wally117 and wally122 are active the arm only has
    one neighbor:
      - wally113.cpu.user-wally122.mem.used
    """
    def __init__(self, arms: np.ndarray, weight: float = 0.5, control_host: str = 'wally113', only_push_arms_that_were_not_picked: bool = True):
        super().__init__(arms, control_host)
        self._weight = weight
        self._edges = self._compute_edges()
        self._only_push_arms_that_were_not_picked = only_push_arms_that_were_not_picked

    def _compute_edges(self):
        representation_first_part = np.array([
            self._is_control_host_for_arm[:, 0],
            self._hosts_active_for_arm[:, 0],
            self._metrics_for_arm[:, 0],
            self._arm_lays_on_same_host
        ]).T

        representation_second_part = np.array([
            self._is_control_host_for_arm[:, 1],
            self._hosts_active_for_arm[:, 1],
            self._metrics_for_arm[:, 1],
            self._arm_lays_on_same_host
        ]).T

        representation_first_part_row_aligned = np.stack(
            [representation_first_part] * self._K)
        representation_first_part_column_aligned = np.tile(
            representation_first_part, self._K).reshape(self._K, self._K, -1)

        representation_second_part_row_aligned = np.stack(
            [representation_second_part] * self._K)
        representation_second_part_column_aligned = np.tile(
            representation_second_part, self._K).reshape(self._K, self._K, -1)

        arms_match = np.logical_or(
            np.logical_and(
                (representation_first_part_row_aligned ==
                 representation_first_part_column_aligned).all(axis=2),
                (representation_second_part_row_aligned ==
                 representation_second_part_column_aligned).all(axis=2)
            ),
            np.logical_and(
                (representation_first_part_row_aligned ==
                 representation_second_part_column_aligned).all(axis=2),
                (representation_second_part_row_aligned ==
                 representation_first_part_column_aligned).all(axis=2)
            )
        )

        np.fill_diagonal(arms_match, False)

        return arms_match * self._weight

    def recompute_properties(self):
        self._edges = self._compute_edges()

    def get_group_of_arms(self):
        assert (self._edges == self._edges.T).all()

        clique_number = 1

        group_of_arms = np.zeros(self._K)
        for row in range(self._K):
            if group_of_arms[row] != 0:
                continue

            if self._edges[row, :].any():
                group_of_arms[row] = clique_number
                group_of_arms[self._edges[row, :] > 0] = clique_number
                clique_number += 1

        return group_of_arms


class SyntheticGraphArmKnowledge(GraphKnowledge):
    edges: np.ndarray

    def __init__(self, arms: np.ndarray, groups: List[int], weight: float =0.5, only_push_arms_that_were_not_picked: bool = True):
        self._K = len(arms)
        self._arms = arms
        self._weight = weight
        self._groups = np.array(groups)
        self._edges = get_adjacency_matrix_from_groups(groups, self._weight)
        self._only_push_arms_that_were_not_picked = only_push_arms_that_were_not_picked

class WrongSyntheticGraphArmKnowledge(SyntheticGraphArmKnowledge):
    def __init__(self, arms: np.ndarray, groups: np.ndarray, error_kind: str, percentage_affected: float, weight: float=0.5, only_push_arms_that_were_not_picked: bool = True, seed=0):
        super().__init__(arms, groups, weight)
        self._rnd = np.random.RandomState(seed)
        self._unique_groups = np.unique(groups[groups != 0])
        self._only_push_arms_that_were_not_picked = only_push_arms_that_were_not_picked
        self._no_groups = self._unique_groups.shape[0]
        self._percentage_affected = percentage_affected
        self._error_kind = error_kind
        
        if error_kind == 'remove':
            self._remove_edges()
        elif error_kind == 'random':
            self._random_group()
    
    def _remove_edges(self):
        candidates = np.arange(self._K)[self._groups > 0]
        number_of_removals = int(len(candidates) * self._percentage_affected)
        remove_group_indicies = self._rnd.choice(
            candidates, number_of_removals, replace=False)

        self._groups[remove_group_indicies] = 0.0

        self._edges = get_adjacency_matrix_from_groups(self._groups, self._weight)

    def _random_group(self):
        no_affected = int(self._K * self._percentage_affected)
        indicies_of_affected = self._rnd.choice(self._K, no_affected, replace=False)
        self._groups[indicies_of_affected] = self._rnd.choice(self._unique_groups, no_affected)
        self._edges = get_adjacency_matrix_from_groups(self._groups, self._weight)


        
class WrongGraphArmknowledge(GraphArmKnowledge):

    def __init__(self, arms: np.ndarray, kind: str, n_affected: int, weight: float = 1.0, control_host: str = 'wally113', only_push_arms_that_were_not_picked: bool = True, random_seed=0):
        self._rnd = np.random.RandomState(random_seed)
        super().__init__(arms, weight, control_host, only_push_arms_that_were_not_picked)

        self._kind = kind
        self._n_affected = n_affected
        if kind == 'remove':
            self._remove_edges(n_affected)
        elif kind == 'add':
            self._add_edges(n_affected)
        elif kind == 'unify':
            self._unify_cliques(n_affected)
        elif kind == 'flip':
            self._flip_edges(n_affected)

    def _remove_edges(self, n_affected_arms):
        groups = self.get_group_of_arms()

        candidates = np.arange(len(groups))[groups > 0]
        remove_group_indicies = self._rnd.choice(
            candidates, n_affected_arms, replace=False)

        groups[remove_group_indicies] = 0.0

        self._edges = get_adjacency_matrix_from_groups(groups, self._weight)

    def _add_edges(self, n_affected_arms):
        groups = self.get_group_of_arms()

        candidates = np.arange(len(groups))[groups == 0]
        add_group_indicies = self._rnd.choice(
            candidates, n_affected_arms, replace=False)

        groups[add_group_indicies] = self._rnd.choice(
            np.arange(groups.max()) + 1, n_affected_arms)

        self._edges = get_adjacency_matrix_from_groups(groups, self._weight)

    def _unify_cliques(self, n_affected_cliques):
        group = self.get_group_of_arms()

        unique_groups = np.delete(np.unique(group), 0)

        unified_groups = self._rnd.choice(
            unique_groups, 2 * n_affected_cliques, replace=False).reshape(-1, 2)

        for ug in unified_groups:
            min_clique_number = ug.min()
            group[(group == ug[0]) | (group == ug[1])] = min_clique_number

        self._edges = get_adjacency_matrix_from_groups(group, self._weight)

    def _flip_edges(self, n_affected):
        self._edges = self._edges.flatten()

        ind = self._rnd.choice(
            np.arange(self._edges.shape[0]), n_affected, replace=False)

        self._edges[ind] = np.logical_not(self._edges[ind])
        self._edges = self._edges.reshape(self._K, self._K)

class RandomGraphKnowledge(Knowledge):

    edges: np.ndarray

    def __init__(self, K: int, weight: float = 0.5, probability_neighbors: List[float] = [0.4, 0.15, 0.15, 0.15, 0.15], seed: int = 0):
        self._K = K
        self._weight = weight
        self._probability_neighbors = probability_neighbors
        self._rnd = np.random.RandomState(seed)
        self._edges = self._get_edges()

    def _get_edges(self):
        edges = []
        for current_arm_idx in np.arange(self._K):
            number_neighbors = self._rnd.choice(
                len(self._probability_neighbors),
                1,
                p=self._probability_neighbors
            )
            neighbors = np.zeros(self._K, dtype=bool)
            neighbors[self._rnd.choice(self._K, number_neighbors)] = True
            neighbors[current_arm_idx] = False
            edges.extend(np.where(neighbors, self._weight, 0.0))

        neighbor_matrix = np.array(edges).reshape(self._K, self._K)
        symetric_neighbor_matrix = np.tril(
            neighbor_matrix) + np.tril(neighbor_matrix, -1).T

        assert (symetric_neighbor_matrix == symetric_neighbor_matrix.T).all()

        return symetric_neighbor_matrix

    @property
    def edges(self):
        """Adjacency matrix of size self._K x self._K that describes
        neighborhod for arms."""
        return self._edges

class SyntheticStaticPushKnowledge(StaticPushArmKnowledge):

    def __init__(self):
        self._arm_likely = np.zeros(100, dtype=bool)
        self._arm_likely[0:30] = True
        self._indicies_of_arms_that_will_not_be_explored = []
        self._indicies_of_arms_that_will_be_explored = np.arange(100)
        self._arm_has_temporal_correlation = np.zeros(100, dtype=bool)


class WrongSyntheticStaticPushKnowledge(SyntheticStaticPushKnowledge):

    def __init__(self, kind='remove', percentage_affected=0.1, random_seed = 0):
        super().__init__()
        self._rnd = np.random.RandomState(random_seed)
        if kind == 'remove':
            number_of_removals = int(30 * percentage_affected)
            self._arm_likely[self._rnd.choice(np.arange(30), number_of_removals, replace=False)] = False
        elif kind == 'random':
            number_of_random_initialisation = int(100 * percentage_affected)
            self._arm_likely[self._rnd.choice(np.arange(100), number_of_random_initialisation, replace=False)] = self._rnd.choice([True, False], number_of_random_initialisation)
            
        
