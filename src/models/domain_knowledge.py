import typing
from typing import List, Tuple
import numpy as np


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


class Knowledge:
    pass


class ArmKnowledge(Knowledge):
    """Contains information about metrics and hosts for the arms."""

    arm: np.ndarray
    hosts_for_arm: np.ndarray
    is_control_host_for_arm: np.ndarray
    hosts_active_for_arm: np.ndarray
    metrics_for_arm: np.ndarray
    arm_lays_on_same_host: np.ndarray
    arm_lays_on_control_host: np.ndarray
    arm_lays_on_different_compute_hosts: np.ndarray
    arm_lays_on_same_compute_host: np.ndarray
    arm_has_temporal_correlation: np.ndarray
    indicies_of_arms_that_will_not_be_explored: np.ndarray
    indicies_of_arms_that_will_be_explored: np.ndarray

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

    @property
    def arm(self) -> np.ndarray:
        """Return name of arms"""
        return self._arms

    @property
    def hosts_for_arm(self) -> np.ndarray:
        """Hosts for each arm"""
        return self._hosts_for_arm

    @property
    def hosts_active_for_arm(self) -> np.ndarray:
        """If hosts of the arms are active"""
        return self._hosts_active_for_arm

    @property
    def is_control_host_for_arm(self) -> np.ndarray:
        """If host is control host"""
        return self._is_control_host_for_arm

    @property
    def metrics_for_arm(self) -> np.ndarray:
        """Metrics for each arm"""
        return self._metrics_for_arm

    @property
    def arm_lays_on_same_host(self) -> np.ndarray:
        """If the arms lay on the same host, meaning if for each arm the
        metrics belong to the same host."""
        return self._arm_lays_on_same_host

    @property
    def arm_lays_on_control_host(self) -> np.ndarray:
        """If the arms lay on the control host, meaning if atleast one of the
        metrics belongs to the control host."""
        return self._arm_lays_on_control_host

    @property
    def arm_lays_on_different_compute_hosts(self) -> np.ndarray:
        """If the arms lay on different compute hosts, meaning one of the
        metrics belongs to one compute host and the other one to another
        compute host.
        """
        return self._arm_lays_on_different_compute_hosts

    @property
    def arm_lays_on_same_compute_host(self) -> np.ndarray:
        """If the arms lay on the same compute hosts, meaning both memtrics
        belong to the same compute host."""
        return self._arm_lays_on_same_compute_host

    @property
    def arm_has_temporal_correlation(self) -> np.ndarray:
        """Some arms are correlated through time, e.g. 
        wally113.load.min1-wally113.load.min5. Returns bool array that indicate
        if temporal correlation exists for this arm.
        """
        return self._arm_has_temporal_correlation

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


class DynamicArmKnowledge(ArmKnowledge):
    """Base class for domain knowledge that can change dynamically."""

    def __init__(self, arms: np.ndarray, control_host: str = 'wally113'):
        super().__init__(arms, control_host)
        self._active_hosts = set(
            ['wally113', 'wally117', 'wally122', 'wally123', 'wally124'])

    def recompute_properties(self):
        """Add recomputation for properties that change dynamically in child
        class."""
        pass

    def update_active_hosts(self, active_hosts: np.ndarray):
        if self._active_hosts != set(active_hosts):
            self._hosts_active_for_arm = np.isin(
                self._hosts_for_arm, active_hosts
            )
            self.recompute_properties()
            self._active_hosts = set(active_hosts)


class PushArmKnowledge(DynamicArmKnowledge):
    """Arms are eligible for a push if either one or both of its hosts are
    active.
    """

    arms_eligible_for_push: np.ndarray

    def __init__(self, arms: np.ndarray,
                 one_active_host_sufficient_for_push: bool = True,
                 control_host: str = 'wally113'
                 ):
        super().__init__(arms, control_host)
        self._one_active_host_sufficient_for_push = one_active_host_sufficient_for_push
        self._arms_eligible_for_push = self._compute_arms_eligible_for_push()

    def _compute_arms_eligible_for_push(self):
        if self._one_active_host_sufficient_for_push:
            return self._hosts_active_for_arm.any(axis=1)
        else:
            return self._hosts_active_for_arm.all(axis=1)

    def recompute_properties(self):
        self._arms_eligible_for_push = self._compute_arms_eligible_for_push()

    @property
    def arms_eligible_for_push(self) -> np.ndarray:
        """Returns for each arm whether or not it is eligible for a push."""
        return self._arms_eligible_for_push


class GraphArmKnowledge(DynamicArmKnowledge):
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

    edges: np.ndarray

    def __init__(self, arms: np.ndarray, weight: float = 0.5, control_host: str = 'wally113'):
        super().__init__(arms, control_host)
        self._weight = weight
        self._edges = self._compute_edges()

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

    def get_adjacency_matrix_from_groups(self, groups):
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
            adjacency[clique_idx] = self._weight
            edges[clique_idx, :] = adjacency

        np.fill_diagonal(edges, 0.0)
        assert (edges == edges.T).all()

        return edges

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
    def name(self):
        return '%.1f-correct-gk' % self._weight


class WrongGraphArmknowledge(GraphArmKnowledge):

    def __init__(self, arms: np.ndarray, kind: str, n_affected: int, weight: float = 1.0, control_host: str = 'wally113', random_seed=0):
        self._rnd = np.random.RandomState(random_seed)
        super().__init__(arms, weight, control_host)

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

        self._edges = self.get_adjacency_matrix_from_groups(groups)

    def _add_edges(self, n_affected_arms):
        groups = self.get_group_of_arms()

        candidates = np.arange(len(groups))[groups == 0]
        add_group_indicies = self._rnd.choice(
            candidates, n_affected_arms, replace=False)

        groups[add_group_indicies] = self._rnd.choice(
            np.arange(groups.max()) + 1, n_affected_arms)

        self._edges = self.get_adjacency_matrix_from_groups(groups)

    def _unify_cliques(self, n_affected_cliques):
        group = self.get_group_of_arms()

        unique_groups = np.delete(np.unique(group), 0)

        unified_groups = self._rnd.choice(
            unique_groups, 2 * n_affected_cliques, replace=False).reshape(-1, 2)

        for ug in unified_groups:
            min_clique_number = ug.min()
            group[(group == ug[0]) | (group == ug[1])] = min_clique_number

        self._edges = self.get_adjacency_matrix_from_groups(group)

    def _flip_edges(self, n_affected):
        self._edges = self._edges.flatten()

        ind = self._rnd.choice(
            np.arange(self._edges.shape[0]), n_affected, replace=False)

        self._edges[ind] = np.logical_not(self._edges[ind])
        self._edges = self._edges.reshape(self._K, self._K)

    @property
    def name(self):
        return '%.1f-wrong-gk-%s-%d' % (self._weight, self._kind, self._affected)


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
