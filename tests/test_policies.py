import sys
import os
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../src')))

from models.domain_knowledge import GraphArmKnowledge
from models.policy import EGreedy, MPTS
import unittest
from unittest.mock import Mock
import pandas as pd
import numpy as np


class PolicyTest(unittest.TestCase):

    def setUp(self):
        self._reward_df = pd.read_csv('./data/test_reward_df.csv')
        self._pick_arm_mock = Mock()

    def _get_graph_knowledge_mock(self):
        graph_knowledge_mock = Mock()
        graph_knowledge_mock.weight = 0.5
        graph_knowledge_mock.edges = np.array([
            [0, 0, 0, 0.5, 0, 0, 0.5, 0, 0],
            [0, 0, 0.5, 0, 0.5, 0.5, 0, 0, 0],
            [0, 0.5, 0, 0, 0.5, 0.5, 0, 0, 0],
            [0.5, 0, 0, 0, 0, 0, 0.5, 0, 0],
            [0, 0.5, 0.5, 0, 0, 0.5, 0, 0, 0],
            [0, 0.5, 0.5, 0, 0.5, 0, 0, 0, 0],
            [0.5, 0, 0, 0.5, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0]
        ])

        return graph_knowledge_mock

    def test_egreedy_without_sliding_window(self):
        egreedy = EGreedy(3, self._reward_df, 0, 0)
        expected_result = pd.read_csv('./data/expected_results_egreedy.csv')[
            'expected_values_without_sliding_window'].values.reshape(-1, egreedy.K)

        self._pick_arm_mock.side_effect = [
            [0, 1, 2], [3, 4, 5], [0, 1, 2], [3, 4, 5], [0, 1, 2], [
                3, 4, 5], [0, 1, 2], [3, 4, 5], [0, 1, 2], [3, 4, 5]
        ]

        egreedy._pick_arms = self._pick_arm_mock

        for i in range(egreedy._T):
            egreedy.perform_iteration()
            np.testing.assert_almost_equal(
                egreedy._expected_values, expected_result[i, :], decimal=3)

    def test_mpts_without_sliding_window(self):
        mpts = MPTS(3, self._reward_df, 0)
        expected_results_mpts_df = pd.read_csv(
            './data/expected_results_mpts.csv')
        expected_alpha = expected_results_mpts_df['alpha_without_sliding_window'].values.reshape(
            -1, mpts.K)
        expected_beta = expected_results_mpts_df['beta_without_sliding_window'].values.reshape(
            -1, mpts.K)

        self._pick_arm_mock.side_effect = [
            [0, 1, 2], [3, 4, 5], [0, 1, 2], [3, 4, 5], [0, 1, 2], [
                3, 4, 5], [0, 1, 2], [3, 4, 5], [0, 1, 2], [3, 4, 5]
        ]

        mpts._pick_arms = self._pick_arm_mock

        for i in range(mpts._T):
            mpts.perform_iteration()
            np.testing.assert_equal(mpts._alpha, expected_alpha[i, :])
            np.testing.assert_equal(mpts._beta, expected_beta[i, :])

    def test_egreedy_with_sliding_window(self):
        egreedy = EGreedy(3, self._reward_df, 0, 0, sliding_window_size=3)

        self._pick_arm_mock.side_effect = [
            [0, 1, 2], [0, 1, 2], [3, 4, 5], [3, 4, 5], [6, 7, 8], [
                6, 7, 8], [0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2]
        ]

        expected_result = pd.read_csv('./data/expected_results_egreedy.csv')[
            'expected_values_with_sliding_window'].values.reshape(-1, egreedy.K)

        egreedy._pick_arms = self._pick_arm_mock

        for i in range(egreedy.T):
            egreedy.perform_iteration()
            np.testing.assert_almost_equal(
                egreedy._expected_values, expected_result[i, :], decimal=3)

    def test_mpts_with_sliding_window(self):
        mpts = MPTS(3, self._reward_df, 0, sliding_window_size=3)

        expected_results_mpts_df = pd.read_csv(
            './data/expected_results_mpts.csv')
        expected_alpha = expected_results_mpts_df['alpha_with_sliding_window'].values.reshape(
            -1, mpts.K)
        expected_beta = expected_results_mpts_df['beta_with_sliding_window'].values.reshape(
            -1, mpts.K)

        self._pick_arm_mock.side_effect = [
            [0, 1, 2], [0, 1, 2], [3, 4, 5], [3, 4, 5], [6, 7, 8], [
                6, 7, 8], [0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2]
        ]

        mpts._pick_arms = self._pick_arm_mock

        for i in range(mpts._T):
            mpts.perform_iteration()
            np.testing.assert_equal(mpts._alpha, expected_alpha[i, :])
            np.testing.assert_equal(mpts._beta, expected_beta[i, :])

    def test_egreedy_with_graph_knowledge(self):
        egreedy = EGreedy(3, self._reward_df, 0, 0,
                          graph_knowledge=self._get_graph_knowledge_mock())
        self._pick_arm_mock.side_effect = [
            [0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [
                5, 6, 7], [6, 7, 8], [7, 8, 0], [8, 0, 1], [0, 1, 2]
        ]

        expected_result_df = pd.read_csv('./data/expected_results_egreedy.csv')
        expected_num_pushes = expected_result_df['num_pushes'].values.reshape(
            -1, egreedy.K)
        expected_sum_pushes = expected_result_df['sum_pushes'].values.reshape(
            -1, egreedy.K)

        egreedy._pick_arms = self._pick_arm_mock
        for i in range(egreedy.T):
            egreedy.perform_iteration()
            np.testing.assert_equal(
                egreedy._num_pushes_by_neighbours, expected_num_pushes[i, :])
            np.testing.assert_equal(
                egreedy._sum_pushes_by_neighbours, expected_sum_pushes[i, :])

    def test_mpts_with_graph_knowledge(self):
        mpts = MPTS(3, self._reward_df, 0,
                    graph_knowledge=self._get_graph_knowledge_mock())
        self._pick_arm_mock.side_effect = [
            [0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [
                5, 6, 7], [6, 7, 8], [7, 8, 0], [8, 0, 1], [0, 1, 2]
        ]
        mpts._pick_arms = self._pick_arm_mock

        expected_result_df = pd.read_csv('./data/expected_results_mpts.csv')
        expected_alpha = expected_result_df['alpha_gk'].values.reshape(
            -1, mpts.K)
        expected_beta = expected_result_df['beta_gk'].values.reshape(
            -1, mpts.K)

        for i in range(mpts.T):
            mpts.perform_iteration()
            np.testing.assert_equal(mpts._alpha, expected_alpha[i, :])
            np.testing.assert_equal(mpts._beta, expected_beta[i, :])

    def test_egreedy_with_sliding_window_and_domain_knowledge(self):
        egreedy = EGreedy(3, self._reward_df, 0, 0, sliding_window_size=3,
                          graph_knowledge=self._get_graph_knowledge_mock())
        self._pick_arm_mock.side_effect = [
            [0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2], [
                0, 1, 2], [6, 7, 8], [6, 7, 8], [6, 7, 8], [6, 7, 8]
        ]

        expected_result_df = pd.read_csv('./data/expected_results_egreedy.csv')
        expected_num_pushes = expected_result_df['num_pushes_sliding'].values.reshape(
            -1, egreedy.K)
        expected_sum_pushes = expected_result_df['sum_pushes_sliding'].values.reshape(
            -1, egreedy.K)

        egreedy._pick_arms = self._pick_arm_mock
        for i in range(egreedy.T):
            egreedy.perform_iteration()
            np.testing.assert_equal(
                egreedy._sum_pushes_by_neighbours, expected_sum_pushes[i, :])
            np.testing.assert_equal(
                egreedy._num_pushes_by_neighbours, expected_num_pushes[i, :])

    def test_mpts_with_sliding_window_and_domain_knowledge(self):
        mpts = MPTS(3, self._reward_df, 0, sliding_window_size=3,
                    graph_knowledge=self._get_graph_knowledge_mock())
        self._pick_arm_mock.side_effect = [
            [0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2], [
                0, 1, 2], [6, 7, 8], [6, 7, 8], [6, 7, 8], [6, 7, 8]
        ]

        expected_result_df = pd.read_csv('./data/expected_results_mpts.csv')
        expected_alpha = expected_result_df['alpha_sliding_gk'].values.reshape(
            -1, mpts.K)
        expected_beta = expected_result_df['beta_sliding_gk'].values.reshape(
            -1, mpts.K)

        mpts._pick_arms = self._pick_arm_mock
        for i in range(mpts.T):
            mpts.perform_iteration()
            np.testing.assert_equal(mpts._alpha, expected_alpha[i, :])
            np.testing.assert_equal(mpts._beta, expected_beta[i, :])


unittest.main()
