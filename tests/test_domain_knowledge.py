from unittest.mock import Mock
from models.domain_knowledge import GraphArmKnowledge, WrongGraphArmknowledge
import unittest
import sys
import os

import numpy as np
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../src')))


class TestDomainKnowledge(unittest.TestCase):

    def test_group_of_arm(self):
        gk = GraphArmKnowledge(np.array([
            'wally113.cpu.user-wally117.cpu.user',
            'wally113.cpu.user-wally122.cpu.user']), 1.0)

        np.testing.assert_equal(gk.get_group_of_arms(), np.array([1, 1]))

        gk = GraphArmKnowledge(np.array([
            'wally113.cpu.user-wally113.mem.used',
            'wally113.cpu.user-wally117.cpu.user',
            'wally113.cpu.user-wally122.cpu.user']), 1.0)

        np.testing.assert_equal(gk.get_adjacency_matrix_from_groups([1, 1]),
                                np.array([
                                    [0, 1],
                                    [1, 0]
                                ]))

        np.testing.assert_equal(gk.get_adjacency_matrix_from_groups([0, 0]),
                                np.array([
                                    [0, 0],
                                    [0, 0]
                                ]))

        np.testing.assert_equal(gk.get_adjacency_matrix_from_groups([0, 1, 1]),
                                np.array([
                                    [0, 0, 0],
                                    [0, 0, 1],
                                    [0, 1, 0]
                                ]))

        np.testing.assert_equal(gk.get_adjacency_matrix_from_groups([0, 1, 0, 2, 1, 2]),
                                np.array([
                                    [0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 1, 0],
                                    [0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 1],
                                    [0, 1, 0, 0, 0, 0],
                                    [0, 0, 0, 1, 0, 0]
                                ]))

        np.testing.assert_equal(gk.get_adjacency_matrix_from_groups([0, 1, 0, 3, 1, 3]),
                                np.array([
                                    [0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 1, 0],
                                    [0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 1],
                                    [0, 1, 0, 0, 0, 0],
                                    [0, 0, 0, 1, 0, 0]
                                ]))

    def test_correct_graph_arm_knowledge(self):
        arms = np.array(['wally113.cpu.user-wally113.mem.used',
                         'wally113.cpu.user-wally117.cpu.user',
                         'wally113.cpu.user-wally122.cpu.user'])

        gk = GraphArmKnowledge(arms, 1.0)

        expected_edges = np.array([
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0]
        ])

        np.testing.assert_equal(gk.edges, expected_edges)

        arms = np.array(['wally113.cpu.user-wally113.mem.used',
                         'wally113.cpu.user-wally117.cpu.user',
                         'wally113.cpu.user-wally122.cpu.user',
                         'wally117.cpu.user-wally117.mem.used',
                         'wally122.cpu.user-wally122.mem.used'])

        gk = GraphArmKnowledge(arms, 1.0)

        expected_edges = np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 1, 0]
        ])

        np.testing.assert_equal(gk.edges, expected_edges)

        arms = np.array(['wally113.cpu.user-wally113.mem.used',
                         'wally113.cpu.user-wally117.cpu.user',
                         'wally113.cpu.user-wally122.cpu.user',
                         'wally117.cpu.user-wally122.mem.used',
                         'wally122.cpu.user-wally123.mem.used'])

        gk = GraphArmKnowledge(arms, 1.0)

        expected_edges = np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 1, 0]
        ])

        np.testing.assert_equal(gk.edges, expected_edges)

    def test_wrong_graph_arm_knowledge_flip(self):
        arms = np.array(['wally113.cpu.user-wally113.mem.used',
                         'wally113.cpu.user-wally117.cpu.user',
                         'wally113.cpu.user-wally122.cpu.user',
                         'wally117.cpu.user-wally122.mem.used',
                         'wally122.cpu.user-wally123.mem.used'])

        correct_gk = GraphArmKnowledge(arms, 1.0)

        for nflip in [1, 5, 10]:
            wrong_gk = WrongGraphArmknowledge(arms, 'flip', nflip)

            np.testing.assert_equal(
                (correct_gk.edges != wrong_gk.edges).sum(), nflip)

    def test_wrong_graph_arm_knowledge_unify(self):
        arms = np.array([
            'wally113.cpu.user-wally113.mem.used',
            'wally113.cpu.user-wally117.cpu.user',
            'wally113.cpu.user-wally122.cpu.user',
            'wally117.cpu.user-wally122.mem.used',
            'wally122.cpu.user-wally123.mem.used'
        ])

        wrong_gk = WrongGraphArmknowledge(arms, 'unify', 1)

        np.testing.assert_equal(wrong_gk.edges, np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 1, 1, 1],
            [0, 1, 0, 1, 1],
            [0, 1, 1, 0, 1],
            [0, 1, 1, 1, 0]
        ]))

    def test_add_edges_wrong_graph_arm_knowledge(self):
        arms = np.array(['wally113.cpu.user-wally113.mem.used',
                         'wally113.cpu.user-wally117.cpu.user',
                         'wally113.cpu.user-wally122.cpu.user',
                         'wally117.cpu.user-wally122.mem.used',
                         'wally122.cpu.user-wally123.mem.used'])

        wrong_gk = WrongGraphArmknowledge(arms, 'add', 1)

        assert (wrong_gk.get_group_of_arms() == np.array([1, 1, 1, 2, 2])).all() or (
            wrong_gk.get_group_of_arms() == np.array([2, 1, 1, 2, 2])).all()

    def test_remove_edges_wrong_graph_arm_knowledge(self):
        arms = np.array(['wally113.cpu.user-wally113.mem.used',
                         'wally113.cpu.user-wally117.cpu.user',
                         'wally113.cpu.user-wally122.cpu.user',
                         'wally117.cpu.user-wally122.mem.used',
                         'wally122.cpu.user-wally123.mem.used'])

        wrong_gk = WrongGraphArmknowledge(arms, 'remove', 1)

        # only one clique left after removing one edge
        assert np.unique(wrong_gk.get_group_of_arms()).shape[0] == 2

        # no cliques left if we remove atleast 3 edges
        for r in [3, 4]:
            wrong_gk = WrongGraphArmknowledge(arms, 'remove', r)
            np.testing.assert_equal(wrong_gk.edges, np.zeros(shape=(5, 5)))


unittest.main()
