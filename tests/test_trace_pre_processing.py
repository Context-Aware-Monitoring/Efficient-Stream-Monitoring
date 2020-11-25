import unittest
import json

import os
import sys
import pdb

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pre_processing import trace_pre_processing as tpp

class TracePreProcessingTest(unittest.TestCase):
    def setUp(self):
        self.json_file = open('./data/test_trace.json')
        self.trace = json.load(self.json_file)

    def test_get_flat_list(self):
        flat_list = tpp.get_flat_list(self.trace)
        expected_result = [['2', 0], ['3', 1], ['4', 3], ['5', 8]]
        self.assertEqual(flat_list, expected_result)

    def test_get_graph_directed_adjacency_list(self):
        adjacency_list = tpp.get_graph_adjacency_list(self.trace)
        expected_result = [['total', '2'], ['total', '4'], ['total', '5'], ['2', '3']]


        for elem in adjacency_list:
            self.assertIn(elem, expected_result)
            expected_result.remove(elem)

        self.assertEqual(0, len(expected_result))

    def test_get_graph_unidirected_adjacency_list(self):
        adjacency_list = tpp.get_graph_adjacency_list(self.trace, False)
        expected_result = [['total', '2'], ['2', 'total'], ['total', '4'], ['4', 'total'],
                           ['total', '5'], ['5', 'total'], ['2', '3'], ['3', '2']]

        for elem in adjacency_list:
            self.assertIn(elem, expected_result)
            expected_result.remove(elem)

        self.assertEqual(0, len(expected_result))

    def tearDown(self):
        self.json_file.close()


if __name__ == '__main__':
    unittest.main()
