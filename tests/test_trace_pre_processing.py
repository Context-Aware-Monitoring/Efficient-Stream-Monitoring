import unittest
import json

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pdb

from pre_processing import trace_pre_processing as tpp

class TracePreProcessingTest(unittest.TestCase):
    def setUp(self):
        self.json_file = open('./test_trace.json')
        self.trace = json.load(self.json_file)

    def test_get_flat_list(self):
        flat_list = tpp.get_flat_list(self.trace)
        expected_result = [('2',0), ('3',1), ('4',3), ('5',8)]
        self.assertEqual(flat_list, expected_result)
         
    def test_get_list(self):
        list_of_lists = tpp.get_list(self.trace)
        expected_result = [[('2', 0), [[('3', 1), []]]], [('4', 3), []], [('5', 8), []]]
        self.assertEqual(list_of_lists, expected_result)

    def test_get_graph_directed_adjacency_list(self):
        adjacency_list = tpp.get_graph_adjacency_list(self.trace)
        expected_result = [('1','2'), ('1','4'), ('1','5'),('2','3')]

        self.assertEqual(len(adjacency_list), len(expected_result))
        for elem in expected_result:
            self.assertIn(elem, adjacency_list)
            
    def test_get_graph_unidirected_adjacency_list(self):
        adjacency_list = tpp.get_graph_adjacency_list(self.trace, False)
        expected_result = [('1','2'), ('2', '1'), ('1','4'), ('4','1'),('1','5'), ('5','1'), ('2','3'), ('3', '2')]

        self.assertEqual(len(adjacency_list), len(expected_result))
        for elem in expected_result:
            self.assertIn(elem, adjacency_list)
    
    def tearDown(self):
        self.json_file.close()

if __name__ == '__main__':
    unittest.main()
