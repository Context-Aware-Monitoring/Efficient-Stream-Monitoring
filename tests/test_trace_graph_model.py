import unittest
import json

import os
import sys
import pdb

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from modeling.trace_graph_model import TraceGraphRepresentation, TraceSpanRepresentation

class TraceGraphModelTest(unittest.TestCase):
    def setUp(self):
        self.json_file = open('./data/test_trace.json')
        self.trace = json.load(self.json_file)

    def compare_edges(self, expected_edges, actual_edges):
        edges_dict = actual_edges.to_dict('index')
        for e in edges_dict.values():
            self.assertIn(e, expected_edges)
            expected_edges.remove(e)

        self.assertEqual(0, len(expected_edges))
        
    def test_get_trace_span_representation(self):
        tsr = TraceSpanRepresentation(self.trace)
        nodes = tsr.get_nodes()
        edges = tsr.get_edges()
        
        expected_indicies = ['total-start', 'total-stop', '2-start', '2-stop', '3-start', '3-stop', '4-start', '4-stop', '5-start', '5-stop']
        actual_indicies = nodes.index
        
        self.assertCountEqual(expected_indicies, actual_indicies)

        expected_edges = [
            {'source' : 'total-start', 'target' : 'total-stop', 'weight' : 11},
            {'source' : 'total-start', 'target' : '2-start', 'weight' : 0},
            {'source' : 'total-start', 'target' : '4-start', 'weight' : 3},
            {'source' : 'total-start', 'target' : '5-start', 'weight' : 8},
            {'source' : '2-stop', 'target' : 'total-stop', 'weight' : 0},
            {'source' : '4-stop', 'target' : 'total-stop', 'weight' : 0},
            {'source' : '5-stop', 'target' : 'total-stop', 'weight' : 0},
            {'source' : '2-start', 'target': '2-stop', 'weight' : 2},
            {'source' : '2-start', 'target': '3-start', 'weight' : 1},
            {'source' : '3-start', 'target': '3-stop', 'weight' : 1},
            {'source' : '3-stop', 'target': '2-stop', 'weight' : 0},
            {'source' : '4-start', 'target': '4-stop', 'weight' : 4},
            {'source' : '5-start', 'target': '5-stop', 'weight' : 3}
        ]

        self.compare_edges(expected_edges, edges)

    def test_trace_graph_representation(self):
        tgr = TraceGraphRepresentation(self.trace)
        nodes = tgr.get_nodes()
        edges = tgr.get_edges()

        expected_indicies = ['total', '2', '3', '4', '5']
        actual_indicies = nodes.index

        self.assertCountEqual(expected_indicies, actual_indicies)
        
        expected_edges = [
            {'source' : 'total', 'target' : '2', 'weight' : 0},
            {'source' : 'total', 'target' : '4', 'weight' : 3},
            {'source' : 'total', 'target' : '5', 'weight' : 8},
            {'source' : '2', 'target': '3', 'weight' : 1}
        ]

        self.compare_edges(expected_edges, edges)
            
    def tearDown(self):
        self.json_file.close()

if __name__ == '__main__':
    unittest.main()
        
