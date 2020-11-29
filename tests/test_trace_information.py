import unittest
import json

import os
import sys
import pdb

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import trace_information as ti

class TraceInformationTest(unittest.TestCase):
    def setUp(self):
        self.json_file = open('./data/test_trace.json')
        self.trace = json.load(self.json_file)

    def test_get_number_of_parents(self):
        self.assertEqual(ti.get_number_of_parents(self.trace), 3)

    def test_get_number_of_children(self):
        self.assertEqual(ti.get_number_of_children(self.trace), 1)

    def test_get_depths_per_parent(self):
        self.assertEqual(ti.get_depths_per_parent(self.trace), [2,1,1])
        
    def test_get_depth(self):
        self.assertEqual(ti.get_depth(self.trace), 2)

    def test_get_average_depth(self):
        self.assertAlmostEqual(ti.get_average_depth(self.trace), 4.0/3)
        
    def tearDown(self):
        self.json_file.close()


if __name__ == '__main__':
    unittest.main()
