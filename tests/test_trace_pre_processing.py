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
        expected_result = ['2', '3', '4', '5']
        self.assertEqual(flat_list, expected_result)

    def test_get_flat_list_with_multiple_attributes_collected_per_event(self):
        flat_list = tpp.get_flat_list(self.trace, lambda x: [[x['trace_id'], x['info']['name']]])
        expected_result = [['2', 'wsgi'], ['3', 'db'], ['4', 'db'], ['5', 'db']]
        self.assertEqual(flat_list, expected_result)

    def test_get_flat_list_with_multiple_objects_collected_per_event(self):
        flat_list = tpp.get_flat_list(self.trace, lambda x: [x['trace_id'] + '-first', x['trace_id'] + '-second'])
        expected_result = ['2-first', '2-second', '3-first', '3-second', '4-first', '4-second', '5-first', '5-second']

        self.assertEqual(flat_list, expected_result)

    def test_get_flat_list_with_multiple_objects_and_multiple_attributes(self):
        flat_list = tpp.get_flat_list(self.trace, lambda x: [[x['trace_id']+'-first', 'second attribute'], [x['trace_id'] + '-second', 'second attribute']])
        expected_result = [['2-first', 'second attribute'], ['2-second', 'second attribute'], ['3-first', 'second attribute'], ['3-second', 'second attribute'], ['4-first', 'second attribute'], ['4-second', 'second attribute'], ['5-first', 'second attribute'], ['5-second', 'second attribute']]
        self.assertEquals(flat_list, expected_result)
                                      
        
    def test_get_list(self):
        list_of_lists = tpp.get_list(self.trace)
        expected_result = [
            ['2',
             [
                 ['3', []]
             ]
            ],
            ['4', []],
            ['5', []]
        ]
        self.assertEqual(list_of_lists, expected_result)
        

    def test_get_list_with_multiple_attributes_collected_per_event(self):
        list_of_lists = tpp.get_list(self.trace, lambda x: (x['trace_id'], x['info']['name']))
        expected_result = [
            [('2', 'wsgi'),
             [
                 [('3', 'db'), []]
             ]
            ],
            [('4', 'db'), []],
            [('5', 'db'), []]
        ]

    def test_get_number_of_parents(self):
        self.assertEqual(tpp.get_number_of_parents(self.trace), 4)

    def test_get_number_of_children(self):
        self.assertEqual(tpp.get_number_of_children(self.trace), 1)
        
    def tearDown(self):
        self.json_file.close()


if __name__ == '__main__':
    unittest.main()
