import unittest
import os
import sys
import filecmp

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pre_processing import metric_pre_processing as mpp

class MetricPreProcessingTest(unittest.TestCase):
    def test_add_rows(self):
        row1 = ['2019-11-25 15:58:59 CEST', '16.0', '11200000000', '8', '1.99', '1.94', '2.13']
        row2 = ['2019-11-25 15:58:59 CEST', '14.0', '11400000000', '8', '2.01', '1.96', '2.15']
        expected_result = ['2019-11-25 15:58:59 CEST', 30.0, 22600000000.0, 16.0, 4.0, 3.9, 4.28]

        self.assertEqual(expected_result, mpp._add_rows(row1, row2))

    def test_get_average_for_row(self):
        row = ['2019-11-25 15:58:59 CEST', 30.0, 22600000000.0, 16.0, 4.0, 3.9, 4.28]
        expected_result = [
            '2019-11-25 15:58:59 CEST',
            '15.0',
            '11300000000.0',
            '8.0',
            '2.0',
            '1.95',
            '2.14']

        self.assertEqual(expected_result, mpp._get_average_for_row(row, 2))

    def test_get_average_for_row_with_strings_as_elements(self):
        row = ['2019-11-25 15:58:59 CEST', '15.0', '11300000000.0', '8.0', '2.0', '1.95', '2.14']
        expected_result = [
            '2019-11-25 15:58:59 CEST',
            '15.0',
            '11300000000.0',
            '8.0',
            '2.0',
            '1.95',
            '2.14']

        self.assertEqual(expected_result, mpp._get_average_for_row(row, 1))

    def test_get_output_file_name(self):
        self.assertEqual('./test_seconds.csv', mpp._get_output_file_name('./test.csv'))
        self.assertEqual('/this/is/a/absolute/path/to/a/file_seconds.csv',
                         mpp._get_output_file_name('/this/is/a/absolute/path/to/a/file.csv'))

    def test_get_metrics_on_seconds_interval(self):
        mpp.get_metrics_on_seconds_interval('./test_metrics.csv')

        output_file_written = os.path.exists('./test_metrics_seconds.csv')
        self.assertTrue(output_file_written)

        self.assertTrue(filecmp.cmp('./test_metrics_seconds.csv', 'test_metrics_expected.csv'))


if __name__ == '__main__':
    unittest.main()
