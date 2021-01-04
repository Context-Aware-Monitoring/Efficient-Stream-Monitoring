import unittest
import os
import sys
import filecmp
import pdb

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

    def test_get_output_file_name_with_suffix(self):
        self.assertEqual('./test_suffix.csv', mpp._get_output_file_name_with_suffix('./test.csv', '_suffix'))
        self.assertEqual('/this/is/a/absolute/path/to/a/file_suffix.csv',
                         mpp._get_output_file_name_with_suffix('/this/is/a/absolute/path/to/a/file.csv', '_suffix'))


    def test_get_interpolated_rows_1_second_gap(self):
        row1 = ['2019-11-19 16:56:32 CEST','12','10000000000','8','0.8','1.02','1.18']
        row2 = ['2019-11-19 16:56:34 CEST','10','11000000000','8','0.8','1.00','1.20']

        expected = [['2019-11-19 16:56:33 CEST', '11.0', '10500000000.0', '8.0', '0.8', '1.01', '1.19']]
        actual = mpp._get_interpolated_rows(row1, row2)

        self.assertEqual(expected, actual)

    def test_get_interpolated_rows_2_second_gap(self):
        row1 = ['2019-11-19 16:56:32 CEST','12','10000000000','8','0.8','1.02','1.18']
        row2 = ['2019-11-19 16:56:35 CEST','9','11500000000','8','1.1','1.02','1.21']

        expected = [
            ['2019-11-19 16:56:33 CEST', '11.0', '10500000000.0', '8.0', '0.9', '1.02', '1.19'],
            ['2019-11-19 16:56:34 CEST', '10.0', '11000000000.0', '8.0', '1.0', '1.02', '1.20']            
        ]
        actual = mpp._get_interpolated_rows(row1, row2)

        
    def test_get_metrics_on_seconds_interval(self):
        mpp.get_metrics_on_seconds_interval('./data/test_metrics.csv')

        output_file_written = os.path.exists('./data/test_metrics_seconds.csv')

        self.assertTrue(output_file_written)
        self.assertTrue(filecmp.cmp('./data/test_metrics_seconds.csv', './data/test_metrics_seconds_expected.csv'))
        os.remove('./data/test_metrics_seconds.csv')
        
    def test_fill_metrics_missing_seconds_using_linear_interpolation(self):
        mpp.fill_metrics_missing_seconds_using_linear_interpolation('./data/test_metrics2.csv')

        output_file_written = os.path.exists('./data/test_metrics2_filled.csv')
        self.assertTrue(output_file_written)
        self.assertTrue(filecmp.cmp('./data/test_metrics2_filled.csv', './data/test_metrics2_expected.csv'))
        # os.remove('./data/test_metrics_seconds.csv')

if __name__ == '__main__':
    unittest.main()
