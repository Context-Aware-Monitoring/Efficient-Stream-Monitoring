"""This module performs various pre processing operations on the metric data."""
import csv
from datetime import datetime, timedelta
import pdb

POS_TIMESTAMP = 0 

def fill_metrics_missing_seconds_using_linear_interpolation(metrics_csv_file_path):
    """This method fills the missing second intervals in the provided csv file by performing linear
    interpolation. It writes a new csv file at the same location with the suffix '_filled'.

    Args:
      metrics_csv_file_path: The path to the csv file containing the metrics data.
    Returns:
      None
    """
    rows = list()
    with open(metrics_csv_file_path) as metrics_csv_file:
        csv_reader = csv.reader(metrics_csv_file, delimiter=',')
        header = None
        for row in csv_reader:
            if header == None:
                header = row
                continue
            if len(rows) == 0:
                rows.append(row)
            else:
                prev_timestamp = _convert_timestamp_string_to_datetime(rows[-1][POS_TIMESTAMP])
                current_timestamp = _convert_timestamp_string_to_datetime(row[POS_TIMESTAMP])

                if (prev_timestamp + timedelta(seconds=1)) == current_timestamp:
                    rows.append(row)
                else:
                    rows.extend(list(_get_interpolated_rows(rows[-1], row)))
                    rows.append(row)

    output_file_name = _get_output_file_name_with_suffix(metrics_csv_file_path, '_filled')
    with open(output_file_name, 'w') as output_file:
        output_file.write(','.join(header))
        output_file.write('\n')
        for row in rows:
            output_file.write(','.join(row))
            output_file.write('\n')
        


def _get_interpolated_rows(row1, row2):
    """Performs linear interpolation for the values inside the passed rows.

    Args:
      row1 (array): Array with values containing the earlier timestamp.
      row2 (array): Array with values containing the later timestamp.

    Returns:
      Array of arrays of size n, where n is the number of seconds between the two timestamps. Each array contains the interpolated values for a timestamp.
    """
    ts1 = _convert_timestamp_string_to_datetime(row1[POS_TIMESTAMP])
    ts2 = _convert_timestamp_string_to_datetime(row2[POS_TIMESTAMP])
    diff_seconds = (ts2 - ts1).seconds

    interpolated_rows = []
    slope = [0] * len(row1)

    for i,_ in enumerate(slope):
        if i == POS_TIMESTAMP:
            continue
        
        diff_values = float(row2[i]) - float(row1[i])
        slope[i] = round(diff_values / diff_seconds, 2)

    for s in range(1, diff_seconds):
        interpolated_rows.append([0] * len(row1))
        
        for i,_ in enumerate(row1):
            if i == POS_TIMESTAMP:
                interpolated_rows[-1][i] = _convert_datetime_to_string_timestamp(ts1 + timedelta(seconds=s))
                continue
            
            interpolated_rows[-1][i] = str(round((float(row1[i]) + slope[i] * s), 2))

    return interpolated_rows
            

def _convert_datetime_to_string_timestamp(datetime):
    """Converts the passed datetime to a string.

    Args:
      datetime (datetime): Timestamp as datetime object.
    Returns:
      String representation of the datetime object.
    """
    return datetime.strftime('%Y-%m-%d %H:%M:%S') + ' CEST'
    
def _convert_timestamp_string_to_datetime(timestamp):
    """Converts the passed string to a datetime object.
    
    Args:
      timestamp (string): Timestamp as string
    Returns:
      Timestamp as datetime objects.
    """
    timestamp = timestamp[:-5] # remove the ' CEST'
    return datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')

def get_metrics_on_seconds_interval(metrics_csv_file_path):
    """This method turns the metrics in the csv_file into on that has seconds intervals between the
    different measurements. To do this it averages all the measurements taken in a second. It will
    write a new csv file at the same location with the suffix '_seconds'.

    Args:
      metrics_csv_file_path: The path to the csv file containing the metrics data.
    Returns:
      None
    """
    with open(metrics_csv_file_path) as metrics_csv_file:
        csv_reader = csv.reader(metrics_csv_file, delimiter=',')
        line_count = 0
        output_lines = []
        aggregated_values = []
        values_per_timestamp = 1
        for row in csv_reader:
            if line_count == 0:
                output_lines.append(row)
            else:
                if len(aggregated_values) == 0:
                    aggregated_values = row
                else:
                    if aggregated_values[POS_TIMESTAMP] == row[POS_TIMESTAMP]:
                        aggregated_values = _add_rows(aggregated_values, row)
                        values_per_timestamp += 1
                    else:
                        output_lines.append(
                            _get_average_for_row(
                                aggregated_values,
                                values_per_timestamp))
                        aggregated_values = row
                        values_per_timestamp = 1
            line_count += 1

        output_lines.append(_get_average_for_row(aggregated_values, values_per_timestamp))

        output_file_name = _get_output_file_name_with_suffix(metrics_csv_file_path, '_seconds')
        with open(output_file_name, 'w') as output_file:
            for line in output_lines:
                output_file.write(','.join(line))
                output_file.write('\n')


def _add_rows(row1, row2):
    """This method performs an element-wise addition of the rows, except for the timestamp element.
    Rows need to have the same timestamp and the same length.

    Args:
      row1: List of values
      row2: List of values

    Returns:
      A list containing the element wise sum, except for the timestamp element which stays the same.
    """
    output = []
    assert len(row1) == len(row2)
    assert row1[POS_TIMESTAMP] == row2[POS_TIMESTAMP]
    for i, _ in enumerate(row1):
        if i == POS_TIMESTAMP:
            output.append(row1[i])
        else:
            output.append(round(float(row1[i]) + float(row2[i]), 2))

    return output


def _get_average_for_row(row, divisor):
    """This method performs element-wise divison for the passed list, except for the element that
    is the timestamp.

    Args:
      row: List of elements
      divisor: Divisor of the divison

    Returns:
      A list containing the element-wise quotient, except for the timestamp element which stays the
      same.
    """
    output = []
    for i, _ in enumerate(row):
        if i == POS_TIMESTAMP:
            output.append(row[i])
        else:
            output.append(str(round(float(row[i]) / divisor, 2)))

    return output


def _get_output_file_name_with_suffix(csv_file_name, suffix):
    """This method returns for a path of a csv file the same path but with a filename that is
    suffixed by the passed suffix.

    Args:
      csv_file_name: Path to a csv file

    Returns:
      The input path where the filename is suffixed by the passed suffix.
    """
    assert csv_file_name[-4:] == '.csv'
    return csv_file_name[:-4] + suffix + '.csv'
