"""This module performs various pre processing operations on the metric data."""
import csv

POS_TIMESTAMP = 0


def get_metrics_on_seconds_interval(metrics_csv_file_path):
    """This method turns the metrics in the csv_file into on that has seconds intervals between the
    different measurements. To do this it averages all the measurements taken in a second. It will
    write a new csv_file at the same location with the suffix seconds.

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

        output_file_name = _get_output_file_name(metrics_csv_file_path)
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


def _get_output_file_name(csv_file_name):
    """This method returns for a path of a csv file the same path but with a filename that is
    suffixed by _seconds.

    Args:
      csv_file_name: Path to a csv file

    Returns:
      The input path where the filename is suffixed by _seconds.
    """
    assert csv_file_name[-4:] == '.csv'
    return csv_file_name[:-4] + '_seconds.csv'
