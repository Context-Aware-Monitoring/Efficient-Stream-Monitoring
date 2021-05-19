"""Generates context csv files that can be used by a contextual bandit
algorithm
"""
import os
import csv
from datetime import datetime, timedelta
import json
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import trace_pre_processing as tpp

start_sequential = np.datetime64('2019-11-19 17:38:49')
end_sequential = np.datetime64('2019-11-20 01:30:00')

start_concurrent = np.datetime64('2019-11-19 15:12:13')
end_concurrent = np.datetime64('2019-11-20 19:45:00')

def generate_context_csv(
        event_extractor,
        paths,
        seq=True,
        window_size=30,
        step=5,
        columns=None,
        outdir='./'
):
    """Writes a context.csv file that can be processed by a contextual bandit
    algorithm. It extracts features for each event in the passed traces.
    Afterwards in aggregates the features using a sliding window approach
    and writes the file to a desired location.

    Args:
      event_extractor (AbstractTraceExtractor): Used to extract the features
      from individual events.
      paths (string[]): Paths to the traces that will be used to generate the
      file.
      start (np.datetime64): Timestamp of the start for the context file. Only
      events occuring between start and end will be used.
      end (np.datetime64): Timestamp of the end for the context file.
      window_size (int): Size of the sliding window.
      step (int): Step of the sliding window.
      columns (string[]): If not None, thae resulting DataFrame will have these
      columns.
      outdir (string): Directory where to write the generated file.

    Returns:
      String: Full path of the written file
    """
    stime = datetime.now()
    print('Generating context.csv...')

    start = start_sequential if seq else start_concurrent
    end = end_sequential if seq else end_concurrent

    start = pd.to_datetime(start)
    end = pd.to_datetime(end)

    context_df = pd.DataFrame(index=pd.date_range(
        start=start, end=end, freq='1S'))

    file_paths = []
    current_path = ''
    for current_path in paths:
        file_paths.extend(
            list(map(lambda x: current_path + x, os.listdir(current_path))))

    print('Total of %d traces will be used to generate the context.csv' %
          len(file_paths))
    i = 0

    context_df_columns_set = set()
    for current_fp in file_paths:
        with open(current_fp) as open_file:
            trace_json = json.load(open_file)
            try:
                event_features = event_extractor.extract_features_for_events(
                    trace_json)
            except KeyError:
                print('Skipped trace %s' % current_fp)
                i = i + 1
                continue

            trace_df = pd.pivot_table(
                pd.DataFrame(data=event_features),
                index=['start'],
                aggfunc=np.sum)
            mask = (trace_df.index >= pd.to_datetime(start)) & (
                trace_df.index <= pd.to_datetime(end))

            if len(trace_df[mask]) == 0:
                print('Found trace with events outside time window')
                continue

            non_existing_columns = set(
                trace_df.columns.values) - context_df_columns_set

            if len(non_existing_columns) > 0:
                context_df[list(non_existing_columns)] = np.zeros(len(context_df.index) * len(non_existing_columns)).reshape(-1, len(non_existing_columns))
                context_df_columns_set |= non_existing_columns

            context_df.loc[trace_df.index.values[mask],
                           trace_df.columns.values] += trace_df.values[mask]

            #_insert_event_features_into_context_df(event_features, context_df)
            i = i + 1

    windowed_context = context_df.rolling(window=window_size).sum().loc[pd.date_range(
        start=start + np.timedelta64(window_size - 1, 's'), end=end, freq='%dS' % step)].fillna(0.0)

    filepath = _generate_context_filepath(
        outdir, event_extractor.get_name(), seq, window_size, step)

    if columns is not None:
        cols_diff = set(columns) - set(windowed_context.columns.values)
        windowed_context.loc[windowed_context.index, list(cols_diff)] = 0.0

    windowed_context.to_csv(filepath, index=True)
    print('Generation finished, took %d seconds' %
          ((datetime.now() - stime).seconds))
    print('Wrote file %s' % filepath)

    return filepath


def transform_context_csv(
        context_filepath, context_transformer, output_path='./test.csv'):
    """Reads in the context csv file and transform it using the
    context_transformer.
    """
    stime = datetime.now()
    print("Transform context")
    with open(context_filepath) as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        with open(output_path, mode='w') as output_file:
            csv_writer = csv.writer(output_file, delimiter=',')
            for row in reader:
                outrow = context_transformer.transform_context_row(row)
                csv_writer.writerow(outrow)
    print('Transformation finished, took %d seconds' %
          ((datetime.now() - stime).seconds))
    print('Wrote file %s' % output_path)


def _compute_windowed_context_df(context_df, window_size, step):
    """Sums up the features in the context DataFrame using a sliding window
    approach.

    Args:
      context_df (DataFrame): DataFrame where rows correspond to timestamps and
      columns correspond to features.
      window_size (int): Size of the sliding window
      step (int): Step of the sliding window

    Returns:
      DataFrame: The windowed context DataFrame
    """
    windowed_context = pd.DataFrame(columns=context_df.columns)
    i = 0

    timedelta_window = timedelta(seconds=window_size - 1)
    cur = context_df.index[0]
    end = context_df.index[-1]
    while(cur + timedelta_window) < end:
        window = context_df.loc[pd.date_range(
            start=cur, end=(cur+timedelta_window), freq='1S')]
        windowed_context.loc[i] = window.sum()
        i += 1
        cur += timedelta(seconds=step)

    return windowed_context


def _generate_context_filepath(outdir, kind, seq, window_size, step):
    """Generates the fullpath of the output file. The name contains information
    on how the context was created.

    Args:
      outdir (string): The output directory
      kind (string): Identifier for how the context was extracted from the
      traces.
      start (datetime): Start of the context data
      end (datetime): End of the context data
      window_size (int): Size of the sliding window
      step (int): Step of the sliding window

    Returns:
      String: The path of the output file
    """
    seq_or_con = 'seq' if seq else 'con'

    return outdir + seq_or_con + '_context_' + kind + '_w' + str(window_size) + '_s' + str(step) + '.csv'


def _get_start_as_datetime(event_json):
    """This method reads the start timestamp of the event and returns it as
    a datetime object.

    Args:
        event_json (json): The event encapsulated as json.
    Returns
        datetime: Timestamp of the start of the event.
    """
    name = event_json['info']['name']
    payload_start = 'meta.raw_payload.' + name + '-start'

    start_timestamp_string = event_json['info'][payload_start]['timestamp']

    start_date_string, start_time_string = start_timestamp_string.split('T')
    start_time_string, _ = start_time_string.split('.')

    date_and_time_string = start_date_string + ' ' + start_time_string

    # datetime.strptime(date_and_time_string, '%Y-%m-%d %H:%M:%S')
    return np.datetime64(date_and_time_string)


def _get_stop_as_datetime(event_json):
    """Reads the stop timestamp of the event and returns it as a datetime
    object.

    Args:
        event_json (json): The event encapsulated as json.
    Returns
        datetime: Timestamp of the stop of the event.
    """
    name = event_json['info']['name']
    payload_stop = 'meta.raw_payload.' + name + '-stop'

    stop_timestamp_string = event_json['info'][payload_stop]['timestamp']

    stop_date_string, stop_time_string = stop_timestamp_string.split('T')
    stop_time_string, _ = stop_time_string.split('.')

    date_and_time_string = stop_date_string + ' ' + stop_time_string

    return datetime.strptime(date_and_time_string, '%Y-%m-%d %H:%M:%S')


class AbstractTraceExtractor(ABC):
    """This is the base class to extract features from events in traces.
    It extracts the start and if desirable the stop timestamp as well.
    Children must implement the _get_features method, which extracts features
    for the current event.

    Attributes
    ----------
    _extract_stop: bool
        Whether the extractor extracts the stop timestamp from the event

    Methods
    -------
    extract_features_for_events(trace_json)
        Extracts the features for each event in the given trace. The trace is
        provided as json. The method returns an array of dicts.
    get_name()
        Returns the name of the extractor.
    """

    def extract_features_for_events(self, trace_json):
        """Returns the features for each event in the trace

        Args:
            trace_json (json): Json representing the trace.

        Returns:
            array: Returns an array containing dicts. For each event in the
            trace there will be one dictionary in the array. The dictionary
            contains the features the extractor extracted.
        """
        return tpp.get_flat_list(
            trace_json,
            lambda event_json: [
                {'start': _get_start_as_datetime(
                    event_json)} | self._get_features(event_json)
            ]
        )

    @abstractmethod
    def _get_features(self, trace_event):
        """Abstract method to do the actual extraction of features from events.

        Args:
            trace_event (json): Json representing the current event.

        Returns:
            dict: Dictionary containing the features.
        """
        return

    @abstractmethod
    def get_name(self):
        """Returns the name of the extractor.

        Returns:
            string: Name of the extractor
        """
        return


class HostExtractor(AbstractTraceExtractor):
    """This extractor extracts the host of the event.
    """

    def _get_features(self, trace_event):
        return {trace_event['info']['host']: 1}

    def get_name(self):
        return 'host-traces'


class WorkloadExtractor(AbstractTraceExtractor):
    """This extractor extracts the arriving events for computing hosts.
    """

    def _get_features(self, trace_event):
        host = trace_event['info']['host']
        name = trace_event['info']['name']
        return {
            '%s-%s' % (host, name): 1
        }

    def get_name(self):
        return 'workload-extractor'
