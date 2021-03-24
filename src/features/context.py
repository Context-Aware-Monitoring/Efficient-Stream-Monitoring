"""Generates context csv files that can be used by a contextual bandit
algorithm
"""
import os
from datetime import datetime, timedelta
import json
from abc import ABC, abstractmethod
import pandas as pd
import trace_pre_processing as tpp

def generate_context_csv(
        event_extractor,
        paths,
        start=datetime(2019, 11, 19, 18, 38, 39),
        end=datetime(2019, 11, 20, 1, 30, 00),
        window_size=30,
        step=5,
        outdir='./'
):
    """Writes a context.csv file that can be processed by a contextual bandit
    algorithm. It extracts features for each event in the passed traces.
    Afterwards in aggregates the features using a sliding window appraoch
    and writes the file to a desired location.

    Args:
      event_extractor (AbstractTraceExtractor): Used to extract the features
      from individual events.
      paths (string[]): Paths to the traces that will be used to generate the
      file.
      start (datetime): Timestamp of the start for the context file. Only
      events occuring between start and end will be used.
      end (datetime): Timestamp of the end for the context file.
      window_size (int): Size of the sliding window.
      step (int): Step of the sliding window.
      outdir (string): Directory where to write the generated file.

    Returns:
      String: Full path of the written file
    """
    stime = datetime.now()
    print('Generating context.csv...')

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

    for current_fp in file_paths:
        with open(current_fp) as open_file:
            print("Processing " + current_fp)
            trace_json = json.load(open_file)
            try:
                event_features = event_extractor.extract_features_for_events(
                    trace_json)
            except KeyError:
                print('Skipped trace %s' % current_fp)
                i = i + 1
                continue

            _insert_event_features_into_context_df(event_features, context_df)
            i = i + 1

    windowed_context = _compute_windowed_context_df(
        context_df,
        window_size,
        step
    )

    filepath = _generate_context_filepath(
        outdir, event_extractor.get_name(), paths, start, end, window_size, step)
    windowed_context.to_csv(filepath, index=False)
    print('Generation finished, took %d seconds' %
          ((datetime.now() - stime).seconds))
    print('Wrote file %s' % filepath)

    return filepath

def _insert_event_features_into_context_df(event_features, context_df):
    """Processes the features generated from events and inserts them into the
    right location in the context DataFrame.

    Args:
      event_features (array): Array of dicts, where the key corresponds to the
      feature name and the value to the feature value.
      context_df (DataFrame): Features will be inserted into this DataFrame.
    """
    start = context_df.index[0]
    end = context_df.index[-1]
    for event_feature in event_features:
        if event_feature['start'] > end or (event_feature['stop'] is not None and event_feature['stop'] < start):
            continue

        for feature_key, feature_value in event_feature.items():
            if feature_key in ('start', 'stop'):
                continue

            if feature_value in context_df.columns:
                if event_feature['stop'] is not None:
                    context_df.loc[pd.date_range(
                        max(start, event_feature['start']), min(end, event_feature['stop'])), feature_value] += 1
                elif event_feature['start'] >= start:
                    context_df.loc[event_feature['start']] += 1
            else:
                context_df[feature_value] = 0
                if event_feature['stop'] is not None:
                    context_df.loc[pd.date_range(
                        max(start, event_feature['start']),
                        min(end, event_feature['stop'])
                    ), feature_value] = 1
                elif event_feature['start'] >= start:
                    context_df.loc[event_feature['start']] = 1


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

def _generate_context_filepath(outdir, kind, paths, start, end, window_size, step):
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
    start = start.strftime('%Y%m%d-%H%M%S')
    end = end.strftime('%Y%m%d-%H%M%S')

    path_representations = []
    for current_path in paths:
        path_r = ''
        if 'sequential' in current_path:
            path_r += 's'
        else:
            path_r += 'c'

        if 'boot_delete' in current_path:
            path_r += 'bd'
        elif 'image_create_delete' in current_path:
            path_r += 'icd'
        else:
            path_r += 'ncd'
        path_representations.append(path_r)

    path_representations = sorted(path_representations)

    return outdir + 'context_' + kind + '_s' + start + '_e' + end + '_' + '-'.join(path_representations) + '_w' + str(window_size) + '_s' + str(step) + '.csv'

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

    return datetime.strptime(date_and_time_string, '%Y-%m-%d %H:%M:%S')

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
    def __init__(self, extract_stop=True):
        self._extract_stop = extract_stop

    def extract_features_for_events(self,trace_json):
        """Returns the features for each event in the trace

        Args:
            trace_json (json): Json representing the trace.

        Returns:
            array: Returns an array containing dicts. For each event in the
            trace there will be one dictionary in the array. The dictionary
            contains the features the extractor extracted.
        """
        if self._extract_stop:
            return tpp.get_flat_list(
                trace_json,
                lambda event_json : [
                    {'start': _get_start_as_datetime(event_json), 'stop': _get_stop_as_datetime(event_json)} | self._get_features(event_json)
                ]
            )
        return tpp.get_flat_list(
            trace_json,
            lambda event_json : [
                {'start': _get_start_as_datetime(event_json)} | self._get_features(event_json)
            ]
        )

    @abstractmethod
    def _get_features(self,trace_event):
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

class StartStopExtractor(AbstractTraceExtractor):
    """Extracts solely the start and stop timestamp of an event.
    """
    def _get_features(self,trace_event):
        return {}

    def get_name(self):
        return 'start-stop'

class NameExtractor(AbstractTraceExtractor):
    """Extracts the name of the event
    """
    def _get_features(self,trace_event):
        return {'name': trace_event['info']['name']}

    def get_name(self):
        if self._extract_stop:
            return 'name-start-stop'
        return 'name-start'

class ServiceExtractor(AbstractTraceExtractor):
    """This class extracts the name of each event.
    """
    def _get_features(self,trace_event):
        return {'name': trace_event['info']['service']}

    def get_name(self):
        if self._extract_stop:
            return 'service-start-stop'
        return 'service-start'

class NumberOfTraces(AbstractTraceExtractor):
    """This extractor extracts a single feature for each element.
    Summing this up leads to the number of traces per timestamp.
    """
    def _get_features(self,trace_event):
        return {'count': 'sum'}

    def get_name(self):
        if self._extract_stop:
            return 'no-traces-start-stop'
        return 'no-traces-start'

class HostExtractor(AbstractTraceExtractor):
    """This extractor extracts the host of the event.
    """
    def _get_features(self,trace_event):
        return {'host': trace_event['info']['host']}

    def get_name(self):
        if self._extract_stop:
            return 'host-start-stop'
        return 'host-traces-start'
