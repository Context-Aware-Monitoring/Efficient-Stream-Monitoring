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


def transform_context_csv(context_filepath, context_transformer, output_path='./test.csv'):
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


def _extract_hosts_from_arm(arm):
    """Computes the hosts of the arm.

    Args:
      arm (string): Name of the arm with the following scheme:
      host1.metric1-host2.metric2
    """
    host_metrics = arm.split('-')
    host1, host2 = host_metrics[0][0:8], host_metrics[1][0:8]

    return host1, host2


def _get_hosts_from_arms(arms):
    """Computes all involved hosts from the arms.

    Args:
      arms (string[]): Name of the arms for which a push shall be computed.

    Returns:
      string[]: Name of the hosts.
    """
    hosts = []
    for arm in arms:
        host1, host2 = _extract_hosts_from_arm(arm)
        hosts.append(host1)
        hosts.append(host2)

    return list(set(hosts))


def _get_appearing_names_from_context_header(context_header):
    """For each element in the context header of a context csv file generated
    by WorkloadExtractor compute the appearing names.

    Args:
      context_header (string[]): Name of the columns in the context csv file.
      Elements have format: host-name

    Returns:
      string[]: The occuring names in the header.
    """
    names = []
    for column_name in context_header:
        name = column_name.split('-')[1]
        names.append(name)

    return list(set(names))


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

    def extract_features_for_events(self, trace_json):
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
                lambda event_json: [
                    {'start': _get_start_as_datetime(event_json), 'stop': _get_stop_as_datetime(
                        event_json)} | self._get_features(event_json)
                ]
            )
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


class StartStopExtractor(AbstractTraceExtractor):
    """Extracts solely the start and stop timestamp of an event.
    """

    def _get_features(self, trace_event):
        return {}

    def get_name(self):
        return 'start-stop'


class NameExtractor(AbstractTraceExtractor):
    """Extracts the name of the event
    """

    def _get_features(self, trace_event):
        return {'name': trace_event['info']['name']}

    def get_name(self):
        if self._extract_stop:
            return 'name-start-stop'
        return 'name-start'


class ServiceExtractor(AbstractTraceExtractor):
    """This class extracts the name of each event.
    """

    def _get_features(self, trace_event):
        return {'name': trace_event['info']['service']}

    def get_name(self):
        if self._extract_stop:
            return 'service-start-stop'
        return 'service-start'


class NumberOfTraces(AbstractTraceExtractor):
    """This extractor extracts a single feature for each element.
    Summing this up leads to the number of traces per timestamp.
    """

    def _get_features(self, trace_event):
        return {'count': 'sum'}

    def get_name(self):
        if self._extract_stop:
            return 'no-traces-start-stop'
        return 'no-traces-start'


class HostExtractor(AbstractTraceExtractor):
    """This extractor extracts the host of the event.
    """

    def _get_features(self, trace_event):
        return {'host': trace_event['info']['host']}

    def get_name(self):
        if self._extract_stop:
            return 'host-start-stop'
        return 'host-traces-start'


class WorkloadExtractor(AbstractTraceExtractor):
    """This extractor extracts the arriving events for computing hosts.
    """

    def _get_features(self, trace_event):
        host = trace_event['info']['host']
        name = trace_event['info']['name']
        features_key = host + '-' + name
        return {
            'event-by-host': features_key
        }

    def get_name(self):
        if self._extract_stop:
            return 'workload-extractor-start-stop'
        return 'workload-extractor-start'


class AbstractContextTransformer(ABC):
    """Base class to transform the rows of a context csv file into the desired
    format.

    Methods
    -------
    transform_context_row(context):
        Transforms the context and returns it.
    get_name()
        Returns the name of the extractor.
    """

    @abstractmethod
    def transform_context_row(self, context):
        """Transforms the provided context into a new format.

        Args:
          context (array): The context array that is read from a context csv
          file.

        Returns:
          array: The transformed context vector that will be written to the
          transformed context csv file.
        """
        return


class PushContextTransformer(AbstractContextTransformer):
    """Reads the context generated by the WorkloadExtractor to determine which
    host is active and pushes relevant arms. Pushes thes arm that lie within an
    active host and further either pushes all arms between active nodes or just
    arms between the control host and active compute hosts.

    The resulting format of the context csv file will look like this:
    arm0, arm1, arm2, ..., armn
    True, False, True, ..., False
    Attributes
    ----------
    _arms: string[]
        The arms of the reward csv file.
    _push_only_controlhost: bool
        If true a push is only performed for arms that lie between control and
        active compute host. If false a push is also performed for arm that lie
        between active compute hosts.
    _hosts: string[]
        Contains the names of the involved hosts.
    _context_index_to_host: string[]
        Maps the index of the column in the context csv file to the name of
        the host on which the event was executed.
    _arm_index_to_hosts: string[][]
        For each index of an arm this array holds the two hosts of the arm.
        The two hosts might be the same host.
    _control_host: string
        The name of the host in self._hosts that is the control host.
    """

    def __init__(self, arms, context_header, push_only_controlhost=False, control_host='wally113'):
        """Constructs the PushContextTransformer.

        Args:
          arms (string[]): Names of the arms. Needs to correspond to the
          position of the arm in the reward.csv file.
          context_header (string[]): Header of the context csv file generated
          by WorkloadExtractor.
          push_only_controlhost (bool): If true only pushes arm thats lie
          between the control host and active compute hosts. Otherwise pushes
          arms that lie between active compute nodes as well.
          control_host (string): Name of the control host
        """
        self._arms = arms
        self._push_only_controlhost = push_only_controlhost
        self._hosts = _get_hosts_from_arms(arms)
        self._context_index_to_host = [
            csv_column_name.split('-')[0] for csv_column_name in context_header
        ]
        self._arm_index_to_hosts = [
             _extract_hosts_from_arm(arm) for arm in self._arms
        ]
        self._control_host = control_host

    def transform_context_row(self, context):
        """Transforms a row of the context csv file generated by the
        WorkloadExtractor to a context that state for each arm and timestamp if
        a ContextualBandit policy should perform a push.

        Args:
          context (int[]): A row from the context csv file generated by
          WorkloadExtractor
        """
        host_active = dict(zip(self._hosts, [False] * len(self._hosts)))

        for i, value in enumerate(context):
            host = self._context_index_to_host[i]

            value = int(value)
            if value > 0:
                host_active[host] = True

        if sum(host_active.values()) > 0:
            host_active[self._control_host] = True

        push = [0] * len(self._arms)

        for i in range(len(self._arms)):
            hosts = self._arm_index_to_hosts[i]
            push[i] = str(int(host_active[hosts[0]] & host_active[hosts[1]]))

        return push


class PushContextDistanceBasedTransformer(PushContextTransformer):
    """Reads the context generated by the WorkloadExtractor to determine which
    host is active and pushes relevant arms. Pushes arms of hosts by computing
    the distance of the context's (events) of the hosts. If the distance is
    smaller than a distance threshold the arm receives a push.


    The resulting format of the context csv file will look like this:
    arm0, arm1, arm2, ..., armn
    True, False, True, ..., False
    Attributes
    ----------
    _push_only_active_hosts (bool): If true, arms of hosts only get pushed if
    atleast one event was processed on that host.
    _distance_threshold (int): For distances of the hosts context that are
    smaller than the threshold respective arms will receive a push.
    _appearing_names_in_context_header (string[]): The context header contains
    names of the different events. This member holds a set of these event
    names.
    _context_index_to_name (string[]): The index of the context array maps to
    an event name. This member holds this name.
    _push_df (DataFrame): Indicies and column names of that DataFrame are the
    names of the hosts, therefore a cell expresses whether arms between column
    and row host should receive a push. This is a quadratic symmetric matrix.
    """

    def __init__(self, arms, context_header, push_only_controlhost=False, control_host='wally113', push_only_active_hosts=False, distance_threshold=100):
        super().__init__(arms, context_header, push_only_controlhost)
        self._push_only_active_hosts = push_only_active_hosts
        self._distance_threshold = distance_threshold
        self._appearing_names_in_context_header = _get_appearing_names_from_context_header(
            context_header)
        self._context_index_to_name = [
            header.split('-')[1] for header in context_header
        ]
        self._push_df = pd.DataFrame(columns=self._hosts, index=self._hosts, data=[
                                     [False] * len(self._hosts) for _ in self._hosts])

    def transform_context_row(self, context):
        context_by_host = {
            host: pd.DataFrame(
                columns=self._appearing_names_in_context_header,
                index=[0],
                data=[[0] * len(self._appearing_names_in_context_header)]
            ) for host in self._hosts
        }

        for col in self._push_df.columns:
            self._push_df[col].values[:] = False

        for i, value in enumerate(context):
            value = int(value)
            host = self._context_index_to_host[i]
            name = self._context_index_to_name[i]

            host_df = context_by_host[host]
            host_df.loc[0, name] = value

        for host1 in self._hosts:
            for host2 in self._hosts:
                push = 0
                if host1 == host2:
                    if host1 == 'wally113':
                        push = 1
                else:
                    distance = np.linalg.norm(
                        context_by_host[host1].loc[0] - context_by_host[host2].loc[0])
                    if distance < self._distance_threshold:
                        push = 1

                self._push_df.loc[host1][host2] = push

        push = [0] * len(self._arms)

        for i in range(len(self._arms)):
            hosts = self._arm_index_to_hosts[i]
            push[i] = str(self._push_df.loc[hosts[0]][hosts[1]])

        return push
