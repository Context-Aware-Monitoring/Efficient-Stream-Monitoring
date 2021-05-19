"""This module computes some basic properties of traces."""

import os
import sys
import json
import context

sys.path.append("./features")


def get_number_of_parents(trace_json):
    """Returns the number of parents. The number of parents is the number of
    events on the first level.

    Args:
      trace_json (json): Json representing a trace

    Returns:
      Number of parents for the trace.
    """
    return len(trace_json.get('children'))


def get_number_of_children(trace_json):
    """Returns the number of children, meaning the amount of total events minus
    the number of parents

    Args:
      trace_json (json): Json representing a trace

    Returns:
      Number of children, equates to the total number of events minus the
      number of parents.
    """
    children = 0
    for parent in trace_json['children']:
        children += _count_number_of_children_recursively(parent)

    return children


def _count_number_of_children_recursively(event):
    """Recursively steps down the children of an event to calculate the number
    of children.

    Args:
      event (json): Json representing the current event.

    Returns:
      The number of children of the current event.
"""
    if len(event['children']) == 0:
        return 0

    children = 0
    for child in event['children']:
        children += 1 + _count_number_of_children_recursively(child)

    return children


def get_depths_per_parent(trace_json):
    """Returns an array representing the depth for each of the parents. The
    depth for an event is the largest number of possible successive descends
    into the children field.

    Args:
      trace_json (json): Json representing the trace.

    Returns:
      An array representing the depth for each of the parents.
    """
    depths = []

    for parent in trace_json['children']:
        depths.append(_compute_depth_recursively(parent))

    return depths


def _compute_depth_recursively(event):
    """Return the depth of the current event. The depth is the largest number
    of possible successive descend into the children field.

    Args:
      event (json): Json representing the current event.

    Returns:
      The depth of the current event.
    """
    if event['children'] == []:
        return 1

    return 1 + max([_compute_depth_recursively(child)
                   for child in event['children']])


def get_depth(trace_json):
    """Returns the deepest level of the trace. This equates to the number of
    possible successive descends into the children field.

    Args:
      trace_json (json): Json representing the trace

    Returns:
      The deepest level of the trace.
"""
    return max(get_depths_per_parent(trace_json))


def get_average_depth(trace_json):
    """Returns the average of the depths for each of the parents. The depth is
    the number of possible successive descends into the children field.

    Args:
      trace_json (json): Json representing the trace

    Returns: The average depth of the parents.
    """
    return sum(
        get_depths_per_parent(trace_json)) / get_number_of_parents(trace_json)


def print_time_window_for_traces(paths):
    """Prints the time window in wich the events if the traces in the paths
    occur.

    Args:
      paths (string[]): Path to directory where traces are located
    """
    file_paths = []
    for p in paths:
        file_paths.extend(list(map(lambda x: p + x, os.listdir(p))))

    start = None
    stop = None
    print("Will process %d traces" % len(file_paths))
    extractor = context.StartStopExtractor()
    for tp in file_paths:
        with open(tp) as f:
            trace_json = json.load(f)
            try:
                start_stop_times = extractor.extract_features_for_events(
                    trace_json)
            except KeyError:
                print("Skipped trace")
                continue
            for sst in start_stop_times:
                start = min(
                    sst['start'],
                    start) if start is not None else sst['start']
                stop = max(sst['stop'], stop) if stop is not None else sst['stop']

    print("Start: %s" % str(start))
    print("Stop: %s" % str(stop))


if __name__ == "__main__":
    print_time_window_for_traces(
        ['../data/raw/sequential_data/traces/boot_delete/'])
    print_time_window_for_traces(
        ['../data/raw/sequential_data/traces/image_create_delete/'])
    print_time_window_for_traces(
        ['../data/raw/sequential_data/traces/network_create_delete/'])
