"""This module turns traces from the json representation into various different representations."""


def collect_id(event):
    """Collects the trace_id from the event.

    Args:
      event (json): Json representing an event of a trace.

    Returns:
      The trace_id of the event
    """
    return event.get('trace_id')


def collect_id_as_array(event):
    """Collects the trace_id from the event.

    Args:
      event (json): Json representing an event of a trace.

    Returns:
      The trace_id of the event.
    """
    return [event.get('trace_id')]


def get_flat_list(json, collect_function=collect_id_as_array):
    """Transforms the trace from the json to the flat list representation.

    Args:
      json: Json representation of trace

    Returns:
      A flat list representation of the trace
    """
    events = list()
    for child in json['children']:
        extract_events_flat(child, collect_function, events)

    return events


def extract_events_flat(json_node, collect_function=collect_id, events=[]):
    """Depth-first search of events returning them as a flat list.

    Args:
      json_node: Json representation of the current events.
      collect_function: Function that collects the desired data for events. Make sure that this
      returns an array.
      events: Earlier traversed events

    Returns:
      A flat list representation that contains all the events traversed by the depth-first search.
    """
    if len(json_node) == 0:
        return events

    events.extend(list(collect_function(json_node)))
    for child in json_node['children']:
        extract_events_flat(child, collect_function, events)

    return events


def get_list(json, collect_function=collect_id):
    """Transforms the trace from the json to the list of lists representation, therefore implicitly
    saving the parent-child relationship between events.

    Args:
      json: Json representation of trace

    Returns:
      A list representation where the children of an event are saved in a list.
    """

    return list([extract_events(c, collect_function) for c in json['children']])


def extract_events(json_node, collect_function):
    """Depth-first search of events returning them as a list of lists, therefore implicitly saving
    the parent-child relationship between events.
    Args:
      json_node: Json representation of the current events.
      collect_function: Function that collects the desired data for events.
    Returns:
      A list of lists representation that contains all the events traversed by the depth-first
      search.
    """
    if len(json_node) == 0:
        return []

    extracted_events = list()
    child_events = json_node['children']

    if child_events == []:
        extracted_events.extend([collect_function(json_node), []])
    else:
        extracted_events.extend([collect_function(json_node), [
                                extract_events(ce, collect_function) for ce in child_events]])

    return extracted_events

    
