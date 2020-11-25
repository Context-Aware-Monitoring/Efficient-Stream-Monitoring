import pdb
"""This module turns traces from the json representation into various different representations."""
def collect_id(event):
    trace_id = event.get('trace_id')

    if trace_id == None:
        return [event['name'] + '-start', event['name'] + '-stop']
    return [trace_id + '-start', trace_id + '-stop']


def get_node_columns():
    return ['name', 'service', 'project', 'host', 'payload']

def collect_id_and_started(event):
    return [[event['trace_id'], event['info']['started']]]

def get_flat_list(json, collect_function=collect_id_and_started):
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


def extract_events_flat(json_node, collect_function, events=[]):
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

def get_graph_adjacency_list(json, directed=True):
    """Transforms the trace from the json to the graph adjacency list representation. Edges exist
    between events that have a parent-child relationship.

    Args:
      json: Json representation of the trace
      directed: Determines whether a directed or unidirected graph is returned

    Returns:
      List of edges for the trace
    """
    edges = list([[json['info']['name'], c['trace_id']]
                  for c in json['children']])

    for child in json['children']:
        extract_events_flat(child, collect_edges_to_children, edges)

    edges = list(filter(lambda x : x != [], edges))
    
    if not directed:
        backward_edges = list(map(lambda x: [x[1], x[0]], edges))
        edges.extend(backward_edges)

    return edges


def collect_edges_to_children(event):
    """Returns a list of edges from the current event to its children.

    Args:
      event: Event encoded as json

    Returns:
      List of edges from the event to its children
    """
    return [[event['trace_id'], child['trace_id']]
            for child in event['children']]
