import pdb

def get_flat_list(json):
    # Returns the flat list representation (events ordered by their starting point) of the trace in the json

    events = list()
    for c in json['children']:
        extract_events_flat(c, lambda x: [(x['trace_id'], x['info']['started'])], events)
    events.sort(key=lambda x: x[1])

    return events

def get_list(json):
    # Return the list of lists representation where children of an event are saved in a list

    return list([extract_events(c, lambda x: (x['trace_id'], x['info']['started'])) for c in json['children']])

def extract_events_flat(json_node, collect_function, events=[]):
    # Recursively steps down the json_node, collects the event using
    # the collect_function and returns the collected event as a flat list
    # Make sure that the collect function returns an array
    
    if len(json_node) == 0:
        return

    events.extend(list(collect_function(json_node)))
    for child in json_node['children']:
        extract_events_flat(child, collect_function, events)

    return events

def extract_events(json_node, collect_function):
    # Recursively steps down the json_node, collects the event using
    # the collect_function and return the collected event as a list of
    # lists, where the children of an element are saved as a list
    
    if len(json_node) == 0:
        return list()
    
    return list([collect_function(json_node), [extract_events(node, collect_function) for node in json_node['children']]])

def get_graph_adjacency_list(json, directed=True):
    # Returns a adjacency list of the json trace where edges exist
    # between events that have a parent-child relationship
    
    edges = list([(json['info']['name'], c['trace_id']) for c in json['children']])

    for child in json['children']:
        extract_events_flat(child, collect_edges_to_children, edges)

    if directed == False:
        backward_edges = list(map(lambda x: (x[1], x[0]), edges))
        edges.extend(backward_edges)
        
    return edges


def collect_edges_to_children(trace):
    # Collects the edges from the trace to its children
    
    return [(trace['trace_id'], child['trace_id']) for child in trace['children']]
    
