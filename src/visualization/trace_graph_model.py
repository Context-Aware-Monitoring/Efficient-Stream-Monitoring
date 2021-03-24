"""This module contains functionality to represent traces as different graph models."""
import os
import sys
sys.path.append("../features")
from abc import ABC, abstractmethod
import numpy as np
import trace_pre_processing as tpp
import pandas as pd

class AbstractTraceGraphModel(ABC):
    """
    Abstract class providing the basic functionality to generate a graph model for a trace.
    Graphs models are constructed the same why:
    1. Nodes are created. Nodes consists of a set of attributes. The way nodes are created is an
       implementation detail of the concrete model. Two methods are used to create nodes:
       _collect_header_nodes: For creating nodes from the informations provided in the head of the
       json.
       _collect_event_nodes: For creating nodes from the regular events in the json.
    2. Edges are created. An edge consists of the following three properties: source, target and
       weight. Like for nodes there exist two different methods to collect the edges. For the head
       of the json the method _collect_header_edges, and for normal events in the json the method
       _collect_event_edges.

    Attributes
    ----------
    _nodes: pd.DataFrame
        nodes of the graph
    _edges: pd.DataFrame
        edges of the graph

    Methods
    -------
    get_nodes()
        Returns the nodes of the graph
    get_edges()
        Returns the edges of the graph
    """

    NODE_INDEX = "NODE_INDEX"

    def __init__(self, json_trace):
        """Initializes the graph from the trace

        Args:
          json_trace (json): Json representing the trace
        """
        self._create_nodes_for_trace(json_trace)
        self._create_edges_for_trace(json_trace)

    def get_nodes(self):
        """Returns the nodes of the graph

        Returns:
          nodes of the graph
        """
        return self._nodes

    def get_edges(self):
        """Returns the edges of the graph

        Returns:
          edges of the graph
        """
        return self._edges

    def _create_nodes_for_trace(self, json_trace):
        """Processes the provided trace and computes the nodes for it.
        The way nodes are created is an implementation detail of the concrete model.

        Args:
          json_trace (json): Json representing the trace
        """
        properties = self._collect_header_nodes(json_trace)
        properties.extend(np.array(tpp.get_flat_list(json_trace, self._collect_event_nodes)))

        self._nodes = self._build_data_frame_containing_all_features(properties)

    def _create_edges_for_trace(self, json_trace):
        """Processes the provided trace and computes the edges for it.
        The way edges are created is an implementation detail of the concrete model.

        Args:
          json_trace: Json representing the trace
        """
        edges = self._collect_header_edges(json_trace)
        edges.extend(tpp.get_flat_list(json_trace, self._collect_event_edges))

        source = np.array(list(map(lambda x: x['source'], edges)))
        target = np.array(list(map(lambda x: x['target'], edges)))
        weight = np.array(list(map(lambda x: x['weight'], edges)))

        self._edges = pd.DataFrame(
            {
                'source': source,
                'target': target,
                'weight': weight
            }
        )

    def _build_data_frame_containing_all_features(self, properties):
        """Creating a data frame representing the nodes of the graph from the information provided
        in properties.

        Args:
          properties (list): List of dicts, where each dict represents one node. The key in the
          dict are the attribute name, the value is the respective value of that attribute.

        Returns:
          Panda data frame representing the nodes.
        """
        column_names = list()
        for prop in properties:
            column_names.extend(list(prop.keys()))
        column_names = set(column_names)
        n_properties = len(properties)

        data = {}
        indicies = list()
        for current_column_name in column_names:
            if current_column_name == self.NODE_INDEX:
                continue

            data[current_column_name] = np.empty(n_properties, dtype='U25')

        i = 0
        for prop in properties:
            for column_name, value in prop.items():
                if column_name == self.NODE_INDEX:
                    indicies.extend([prop[self.NODE_INDEX]])
                else:
                    data[column_name][i] = value
            i += 1

        return pd.DataFrame(data, index=indicies)

    @abstractmethod
    def _collect_header_nodes(self, json_trace):
        """Collects nodes from the header data of the json.

        Args:
          json_trace (json): Json representing the trace.
        Returns:
          List of dicts, where a dict represents a node. Keys in the dict correspond to attributes,
          values to the respective value of that attribute.
        """
        return

    @abstractmethod
    def _collect_header_edges(self, json_trace):
        """Collects the edges from the head of the json.

        Args:
          json_trace (json): Json representing the trace.

        Returns:
          List of dicts, where a dict represents an edge. Dict has three keys: source, target
          and weight. Source and target should be the ids of nodes, weight an int.
        """
        return

    @abstractmethod
    def _collect_event_nodes(self, event):
        """Collects nodes for regular events of traces.

        Args:
          event (json): Json representing the currently processed event.
        Returns:
          List of dicts, where a dict represents a node. Keys in the dict correspond to attributes,
          values to the respective value of that attribute.
        """
        return

    @abstractmethod
    def _collect_event_edges(self, event):
        """Collects the edges for regular events of traces.

        Args:
          event (json): Json representing the currently processed event.

        Returns:
          List of dicts, where a dict represents an edge. Dict has three keys: source, target
          and weight. Source and target should be the ids of nodes, weight an int.
        """
        return

    def _get_edge_dict(self, source, target, weight):
        """Returns a dictionary representing an edge.

        Args:
          source (string): Id of the source node
          target (string): Id of the target node
          weight (int): weight of the edges
        """
        return {
            'source': source,
            'target': target,
            'weight': weight
        }


class TraceSpanRepresentation(AbstractTraceGraphModel):
    """Class representing traces as a span. In a span each event creates two nodes. Edges exist
    between both of these nodes and its children. The weight of the edge corresponds to the delay
    of the service."""

    COLUMN_NAME_NAME = 'name'
    COLUMN_NAME_SERVICE = 'service'
    COLUMN_NAME_PROJECT = 'project'
    COLUMN_NAME_HOST = 'host'
    COLUMN_NAME_PAYLOAD = 'payload'

    def _collect_header_nodes(self, json_trace):
        properties = [
            {
                self.NODE_INDEX: 'total-start',
                self.COLUMN_NAME_NAME: 'total-start'
            },
            {
                self.NODE_INDEX: 'total-stop',
                self.COLUMN_NAME_NAME: 'total-stop'
            }
        ]

        return properties

    def _collect_header_edges(self, json_trace):
        edges = list()

        edges.extend(
            [
                self._get_edge_dict(
                    'total-start',
                    'total-stop',
                    int(json_trace['info']['finished'])
                )
            ]
        )

        edges.extend(self._get_edges_to_children(json_trace, lambda x: 'total'))

        return edges

    def _collect_event_nodes(self, event):
        index = event['trace_id']
        name = event['info'].get('name')
        service = event['info'].get('service')
        project = event['info'].get('project')
        host = event['info'].get('host')
        payload_start_name = 'meta.raw_payload.' + name + '-start'
        payload_stop_name = 'meta.raw_payload.' + name + '-stop'
        payload_start = event['info'].get(payload_start_name)
        payload_stop = event['info'].get(payload_stop_name)

        return [
            {
                self.NODE_INDEX: index + '-start',
                self.COLUMN_NAME_NAME: name + '-start',
                self.COLUMN_NAME_SERVICE: service,
                self.COLUMN_NAME_PROJECT: project,
                self.COLUMN_NAME_HOST: host,
                self.COLUMN_NAME_PAYLOAD: str(payload_start)
            },
            {
                self.NODE_INDEX: index + '-stop',
                self.COLUMN_NAME_NAME: name + '-stop',
                self.COLUMN_NAME_SERVICE: service,
                self.COLUMN_NAME_PROJECT: project,
                self.COLUMN_NAME_HOST: host,
                self.COLUMN_NAME_PAYLOAD: str(payload_stop)
            }
        ]

    def _collect_event_edges(self, event):
        edges = list()
        edges.extend(
            [
                self._get_edge_dict(
                    event['trace_id'] + '-start',
                    event['trace_id'] + '-stop',
                    int(event['info']['finished']) - int(event['info']['started'])
                )
            ]
        )

        edges.extend(self._get_edges_to_children(event, lambda x: x['trace_id']))

        return edges

    def _get_edges_to_children(self, event, access_id_function):
        """Returns the edges for the current event. Edges exist between the start node of the event
        and the start node of the child and between the stop node of the child and the stop node of
        the event.

        Args:
          event (json): Json representing the event
          access_id_function (function): Function to retrieve the key for the event.

        Returns:
          Array containing the edges. Edges are represented as dictionaries. The dictionaries
          contain three keys: source, target and weight. Source/ target are set to a node ids and
          weight is an int.
        """
        edges = list()
        for child in event['children']:
            event_id = access_id_function(event)
            edges.extend(
                [
                    self._get_edge_dict(
                        event_id + '-start',
                        child['trace_id'] + '-start',
                        int(child['info']['started']) - int(event['info']['started'])
                    ),
                    self._get_edge_dict(
                        child['trace_id'] + '-stop',
                        event_id + '-stop',
                        0
                    )
                ]
            )

        return edges


class TraceGraphRepresentation(AbstractTraceGraphModel):
    """Class representing the trace as a graph with a single node per event. Edges exist between
    events and its children. Weights on edges correspond to the delay between events."""

    COLUMN_NAME_NAME = 'name'
    COLUMN_NAME_SERVICE = 'service'
    COLUMN_NAME_PROJECT = 'project'
    COLUMN_NAME_HOST = 'host'
    COLUMN_NAME_PAYLOAD_START = 'payload-start'
    COLUMN_NAME_PAYLOAD_STOP = 'payload-stop'
    COLUMN_NAME_DURATION = 'duration'

    def _collect_header_nodes(self, json_trace):
        return [
            {
                self.NODE_INDEX: 'total',
                self.COLUMN_NAME_NAME: 'total',
                self.COLUMN_NAME_DURATION: json_trace['info']['finished']
            }
        ]

    def _collect_header_edges(self, json_trace):
        edges = list()
        edges.extend(self._get_edges_to_children(json_trace, lambda x: 'total'))

        return edges

    def _collect_event_nodes(self, event):
        name = event['info'].get('name')
        payload_start_name = 'meta.raw_payload.' + name + '-start'
        payload_stop_name = 'meta.raw_payload.' + name + '-stop'
        payload_start = event['info'].get(payload_start_name)
        payload_stop = event['info'].get(payload_stop_name)
        started = event['info'].get('started')
        finished = event['info'].get('finished')
        return [
            {
                self.NODE_INDEX: event['trace_id'],
                self.COLUMN_NAME_NAME: event['info'].get('name'),
                self.COLUMN_NAME_SERVICE: event['info'].get('service'),
                self.COLUMN_NAME_PROJECT: event['info'].get('project'),
                self.COLUMN_NAME_HOST: event['info'].get('host'),
                self.COLUMN_NAME_PAYLOAD_START: str(payload_start),
                self.COLUMN_NAME_PAYLOAD_STOP: str(payload_stop),
                self.COLUMN_NAME_DURATION: str(int(finished) - int(started))
            }
        ]

    def _collect_event_edges(self, event):
        edges = list()
        edges.extend(self._get_edges_to_children(event, lambda x: x['trace_id']))

        return edges

    def _get_edges_to_children(self, event, access_id_function):
        """Returns the edges for the current event. Edges exist between the event and all of its
        children.

        Args:
          event (json): Json representing the event
          access_id_function (function): Function to retrieve the key for the event.

        Returns:
          Array containing the edges. Edges are represented as dictionaries. The dictionaries
          contain three keys: source, target and weight. Source/ target are set to a node ids and
          weight is an int.
        """
        edges = list()
        for child in event['children']:
            event_id = access_id_function(event)
            edges.extend(
                [
                    self._get_edge_dict(
                        event_id,
                        child['trace_id'],
                        int(child['info']['started']) - int(event['info']['started'])
                    )
                ]
            )

        return edges
