"""This module capsulates a trace as a graph"""
import os
import sys
import numpy as np
from pre_processing import trace_pre_processing as tpp
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class TraceGraphModel:
    """
    Class to represent a trace as a graph

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

    """

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
        For every event of the trace two nodes are created: one start node and one stop node.

        Args:
          json_trace (json): Json representing the trace
        """
        header_properties = self._collect_header_information()
        properties = np.array(tpp.get_flat_list(json_trace, self._collect_node_properties))

        header_right_dimensions = self._transform_header_properties_to_proper_dimensions(
            header_properties, properties)

        properties_with_header = np.concatenate((header_right_dimensions, properties))

        indicies = list(['total-start', 'total-stop'])
        indicies.extend(tpp.get_flat_list(json_trace, tpp.collect_id))
        self._nodes = pd.DataFrame(
            properties_with_header,
            columns=tpp.get_node_columns(),
            index=indicies
        )

    def _create_edges_for_trace(self, json_trace):
        """Processes the provided trace and computes the edges for it

        Args:
          json_trace: Json representing the trace
        """
        edges = self._collect_header_edges(json_trace)
        edges.extend(tpp.get_flat_list(json_trace, self._collect_non_header_edges))
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

    def _transform_header_properties_to_proper_dimensions(self, header_properties, properties):
        """This function returns a numpy array that matches the number of columns of the properties
        array. It copies the values from the header_properties and adds empty missing columns.

        Args:
          header_properties (array): Array with dimensionality (h_rows, h_cols) containing the
          features from the header event in the trace
          properties (array): Array with dimensionality (p_rows, p_cols) containing the features
          from the non-header events in the trace

        Returns:
          numpy array with dimensionality (h_rows + p_rows, p_cols)
        """
        n_features = np.shape(properties)[1]
        header_rows = len(header_properties)
        header_right_dimensions = np.empty((header_rows, n_features), dtype='U25')
        for i, row in enumerate(header_properties):
            for j, value in enumerate(row):
                header_right_dimensions[i][j] = value

        return header_right_dimensions

    def _collect_header_information(self):
        """Collects features from the header event of the trace

        Returns:
          array of features collected for the header event, where features are arrays as well.
        """
        return [
            ['total-start'],
            ['total-stop']
        ]

    def _collect_header_edges(self, header_event):
        """Collects the edges from the header event. Edges exist between the start and stop
        header nodes, from the start header node to all the start nodes of children and from the
        stop node of all children to the stop header node.

        Args:
          header_event (json): Json representing the header event

        Returns:
          List containing dictionaries that represent the edge. Keys in the dictionary are source,
          target and weight.
        """
        edges = list()

        edges.extend(
            [
                self._get_edge_dict(
                    'total-start',
                    'total-stop',
                    int(header_event['info']['finished'])
                )
            ]
        )

        edges.extend(self._get_edges_to_children(header_event, lambda x: 'total'))

        return edges

    def _collect_node_properties(self, event):
        """Collects the node properties for non-header events. For each event two nodes will be
        returned, one start and one stop node.

        Args:
          event (json): Json representing the event

        Returns:
          Array of features, where features are arrays as well.
        """
        name = event['info'].get('name')
        service = event['info'].get('service')
        project = event['info'].get('project')
        host = event['info'].get('host')
        payload_start_name = 'meta.raw_payload.' + name + '-start'
        payload_stop_name = 'meta.raw_payload.' + name + '-stop'
        payload_start = event['info'].get(payload_start_name)
        payload_stop = event['info'].get(payload_stop_name)

        return [
            [
                name + '-start',
                service,
                project,
                host,
                hash(str(payload_start))
            ],
            [
                name + '-stop',
                service,
                project,
                host,
                hash(str(payload_stop))
            ]
        ]

    def _collect_non_header_edges(self, event):
        """Collects the edges from the non-header event. Edges exist between the start and stop
        event nodes, from the start node to all the start nodes of children and from the stop
        node of all children to the stop node.

        Args:
          event (json): Json representing the non-header event

        Returns:
          List containing dictionaries that represent the edge. Keys in the dictionary are source,
          target and weight.
        """
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

        edges.extend(self._get_edges_to_children(event))

        return edges

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

    def get_id(self, event):
        """Function to retrieve the id of a non-header event

        Args:
          event (json): Json representing a non-header event

        Returns:
          The id of the event as string.
        """
        return event['trace_id']

    def _get_edges_to_children(self, event, access_id_function=get_id):
        """Returns the edges from the current event to all its children.

        Args:
          event (json): Json representing the event
          access_id_function (function): Function to retrieve the key for the event.
            Defaults to self.get_id

        Returns:
          Array containing the edges. Edges are represented as dictionaries.
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
