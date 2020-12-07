from modeling.trace_graph_model import TraceGraphRepresentation
from bokeh.plotting import figure, show, curdoc
from bokeh.models import ColumnDataSource, HoverTool, FileInput, RadioButtonGroup, Range1d, Div
from bokeh.layouts import column, row
import json
import pandas as pd
import pdb
from pre_processing import trace_pre_processing as tpp
import bokeh.colors
import base64

HEIGHT_PER_LEVEL = 50
MIN_HEIGHT = 200
HEIGHT_INCREMENT = -1
SOURCE_NAME = 'name'
SOURCE_HOST = 'host'
SOURCE_SERVICE = 'service'
sources = {
    SOURCE_NAME : {},
    SOURCE_SERVICE : {},
    SOURCE_HOST : {}
}

levels = 0


p1 = None
p2 = None
p3 = None
labels = ['name', 'service', 'host']
radio_button = RadioButtonGroup(labels=labels, active=0)
div = Div()
HOST_COLOR_MAP = {
    'wally113' : 'blue',
    'wally117' : 'green',
    'wally122' : 'yellow',
    'wally123' : 'red',
    'wally124' : 'purple'
}

NAME_COLOR_MAP = {
    'neutron.db' : 'darkorange',
    'db' : 'dodgerblue',
    'neutron_api' : 'darkviolet',
    'wsgi' : 'lightgreen',
    'rpc' : 'peru',
    'compute_api' : 'gold',
    'nova_image' : 'pink'
}

SERVICE_COLOR_MAP = {
    'neutron-server' : 'darkorange',
    'osapi_compute' : 'dodgerblue',
    'public' : 'darkviolet',
    'api' : 'lightgreen',
    'nova-conductor' : 'peru',
    'nova-compute' : 'gold',
    'nova-scheduler' : 'pink',
    'neutron-dhcp-agent' : 'black'
}

def group_by_name(event):
    return event['name']

def group_by_service(event):
    return event['service']

def group_by_host(event):
    return event['host']

def compute_sources(trace_json):
    events = tpp.get_flat_list(trace_json, lambda x : [{'name' : x['info']['name'], 'service' : x['info']['service'], 'host': x['info']['host'], 'project': x['info']['project'], 'started' : x['info']['started'], 'finished' : x['info']['finished'], 'trace_id' : x['trace_id']}])
    group_events_and_copy_into_source(sources[SOURCE_NAME], events, group_by_name, int(trace_json['info']['finished']))
    group_events_and_copy_into_source(sources[SOURCE_SERVICE], events, group_by_service, int(trace_json['info']['finished']))
    group_events_and_copy_into_source(sources[SOURCE_HOST], events, group_by_host, int(trace_json['info']['finished']))

def group_events_and_copy_into_source(source, events, group_by_function, finished_trace):
    height_at_pos = [0] * finished_trace
    for e in events:
        key = group_by_function(e)
        if key not in source:
            source[key] = ColumnDataSource({
            'x_values' : [],
            'y_values' : [],
            'color' : [],
            'name' : [],
            'service' : [],
            'host' : [],
            'project' : [],
            'trace_id' : [],
            'started' : [],
            'finished' : []    
        })

        cds = source[key]
        started = int(e['started'])
        finished = int(e['finished'])
        cds.data['x_values'].append([started, finished])
        if finished - started == 0:
            height = height_at_pos[started]
        else:
            height = min(height_at_pos[started:finished])
        height_at_pos[started:finished] = (finished - started) * [height + HEIGHT_INCREMENT]
        cds.data['y_values'].append([height, height])
        if group_by_function.__name__ == 'group_by_name':
            cds.data['color'].append(NAME_COLOR_MAP[e['name']])
        elif group_by_function.__name__ == 'group_by_service':
            cds.data['color'].append(SERVICE_COLOR_MAP[e['service']])
        else:
            cds.data['color'].append(HOST_COLOR_MAP[e['host']])
        cds.data['name'].append(e['name'])
        cds.data['service'].append(e['service'])
        cds.data['host'].append(e['host'])
        cds.data['project'].append(e['project'])
        cds.data['trace_id'].append(e['trace_id'])
        cds.data['started'].append(e['started'])
        cds.data['finished'].append(e['finished'])

    global levels
    levels = abs(min(height_at_pos)) + 1
    
def plot_trace(plot, group):
    for legend, data in sources[group].items():
        plot.multi_line(xs='x_values',ys='y_values', line_color='color',source=data, line_width=20,line_alpha=0.6, hover_line_alpha=1.0, legend_label=legend,muted_color='color', muted_alpha=0.2)

    plot.legend.location = 'bottom_right'
    plot.legend.click_policy = 'mute'
        
TOOLTIPS = [
    ('name', '@name'),
    ('service', '@service'),
    ('host', '@host'),
    ('project', '@project'),
    ('trace_id', '@trace_id'),
    ('started', '@started'),
    ('finished', '@finished')
]


def load_trace_data(attr, old, new):
    trace_json = json.loads(base64.b64decode(new))
    compute_sources(trace_json)
    setup_figures()
    plot_trace(p1, SOURCE_NAME)
    plot_trace(p2, SOURCE_SERVICE)
    plot_trace(p3, SOURCE_HOST)
    
hover = HoverTool(
    tooltips=TOOLTIPS,
    show_arrow=False,
    mode='mouse',
    line_policy='next',
    point_policy='snap_to_data'
)

def setup_figures():
    global p1, p2, p3, HEIGHT_PER_LEVEL, levels
    height = max(HEIGHT_PER_LEVEL * levels, MIN_HEIGHT)
    y_range = Range1d(-levels + 1, 1)
    p1 = figure(title='trace visualization by name', x_axis_label='x', y_axis_label='y', tools='reset,pan,wheel_zoom', height=height, y_range=y_range)
    p2 = figure(title='trace visualization by service', x_axis_label='x', y_axis_label='y', tools='reset,pan,wheel_zoom', height=height,y_range=y_range)
    p3 = figure(title='trace visualization by host', x_axis_label='x', y_axis_label='y', tools='reset,pan,wheel_zoom', height=height, y_range=y_range)
    plts = [p1, p2, p3]

    for p in plts:
        p.xaxis.axis_label = 'Time'
        p.yaxis.visible = False
        p.add_tools(hover)
    curdoc().clear()
    div.text = '<h1>' + file_input.filename + '</h1>'

    curdoc().add_root(column(file_input, div, radio_button, get_current_plot()))

def get_current_plot():
    if radio_button.active == 0:
        return p1
    elif radio_button.active == 1:
        return p2
    else:
        return p3
    
def change_current_plot(attr, old, new):
    curdoc().clear()
    curdoc().add_root(column(file_input, div, radio_button, get_current_plot()))    
    
file_input = FileInput(accept=".json")
file_input.on_change('value', load_trace_data)
radio_button.on_change('active', change_current_plot)

curdoc().add_root(file_input)

