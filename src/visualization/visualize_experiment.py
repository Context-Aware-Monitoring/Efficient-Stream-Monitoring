"""Plots various dynamic graphs to vizualize the results of experiments.
"""

import sys
sys.path.append('..')
                
from bokeh.plotting import figure, show, curdoc
from bokeh.models import ColumnDataSource, HoverTool, FileInput, RadioButtonGroup, Range1d, Div, MultiSelect, DataTable, TableColumn
from bokeh.layouts import column, row
import json
import pandas as pd
import pdb
import bokeh.colors
import base64
import yaml
import pickle
import pdb
import experiment
from experiment import Experiment
from bokeh.core.enums import MarkerType
from numpy import histogram
import glob, os

EXPERIMENT_SERIALIZATION_DIR = '../../data/processed/experiment_results/'


def load_experiment(attr, old, new):
    """Loads the selected experiments from the table and plots the graphs.
    """
    global data_table
    if len(new) == 0:
        return

    experiment_configs = []
    for index in new:
        experiment_configs.append(table_source.data['names'][index])
    
    curdoc().clear()
    curdoc().add_root(data_table)
    
    for name in experiment_configs:
        div = Div()
        div.sizing_mode = 'stretch_width'
        with open(EXPERIMENT_SERIALIZATION_DIR + name, 'r') as ymlfile:
            config = yaml.safe_load(ymlfile)
            div.text = str(config)
            curdoc().add_root(div)
            
            plot_experiment_results(config)

def _get_color_palette(no_policies):
    """Computes different colors for each policy of the experiments

    Args:
      no_policies (int): Number of policy in the experiment

    Returns:
      string[]: Hex codes of no_policies colors
    """
    palette = [''] * no_policies

    step = 255 // no_policies

    for i, offset in enumerate(range(0, 255, step)):
        if i == no_policies:
            break;
        palette[i] = bokeh.palettes.Plasma256[offset]

    return palette
            
def fill_experiment_table():
    """Populates the table that displays the results from previously ran
    experiments. The results are stored as yaml files in the directory
    that the global variable EXPERIMENT_SERIALIZATION_DIR points to.
    """
    global data_table, table_source
    
    os.chdir(EXPERIMENT_SERIALIZATION_DIR)
    experiment_files = glob.glob("*.yml")
    
    names = [''] * len(experiment_files)
    Ls = [0] * len(experiment_files)
    reward_types = [''] * len(experiment_files)
    policies = [''] * len(experiment_files)
    seq_or_con = [''] * len(experiment_files)
    
    for i, filename in enumerate(experiment_files):
        with open(filename, 'r') as ymlfile:
            experiment_data = yaml.safe_load(ymlfile)

            names[i] = filename
            Ls[i] = experiment_data['L']
            reward_path = experiment_data['reward_path']
            if 'threshold' in reward_path:
                reward_types[i] = 'threshold'  + reward_path.split('threshold_')[1][0:3]
            elif 'top' in experiment_data['reward_path']:
                reward_types[i] = 'top'
            else:
                if 'pear' in reward_path:
                    reward_types[i] = 'continous-pearson'
                elif 'MI_n1' in reward_path:
                    reward_types[i] = 'continous-MI-normalized'
                else:
                    reward_types[i] = 'continous-MI'

            seq_or_con[i] = reward_path.split('/')[-1][0:3]
            
            policies_string = ''
            for policy in experiment_data['policies']:
                if policies_string == '':
                    policies_string += policy['name']
                else:
                    policies_string += '-' + policy['name']
            policies[i] = policies_string

    table_data = dict(
        names = names,
        seq_or_con = seq_or_con,
        Ls = Ls,
        reward_types = reward_types,
        policies = policies
    )
    table_source = ColumnDataSource(table_data)
    table_source.selected.on_change('indices', load_experiment)
    
    columns = [
            TableColumn(field="names", title="Experiment"),
            TableColumn(field="seq_or_con", title="Sequential/ concurrent"),
            TableColumn(field="Ls", title="L"),
            TableColumn(field="reward_types", title="Reward Type"),
            TableColumn(field="policies", title="Policies")
    ]
    data_table = DataTable(source=table_source, columns=columns, height=200)
    data_table.sizing_mode = 'stretch_width'
    
fill_experiment_table()

def plot_experiment_results(config):
    """Loads the experiment results and plots a plot showing the total regret
    cumulated regret of all policies, a plot that shows the regret at each
    iteration for every policy and a histrogram showing the distribution of the
    regret for the policies.

    Args:
      config (dict): Dictionary that contains the information about the
      experiment.
    """
    experiment = unpickle(config['pickle_filepath'])    
    plot_total_regret(experiment)
    plot_regret_of_policies(experiment)
    
def plot_total_regret(experiment):
    """Plots a plot that shows the cumulated regret of all policies in a single
    figure.

    Args:
      experiment (Experiment): The experiment for that the plot will be
      created.
    """
    fig = figure(title='Overall regret', height=300)
    fig.sizing_mode = 'stretch_width'
    T = experiment.get_T()
    x = list(range(T))

    no_policies = len(experiment.get_average_regret().keys())
    ordered_policy_names = experiment.get_average_cum_regret().keys()

    for i,policy_name, line_color in zip(range(no_policies), ordered_policy_names, _get_color_palette(no_policies)):
        average_cum_regret = experiment.get_average_cum_regret()[policy_name]
        total_average_cum_regret = average_cum_regret[-1]
        label = "%d. %s (%d regret)" % (i+1, policy_name, total_average_cum_regret)
        fig.line(x,average_cum_regret, color=line_color, legend_label = label)

    fig.legend.location = "top_left"
    fig.legend.click_policy="hide"

    curdoc().add_root(fig)

def plot_regret_of_policies(experiment):
    """Plots the regret in each round for every policy of the experiment in a
    single figure.

    Args:
      experiment (Experiment): The experiment for that the plot will be
      created.
    """
    fig_hist = figure(title="Histogram of regret for policies",height=300)
    fig_hist.sizing_mode = 'stretch_width'
    policy_names = list(experiment.get_average_regret().keys())

    fig_regret = figure(title='Regret of individual rounds for policies', height=300)
    fig_regret.sizing_mode = 'stretch_width'
    x = range(experiment.get_T())

    no_policies = len(experiment.get_average_regret().keys())
    
    for policy_name, color,marker in zip(policy_names, _get_color_palette(no_policies), MarkerType):
        regret_for_policy = experiment.get_average_regret()[policy_name]
        
        fig_regret.dot(x, regret_for_policy, color=color, marker=marker, legend_label=policy_name, size=15,alpha=0.5)
        hist, edges = histogram(regret_for_policy, bins=experiment.get_L())
        fig_hist.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:], color=color, legend_label=policy_name, alpha=0.5)

    fig_regret.legend.location = 'top_right'
    fig_regret.legend.click_policy='hide'

    fig_hist.legend.location = 'top_right'
    fig_hist.legend.click_policy = 'hide'

    curdoc().add_root(row(fig_regret, fig_hist))
    
def unpickle(filepath):
    """Unpickles the experiment object from the pickle that is located at the
    passed filepath.

    Args:
      filepath (string): Filepath that points to the pickle.

    Returns:
      Experiment: The unpickled experiment
    """
    with open(filepath, 'rb') as pickle_file:
        data = pickle.load(pickle_file)

    return data

curdoc().add_root(data_table)

