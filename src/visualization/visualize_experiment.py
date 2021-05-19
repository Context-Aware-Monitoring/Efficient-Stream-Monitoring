"""Plots various dynamic graphs to vizualize the results of experiments.
"""

import os
import glob
from os.path import dirname, abspath
import re
import numpy as np
from numpy import histogram
import yaml
import bokeh.colors
import pandas as pd
from bokeh.layouts import row
from bokeh.models import ColumnDataSource, DataTable, TableColumn, Button, Select
from bokeh.plotting import figure, curdoc


MOVING_AVERAGE_LENGTH = 10
MAX_NO_POLICIES = 10
DATA_DIR = '%s/data' % dirname(dirname(dirname(abspath(__file__))))
EXPERIMENT_SERIALIZATION_DIR = '%s/processed/experiment_results/' % DATA_DIR

COLOR_PALETTE = bokeh.palettes.Paired[MAX_NO_POLICIES]

no_policies = 0
current_experiment_config = None


def moving_average(values, window_size):
    return np.convolve(values, np.ones(window_size), 'valid') / window_size


def reset_plots():
    global no_policies
    no_policies = 0

    plot_data.data = {}
    moving_plot_data.data = {}
    hist_data.data = {}

    for current_fig in [
            hist_fig, overall_regret_fig, regret_fig, rolling_regret_fig]:
        current_fig.legend.items = []

    pol_source.selected.indices = []


def load_experiment(attr, old, new):
    """Loads the information about the currently selected experiment and fills
    the policy table.
    """
    global data_table, pol_table, current_experiment_config

    if len(new) != 1:
        return

    name = table_source.data['names'][new[0]]

    with open(EXPERIMENT_SERIALIZATION_DIR + name, 'r') as ymlfile:
        config = yaml.safe_load(ymlfile)

    current_experiment_config = config
    show_policies_in_table(config)

    global current_average_regret, current_cum_regret
    current_average_regret = pd.read_csv(config['average_regret_csv_file'])
    current_cum_regret = pd.read_csv(config['cum_regret_csv_file'])


def _fill_plot_data_sources(average_regret, cum_regret):
    plot_data.data['x'] = current_average_regret.index.values
    plot_data.data['regret_per_round_pol%d' % no_policies] = average_regret
    plot_data.data['cum_regret_pol%d' % no_policies] = cum_regret

    moving_average_y = moving_average(average_regret, MOVING_AVERAGE_LENGTH)
    moving_plot_data.data['x'] = np.arange(len(moving_average_y))
    moving_plot_data.data['rolling_regret_pol%d' %
                          no_policies] = moving_average_y

    hist, edges = histogram(
        average_regret, bins=current_experiment_config['L'])
    hist_data.data['bottom'] = np.zeros(len(hist))
    hist_data.data['hist_pol%d' % no_policies] = hist
    hist_data.data['left_pol%d' % no_policies] = edges[:-1]
    hist_data.data['right_pol%d' % no_policies] = edges[1:]


def _add_policy_to_plots(pol_name):
    hist_fig.quad(top='hist_pol%d' % no_policies, bottom='bottom',
                  left='left_pol%d' % no_policies, right="right_pol%d" %
                  no_policies, color=COLOR_PALETTE[no_policies],
                  legend_label=pol_name, alpha=0.5, source=hist_data)

    overall_regret_fig.line(
        x='x', y='cum_regret_pol%d' % no_policies,
        color=COLOR_PALETTE[no_policies],
        source=plot_data, legend_label=pol_name)
    regret_fig.line(x='x', y='regret_per_round_pol%d' % no_policies,
                    color=COLOR_PALETTE[no_policies],
                    source=plot_data, legend_label=pol_name)
    rolling_regret_fig.line(x='x', y='rolling_regret_pol%d' % no_policies,
                            color=COLOR_PALETTE[no_policies],
                            source=moving_plot_data, legend_label=pol_name)


def plot_policy(attr, old, new):
    """Adds the newly selected policy to the four plots of the policy.
    """
    global no_policies

    if len(new) != 1 or no_policies >= MAX_NO_POLICIES:
        return

    new_index = new[0]

    pol_name = pol_source.data['pol_name'][new_index]

    average_regret = current_average_regret[pol_name].values
    cum_regret = current_cum_regret[pol_name].values

    _fill_plot_data_sources(average_regret, cum_regret)
    _add_policy_to_plots(pol_name)

    no_policies += 1


def fill_experiment_table():
    """Populates the table that displays the results from previously ran
    experiments. The results are stored as yaml files in the directory
    that the global variable EXPERIMENT_SERIALIZATION_DIR points to.
    """
    global data_table, table_source, table_data_df

    os.chdir(EXPERIMENT_SERIALIZATION_DIR)
    experiment_files = glob.glob("*.yml")

    names = [''] * len(experiment_files)
    Ls = [0] * len(experiment_files)
    reward_types = [''] * len(experiment_files)
    window_size = [''] * len(experiment_files)
    window_step = [''] * len(experiment_files)
    seq_or_con = [''] * len(experiment_files)

    for i, filename in enumerate(experiment_files):
        with open(filename, 'r') as ymlfile:
            experiment_data = yaml.safe_load(ymlfile)

            names[i] = filename
            Ls[i] = experiment_data['L']
            reward_path = experiment_data['reward_path']
            if 'threshold' in reward_path:
                reward_types[i] = 'threshold' + reward_path.split('threshold_')[
                    1][0:3]
            elif 'top' in experiment_data['reward_path']:
                reward_types[i] = 'top'
            else:
                if 'pear' in reward_path:
                    reward_types[i] = 'continous-pearson'
                elif 'MI_n1' in reward_path:
                    reward_types[i] = 'continous-MI-normalized'
                else:
                    reward_types[i] = 'continous-MI'

            window_size[i], window_step[i] = re.findall(
                r'\d+', reward_path)[0:2]

            seq_or_con[i] = reward_path.split('/')[-1][0:3]

    table_data = dict(
        names=names,
        seq_or_con=seq_or_con,
        Ls=Ls,
        reward_types=reward_types,
        window_size=window_size,
        window_step=window_step,
    )

    table_data_df = pd.DataFrame(data=table_data)
    table_data_df = table_data_df.sort_values(
        ['seq_or_con', 'Ls', 'window_size', 'window_step', 'reward_types'])

    table_source = ColumnDataSource(data=table_data_df)
    table_source.selected.on_change('indices', load_experiment)

    columns = [
        TableColumn(field="names", title="Experiment"),
        TableColumn(field="seq_or_con", title="Sequential/ concurrent"),
        TableColumn(field="Ls", title="L"),
        TableColumn(field="reward_types", title="Reward Type"),
        TableColumn(field='window_size', title="Window Size"),
        TableColumn(field='window_step', title="Window Step"),
    ]
    data_table = DataTable(source=table_source, columns=columns, height=200)
    data_table.sizing_mode = 'stretch_width'

    global pol_table, pol_source

    columns_pol_table = [
        TableColumn(field="pol_name", title="Name"),
        TableColumn(field="overall_regret", title="Overall regret")
    ]

    pol_source = ColumnDataSource(dict(pol_name=[], overall_regret=[]))
    pol_source.selected.on_change('indices', plot_policy)
    pol_table = DataTable(
        source=pol_source, columns=columns_pol_table, height=300)
    pol_table.sizing_mode = 'stretch_width'


fill_experiment_table()


def show_policies_in_table(config):
    """Populates the policy table with the polices of the experiment.

    Args:
      config (dict): Config describing the experiment results.
    """
    pol_source.selected.indices = []
    result_for_policies = config['results']

    pol_name = list(result_for_policies.keys())
    overall_regret = list(result_for_policies.values())

    table_data = dict(
        pol_name=pol_name,
        overall_regret=overall_regret
    )

    table_data_df = pd.DataFrame(data=table_data)
    table_data_df = table_data_df.sort_values(['overall_regret'])

    pol_source.data = table_data_df


def filter_experiment_table(attr, old, new):
    """Filters the data shown in the experiment table according to the sliders.
    """
    filtered_df = table_data_df

    if select_seq.value != '':
        seq_or_con = 'seq' if bool(select_seq.value) else 'con'
        filtered_df = filtered_df[filtered_df.seq_or_con == seq_or_con]

    if select_L.value != '':
        filtered_df = filtered_df[filtered_df.Ls == int(select_L.value)]

    if select_reward_type.value != '':
        filtered_df = filtered_df[filtered_df.reward_types ==
                                  select_reward_type.value]

    if select_window_size.value != '':
        filtered_df = filtered_df[filtered_df.window_size ==
                                  select_window_size.value]

    if select_window_step.value != '':
        filtered_df = filtered_df[filtered_df.window_step ==
                                  select_window_step.value]

    table_source.data = filtered_df
    table_source.selected.indices = []
    pol_source.data = {}


select_seq = None
select_L = None
select_reward_type = None
select_window_size = None
select_window_step = None


def add_selects_to_document():
    """Adds select to the document to filter the experiment table.
    """
    global select_seq, select_L, select_reward_type, select_window_size, select_window_step

    select_seq = Select(title='Sequential', options=[
        '', 'True', 'False'], width=100)
    select_L = Select(
        title="L", options=['', '5', '10', '20', '50', '100'],
        width=100)
    select_reward_type = Select(
        title="Reward type",
        options=['', 'threshold0.6', 'threshold0.7', 'threshold0.8', 'top',
                 'continous'],
        width=150)
    select_window_size = Select(title="Window size", options=[
        '', '10', '30', '60'], width=100)
    select_window_step = Select(title="Window step", options=[
        '', '1', '5', '10'], width=100)

    for current_select in [
            select_seq, select_L, select_reward_type, select_window_size,
            select_window_step]:
        current_select.on_change('value', filter_experiment_table)

    curdoc().add_root(row(select_seq, select_L, select_reward_type,
                          select_window_size, select_window_step))


add_selects_to_document()
curdoc().add_root(data_table)
curdoc().add_root(pol_table)

plot_data = ColumnDataSource()
moving_plot_data = ColumnDataSource()
hist_data = ColumnDataSource()


button = Button(label="Reset plots")
button.on_click(reset_plots)

curdoc().add_root(button)

overall_regret_fig = figure(title='Overall regret', height=300)
regret_fig = figure(title='Regret per round', height=300)
rolling_regret_fig = figure(title='Rolling regret', height=300)
hist_fig = figure(title='Histogram of regret for polices', height=300)

for current_fig in [
        overall_regret_fig, regret_fig, rolling_regret_fig, hist_fig]:
    current_fig.sizing_mode = 'stretch_width'

curdoc().add_root(overall_regret_fig)
curdoc().add_root(regret_fig)
curdoc().add_root(rolling_regret_fig)
curdoc().add_root(hist_fig)
