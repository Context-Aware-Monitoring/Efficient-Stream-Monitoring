"""Generates rewards csv files for a bandit algorithm.
"""
from os.path import dirname, abspath
import pandas as pd
import numpy as np

DATA_DIR = '%s/data' % dirname(dirname(dirname(abspath(__file__))))
REWARDS_DIR = '%s/processed/rewards' % DATA_DIR
EXPERIMENT_CONFIG_DIR = '%s/interim/experiment_configs' % DATA_DIR


def generate_reward_csv(
        hosts,
        window_size=30,
        step=5,
        correlation_method='pearson',
        sequential=True,
        outdir=REWARDS_DIR,
        kind='continous',
        **kwargs
):
    """Writes a reward.csv file that can be processed by a bandit
    algorithm.

    Args:
      hosts (string[]): Hosts used for the computation of the reward csv file.
      Possible values are wally113, wally117, wally122, wally123, wally124.
      window_size (int): For the calculation of the reward function a sliding
      window is used. window_size specifies the size of this sliding window.
      step (int): Specifies the step of the sliding window.
      correlation_method (string): Specifies the measure of correlation used to compute the
      correlation between metrics. Available options are 'MI'.
      sequential (bool): Specifies whether the metrics for sequential or
      concurrent program execution should be used.
      outdir (string): Specifies the directory where to write the generated
      file.
      kind (string): One of 'continous', 'threshold' or 'top'. Specifies how
      the rewards will be computed. 'Continous' is a continous reward, where
      the reward for each arm is the correlation, 'top' and 'threshold' are
      binary rewards where 'top' rewards a 1 if the arm is one of the top L
      correlated arms and 'threshold' rewards a 1 if the arms correlation
      exceeds a threshold.
      kwargs (dict): The following parameters can be passed: L (int) for kind
      'top' and threshold (float) for kind 'threshold'.
    """
    unified_metrics_df = _generate_unified_metrics_dataframe(hosts, sequential)

    reward_df = _generate_windowed_reward_df(
        unified_metrics_df,
        window_size,
        step,
        correlation_method,
        kind,
        **kwargs
    )


    filepath = _generate_filepath(
        window_size, step, correlation_method, sequential, outdir, kind, **kwargs)

    reward_df.to_csv(filepath, index=True)


def _generate_filepath(
        window_size, step, correlation_method, sequential, outdir, kind, **
        kwargs):
    """Computes a filepath for the reward csv file that allows us to identify
    the properties of the experiment.

    Args:
      window_size (int): Size of the sliding window
      step (int): Step of the sliding window
      correlation_method (string): One of 'pearson',
      sequential (bool): If the experiment is for the sequential or concurrent
      metrics data.
      outdir (string): Directory where to write the csv file
      kind (string): One of 'continous', 'top' or 'threshold'

    Returns:
      string: The filepath of the reward csv file
    """
    seq_or_con = 'seq' if sequential else 'con'
    if kind == 'top':
        kind += '_' + str(kwargs['L'])
    elif kind == 'threshold':
        kind += '_' + str(kwargs['threshold'])
    filepath = (
        "%s%s_rewards_w%d_s%d_%s_%s.csv" %
        (outdir, seq_or_con, window_size, step, correlation_method, kind))

    return filepath


def _read_host_df(host, seq=True):
    """Reads the metrics data for the host and returns a DataFrame.

    Args:
      host (str): Hostname, one of wally113, wally117, wally122, wally123,
      wally124
      seq (bool): If sequential or concurrent metrics should be read

    Returns:
      DataFrame: Containing all the metrics as columns
    """
    filepath = ''
    if seq:
        filepath = '%s/interim/sequential_data/metrics/%s_metrics.csv' % (DATA_DIR, host)
    else:
        filepath = '%s/interim/concurrent_data/metrics/%s_metrics_concurrent.csv' % (DATA_DIR, host)

    metrics_df = pd.read_csv(
        filepath,
        dtype={'now': str, 'load.cpucore': np.float64, 'load.min1': np.float64,
               'load.min5': np.float64, 'load.min15': np.float64,
               'mem.used': np.float64})

    metrics_df['now'] = pd.to_datetime(metrics_df['now'])
    metrics_df = metrics_df.set_index('now')

    metrics_df = metrics_df.add_prefix('%s.' % host)

    return metrics_df.pivot_table(metrics_df, index=['now'], aggfunc='mean')


def _generate_unified_metrics_dataframe(hosts, seq):
    """Generates a unified data frame for the metrics of all the hosts that are
    passed as a parameter.

    Args:
      hosts (string[]): Contains all the hosts. Possible values are wally113,
      wally117, wally122, wally123, wally124.
      seq (bool): Weather seequential or concurrent metrics data is read.

    Returns:
      DataFrame containing the metrics for all hosts
    """

    hosts_df = [_read_host_df(host, seq) for host in hosts]

    unified_df = hosts_df[0]

    for h_df in hosts_df[1:]:
        unified_df = unified_df.merge(
            h_df, how='left', left_index=True, right_index=True)

    return unified_df

def _compute_pairwise_correlation_of_arms(
        metrics_df, window_size, window_step, method):
    """Computes a DataFrame that contains the correlation for arms (pairs
    of metrics) as columns and the iterations as rows.

    Args:
      metrics_df (DataFrame): A DataFrame that contains the metrics as columns
      and the iteration as rows.
      window_size (int): Size of the sliding window
      window_step (int): Step of the sliding window
      method (string): One of 'pearson',

    Returns:
      DataFrame: Containing the correlation for the arms as columns and a row
      for each iteration.
    """
    start = metrics_df.index.values[0]
    # skip the windows that contain nan values because the window size hasn't
    # reached the desired window size
    start = start + np.timedelta64(window_size - 1, 's')
    end = metrics_df.index.values[-1]

    arm_names = metrics_df.columns.values
    no_arms = len(arm_names)
    indicies_for_new_df = pd.date_range(start, end, freq='%dS' % window_step)
    no_windows = len(indicies_for_new_df)
    values_in_window = no_arms * no_arms

    correlation_matrix = metrics_df.rolling(
        window=window_size).corr().values.flatten()
    # first window_size - 1 values are NaN because the sliding window is not yet full
    correlation_matrix = correlation_matrix[values_in_window *
                                            (window_size - 1):]

    indicies_of_relevant_windows = np.array(
        [np.arange(offset, offset + values_in_window)
         for offset in range(
             0, correlation_matrix.shape[0],
             values_in_window * window_step)]).flatten()

    correlation_matrix = correlation_matrix[indicies_of_relevant_windows]

    indicies_arms_access_order = np.array(
        [tuple([i, j])
         for i in range(len(arm_names)) for j in range(len(arm_names))
         if i < j])
    column_names_reward_df = list(map(
        lambda x: arm_names[x[0]] + '-' + arm_names[x[1]], indicies_arms_access_order))

    # the correlation matrix is a quadratic symetric matrix, we will access the
    # top half above the diagonal to get the values for the pairwise
    # correlation of the arms
    access_top_of_diagonal = np.array([i < j
                                       for i in range(len(arm_names))
                                       for j in range(len(arm_names))])

    return pd.DataFrame(
        data=correlation_matrix
        [np.tile(access_top_of_diagonal, no_windows)].reshape(
            -1, len(column_names_reward_df)),
        columns=column_names_reward_df,
        index=indicies_for_new_df)


def _generate_windowed_reward_df(
        unified_metrics_df,
        window_size,
        step,
        correlation_method,
        kind='continous',
        **kwargs
):
    """Computes the DataFrame that contains the reward for each arm as columns
    and a row for each iteration.

    Args:
      unified_metrics_df (DataFrame): A DataFrame that contains the value of
      metrics as columns and a row for each iteration.
      window_size (int): Size of the sliding window
      window_step (int): Step of the sliding window
      correlation_method (string): Method use to compute the correlation for
      arms. One of 'pearson',
      kind (string): Type of reward function. One of 'continous', 'top' or
      'threshold'. Continous takes the correlation for arms as reward. Top
      computes a binary reward, where an arm gets an 1 if it is one of the
      top L highest correlated arms this iteration. L is passed in the kwargs.
      Threshold computes a binary reward where an arm gets a 1 if its
      correlation exceeds a certain threshold. The threshold is passed in the
      kwargs.
      **kwargs (dict): Possible keys are 'L' or 'threshold'.

    Returns:
      DataFrame: The DataFrame where columns contain the reward for arms and
      rows the iterations.

    """
    reward_df = _compute_pairwise_correlation_of_arms(
        unified_metrics_df, window_size, step, correlation_method)
    reward_df = reward_df.abs().replace([np.inf, -np.inf], np.nan).fillna(0.0)

    if kind == 'continous':
        return reward_df
    if kind == 'top':
        L = kwargs['L']
        L_highest_vals = np.sort(reward_df.values)[:, -L]
        highest_vals_ndarray = np.repeat(
            L_highest_vals, reward_df.shape[1]).reshape(-1, reward_df.shape[1])
        transformed_values = (
            reward_df.values >= highest_vals_ndarray).astype(int)
    elif kind == 'threshold':
        threshold = kwargs['threshold']
        transformed_values = (reward_df.values >= threshold).astype(int)

    reward_df = pd.DataFrame(
        columns=reward_df.columns, index=reward_df.index,
        data=transformed_values)
    return reward_df
