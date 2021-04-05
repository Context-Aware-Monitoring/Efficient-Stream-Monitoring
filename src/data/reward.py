"""Generates rewards csv files for a bandit algorithm.
"""
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from sklearn.metrics import mutual_info_score

corr_methods = {
    'MI': lambda window, arms: _calculate_mutual_information_for_window_df(window, arms),
    'pear': lambda window, arms: _calculate_pearson_coefficient_for_window_df(window, arms)
}


def generate_reward_csv(
        metrics_file_paths,
        window_size=30,
        step=5,
        corr='MI',
        normalize=True,
        sequential=True,
        outdir='../../data/processed/',
        kind='continous',
        **kwargs
):
    """Writes a reward.csv file that can be processed by a bandit
    algorithm.

    Args:
      metrics_file_paths (string[]): Metrics files that are used to compute the
      reward.
      window_size (int): For the calculation of the reward function a sliding
      window is used. window_size specifies the size of this sliding window.
      step (int): Specifies the step of the sliding window.
      corr (string): Specifies the measure of correlation used to compute the
      correlation between metrics. Available options are 'MI'.
      normalize (bool): Weather or not to normalize the reward.
      outdir (string): Specifies the directory where to write the generated
      file.

    Returns:
      String: Filepath of the generated file.
    """
    unified_metrics_df = _generate_unified_metrics_dataframe(
        metrics_file_paths)

    reward_df = _generate_windowed_reward_df(
        unified_metrics_df,
        window_size,
        step,
        corr,
        kind,
        **kwargs
    )

    if normalize:
        _normalize_reward_df(reward_df)

    seq_or_con = 'seq' if sequential else 'con'
    if kind == 'top':
        kind += '_' + str(kwargs['L'])
    elif kind == 'threshold':
        kind += '_' + str(kwargs['threshold'])
    filepath = ("%s%s_rewards_w%d_s%d_%s_n%d_%s.csv" %
                (outdir, seq_or_con, window_size, step, corr, normalize, kind))
    reward_df.to_csv(filepath, index=False)

    print('Wrote file %s' % filepath)

    return filepath


def convert_now_column_to_datetime(dataframe):
    """Converts the now column in the DataFrame to a datetime object.

    Args:
      df (DataFrame): DataFrame containing a now column

    """
    def ts_to_datetime(ts): return datetime.strptime(
        ts.split(' CEST')[0], '%Y-%m-%d %H:%M:%S')
    dataframe['now'] = dataframe['now'].apply(ts_to_datetime)


def _generate_unified_metrics_dataframe(paths):
    """Generates a unified data frame containing all the individual metrics csv
    files passed as a parameter.

    Args:
      path (string): Path to folder containing individual csv files of metrics
    Returns:
      DataFrame containing the metrics for all hosts
    """
    hostnames = [x.split('/')[-1].split('_')[0] for x in paths]
    hosts_df = [None] * len(hostnames)

    i = 0
    for i, current_path in enumerate(paths):
        metrics_df = pd.read_csv(current_path)
        convert_now_column_to_datetime(metrics_df)
        metrics_df = metrics_df.set_index('now')
        hosts_df[i] = pd.pivot_table(metrics_df, index=['now'], aggfunc='mean')

        hosts_df[i].columns = list(
            map(lambda x: hostnames[i] + '.' + x, hosts_df[i].columns.values))

    unified_df = hosts_df[0]

    for h_df in hosts_df[1:]:
        unified_df = unified_df.merge(
            h_df, how='left', left_index=True, right_index=True)

    return unified_df


def _generate_windowed_reward_df(
        unified_metrics_df,
        window_size,
        step,
        corr,
        kind='continous',
        **kwargs
):
    """
    Generated a DataFrame for the rewards based on the passed dataframe
    containing the metrics data.
    For each pair of columns in the metrics DataFrame a column in the reward
    DataFrame will be created, representing an arm.
    The columnes will be aggregated using a sliding window approach.

    Args:
      unified_metrics_df (DataFrame): DataFrame where the columns represent a
      metric and the rows represent a timestamp.
      window_size (int): Size of the sliding window
      step (int): Step of the sliding window
      corr (string): Measure of correlation that is used. Possible options are:
      'MI'
      kind: Specify how the reward is computed.
      Options:
        continuos: Just reward the measure of correlation
        top: A binary reward, 1 if arm is one of L highest correlated arms, 0
        if not. L is passed in **kwargs.
        threshold: A binary reward is rewarded for arms whose correlation
        exceeds a threshold. The threshold is passed in **kwargs.

    Returns:
      DataFrame: An empty DataFrame where each column represents an arm.
    """
    arms = []
    for i, column1 in enumerate(unified_metrics_df.columns):
        for j, column2 in enumerate(unified_metrics_df.columns):
            if i == j or j < i:
                continue
            # please dont change -, gets split later based on this
            arms.append(column1 + '-' + column2)

    reward_df = pd.DataFrame(columns=arms)

    i = 0
    cur = unified_metrics_df.index[0]
    end = unified_metrics_df.index[-1]
    timedelta_window = timedelta(seconds=window_size - 1)

    while (cur + timedelta_window) <= end:
        window = unified_metrics_df.loc[pd.date_range(
            start=cur, end=(cur + timedelta_window), freq='1S')]
        correlation_for_arms = corr_methods[corr](window, arms)
        if kind == 'continous':
            reward_df.loc[i] = correlation_for_arms
        elif kind == 'threshold':
            threshold = kwargs['threshold']
            reward_df.loc[i] = list(
                map(lambda cv: cv >= threshold, correlation_for_arms))
        elif kind == 'top':
            L = kwargs['L']
            Lhighest_value = sorted(correlation_for_arms)[-L]
            reward_df.loc[i] = list(
                map(lambda cv: cv >= Lhighest_value, correlation_for_arms))
        i += 1
        cur += timedelta(seconds=step)

    return reward_df


def _normalize_reward_df(reward_df):
    """
    Normalizes the reward DataFrame using the min-max normalization.

    Args:
      reward_df (DataFrame): DataFrame containing the measure of correlation
      for each arm and timestamp.
    """
    # if normalize:
    #     xmax = max(reward_df.max())
    #     xmin = min(reward_df.min())

    #     norm = lambda x : (x - xmin) / (xmax - xmin)
    #     reward_df = reward_df.apply(norm)
    row_min = 0
    row_max = 0
    for i in reward_df.index:
        row_max = reward_df.loc[i].max()
        row_min = reward_df.loc[i].min()

        def norm(cell_value):
            return (cell_value-row_min) / (row_max-row_min)
        reward_df.loc[i] = reward_df.loc[i].apply(norm)


def _calculate_mutual_information_for_window_df(window_df, arms):
    """Computes the pairwise mutual information for the arms (pairs of metrics)
    in the window_df dataframe.

    Args:
      window_df (DataFrame): DataFrame containing the metrics values for each
      arm.
      arms (string[]): Array of strings representing the arms (pairs of
      metrics).
      An arm is represented by the string representation metric1-metric2.

    Returns
      Float[]: The mutual information for the arms.
    """
    reward = [0] * len(arms)

    for i, column_name in enumerate(arms):
        individual_columns = column_name.split('-')
        mutual_information = mutual_info_score(
            window_df[individual_columns[0]], window_df[individual_columns[1]]
        )
        reward[i] = mutual_information

    return reward


def _calculate_pearson_coefficient_for_window_df(window_df, arms):
    """Computes the pairwise absolute pearson coefficient for the arms (pairs
    of metrics) in the window_df dataframe.

    Args:
      window_df (DataFrame): DataFrame containing the metrics values for each
      arm.
      arms (string[]): Array of strings representing the arms (pairs of
      metrics).
      An arm is represented by the string representation metric1-metric2.

    Returns
      Float[]: The absolute of the pearson correlation for the arms.
    """
    reward = [0] * len(arms)
    for i, column_name in enumerate(arms):
        individual_columns = column_name.split('-')
        correlation_matrix = np.corrcoef(
            window_df[individual_columns[0]], window_df[individual_columns[1]]
        )
        reward[i] = 0.0 if np.isnan(correlation_matrix[0][1]) else abs(
            correlation_matrix[0][1])

    return reward
