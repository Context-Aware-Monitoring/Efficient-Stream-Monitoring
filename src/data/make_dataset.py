"""Creates the required data for notebooks and training models."""

import os
import sys
from datetime import datetime
import pandas as pd
import reward

def clean_metrics_data(metrics_dir, start, end):
    """Cleans the metrics csv files by removing the rows that don't lie within
    the time window specified by start and end. Further linear interpolation is
    used to fill missing values.

    Args:
      metrics_dir (string): Directory where the metrics csv files are located.
      start (datetime): Start of the time window
      end (datetime): End of the time window
    """
    metrics_file_paths = []
    metrics_file_paths.extend(list(map(lambda x : metrics_dir + x, os.listdir(metrics_dir))))

    for current_path in metrics_file_paths:
        metrics_df = pd.read_csv(current_path)
        reward.convert_now_column_to_datetime(metrics_df)
        metrics_df = metrics_df.loc[(metrics_df.now >= start) & (metrics_df.now <= end),]

        metrics_df_single_timestamp = pd.pivot_table(metrics_df, index=['now'], aggfunc='mean')

        df_without_missing_timestamps = pd.DataFrame(
            index=pd.date_range(start=start, end=end, freq='1S')
        )
        df_without_missing_timestamps = df_without_missing_timestamps.merge(
            metrics_df_single_timestamp,
            how='left',
            left_index=True,
            right_index=True
        )
        df_without_missing_timestamps.interpolate(method='linear', axis=0, inplace=True)


        new_filepath = current_path.replace('raw', 'interim')
        new_file_dir = metrics_dir.replace('raw', 'interim')
        if not os.path.exists(new_file_dir):
            os.makedirs(new_file_dir)
        df_without_missing_timestamps.to_csv(new_filepath, index=True,index_label='now')

if __name__ == '__main__':
    args = sys.argv
    if len(args) == 1:
        print("Possible arguments are --rewards, --context, --all")
    args = args[1:]
    if "--all" in args:
        args = ['--clean', '--rewards']

    for arg in args:
        stime = datetime.now()
        if arg == '--clean':
            print("Cleaning metrics data")
            clean_metrics_data(
                '../../data/raw/sequential_data/metrics/',
                datetime(2019,11,19,18,38,39),
                datetime(2019,11,20,1,30,0)
            )
            clean_metrics_data(
                '../../data/raw/concurrent_data/metrics/',
                datetime(2019,11,19,16,12,13),
                datetime(2019,11,20,20,45,00)
            )
        elif arg == '--rewards':
            print("Generating rewards")
            reward.generate_reward_csv(
                [
                    '../../data/interim/sequential_data/metrics/wally113_metrics.csv',
                    '../../data/interim/sequential_data/metrics/wally117_metrics.csv',
                    '../../data/interim/sequential_data/metrics/wally122_metrics.csv',
                    '../../data/interim/sequential_data/metrics/wally123_metrics.csv',
                    '../../data/interim/sequential_data/metrics/wally124_metrics.csv'
                ]
            )
        else:
            sys.exit('Invalid argument %s found' %arg)
        print('Finished, took %d seconds' %
              ((datetime.now() - stime).seconds))
