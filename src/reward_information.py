"""Functionality to explore the reward csv file
"""

import pandas as pd
import numpy as np


class RewardInformation:
    """Provides explorative methods for a reward csv file.

    Attributes:
    -----------
    _reward_df (DataFrame): Contains the reward for each iteration

    Methods:
    --------
    get_top_correlated_arms(t):
        Gets the arms ordered by the measure of correlation for iteration t.
    """

    def __init__(self, reward_filepath):
        self._reward_df = pd.read_csv(reward_filepath, index_col=0)

    def get_top_correlated_arms(self, t):
        """Gets the arms ordered by the measure of correlation.

        Args:
          t (int): Iteration

        Returns:
          string[]: Arm names ordered by correlation
        """
        rewards = self._reward_df.loc[t].values
        arg_sorted = np.argsort(-1 * rewards)

        arms = [''] * len(rewards)
        for i, index in enumerate(arg_sorted):
            arms[i] = self._reward_df.columns[index]

        return arms

    def get_top_rank_count_for_arms(self):
        """Gets for each arm the number of times that he is the highest
        correlated arm. If more than one arm has the highest reward, the count
        for each of them gets incremented.

        Returns:
          dict: Maps arm to the number of times he is the highest correlated
          arm.
        """
        T = len(self._reward_df.index)

        arm_to_count = dict(zip(self._reward_df.columns.values, [0] * T))

        for t in range(T):
            top_correlated_arms = self.get_top_correlated_arms(t)
            top_reward = self._reward_df.loc[t, top_correlated_arms[0]]
            for arm in top_correlated_arms:
                if top_reward != self._reward_df.loc[t, arm]:
                    break
                arm_to_count[arm] += 1

        print('A total of %d different arms atleast once receive the highest reward' % len(
            list(filter(lambda x: x > 0, list(arm_to_count.values())))))
        return sorted(
            list(arm_to_count.items()),
            reverse=True, key=lambda x: x[1])

    def get_top_L_rank_count_for_arms(self, L):
        """Gets for each arm the number of times that he is the one of the
        highest L correlated arms. If the L-highest reward is shared by
        multiple arms, the count for each of them gets incremented.

        Args:
          L (int): Defines the threshold
        Returns:
          dict: Maps arm to the number of times he is in the highest L
          correlated arms.
        """
        T = len(self._reward_df.index)

        arm_to_count = dict(zip(self._reward_df.columns.values, [0] * T))

        for t in range(T):
            top_correlated_arms = self.get_top_correlated_arms(t)
            current_reward = self._reward_df.loc[t, top_correlated_arms[0]]
            for i, arm in enumerate(top_correlated_arms):
                if i >= L and current_reward != self._reward_df.loc[t, arm]:
                    break

                current_reward = self._reward_df.loc[t, arm]
                arm_to_count[arm] += 1

        print('A total of %d different arms atleast once receive a reward within the top-%d' %
              (len(list(filter(lambda x: x > 0, list(arm_to_count.values())))), L))
        return sorted(
            list(arm_to_count.items()),
            reverse=True, key=lambda x: x[1])

    def get_average_reward(self):
        """Returns a sorted dict containing the average regret for the arms.

        Returns:
          dict (string->float)
        """
        avg = self._reward_df.mean()

        arms_to_avg_reward = dict(avg)
        return sorted(list(arms_to_avg_reward.items()),
                      reverse=True, key=lambda x: x[1])
