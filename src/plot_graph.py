import streamlit as st
import pandas as pd
import time
import os
import matplotlib.pyplot as plt


class Methods:

    def __init__(self, df):

        self.df = df

        pass

    def avgofperformance(self, s):

        avgcod = self.df[s].sum()

        indexcnt = self.df.shape[0]

        if indexcnt != 0:

            averageofcoding = (round((avgcod / indexcnt)))

            return averageofcoding

    def averages_of_team(self, wholedf, name_of_col):

        group_number = self.df['group'].iloc[0]

        team_df = wholedf[(wholedf['group'] == group_number)]

        team_df = team_df.drop_duplicates(subset='staff_name')

        tempsum = team_df[name_of_col].sum()

        tempindexcnt = team_df.shape[0]

        avg_team_cnt = round(tempsum / tempindexcnt)

        return avg_team_cnt
