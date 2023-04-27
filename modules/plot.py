"""
plot.py - Module for data visualization

This module provides functions for visualizing data using various plotting techniques.
It contains functions for creating line plots, scatter plots, bar plots, and other types
of plots to visualize data in different formats, such as time traces, images, and other
types of data.

Functions:
- plot_timetrace: Create a timetrace plot from data.

Usage:
from modules import plot


"""

# import statements
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------#
# --------------------- TIMESTAMPS PLOTTING ---------------------------#
# ---------------------------------------------------------------------#

def timetrace(timestamps, bin_width=0.01):
    '''
    Create a timetrace plot from timestamps data.
    '''
    timetrace_len = timestamps[-1]
    timetrace_len_in_s = timetrace_len * 1e-9
    n_bins = timetrace_len_in_s/bin_width

    sns.histplot(timestamps, element="poly", fill=False, bins=int(np.floor(n_bins)))
    plt.show()














# ---------------------------------------------------------------------#
# ------------------------ IMAGE PLOTTING -----------------------------#
# ---------------------------------------------------------------------#