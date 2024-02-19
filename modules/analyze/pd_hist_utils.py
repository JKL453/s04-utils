'''
pd_hist_utils.py - Module for calculating and plotting probability density functions

Description:
This module provides functions for calculating and plotting 
probability density functions (PDFs) from dwell time histogram data.

Functions:
- ...

Usage:
- ...

'''

# Import modules
from math import log
from typing import Any, Tuple
from click import style
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ---------------------------------------------------------------------#
# ---------------------- Histogram FUNCTIONS --------------------------#
# ---------------------------------------------------------------------#

def create_histogram(dwell_times:list[int], 
                     bin_width:int=10
                     )-> Tuple[np.ndarray, np.ndarray]:
    '''
    Create a histogram data from list of dwelltimes.
    
    Parameters
    ----------
    data : list or array of int
        List or array of dwell times in milliseconds.
    bin_width : int
        Width of histogram bins in milliseconds.

    Returns
    -------
    counts : array
        Array of histogram counts.
    bin_edges : array
        Array of bin edges.
    '''

    # Calculate number of bins and counts given dwell times
    n_bins = int(np.max(dwell_times)/ bin_width)
    counts, bin_edges = np.histogram(dwell_times, bins=n_bins)

    return counts, bin_edges



def plot_histogram(dwell_times:list[int],
                   bin_width:int=10,
                   title:str='Dwell Time Histogram', 
                   xlabel:str='Dwell Time (ms)', 
                   ylabel:str='Counts',
                   log_scale:Tuple[bool, bool] = (False, True)
                   ) -> None:
    '''
    Plot histogram data.
    
    Parameters
    ----------
    counts : array
        Array of histogram counts.
    bin_edges : array
        Array of bin edges.
    bin_width : int
        Width of histogram bins in milliseconds.
    title : str
        Title of histogram plot.
    xlabel : str
        Label for x-axis.
    ylabel : str
        Label for y-axis.

    Returns
    -------
    None
    '''
    n_bins = int(np.max(dwell_times)/ bin_width)

    # Create figure for seaborn histogram
    plt.figure(figsize=(4, 2))

    # Plot histogram data with seaborn
    sns.set_theme(style='ticks')
    sns.histplot(dwell_times, bins=n_bins, kde=False, log_scale=log_scale)

    # Set plot title and axis labels
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # Show plot
    plt.show()


def get_probabilities(dwell_times:list[int], bin_width:int=10):
    '''
    Calculate probability densities from dwell time histogram data.

    Parameters
    ----------
    dwell_times : list or array of int
        List or array of dwell times in milliseconds.
    bin_width : int
        Width of histogram bins in milliseconds.
    
    Returns
    -------
    prob_densities : array
        Array of probability densities.
    '''
    # Get histogram counts and bin edges from dwell times
    counts, _ = create_histogram(dwell_times=dwell_times, bin_width=bin_width)

    # Get weight factors for calculating probability densities
    weight_factors = get_distances_to_neighbours(counts)


    # Calculate probability densities
    prob_desities = counts * weight_factors

    # Remove zeroes from probabilities
    prob_desities = remove_zero_counts(prob_desities)

    return prob_desities



def plot_prob_density():
    pass



def remove_zero_counts(counts):
    '''
    Remove zero counts from histogram data.

    Parameters
    ----------
    counts : array
        Array of histogram counts.

    Returns
    -------
    counts : array
        Array of histogram counts with zero counts removed.
    '''
    # Remove zero counts from histogram data
    counts = counts[counts > 0]

    return counts



def get_distances_to_neighbours(counts:np.ndarray):
    '''
    
    '''
    # for each value in x that is greater than 0, 
    # get the distances to the next non-zero value to the left and right
    distances = []

    for i in range(len(counts)):
        if counts[i] > 0:
            if i == 0:
                distances.append(1.0)
            elif i == len(counts) - 1:
                distances.append(0)
            else:
                # get the distance to the next non-zero value to the left
                left = 1
                while counts[i - left] == 0:
                    left += 1
                # get the distance to the next non-zero value to the right
                right = 1
                while counts[i + right] == 0:
                    right += 1

                alg_mean_distance = (left + right) / 2
                distances.append(alg_mean_distance)


        else:
            distances.append(0)

    #print('distances')
    #print(distances)

    # print non-zero values in distances
    non_zero_distances = []
    for i in range(len(distances)):
        if distances[i] > 0:
            non_zero_distances.append(distances[i])

    #print('non_zero_distances_on')
    #print(non_zero_distances)

    # calculate recursive value for every non-zero value in distances
    recursive_distances = []
    for i in range(len(distances)):
        if i == 0:
            recursive_distances.append(distances[i])
        else:
            if distances[i] == 0:
                recursive_distances.append(0)
            else:
                recursive_distances.append(1/distances[i])
    
    #print('recursive_distances')
    #print(recursive_distances)

    return recursive_distances


    



