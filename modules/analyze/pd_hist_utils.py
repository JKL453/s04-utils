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
from typing import Any, Tuple
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------#
# ---------------------- Histogram FUNCTIONS --------------------------#
# ---------------------------------------------------------------------#

def get_hist(dwell_times:list[int], 
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



def plot_hist(dwell_times:list[int],
                   bin_width:int=10,
                   title:str='Dwell Time Histogram', 
                   xlabel:str='Dwell Time (ms)', 
                   ylabel:str='Counts',
                   ) -> None:
    '''
    Plot histogram data.
    
    Parameters
    ----------
    dwell_times : list or array of int
        List or array of dwell times in milliseconds.
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
    _, ax = plt.subplots(1, 1, figsize=(4, 2))

    # Plot histogram data
    plt.hist(dwell_times, bins=n_bins, histtype='stepfilled')

    # Set plot title and axis labels
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Set logaritmic scale for y axis
    ax.set_yscale('log')

    # Show plot
    plt.show()



def get_prob(dwell_times:list[int], bin_width:int=10):
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
    bins : array
        Array of bin edges / x values.
    '''
    # Get histogram counts and bin edges from dwell times
    counts, bins = get_hist(dwell_times=dwell_times, bin_width=bin_width)

    # Get weight factors for calculating probability densities
    weight_factors = get_distances_to_neighbours(counts)

    # Calculate probability densities
    prob_desities = counts * weight_factors

    # remove bins with zero counts
    bins = bins[:-1]
    bins = bins[prob_desities > 0]

    # Remove zeroes from probabilities
    prob_desities = remove_zero_counts(prob_desities)

    return prob_desities, bins



def plot_prob(dwell_times:list[int],
                   bin_width:int=10,
                   title:str='Probability Density Plot', 
                   xlabel:str='Time (ms)', 
                   ylabel:str='Probability Density'
                   ) -> None:
    '''
    Plot probability density distribution.

    Parameters
    ----------
    dwell_times : array
        Array of dwell times in milliseconds.
    bin_width : int
        Width of histogram bins in milliseconds.
    title : str
        Title of probability density plot.
    xlabel : str
        Label for x-axis.
    ylabel : str
        Label for y-axis.

    Returns
    -------
    None
    '''
    # Get probability densities and bin edges from dwell times
    prob_densities, bins = get_prob(dwell_times, bin_width)

    # Create figure for probability density plot
    _, ax = plt.subplots(1, 1, figsize=(4, 2))

    # Plot probability density data
    ax.plot(bins, prob_densities, '.')

    # Set plot title and axis labels
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Set logaritmic scale for y axis
    #ax.set_xscale('log')
    ax.set_yscale('log')

    # Show plot
    plt.show()



def plot_hist_and_prob(dwell_times:list[int],
                    bin_width:int=10,
                    title:str='Probability Density Plot', 
                    xlabel:str='Time (ms)', 
                    ylabel:list[str]=['Counts', 'Probability Density']
                    ) -> None:
    '''
    Plot histogram of dwell times and the corresponding probability 
    density distribution.

    Parameters
    ----------
    dwell_times : array
         Array of dwell times in milliseconds.
    bin_width : int
         Width of histogram bins in milliseconds.
    title : str
         Title of probability density plot.
    xlabel : str
         Label for x-axis.
    ylabel : list of str
            Labels for y-axis.

    Returns
    -------
    None
    '''
    # Create figure for probability density plot
    _, ax = plt.subplots(2, 1, figsize=(4, 5))

    # Plot histogram data
    n_bins = int(np.max(dwell_times)/ bin_width)
    ax[0].hist(dwell_times, bins=n_bins, histtype='stepfilled')
    
    # Set plot title and axis labels
    ax[0].set_title(title)
    ax[0].set_xlabel(xlabel)
    ax[0].set_ylabel(ylabel[0])

    # Set logaritmic scale for y axis
    ax[0].set_yscale('log')

    # Get probability densities and bin edges from dwell times
    prob_densities, bins = get_prob(dwell_times, bin_width)

    # Plot probability density data
    ax[1].plot(bins, prob_densities, '.')

    # Set plot title and axis labels
    ax[1].set_title(title)
    ax[1].set_xlabel(xlabel)
    ax[1].set_ylabel(ylabel[1])

    # Set logaritmic scale for y axis
    #ax[1].set_xscale('log')
    ax[1].set_yscale('log')

    # Show plot
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------#
# ---------------------- Helper FUNCTIONS -----------------------------#
# ---------------------------------------------------------------------#



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

    # print non-zero values in distances
    non_zero_distances = []
    for i in range(len(distances)):
        if distances[i] > 0:
            non_zero_distances.append(distances[i])

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

    return recursive_distances


    



