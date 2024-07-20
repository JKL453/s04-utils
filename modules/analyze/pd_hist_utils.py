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
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.ticker import FuncFormatter



# ---------------------------------------------------------------------#
# --------------------- CALCULATION FUNCTIONS -------------------------#
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



def get_log_hist(dwell_times:list[int], 
                 number_of_bins:int=20,
                 density:bool=True
                ) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Create a logarithmic histogram data from list of dwelltimes.

    Parameters
    ----------
    dwell_times : list or array of int
        List or array of dwell times in milliseconds.
    number_of_bins : int
        Number of bins for logarithmic histogram.
    density : bool
        If True, the result is the value of the probability density function 
        at the bin, normalized such that the integral over the range is 1.

    Returns
    -------
    counts : array
        Array of histogram counts.
    bins : array
        Array of bin edges.

    '''
    # Create log bins
    max_value = np.max(dwell_times)

    # Create logarithmic bins
    log_bins = np.logspace(np.log10(1), np.log10(max_value), number_of_bins)

    # Create histogram data
    counts, bins = np.histogram(dwell_times, bins=log_bins, density=density)

    # Center bins
    bins = (bins[1:] + bins[:-1]) / 2

    # Remove zeroes from probabilities
    counts_cleaned = counts[counts > 0]
    bins_cleaned = bins[counts > 0] 

    return counts_cleaned, bins_cleaned

        


def get_prob(dwell_times:list[int], 
             bin_width:int=10,
             method:str='neighbour'
             ) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Calculate probability densities from dwell time histogram data.

    Parameters
    ----------
    dwell_times : list or array of int
        List or array of dwell times in milliseconds.
    bin_width : int
        Width of histogram bins in milliseconds.
    method : str
        Method for calculating probability densities.
        'neighbour' : Use distances to neighbours for weighting.
        'bin_width' : Use bin width for weighting.
    
    Returns
    -------
    prob_densities : array
        Array of probability densities.
    bins : array
        Array of bin edges / x values.
    '''
    # Get histogram counts and bin edges from dwell times
    counts, bins = get_hist(dwell_times=dwell_times, bin_width=bin_width)

    if method == 'neighbour':
        # Get weight factors for calculating probability densities
        weight_factors = get_distances_to_neighbours(counts)
    elif method == 'bin_width':
        # Get weight factors for calculating probability densities
        weight_factors = np.ones(len(counts)) * 1/bin_width

    # Calculate probability densities
    prob_desities = counts * weight_factors

    # remove bins with zero counts
    bins = bins[:-1]
    bins = bins[prob_desities > 0]

    # Remove zeroes from probabilities
    prob_desities = remove_zero_counts(prob_desities)

    return prob_desities, bins



def get_log_prob(dwell_times:list[int], 
                 bin_width:int=10,
                 method:str='neighbour'
                 ) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Calculate logarithmic probability densities from dwell time data.

    Parameters
    ----------
    dwell_times : list or array of int
        List or array of dwell times in milliseconds.
    bin_width : int
        Width of histogram bins in milliseconds.
    method : str
        Method for calculating probability densities.
        'neighbour' : Use distances to neighbours for weighting.
        'bin_width' : Use bin width for weighting.
    
    Returns
    -------
    log_prob_densities : array
        Array of logarithmic probability densities.
    bins : array
        Array of bin edges / x values.
    '''
    # Get probability densities and bin edges from dwell times
    prob_densities, bins = get_prob(dwell_times, bin_width, method=method)

    '''if method == 'bin_width':
        # Remove specified value from prob_densities and corresponding bins
        prob_densities, bins = clean_prob_and_bins(prob_densities, bins, bin_width)'''

    # add small offset to bins to avoid log(0)
    bins[0] = bin_width

    # get log values for on_bins and weighted_on_counts
    log_bins = np.log(bins)
    log_prob_densities = np.log(prob_densities)

    # remove infinite from arrays
    log_bins = log_bins[np.isfinite(log_prob_densities)]
    log_prob_densities = log_prob_densities[np.isfinite(log_prob_densities)]

    # remove NaN from arrays
    log_bins = log_bins[~np.isnan(log_prob_densities)]
    log_prob_densities = log_prob_densities[~np.isnan(log_prob_densities)]

    # Remove zeroes from probabilities and remove corresponding bins
    
    # Set first bin to 0 
    #log_bins[0] = 0

    return log_prob_densities, log_bins



# ---------------------------------------------------------------------#
# ---------------------- PLOTTING FUNCTIONS ---------------------------#
# ---------------------------------------------------------------------#



def plot_hist(dwell_times:list[int],
                   bin_width:int=10,
                   title:str='Dwell Time Histogram', 
                   xlabel:str='Dwell Time (s)', 
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

    # Create formatter
    formatter = FuncFormatter(format_func)

    # Apply formatter to x axis
    plt.gca().xaxis.set_major_formatter(formatter)

    plt.minorticks_on()

    # Show plot
    plt.show()



def plot_prob(dwell_times:list[int],
                   bin_width:int=10,
                   method:str='neighbour',
                   title:str='Probability Density Plot', 
                   xlabel:str='Time (s)', 
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
    method : str
        Method for calculating probability densities.
        'neighbour' : Use distances to neighbours for weighting.
        'bin_width' : Use bin width for weighting.
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
    prob_densities, bins = get_prob(dwell_times, bin_width, method=method)

    # Create figure for probability density plot
    _, ax = plt.subplots(1, 1, figsize=(4, 2))

    # Plot probability density data
    ax.plot(bins, prob_densities, '.')

    # Set plot title and axis labels
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Set logaritmic scale for y axis
    ax.set_yscale('log')

    # Set minor ticks for y axis
    minor_locator = ticker.LogLocator(base=10.0,subs=(0.2,0.4,0.6,0.8),numticks=12)
    ax.yaxis.set_minor_locator(minor_locator)
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())

    # Set major ticks for y axis
    major_locator = ticker.LogLocator(numticks=12)
    ax.yaxis.set_major_locator(major_locator)

    # Create formatter
    formatter = FuncFormatter(format_func)

    # Apply formatter to x axis
    plt.gca().xaxis.set_major_formatter(formatter)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())

    # Show plot
    plt.show()



def plot_hist_and_prob(dwell_times:list[int],
                    bin_width:int=10,
                    method:str='neighbour',
                    title:str='Probability Density Plot', 
                    xlabel:str='Time (s)', 
                    ylabel:list[str]=['Counts', 'Probability Density'],
                    tick_position:str='outside'
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
    method : str
        Method for calculating probability densities.
        'neighbour' : Use distances to neighbours for weighting.
        'bin_width' : Use bin width for weighting.
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
    _, ax = plt.subplots(2, 1, figsize=(4, 4), sharex=True, gridspec_kw={'hspace': 0.05})

    # Plot histogram data
    n_bins = int(np.max(dwell_times)/ bin_width)
    ax[0].hist(dwell_times, bins=n_bins, histtype='stepfilled')

    print(n_bins)
    print(np.max(dwell_times))
    
    # Set plot title and axis labels
    ax[0].set_title(title)
    #ax[0].set_xlabel(xlabel)
    ax[0].set_ylabel(ylabel[0])

    # Set logaritmic scale for y axis
    ax[0].set_yscale('log')


    # Get probability densities and bin edges from dwell times
    prob_densities, bins = get_prob(dwell_times, bin_width, method=method)

    # Plot probability density data
    ax[1].plot(bins, prob_densities, '.')

    # Set plot title and axis labels
    #ax[1].set_title(title)
    ax[1].set_xlabel(xlabel)
    ax[1].set_ylabel(ylabel[1])

    # Set logaritmic scale for y axis
    #ax[1].set_xscale('log')
    ax[1].set_yscale('log')
    
    # Set minor ticks for y axis
    minor_locator = ticker.LogLocator(base=10.0,subs=(0.2,0.4,0.6,0.8),numticks=12)
    ax[1].yaxis.set_minor_locator(minor_locator)
    ax[1].yaxis.set_minor_formatter(ticker.NullFormatter())

    # Set major ticks for x axis
    major_locator = ticker.LogLocator(numticks=12)
    ax[1].yaxis.set_major_locator(major_locator)

    # Format x axis labels to seconds
    formatter = FuncFormatter(format_func)

    # Apply formatter to x axis
    plt.gca().xaxis.set_major_formatter(formatter)

    # Set minor ticks for x axis
    ax[0].xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax[1].xaxis.set_minor_locator(ticker.AutoMinorLocator())

    if tick_position == 'inside':
        for a in ax:
            # Set ticks position
            a.yaxis.set_ticks_position('both')
            a.xaxis.set_ticks_position('both')

            # Set ticks direction
            a.yaxis.set_tick_params(direction='in', which='both')
            a.xaxis.set_tick_params(direction='in', which='both')

    # Show plot
    plt.show()



def plot_power_law_fit(dwell_times:list[int],
                        bin_width:int=10,
                        method:str='neighbour',
                        title:str='Power Law Fit',
                        xlabel:str='Time (s)',
                        ylabel:str='Probability Density',
                        index_str:str=''
                          ) -> None:
    '''
    Plot power law fit to probability density distribution.

    Parameters
    ----------
    dwell_times : array
        Array of dwell times in milliseconds.
    bin_width : int
        Width of histogram bins in milliseconds.
    method : str
        Method for calculating probability densities.
        'neighbour' : Use distances to neighbours for weighting.
        'bin_width' : Use bin width for weighting.
    title : str
        Title of power law fit plot.
    xlabel : str
        Label for x-axis.
    ylabel : str
        Label for y-axis.
    index_str : str
        SUbscript text for slope displayed on plot.

    Returns
    -------
    None
    '''
    # Create figure for power law fit plot
    _, ax = plt.subplots(1, 1, figsize=(3, 4))
    #ax.set_aspect('equal')

    if method == 'neighbour':
        # Get probability densities and bin edges from dwell times
        prob_densities, bins = get_prob(dwell_times, bin_width, method=method)

        # Get logarithmic probability densities and bin edges from dwell times
        log_prob_densities, log_bins = get_log_prob(dwell_times, bin_width, method=method)

        # Fit power law to logarithmic probability density distribution
        pl_fit_func, coeffs = get_power_law_fit(log_prob_densities, log_bins, method=method)

        # add small offset to log_bins_on and on_bins to avoid log(0)
        bins[0] = bin_width

        # Plot probability density data
        ax.plot(bins, prob_densities, '.')

        # Plot power law fit
        ax.loglog(bins, pl_fit_func(bins))

    elif method == 'bin_width':
        counts_pdh, bins_pdh = get_log_hist(dwell_times, number_of_bins=bin_width, density=True)
        fit_func, coeffs = get_power_law_fit(counts_pdh, bins_pdh, method=method)

        # Plot probability density data
        ax.plot(bins_pdh, counts_pdh, '.')

        # Plot the fitted line
        log_x_lin_space = coeffs[2:]
        plt.loglog(np.power(10, log_x_lin_space), np.power(10, fit_func(log_x_lin_space)))

    else:
        raise ValueError('Invalid method for power law fit.')

    # Display the power law exponent in the plot
    tex_label = r'$m_{{{}}}: $'.format(index_str)
    ax.text(0.95, 0.95, tex_label+f'{coeffs[0]:.2f}', 
            ha='right', va='top', transform=plt.gca().transAxes)

    # Set plot title and axis labels
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Set minor ticks
    minor_locator = ticker.LogLocator(base=10.0, subs='auto', numticks=12)
    ax.xaxis.set_minor_locator(minor_locator)
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())

    # Set major ticks
    major_locator = ticker.LogLocator(numticks=12)
    ax.xaxis.set_major_locator(major_locator)

    # Show plot
    plt.show()



# ---------------------------------------------------------------------#
# ---------------------- FITTING FUNCTIONS ----------------------------#
# ---------------------------------------------------------------------#


def get_power_law_fit(log_prob_densities:np.ndarray, 
                      log_bins:np.ndarray,
                      method:str='neighbour'
                      ) -> Tuple[Any, np.ndarray]:
    '''
    Fit power law to probability density distribution and return fit 
    function and fit coefficients.

    Parameters
    ----------
    prob_densities : array
        Array of logarithmic probability densities.
        (non log values for bin width method)
    bins : array
        Array of logarithmic bin edges / x values.
        (non log values for bin width method)
    
    Returns
    -------
    fit_func : function
        Fit function for power law.
    coeffs : array
        Coefficients of power law fit.
        coeffs[0] = slope
        coeffs[1] = intercept
        coeffs[2:] = x values for fit function (when using bin width method)
    '''
    if method == 'neighbour':
        # Fit power law to probability density distribution
        coeffs = np.polyfit(log_bins, log_prob_densities, deg=1)
        poly = np.poly1d(coeffs)
        fit_func = lambda x: np.exp(poly(np.log(x)))

        return fit_func, coeffs
    
    elif method == 'bin_width':
        log_bins = np.log10(log_bins)
        log_counts = np.log10(log_prob_densities)

        # Fit power law to probability density distribution
        coeffs = np.polyfit(log_bins, log_counts, deg=1)
        fit_func = np.poly1d(coeffs)

        # Create linear space for x values
        log_x_lin_space = np.linspace(min(log_bins), max(log_bins), 500)

        # Append linspace to coefficients
        coeffs = np.append(coeffs, log_x_lin_space)

        return fit_func, coeffs
    
    else:
        raise ValueError('Invalid method for power law fit.')



# ---------------------------------------------------------------------#
# ---------------------- HELPER FUNCTIONS -----------------------------#
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



def get_distances_to_neighbours(counts:np.ndarray) -> np.ndarray:
    '''
    Calculates the distances to the next non-zero value to the left and right
    for each value in the histogram data.

    Parameters
    ----------
    counts : array
        Array of histogram counts.

    Returns
    -------
    recursive_distances : array
        Array of recursive distances to the next non-zero value.
    '''
    # for each value in x that is greater than 0, 
    # get the distances to the next non-zero value to the left and right
    distances = []

    for i in range(len(counts)):
        if counts[i] > 0:
            # Set distance of first value to 1
            if i == 0:
                right = 1
                while counts[i + right] == 0:
                    right += 1
                distances.append(right/2)
                print('right:', right)
            # Set distance of last value to distance to the left
            elif i == len(counts) - 1:
                left = 1
                while counts[i - left] == 0:
                    left += 1
                distances.append(left/2)
            else:
                # get the distance to the next non-zero value to the left
                left = 1
                while counts[i - left] == 0:
                    left += 1
                # get the distance to the next non-zero value to the right
                right = 1
                while counts[i + right] == 0:
                    right += 1

                alg_mean_distance = (left + right)/2
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

    return np.array(recursive_distances)



def clean_prob_and_bins(prob_densities:np.ndarray, 
                        bins:np.ndarray, 
                        bin_width:int
                        ) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Clean probability densities and corresponding bins by removing
    specified values.
    '''
    # Remove specified value from prob_densities and corresponding bins
    prob_densities_cleaned = prob_densities[prob_densities != (1/bin_width)]
    bins_cleaned = bins[prob_densities != (1/bin_width)]

    #print('prob_densities_cleaned:', prob_densities_cleaned)
    #print('bins_cleaned:', bins_cleaned)

    return prob_densities_cleaned, bins_cleaned



def format_func(value, tick_number):
    '''
    Format function for histogram x-axis.
    '''
    return int(value / 1000)
