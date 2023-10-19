'''
HMM_analysis.py - Module for analyzing time traces with hidden Markov models

Description:
This module provides a class, which can be used to analyze time traces with
hidden Markov models. It contains methods for fitting the model to time traces, as well as
performing basic statistical analysis on the fitted model.

Classes:
- ...

Functions:
- ...

Usage:
...

'''

# Import modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from bokeh.plotting import figure, show, output_notebook, curdoc
from matplotlib_inline.backend_inline import set_matplotlib_formats
from pybaselines import Baseline
from sfHMM import sfHMM1
from sfHMM.gmm import GMMs

from s04utils.modules.load.Timestamps import Timestamps
from s04utils.modules.load.BinnedTimestamps import BinnedTimestamps
    

def correct_baseline(binned_timtrace):
    '''
    '''

    # Generate x data
    x_data = np.linspace(0, len(binned_timtrace), len(binned_timtrace))

    # Gete y data
    y_data = binned_timtrace
    
    # Create baseline fitter
    baseline_fitter = Baseline(x_data=x_data)

    # Fit baseline with improved modified polynomial 
    smoothed_imodpoly = baseline_fitter.imodpoly(y_data, poly_order=3, num_std=0.7)[0]

    return smoothed_imodpoly



def substract_baseline(binned_timtrace, baseline):
    '''
    '''

    # Substract baseline from binned time trace
    binned_timtrace = binned_timtrace - baseline

    # Sift data to positive values
    binned_timtrace = binned_timtrace - np.min(binned_timtrace)

    return binned_timtrace



def timetrace_to_dataframe(timetrace_data) -> pd.DataFrame:
    """Converts a timetrace to a pandas dataframe.

    Args:
        timetrace (dict): The timetrace dictionary.

    Returns:
        df (pd.DataFrame): The timetrace data as a pandas dataframe.
    """

   # Get the binned timetrace data for each individual detector
    detector_0 = timetrace_data['detector0'][0]
    detector_1 = timetrace_data['detector1'][0]

    # Get the binned timetrace data for the sum of both detectors
    detector_sum = timetrace_data['detector0'][0] + timetrace_data['detector1'][0]

    # create a pandas dataframe
    df = pd.DataFrame({'detector0': detector_0, 'detector1': detector_1, 'detector_sum': detector_sum})

    return df


def generate_x_data(binned_timestamps:BinnedTimestamps) -> np.ndarray:
    '''
    Returns an array with the x data for the timetrace plot.
    '''
    
    # Get bintime from BinnedTimestamps object
    bin_time = binned_timestamps.bin_width

    # Get timetrace length in seconds
    timetrace_len_in_s = binned_timestamps.len_seconds

    # Get number of data points in timetrace
    n_data_points = binned_timestamps.as_dataframe.shape[0]

    # Get digits of bin_time
    bin_time_digits = len(str(bin_time).split('.')[1])

    # Round timetrace_len_in_s to same number of digits as bin_time
    timetrace_len_in_s = round(timetrace_len_in_s, bin_time_digits) - 2*bin_time

    return np.linspace(0, timetrace_len_in_s, n_data_points)


def create_bokeh_plot(
    binned_timestamps:BinnedTimestamps, 
    width:int=800,
    height:int=300
    ) -> None:
    """
    Creates a bokeh plot for the timetrace data.
    """

    # Create figure
    p = figure(width=width, height=height)

    # Generate x data for the plot
    x = generate_x_data(binned_timestamps)

    # Get binned timestamps data
    detector_0 = binned_timestamps.as_dataframe['detector_0']
    detector_1 = binned_timestamps.as_dataframe['detector_1']
    detector_sum = binned_timestamps.as_dataframe['detector_sum']

    # Get bin time
    bin_time = binned_timestamps.bin_width

    # Plot the data
    p.line(x=x, y=detector_0 , line_width=1, color='red', legend_label='detector_0')
    p.line(x=x, y=detector_1, line_width=1, color='blue', legend_label='detector_1')
    p.line(x=x, y=detector_sum, line_width=1, color='lightgrey', legend_label='detector_sum')
    p.legend.location = 'top_right'

    # Set the axis labels
    p.xaxis.axis_label = 'time (s)'
    p.yaxis.axis_label = 'counts per ' + str(int(bin_time*1e3)) + ' ms'

    # Show the plot
    show(p)


def find_last_step(binned_timestamps:BinnedTimestamps) -> float:
    '''
    Returns the x value of the last step in binned timestamps.
    '''

    # Get binned timestamps data
    detector_0 = binned_timestamps.as_dataframe['detector_0']
    detector_1 = binned_timestamps.as_dataframe['detector_1']
    detector_sum = binned_timestamps.as_dataframe['detector_sum']

    
    sf_0 = sfHMM1(detector_0, krange=(2, 2), model='p').run_all(plot=False)

    # get the viterbi path
    steps_viterbi = sf_0.viterbi

    # get unique values
    unique, counts = np.unique(steps_viterbi, return_counts=True)
    print(dict(zip(unique, counts)))

    high_state = unique[1]
    low_state = unique[0]

    print('High state: ' + str(high_state))
    print('Low state: '+ str(low_state))

    steps = steps_viterbi.copy()

    # assign 0 and 1 to the states in viterbi path
    steps[steps_viterbi == high_state] = 1
    steps[steps_viterbi == low_state] = 0

    # find indices where the state changes
    value_change = np.where(np.diff(steps))[0]

    last_value_change = value_change[-1]

    # set cutoff value
    #CUT_OFFSET = int(last_value_change / 2)
    CUT_OFFSET = 10
    print('Last value change: ' + str(last_value_change))
    print('Cut offset: ' + str(CUT_OFFSET))
    print(len(steps))

    # plot the last value chnage as a vertical line
    plt.figure(figsize=(6, 2))
    plt.plot(steps)
    plt.axvline(last_value_change+CUT_OFFSET, color='red')
    plt.title('Viterbi path and cutoff after bleaching')
    plt.show()

    return last_value_change+CUT_OFFSET



    
    
