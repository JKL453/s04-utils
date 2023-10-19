"""
Timestamps.py - Module for loading and processing timestamps data

This module provides functions for loading and processing timestamps 
data from various sources.
It contains functions for reading timestamps data from CSV files, 
filtering and processing timestampss, and performing basic statistical 
analysis on timestamps data.

Functions:
- load_timetrace: Load timestamps data from a H5 file and return as a
    timestamps class object.

Classes:
- timestamps: Class for loading and processing timestamps data.

"""

# import statements
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from IPython.display import set_matplotlib_formats
from bokeh.plotting import figure, show, output_notebook, curdoc
from bokeh.models import FuncTickFormatter
import seaborn as sns
import h5py

# loading function that returns an image class object
def load_from_path(path):
    '''
    Loads image data from a binary file and returns an image class object.
    '''
    return Timestamps(path)


# ---------------------------------------------------------------------#
# -------------------- TIMESTAMPS DATA CLASS --------------------------#
# ---------------------------------------------------------------------#

class Timestamps():
    """
    Class for loading and processing timestamps data.
    """

    def __init__(self, path):
        """
        Initialize the timestamps class object.
        """
        self.path = path
        self.file_name = self.path.split('/')[-1]
        self.h5_content = h5py.File(self.path, 'r')

        self.groups = self.get_h5_group_list()
        self.timestamps_raw = self.get_h5_dataset('photon_data0', 'timestamps')
        self.detectors_raw = self.get_h5_dataset('photon_data0', 'detectors')
        self.detector_count, self.detector_number = self.get_detector_count()

        self.data = self.get_timestamps_data()
        
        # initialize metadata, like number of 
        # - time stamps
        # - time trace length
        # - number of detectors
        # - comment string
        # - timestamps
        # - all h5 groups need to be read
    
    
    def __len__(self):
        """
        Return the number of timestamps.
        """
        return self.timestamps.shape[0]
    

    def get_h5_group_list(self):
        """
        Return a list of all h5 groups in the timestamps data.
        """
        h5_groups = []
        for group in self.h5_content.keys():
            if group != 'comment':
                h5_groups.append(group)

        return list(h5_groups)
    
    
    #def get_h5_group(self, group_name):
    #    """
    #    Return a h5 group from the timestamps data.
    #    """
    #    return self.h5_content[group_name]
    
    
    def get_h5_dataset_list(self, group_name):
        """
        Return a list of all h5 datasets in a h5 group.
        """
        return list(self.h5_content[group_name].keys())
    
    
    def get_h5_dataset(self, group_name, dataset_name):
        """
        Return a h5 dataset from a h5 group.
        """
        return self.h5_content[group_name][dataset_name][()]
    
    def get_h5_overview(self):
        """
        Print a list of all h5 groups and datasets in the timestamps data.
        """
        for group in self.groups:
            print(group)
            for dataset in self.get_h5_dataset_list(group):
                print('\t', dataset)
            print()


    def get_timestamps_data(self):
        """
        Return the timestamps data as Pandas Series.
        """
        # sort timestamps by detector
        sorted_timestamps = self.sort_timestamps(self.timestamps_raw, self.detectors_raw)

        # create series with timestamps of detector 0
        timestamps_0 = pd.Series(sorted_timestamps['timestamps_0'])
        timestamps_0.name = 'timestamps_0'

        # create series with timestamps of detector 1
        timestamps_1 = pd.Series(sorted_timestamps['timestamps_1'])
        timestamps_1.name = 'timestamps_1'

        # drop NaN values and cast to int64
        timestamps_0 = timestamps_0.dropna().astype('int64')
        timestamps_1 = timestamps_1.dropna().astype('int64')
        
        # create dictionary with timestamps series from both detectors
        timestaps_series_dict = {'detector_0': timestamps_0, 'detector_1': timestamps_1}

        # check if timestamps series contain data and if not, delete it
        #if timestaps_series_dict['detector_0'].empty:
        #    del timestaps_series_dict['detector_0']
        #elif timestaps_series_dict['detector_1'].empty:
        #    del timestaps_series_dict['detector_1']

        return timestaps_series_dict
    
    
    def sort_timestamps(self, timestamps, detectors):
        """
        Return the sorted timestamps data as Pandas dataframe.
        """
        h5_array_timestamps = np.array(timestamps)
        h5_array_detectors = np.array(detectors)

        # create dataframe with timestamps and detectors
        df = pd.DataFrame({'timestamps': h5_array_timestamps, 'detectors': h5_array_detectors})

        # sort timestamps by detector
        timestamps_detector_0 = df.replace(1, np.nan).dropna()['timestamps']
        timestamps_detector_1 = df.replace(0, np.nan).dropna()['timestamps']
        
        # reset index
        timestamps_detector_0 = timestamps_detector_0.reset_index(drop=True)
        timestamps_detector_1 = timestamps_detector_1.reset_index(drop=True)

        # create dataframe with sorted timestamps
        h5_data_sorted = pd.DataFrame({'timestamps_0': timestamps_detector_0, 
                                       'timestamps_1': timestamps_detector_1})
       
        return  h5_data_sorted
    

    def get_detector_count(self):
        """
        Return the number of datasets from different detectors.
        """
        h5_detectors = np.array(self.detectors_raw)
        detector_number = np.unique(h5_detectors)
        detector_count = len(detector_number)

        return detector_count, detector_number
    
    # Define the time-to-seconds conversion function
    def seconds_formatter(self, x, pos):
        return f"{x*5e-9:.0f}"  # Convert time to seconds and format as string
    
    def preview(self, bin_width=0.01):
        """
        Plot a preview of the timestamps data.
        """
        timestamps_0 = self.data['detector_0'].to_numpy()
        timestamps_1 = self.data['detector_1'].to_numpy()

        #timestamps = self.data['detector_0'].to_numpy()
        timetrace_len = timestamps_0[-1]
        timetrace_len_in_s = timetrace_len * 5e-9
        print(timetrace_len_in_s)
        print(timetrace_len)
        n_bins = timetrace_len_in_s/bin_width
        print(n_bins)

        _, ax = plt.subplots(figsize=(8, 2))

        preview = sns.histplot(timestamps_0, 
                               element="poly", 
                               fill=False, 
                               bins=int(np.floor(n_bins)), 
                               ax=ax,
                               linewidth=1,
                               color='#517BA1')
        
        sns.histplot(timestamps_1, 
                               element="poly", 
                               fill=False, 
                               bins=int(np.floor(n_bins)), 
                               ax=ax,
                               linewidth=1,
                               color='#CA4B43')
        
        
        #plt.xlim(0)
        plt.ylim(0)
        preview.set(xlabel='time (s)', 
                    ylabel='counts per ' + str(int(bin_width*1e3)) + ' ms',
                    title=str(self.file_name))

        # Set x-axis labels to seconds using FuncFormatter
        plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(self.seconds_formatter))

        plt.minorticks_on()
        plt.grid(linewidth = 0.5, alpha = 0.3, which = 'major')
        set_matplotlib_formats('retina')

        plt.show()

    def explore(self, bin_width=0.01):
        '''
        Displays an interactive Bokeh image plot for exploratory data analysis.
        '''
        timestamps_0 = self.data['detector_0'].to_numpy()
        timestamps_1 = self.data['detector_1'].to_numpy()


        timetrace_len = timestamps_0[-1]
        timetrace_len_in_s = timetrace_len * 5e-9

        n_bins = timetrace_len_in_s/bin_width
        bins = int(np.floor(n_bins))
        
        counts0, bins0 = np.histogram(timestamps_0, bins=bins)
        counts1, bins1 = np.histogram(timestamps_1, bins=bins)    


        p = figure(width=1000, height=250, title='Photon count histogram')
        p.line(bins0*5e-9, np.append(counts0, 5), line_color='#517BA1')
        p.line(bins1*5e-9, np.append(counts1, 5), line_color='#CA4B43')

        p.xaxis.axis_label = 'time (s)'
        p.yaxis.axis_label = 'counts per ' + str(int(bin_width*1e3)) + ' ms'
        p.title = str(self.file_name)

        p.xaxis.major_label_orientation = "horizontal"  # Set the orientation of the tick labels

        show(p)