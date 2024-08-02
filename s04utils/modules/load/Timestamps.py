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
from matplotlib_inline.backend_inline import set_matplotlib_formats
from bokeh.plotting import figure, show, output_notebook, curdoc
import seaborn as sns

from s04utils.modules.load import h5

# loading function that returns an image class object
def load_from_path(path, load_cursor_pos=False, load_event_dict=False):
    '''
    Loads image data from a binary file and returns an image class object.
    '''
    return Timestamps(path, load_cursor_pos=False, load_event_dict=False)


# ---------------------------------------------------------------------#
# -------------------- TIMESTAMPS DATA CLASS --------------------------#
# ---------------------------------------------------------------------#

class Timestamps():
    """
    Class for loading and processing timestamps data.
    """

    def __init__(self, path=None, data_dict=None, load_cursor_pos=False, load_event_dict=False):
        """
        Initialize the timestamps class object.
        """
        if data_dict is not None:
            self.from_dict(data_dict)
        elif path is not None:
            self.from_file(path, load_cursor_pos, load_event_dict)
        else:
            raise ValueError("Either path or data_dict must be provided")

    def from_file(self, path, load_cursor_pos, load_event_dict):
        """
        Initialize the timestamps class object from a file.
        """
        self.path = path
        self.file_name = self.path.split('/')[-1]
        self.h5_content = h5.load_h5_file(self.path)
        self.groups = h5.get_group_names(self.h5_content)
        self.timestamps_raw = h5.get_dataset_content(self.h5_content, 'photon_data0', 'timestamps')
        self.detectors_raw = h5.get_dataset_content(self.h5_content, 'photon_data0', 'detectors')
        self.comment = h5.get_comment_str(self.h5_content)
        self.detector_count, self.detector_number = self.get_detector_count()
        self.data = self.get_timestamps_data()

        if load_cursor_pos:
            self.cursor_pos = self.get_cursor_pos(self.comment)
        else:
            self.cursor_pos = None

        if load_event_dict:
            self.event_dict = self.get_event_dict(self.comment)
        else:
            self.event_dict = None

    def from_dict(self, data_dict):
        """
        Initialize the timestamps class object from a dictionary.
        """
        self.path = None
        self.file_name = None
        self.h5_content = None
        self.groups = None
        self.timestamps_raw = None
        self.detectors_raw = None
        self.comment = None
        self.detector_count, self.detector_number = None, None
        self.data = data_dict
        self.cursor_pos = None
        self.event_dict = None



    def __repr__(self):
        return f'Timestamps object with {len(self.data)} datasets.'
    


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
    

    def seconds_formatter(self, x, pos):
        """
        Time-to-seconds conversion function for matplotlib FuncFormatter
        """
        # Convert time to seconds and format as string
        return f"{x*5e-9:.0f}"
    

    def preview(self, bin_width=0.01):
        """
        Plot a preview of the binned timestamps data.
        """
        timestamps_0 = self.data['detector_0'].to_numpy()
        timestamps_1 = self.data['detector_1'].to_numpy()

        timetrace_len = timestamps_0[-1]
        timetrace_len_in_s = timetrace_len * 5e-9
        n_bins = timetrace_len_in_s/bin_width
        

        _, ax = plt.subplots(figsize=(8, 2))

        preview = sns.histplot(timestamps_0, 
                               element="poly", 
                               fill=False, 
                               bins=int(np.floor(n_bins)), 
                               ax=ax,
                               linewidth=0.8,
                               color='#517BA1',
                               alpha = 0.8,
                               label='Detector 0')
        
        sns.histplot(timestamps_1, 
                               element="poly", 
                               fill=False, 
                               bins=int(np.floor(n_bins)), 
                               ax=ax,
                               linewidth=0.8,
                               color='#CA4B43',
                               alpha = 0.8,
                               label='Detector 1')
        
        plt.ylim(0)
        preview.set(xlabel='time (s)', 
                    ylabel='counts per ' + str(int(bin_width*1e3)) + ' ms',
                    title=str(self.file_name))

        # Set x-axis labels to seconds using FuncFormatter
        plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(self.seconds_formatter))

        plt.minorticks_on()
        plt.grid(linewidth = 0.5, alpha = 0.3, which = 'major')
        set_matplotlib_formats('retina')
        plt.legend()
        plt.show()


    def explore(self, bin_width=0.01):
        """
        Explore the binned timestamps data in interactive plot.
        """
        timestamps_0 = self.data['detector_0'].to_numpy()
        timestamps_1 = self.data['detector_1'].to_numpy()

        timetrace_len = timestamps_0[-1]
        timetrace_len_in_s = timetrace_len * 5e-9

        n_bins = timetrace_len_in_s/bin_width
        bins = int(np.floor(n_bins))
        
        counts0, bins0 = np.histogram(timestamps_0, bins=bins)
        counts1, bins1 = np.histogram(timestamps_1, bins=bins)    

        p = figure(width=1000, height=250, title='Photon count histogram')
        p.line(bins0*5e-9, np.append(counts0, 5), line_color='#517BA1', alpha=0.8, legend_label='Detector 0')
        p.line(bins1*5e-9, np.append(counts1, 5), line_color='#CA4B43', alpha=0.8, legend_label='Detector 1')

        p.xaxis.axis_label = 'time (s)'
        p.yaxis.axis_label = 'counts per ' + str(int(bin_width*1e3)) + ' ms'
        p.title = str(self.file_name)
        p.xaxis.major_label_orientation = "horizontal"

        output_notebook()

        show(p)



    def get_cursor_pos(self, comment: str) -> dict:
        """
        Returns the cursor position in the h5 file.
        """
    
        # Check if comment string is empty
        if comment == '':
            return None
    
        else:
            # Parse the comment
            comment = comment.split('\n')

            # Strip spaces from the comment
            comment = [line.strip() for line in comment]
            cursor_pos = comment[1:3]
    
            x_pos = cursor_pos[0].split(' ')[-2]
            y_pos = cursor_pos[1].split(' ')[-2]

            x_pos = float(x_pos.replace(',', '.'))
            y_pos = float(y_pos.replace(',', '.'))
    
            return {'x_pos': x_pos, 'y_pos': y_pos}



    def get_event_dict(self, comment: str) -> dict:
        """
        Returns a dictionary containing event information parsed from the given comment.

        Args:
            comment (str): The comment string containing event information.
        Returns:
            dict: A dictionary containing event information.
        """
        if comment == '':
            return None
        else:
            event_dict = self.parse_comment(comment)
            event_list = self.create_event_list(event_dict)
            event_keys = self.get_event_keys(event_list)
            event_dict = self.create_event_dict(event_keys, event_list)
            start_time, end_time = self.calculate_time_difference(event_dict)
            return event_dict

    def parse_comment(self, comment: str) -> list:
        comment = comment.split('\n')
        comment = [line.strip() for line in comment]
        return comment[1:]

    def create_event_list(self, event_dict: list) -> list:
        event_list = [event.split(':') for event in event_dict]
        event_list = event_list[3:-1]
        return event_list
    
    def get_event_keys(self, event_list: list) -> list:
        event_keys = []
        for item in event_list:
            if item[1].strip() not in event_keys:
                event_keys.append(item[1].strip())
        return event_keys
    
    def create_event_dict(self, event_keys: list, event_list: list) -> dict:
        event_dict = {key: list() for key in event_keys}
        for item in event_list:
            key = item[1].strip()
            value = item[0]
            time_str = value.replace(',', '.')
            time_float = float(time_str)
            event_dict[key].append(time_float)
        return event_dict
    
    def calculate_time_difference(self, event_dict: dict) -> tuple:
        start_time = event_dict['measurementStart'][0]
        end_time = event_dict['measurementStop'][0]
        for key in event_dict:
            event_dict[key] = [round(time - start_time, 2) for time in event_dict[key]]
        return start_time, end_time
