"""
BinnedTimestamps.py - Module for loading and processing binned timestamps data.

Classes:
    - BinnedTimestamps
"""

# import statements
import numpy as np
import pandas as pd

from s04utils.modules.load.Timestamps import Timestamps

# ---------------------------------------------------------------------#
# ----------------  BINNED TIMESTAMPS DATA CLASS ----------------------#
# ---------------------------------------------------------------------#


class BinnedTimestamps():
    '''
    Small test class for binned timetrace data.
    '''

    def __init__(self, path:str=None, bin_width:float=0.01, data_dict:dict=None):
        if data_dict:
            self.from_dict(data_dict, bin_width)
        else:
            self.from_path(path, bin_width)

    def from_dict(self, data_dict, bin_width:float=0.01):
        """
        Initialize the BinnedTimestamps class object from a dictionary.
        """
        self.path = None
        self.file_name = None
        self.h5_content = None
        self.groups = None
        self.timestamps_raw = None
        self.detectors_raw = None
        self.comment = None
        self.detector_count, self.detector_number = None, None
        self.cursor_pos = None
        self.event_dict = None
        self.bin_width = bin_width
        self.timestamps = Timestamps(data_dict=data_dict)
        self.data = self.bin_timestamps()
        self.raw = self.timestamps.data
        self.len_seconds = self.get_length()

    def from_path(self, path:str, bin_width:float=0.01):
        """
        Initialize the BinnedTimestamps class object from a file path.
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
        self.bin_width = bin_width
        self.timestamps = Timestamps(path)
        if load_cursor_pos:
            self.cursor_pos = self.get_cursor_pos(self.comment)
        else:
            self.cursor_pos = Non   
        if load_event_dict:
            self.event_dict = self.get_event_dict(self.comment)
        else:
            self.event_dict = None


    def bin_timestamps(self) -> dict:
        """
        Return the timetrace data of both detectors as a dictionary of numpy arrays.
        """

        # Get timestamps data
        timestamps_0 = self.timestamps.data['detector_0'].to_numpy()
        timestamps_1 = self.timestamps.data['detector_1'].to_numpy()
        
        # Get timetrace length in seconds
        timetrace_len = timestamps_0[-1]
        timetrace_len_in_s = timetrace_len * 5e-9

        # Calculate number of bins
        n_bins = timetrace_len_in_s/self.bin_width
        bins = int(np.floor(n_bins))

        # Calculate counts per bin
        counts_0, bins_0 = np.histogram(timestamps_0, bins=bins)
        bins_0 = bins_0[0:-1]
        counts_1, bins_1 = np.histogram(timestamps_1, bins=bins)
        bins_1 = bins_1[0:-1]

        return {'detector_0': [counts_0, bins_0], 'detector_1': [counts_1, bins_1]}
    
    
    def get_length(self) -> float:
        '''
        Returns the length of the binned timetrace in seconds.
        '''
        return round(self.data['detector_0'][1][-1] * 5e-9, 2)

    
    @property
    def as_dataframe(self) -> pd.DataFrame:
        '''
        Returns the binned timetrace data as a pandas dataframe.
        '''
        
        # Get the binned timetrace data for each individual detector
        detector_0 = self.data['detector_0'][0]
        detector_1 = self.data['detector_1'][0]

        # Get the binned timetrace data for the sum of both detectors
        detector_sum = detector_0 + detector_1

        return pd.DataFrame({'detector_0': detector_0, 'detector_1': detector_1, 'detector_sum': detector_sum})

        
    @property
    def as_dict(self) -> dict:
        '''
        Returns the binned timetrace data as a dictionary.
        '''

        # Make copy of data dictionary
        data = self.data.copy()

        # Add detector sum to dictionary
        data['detector_sum'] = self.data['detector_0'][0] + self.data['detector_1'][0]

        return data
    
    
    def preview(self):
        """
        Plot a preview of the binned timestamps data.
        """
        self.timestamps.preview()

    
    def explore(self):
        """
        Explore the binned timestamps data in interactive plot.
        """
        self.timestamps.explore()

        
    def __repr__(self):
        return f'BinnedTimestamps object with {len(self.data)} datasets.'
    


        