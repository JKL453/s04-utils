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

    def __init__(self, path:str, bin_width:float=0.01):
        self.path = path
        self.bin_width = bin_width
        self.timestamps = Timestamps(path)
        self.data = self.bin_timestamps()
        self.raw = self.timestamps.data
        self.len_seconds = self.get_length() 


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
    


        