'''
ld_analyzer.py - Linear Dichroism Analyzer module

This module contains functions for analyzing linear dichroism data.

Classes:
   

Functions:
    

Usage:
    
'''


# Import modules
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
from s04utils.modules.load.BinnedTimestamps import BinnedTimestamps


# ----------------------------------------------------------------------#
# ---------------- LINEAR DICHROISM ANALYZER CLASS ---------------------#
# ----------------------------------------------------------------------#


class LDAnalyzer:
    """
    LDAnalyzer - Linear Dichroism Analyzer class.

    This class contains functions for computing and analyzing linear dichroism data.

    Methods:
        - ...
    """
    
    def __init__(self):
        '''
        Initialize LDAnalyzer class.
        '''
        self.bts_object:BinnedTimestamps = None     # type: ignore
        self.bin_width:float = 0.0
        self.bts_data:pd.DataFrame = pd.DataFrame()
        self.linear_dichroism:np.ndarray = np.array([])



    def load_data(self, binned_timestamps:BinnedTimestamps):
        '''
        Load binned timestamps data.

        Parameters:
            data: BinnedTimestamps object
                Binned timestamps data.

        Returns:
            None
        '''
        self.bts_object = binned_timestamps
        self.bin_width = binned_timestamps.bin_width
        self.bts_data = self.get_bts_data()
        self.linear_dichroism = self.get_linear_dichroism()
        self.angular_change = self.get_angular_change()



    def get_bts_data(self) -> pd.DataFrame:
        '''
        Get binned timestamps data for each detector and the 
        sum of both detectors.

        BinnedTimestamps object must be loaded before calling this method.

        Returns:
            bts_data: pd.DataFrame
                Data frame that contains the timestamps data of
                each detectot and the sum of both detectors.
        '''
        # Check if binned timestamps data has been loaded
        if self.bts_object is None:
            raise ValueError('Binned timestamps data has not been loaded.')
        
        # Get the binned timetrace data for each individual detector
        detector_0_data = self.bts_object.as_dataframe['detector_0']
        detector_1_data = self.bts_object.as_dataframe['detector_1']

        # Get the binned timetrace data for the sum of both detectors
        detector_sum_data = detector_0_data + detector_1_data

        # Create a pandas dataframe
        bts_df = pd.DataFrame({'detector_0': detector_0_data,
                               'detector_1': detector_1_data,
                               'detector_sum': detector_sum_data})

        return bts_df
    


    def get_linear_dichroism(self) -> np.ndarray:
        '''
        Compute linear dichroism data.

        Returns:
            linear_dichroism: array
                Linear dichroism data.
        '''
        # Check if binned timestamps data has been loaded
        if self.bts_data.empty:
            raise ValueError('Binned timestamps data has not been loaded.')
        
        # Load binned timestamps data
        data0 = self.bts_data['detector_0'].to_numpy()
        data1 = self.bts_data['detector_1'].to_numpy()

        # Compute linear dichroism
        linear_dichroism = (data0 - data1)/(data0 + data1)

        # Replace NaN values with 0
        linear_dichroism = np.nan_to_num(linear_dichroism)
        
        # Also add the linear dichroism data to the bts_data dataframe
        self.bts_data['linear_dichroism'] = linear_dichroism

        return linear_dichroism
    


    def get_angular_change(self) -> np.ndarray:
        '''
        Compute angular change data.

        Returns:
            angular_change: array
                Angular change data.
        '''
        # Check if linear dichroism data has been computed
        if len(self.linear_dichroism) == 0:
            raise ValueError('Linear dichroism data has not been computed.')
        
        # calculate the angular orientation of the dipole moment from r values
        # the angle is given in degrees
        angle = np.arccos(self.linear_dichroism)/2
        angle = np.degrees(angle)

        # Compute angular change
        angular_change = np.diff(angle)*self.bin_width

        # Also add the angular change data to the bts_data dataframe
        self.bts_data['angular_change'] = np.append(angular_change, 0)
        
        return angular_change
    




