'''
dwelltime_analyzer_v2.py - Dwell Time Analyzer v2

This module contains the DwelltimeAnalyzer class, which is 
used extract dwell times from binned timestamps data.


Classes:
    DwelltimeAnalyzer: Extract dwell times from binned timestamps data.

Functions:
    None

Usage:
    from modules.analyze.dwelltime_analyzer import DwelltimeAnalyzer

    # Create an instance of DwelltimeAnalyzer
    dta = DwelltimeAnalyzer()

    # Load binned timestamps data
    dta.load_data(data)

    # Get dwell times
    dwell_times = dta.get_dwell_times()
'''



# Import modules
from itertools import count
import numpy as np
import pandas as pd
from typing import Tuple
import matplotlib.pyplot as plt
from sfHMM import sfHMM1
from sfHMM.gmm import GMMs

from s04utils.modules.load.BinnedTimestamps import BinnedTimestamps



# ---------------------------------------------------------------------#
# ---------------- DWELL TIME ANALYZER CLASS --------------------------#
# ---------------------------------------------------------------------#



class DwelltimeAnalyzer():
    '''
    DwelltimeAnalyzer - Extract dwell times from binned timestamps data.

    This class is used to extract dwell times from binned timestamps data.

    Methods:
        load_data: Load binned timestamps data.
        get_bts_data: Get binned timestamps data for each detector and 
            the sum of both detectors.
        set_analysis_method: Set the analysis method to be used.
    '''

    def __init__(self):
        '''
        Initialize DwelltimeAnalyzer class.
        '''
        self.bts_object:BinnedTimestamps = None          # type: ignore
        self.bin_width:float = 0
        self.bts_data:pd.DataFrame = pd.DataFrame()
        self.analysis_method:str = 'sfHMM'
        self.detector_selection:str = ''
        self.detector:str = ''
        self.signal:np.ndarray = np.array([])
        self.ok_for_sfHMM:bool = True
        self.signal_trimmed:np.ndarray = np.array([])
        self.opt_num_states:int = 0
        self.num_states:int = 0
        self.sfHMM:sfHMM1 = sfHMM1()
        self.viterbi_path:np.ndarray = np.array([])
        self.last_state_change:int = 0
        self.viterbi_path_fixed:np.ndarray = np.array([])
        self.high_state:int = 0
        self.low_state:int = 0
        self.high_state_count:int = 0
        self.low_state_count:int = 0
        self.dwelltimes:dict[str, np.ndarray] = {}
        self.survival_time:int = 0
        


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


    
    def analyze(self, analysis_method:str='sfHMM', 
                detector_selection:str='auto', 
                plot:bool=False):
        '''
        Analyze the binned timestamps data.

        Parameters:
            analysis_method: str
                Analysis method to be used. Must be "sfHMM" or "threshold".
            detector_selection: str
                Detector to be analyzed. Must be "detector_0", "detector_1", "detector_sum" 
                or "auto" (default). If "auto" is selected, the detector with the highest
                energy in the data will be selected.
        Returns:
            None
        '''

        # Set the detector
        self.set_detector(detector_selection)

        # Set the analysis method
        self.set_analysis_method(analysis_method)

        # Get the last state change
        self.get_last_state_change()

        if self.ok_for_sfHMM:
            # Get the optimal number of states
            self.get_opt_num_states()

            # Trim the signal
            self.trim_signal()

            # Fit the sfHMM model
            self.fit_sfHMM(plot=plot)

            # Get dwell times
            self.get_dwell_times()

            # Get the survival time
            self.get_survival_time()

            # Rename dwell time states
            self.rename_dwell_time_states()

        else:
            self.dwelltimes = {}
            print('Signal is not suitable for sfHMM analysis.')
            print('----------------------------------------')
            print('Dwell times have not been calculated.')
            print('----------------------------------------')
            



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
    


    def set_analysis_method(self, analysis_method:str) -> None:
        '''
        Set the analysis method to be used.

        Parameters:
            analysis_method: str
                Analysis method to be used. Must be "sfHMM" or "threshold".
        Returns:
            analysis_method: str
                Analysis method to be used.
        '''
        # Check if input is valid
        if analysis_method not in ['sfHMM', 'threshold']:
            raise ValueError('Invalid analysis method. Must be "sfHMM" or "threshold".')
        
        self.analysis_method = analysis_method



    def set_detector(self, detector_selection:str) -> None:
        '''
        Choose the detector to be analyzed.

        Parameters:
            detector: str
                Detector to be analyzed. Must be "detector_0", "detector_1", "detector_sum" 
                or "auto" (default). If "auto" is selected, the detector with the highest
                energy in the data will be selected.
        Returns:
            None
        '''
        # Check if input is valid
        err_str = 'Invalid detector. Must be "detector_0", "detector_1", "detector_sum" or "auto".'

        if detector_selection not in ['detector_0', 'detector_1', 'detector_sum', 'auto']:
            raise ValueError(err_str)
        
        print('Detector selection:')
        print('-------------------')

        # If "auto" is selected, choose the detector with the highest energy
        if detector_selection == 'auto':
            detector_0_data = self.bts_data['detector_0']
            detector_1_data = self.bts_data['detector_1']

            # Calculate the energy of each detector
            detector_0_energy = np.sum(np.square(detector_0_data))
            detector_1_energy = np.sum(np.square(detector_1_data))

            print('Energy detector_0: ', detector_0_energy)
            print('Energy detector_1: ', detector_1_energy)

            # Choose the detector with the highest energy
            if detector_0_energy > detector_1_energy:
                self.detector = 'detector_0'
                self.set_signal()
                print('detector_0 selected')
                print('-------------------')
            else:
                self.detector = 'detector_1'
                self.set_signal()
                print('detector_1 selected')
                print('-------------------')
                
        # If a specific detector is selected, choose that detector
        else:
            self.detector = detector_selection
            self.set_signal()
            print(self.detector, ' selected')
            print('-------------------')
            
        

    def set_signal(self) -> None:
        '''
        Set the signal to be analyzed.

        Parameters:
            signal: np.ndarray
                Signal to be analyzed.
        Returns:
            None
        '''
        # Check if input is valid
        err_str = 'Detector has not been selected. Use set_detector method to select a detector.'

        if self.detector not in ['detector_0', 'detector_1', 'detector_sum', 'auto']:
            raise ValueError(err_str)

        self.signal = self.bts_data[self.detector].to_numpy()
        


    def get_last_state_change(self) -> None:
        '''
        Returns the x-axis value of the last state change.

        Parameters:
        -----------
        None

        Returns:
        --------
        last_state_change: int
            x-axis value of the last state change.
        
        '''
        # Check if signal has been set
        self.check_signal()
        
        # Fit the 2-state sfHMM model to the signal
        sf = sfHMM1(self.signal, krange=(1, 2), model='p').run_all(plot=False)

        # Get the viterbi path
        self.viterbi_path_fixed = np.array(sf.viterbi)
        

        # Get unique values
        unique, counts = np.unique(self.viterbi_path_fixed, return_counts=True)
        print('Unique: ', unique)
        print('Counts: ', counts)

        # Check if there is a high and low state
        if len(unique) < 2:
            unique = np.append(unique, 0)
            counts = np.append(counts, 0)
            self.ok_for_sfHMM = False

        # Get high and low state
        self.high_state = unique[1]
        self.low_state = unique[0]

        self.high_state_count = counts[1]
        self.low_state_count = counts[0]

        # Assign 0 and 1 to the states in viterbi path
        states = self.viterbi_path_fixed.copy()
        states[self.viterbi_path_fixed == self.high_state] = 1
        states[self.viterbi_path_fixed == self.low_state] = 0

        # Find indices where the state changes
        value_change = np.where(np.diff(states))[0]

        # Get last value change
        if len(value_change) == 0:
            self.last_state_change = 0
        else:
            self.last_state_change = value_change[-1]



    def get_opt_num_states(self) -> None:
        '''
        
        '''
        # Check if signal has been set
        self.check_signal()

        # Get the optimal number of states for the signal
        gmms = GMMs(self.signal, krange=(2, 10))
        gmms.fit()
        #gmms.show_aic_bic()
        #gmms.plot_all()
        gmm_optimal = gmms.get_optimal(criterion="bic").n_components # type: ignore
        print('Optimal number of states: ', gmm_optimal)

        self.opt_num_states = gmm_optimal
        


    def fit_sfHMM(self, mode:str='fixed', num_states:int=2, plot:bool=False) -> None:
        '''
        Fit a sfHMM model to the signal.

        Parameters:
            None
        Returns:
            None
        '''
        # Check if signal has been
        self.check_signal()

        # Set the number of states
        if mode == 'fixed':
            self.num_states = num_states
        elif mode == 'optimal':
            self.get_opt_num_states()
            self.num_states = self.opt_num_states
        else:
            raise ValueError('Invalid mode. Must be "fixed" or "optimal".')

        # Fit the 2-state sfHMM model to the signal
        sf = sfHMM1(self.signal_trimmed, krange=(1, self.num_states), model='p').run_all(plot=False)
        self.sfHMM = sf

        # Get the viterbi path
        self.viterbi_path = np.array(sf.viterbi)

        # Plot the results
        if plot:
            plt.figure(figsize=(6, 1))
            plt.plot(self.signal_trimmed, linewidth=0.5)
            plt.plot(self.viterbi_path_fixed)
            plt.axvline(self.last_state_change, color='r')
            plt.minorticks_on()

            plt.grid(which='both', linestyle='--', linewidth=0.5)
            
            plt.show()



    
    def trim_signal(self, start:int=0, end:int=0) -> None:
        '''
        Trim the signal.

        Parameters:
            start: int
                Start index.
            end: int
                End index.
        Returns:
            None
        '''
        # Check if signal has been set
        self.check_signal()

       # Get the offset
        offset = self.get_offset()

        end = self.last_state_change + offset

        # Trim the signal
        self.signal_trimmed = self.signal[start:end]



    def get_offset(self) -> int:
        '''
        Get the offset.

        Parameters:
            None
        Returns:
            None
        '''
        # Check if signal has been set
        self.check_signal()

        states = self.viterbi_path_fixed[:self.last_state_change]

        # Count number of data points in each state
        unique, counts = np.unique(states, return_counts=True)
        print('Unique: ', unique)
        print('Counts: ', counts)

        # Check if there is a high and low state
        if len(unique) == 0:
            return 0
        elif len(unique) < 2:
            if counts[0] > 100:
                return int((counts[0])*2)
            else:
                return 200
        
        counts_low = counts[0]
        counts_high = counts[1]

        if counts_low < counts_high:
            offset = int((counts_high)*2)
        else:
            offset = 100

        return offset
    


    def get_dwell_times(self):
        '''
        Get dwell times.

        Parameters:
            None
        Returns:
            dwelltimes: np.ndarray
                Dwell times.
        '''
        # Check if signal has been set
        self.check_signal()

        # Get step finder object
        sf = self.sfHMM

        # Get unique number of states in sfHMM object
        unique = np.unique(np.array(sf.viterbi))

        # Initialize dictionary
        dwell_times = {}
    
        # Iterate over unique states
        for state in unique:
            # Get indices for state
            indices = np.where(sf.viterbi == state)[0]
            
            # Find the indices where the signal enters and exits the state
            enter_indices = np.where(np.diff(indices) != 1)[0] + 1
            exit_indices = np.where(np.diff(indices) != 1)[0]
            
            # Add the first index and the last index
            enter_indices = np.insert(enter_indices, 0, 0)
            exit_indices = np.append(exit_indices, len(indices) - 1)

            # Calculate the dwell times
            dwell_times[state.astype(int)] = indices[exit_indices] - indices[enter_indices] + 1

        # Check if there is only one state in the viterbi path
        if len(unique) < 2:
            dwell_times['0'] = [0]


        # Multiply every dwell time by the bin width
        for key in dwell_times:
            dwell_times[key] = dwell_times[key] * 10

        self.dwelltimes = dwell_times


    
    def rename_dwell_time_states(self):
        '''
        Rename dwell times.

        Parameters:
            None
        Returns:
            None
        '''
        # Check if dwell times have been calculated
        err_str = 'Dwell times have not been calculated. Use get_dwell_times method to calculate dwell times.'
        if len(self.dwelltimes) == 0:
            raise ValueError(err_str)

        # Get keys from dwell times dictionary
        keys = list(self.dwelltimes.keys())

        # Rename the keys
        first_key = list(keys)[0]
        second_key = list(keys)[1]

        # Check if the first key is the lower state
        if first_key < second_key:
            self.dwelltimes['off'] = self.dwelltimes.pop(first_key)
            self.dwelltimes['on'] = self.dwelltimes.pop(second_key)
        else:
            self.dwelltimes['on'] = self.dwelltimes.pop(first_key)
            self.dwelltimes['off'] = self.dwelltimes.pop(second_key)



    def get_survival_time(self):
        '''
        Get the survival time.

        Parameters:
            None
        Returns:
            None
        '''
        # Get the survival time which is the position of the last state change
        self.survival_time = self.last_state_change



    def check_signal(self) -> None:
        '''
        Check the signal.

        Parameters:
            None
        Returns:
            None
        '''
        # Check if signal has been set
        err_str = 'Signal has not been set. Use set_detector method to set the signal.'
        if len(self.signal) == 0:
            raise ValueError(err_str)
        
