'''
dwell-time-analyzer.py - Module for processing binned timestamps data and anaylzing dwell times.

Description:
    - ...

Classes:
    - DwellTimeAnalyzer

Functions:
    - ...

Usage:
    - ...

'''

# import statements
import numpy as np
import pandas as pd
from typing import Tuple
import matplotlib.pyplot as plt
from bokeh.plotting import figure, output_notebook
from bokeh.io import show
from sfHMM import sfHMM1
from sfHMM.gmm import GMMs

from s04utils.modules.load.Timestamps import Timestamps
from s04utils.modules.load.BinnedTimestamps import BinnedTimestamps

# ---------------------------------------------------------------------#
# ---------------- DWELL TIME ANALYZER CLASS --------------------------#
# ---------------------------------------------------------------------#

# TODO:
# - make functions usable from the outside (input parameters with defaults)
# - add docstrings to functions
# - add type hints to functions
# - add type hints to class attributes
# - add dataframe that contains the results of the analysis
# - add function that plots the results of the analysis
# - move analysis function to separate module for HMM and thresholding

# - GMMS optimal number of states 
# - GMMS plot
# - handle case where only one state is found in viterbi path
# - add function that returns the results of the analysis as a dataframe
# - add function that plots the results of the analysis

# - add function that decides if the signal from the detector is good enough to be analyzed
#   and decides if detector sum or which detector should be used


class DwellTimeAnalyzer():
    '''
    Class for analyzing dwell times from binned timestamps data.

    Attributes:
        - binned_timestamps_object: BinnedTimestamps
        - binned_timestamps_dataframe: pd.DataFrame
        - viterbi_steps: dict
        - last_steps: dict
        - step_finder_HMM: sfHMM1

    Methods:
        - get_dwell_times
        - get_step_finder_HMM
        - find_last_step_in_binned_data
        - find_cutoffset_single
        - plot_viterbi
        - binned_timestamps_as_dataframe

    Usage:
        - ...

    '''

    def __init__(self, binned_timestamps:BinnedTimestamps, min_states: int = 2, max_states: int =2) -> None:
        # Contains the BinnedTimestamps object that was used to create the class
        self.binned_timestamps_object = binned_timestamps

        # Contains the binned timestamps data as a pandas dataframe for each detector and sum
        self.binned_timestamps_dataframe = self.binned_timestamps_as_dataframe()
        
        # Contains the viterbi path for each detector
        _, self.viterbi_steps = self.find_last_step_in_binned_data()

        # Contains the position of last value change in viterbi for each detector and sum
        self.last_steps, _ = self.find_last_step_in_binned_data()

        # Contains the step finder objects for each detector and sum
        self.step_finder_HMM = self.get_step_finder_HMM(min_states=min_states, max_states=max_states)

        # Contains the viterbi path for each detector and sum
        self.viterbi_path = self.get_viterbi_paths()

        # Contains a dictionary with the dwell times for each state in the model
        self.dwell_times = self.get_dwell_times()

        # Contains the name of the detector with the highest energy
        self.best_detector = self.get_best_detector(binned_timestamps)



    def get_dwell_times(self, 
                        step_finder_HMM: dict[str, sfHMM1] = None,  # type: ignore
                        rename_states = True
                        ) -> dict[str, dict]:                       # type: ignore
        '''
        Returns a dictionary with the dwell times for each state in the model.
        '''
        # Set step_finder_HMM if not specified
        if step_finder_HMM is None:
            step_finder_HMM = self.step_finder_HMM

        # Create dictionary for dwell times for each detector and sum
        dwell_times_dict = {}

        # Iterate over detectors in step_finder_HMM
        for detector in step_finder_HMM:
            # Get step finder object
            sf = step_finder_HMM[detector]

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
                dwell_times[state.astype(int)] = indices[exit_indices] - indices[enter_indices]

            # Check if there is only one state in the viterbi path
            if len(unique) < 2:
                dwell_times['0'] = [0]        

            dwell_times_dict[detector] = dwell_times

        return dwell_times_dict



    def get_step_finder_HMM(self, 
                            binned_timestamps: BinnedTimestamps = None,     # type: ignore
                            max_states: int = None,                         # type: ignore
                            min_states: int = None,                         # type: ignore
                            model: str = None,                              # type: ignore
                            plot: bool = None) -> dict[str, sfHMM1]:        # type: ignore
        '''
        Returns a dict that contains a step finder object for each detector
        and sum.
        '''

        # Set binned timestamps data if not specified
        if binned_timestamps is None:
            binned_timestamps = self.binned_timestamps_object
        
        # Set min and max states if not specified
        if max_states is None:
            max_states = 10
        if min_states is None:
            min_states = 1

        # Set model if not specified
        if model is None:
            model = 'p'
        
        # Set plot if not specified
        if plot is None:
            plot = False
        
        # Create dictionary for step finders for each detector
        step_finder_dict = {}

        # Get cutoff positions for each detector and sum
        cutoff_pos = self.last_steps

        # Get binned timestamps data as dataframe for each detector and sum
        binned_timestamps_as_dataframe = self.binned_timestamps_as_dataframe()

        # Create step finder for each detector and sum
        for detector in binned_timestamps_as_dataframe.columns:
            # Set cutoff position
            cutoff = cutoff_pos[detector]
            
            # Set signal data
            signal = binned_timestamps_as_dataframe[detector][0:cutoff]
            
            # Create step finder
            step_finder_HMM = sfHMM1(signal, 
                        krange=(min_states, max_states), 
                        model=model, 
                        name=detector).run_all(plot=plot)
            
            # Add step finder to dictionary
            step_finder_dict[detector] = step_finder_HMM
        
        return step_finder_dict
    


    def get_viterbi_paths(self) -> dict[str, np.ndarray]:
        '''
        Returns the viterbi path for each detector and sum.
        '''
        # Create dictionary for viterbi paths for each detector
        viterbi_path_dict = {}

        # Get the step finder objects for each detector and sum
        step_finder_HMM_dict = self.step_finder_HMM

        for detector in step_finder_HMM_dict:
            # Get step finder object
            step_finder_HMM = step_finder_HMM_dict[detector]

            # Get viterbi path
            viterbi_path = step_finder_HMM.viterbi

            # Add viterbi path to dictionary
            viterbi_path_dict[detector] = viterbi_path

        return viterbi_path_dict
    

    
    def find_last_step_in_binned_data(self) -> Tuple[dict, dict]:
        '''
        Returns the x value of the last step for detector_0, detector_1
        and detector_sum in binned timestamps.
        '''

        # Initialize dictionary
        last_steps = {}
        viterbi_steps = {}

        # Iterate over detectors in binned timestamps
        for detector in self.binned_timestamps_object.as_dataframe.columns:

            # Get binned timestamps data
            detector_data = self.binned_timestamps_object.as_dataframe[detector]

            # Fit the model to the data (two states fixed)
            sf = sfHMM1(detector_data, krange=(2, 2), model='p').run_all(plot=False)

            # Get the viterbi path
            steps_viterbi = np.array(sf.viterbi)

            # Get unique values
            unique, counts = np.unique(steps_viterbi, return_counts=True)

            # Check if there is a high and low state
            # If not, set last step to 0
            if len(unique) < 2:
                print('Only one state found in viterbi path.')
                print('---------------------------------')
                last_steps[detector] = 0
                low_state = unique[0]
                steps = steps_viterbi.copy()
                steps[steps_viterbi == low_state] = 0

                # Count number of data points in each state
                unique, counts = np.unique(steps, return_counts=True)
                counts_low = counts[0]

                pass

            else:
                # Get high and low state
                high_state = unique[1]
                low_state = unique[0]

                # Assign 0 and 1 to the states in viterbi path
                steps = steps_viterbi.copy()
                steps[steps_viterbi == high_state] = 1
                steps[steps_viterbi == low_state] = 0

                # Find indices where the state changes
                value_change = np.where(np.diff(steps))[0]

                # Get last value change
                last_value_change = value_change[-1]

                # Set cutoff value
                CUT_OFFSET = self.find_cutoffset_single(steps_viterbi[0:last_value_change])

                # Add last value change to dictionary
                last_steps[detector] = last_value_change+CUT_OFFSET

                # Add viterbi steps to dictionary
                viterbi_steps[detector] = steps_viterbi[0:last_value_change+CUT_OFFSET].astype(int)

        return last_steps, viterbi_steps
    



    def find_cutoffset_single(self, viterbi_steps) -> int:
        '''
        Returns the number of data points to add to cut off position after bleaching.
        '''

        # Count number of data points in each state
        unique, counts = np.unique(viterbi_steps, return_counts=True)

        # Check if there is a high and low state
        if len(unique) < 2:
            return 0

        # Get high and low state
        high_state = unique[1]
        low_state = unique[0]

        # Assign 0 and 1 to the states in viterbi path
        steps = viterbi_steps.copy()
        steps[viterbi_steps == high_state] = 1
        steps[viterbi_steps == low_state] = 0

        # Count number of data points in each state
        unique, counts = np.unique(steps, return_counts=True)

        counts_low = counts[0]
        counts_high = counts[1]

        if counts_low < counts_high:
            cutoffset = int((counts_high-counts_low)/2)
        else:
            cutoffset = 10

        return cutoffset
    


    def plot_viterbi(self, sf:sfHMM1) -> None:
        '''
        Plots the viterbi path of the fitted model to the raw timetrace data.
        '''
        output_notebook()

        # Get bin time
        bin_time = self.binned_timestamps_object.bin_width

        # generate x data for bokeh plot
        #x = generate_x_data(binned_timestamps)
        x = np.linspace(0, len(sf.data_raw), len(sf.data_raw))

        # create the same plot in bokeh
        p = figure(width=800, height=300)

        # Get unique number of states in sfHMM object
        unique = np.unique(np.array(sf.viterbi))

        #p.line(x=df.index, y=sfp_two_states.data_raw, line_width=1, color='lightgrey', legend_label='detector_sum')
        p.line(x=x*bin_time, y=sf.data_raw, line_width=1, color='#517BA1', legend_label='signal')
        p.line(x=x*bin_time, y=sf.viterbi, line_width=2, color='red', legend_label='viterbi')
        p.legend.location = 'top_right'

        p.title.text = "sfHMM1 of {} with {} states (optimal)".format(sf.name, len(unique))

        # Set the axis labels
        p.xaxis.axis_label = 'time (s)'
        p.yaxis.axis_label = 'counts per ' + str(int(bin_time*1e3)) + ' ms'

        show(p)



    def binned_timestamps_as_dataframe(self) -> pd.DataFrame:
        '''
        Returns the binned timestamps data for each detector and the sum of both detectors 
        as a pandas dataframe.
        '''
        # Get the binned timetrace data for each individual detector
        detector_0 = self.binned_timestamps_object.as_dataframe['detector_0']
        detector_1 = self.binned_timestamps_object.as_dataframe['detector_1']

        # Get the binned timetrace data for the sum of both detectors
        detector_sum = detector_0 + detector_1

        # Create a pandas dataframe
        df = pd.DataFrame({'detector_0': detector_0, 'detector_1': detector_1, 'detector_sum': detector_sum})

        return df
    

    def get_best_detector(self, binned_timestamps:BinnedTimestamps) -> str:
        '''
        Returns the detector with the highest energy.
        '''
        # Get the binned timetrace data for each individual detector
        detector_0_data = binned_timestamps.as_dataframe['detector_0']
        detector_1_data = binned_timestamps.as_dataframe['detector_1']

        # Calculate the energy of the signal for each detector
        energy_detector_0 = np.sum(np.square(detector_0_data))
        energy_detector_1 = np.sum(np.square(detector_1_data))

        print('Energy detector_0:', energy_detector_0)
        print('Energy detector_1:', energy_detector_1)

        return 'detector_0' if energy_detector_0 > energy_detector_1 else 'detector_1'



    def rename_states(self, dwell_times: dict, state_names: list) -> dict:
        '''
        Returns a dictionary with renamed states.
        '''
        # Initialize the lists
        on_times = []
        off_times = []

        # Iterate over the list of dictionaries
        for dwell_times in dwell_times_list:
            # Append the 'on' and 'off' times to the respective lists
            off_times.extend(dwell_times['off'])
            on_times.extend(dwell_times['on'])

        print(len(on_times))
        print(len(off_times))
        print(on_times)

        return renamed_states
