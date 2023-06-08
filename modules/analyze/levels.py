"""
analyze.py - Module for preprocessing timetrace data

Insert description here.

Functions:
- ...

Classes:
- ...

"""

# import statements
from scipy import sparse, signal
from scipy.sparse import linalg
import numpy as np
from numpy.linalg import norm

def find_levels(timetrace, width=20, rel_height=60, prominence=1.2):

    # Convolution part
    dary = np.array(timetrace)
    dary -= np.average(dary)

    step = np.hstack((np.ones(len(dary)), -1*np.ones(len(dary))))

    dary_step = np.convolve(dary, step, mode='valid')

    ceil_dary_step = dary_step/100 - np.min(dary_step/100)

    # Get the peaks of the convolution
    peaks = signal.find_peaks(-ceil_dary_step, 
                              width=width, 
                              rel_height=rel_height, 
                              prominence=prominence)[0]
    
    return peaks