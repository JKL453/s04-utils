'''
h5.py - Module for loading and processing h5 data

This module provides functions for loading and processing Photon-HDF5 files.
See http://photon-hdf5.readthedocs.io for more information on Photon-HDF5.
'''

# Imports
import h5py


# Functions

def load_h5_file(path: str) -> h5py.File:
    """
    Loads Photon-HDF5 data from a file and returns a h5py File object.
    """
    return h5py.File(path, 'r')



def get_group_names(h5_file: h5py.File) -> list:
        """
        Returns a list of all group names in the h5 file.
        """
        h5_groups = []
        for group in h5_file.keys():
            if group != 'comment':
                h5_groups.append(group)

        return h5_groups



def get_dataset_names(h5_file: h5py.File, group_name: str) -> list:
        """
        Returns a list of all datasets in a h5 group.
        """
        return list(h5_file[group_name].keys())
    
    
    
def get_dataset_content(h5_file: h5py.File, group_name: str, dataset_name: str) -> list:
    """
    Returns a h5 dataset from a h5 group.
    """
    return h5_file[group_name][dataset_name][()]


    
def print_h5_overview(h5_file: h5py.File) -> None:
    """
    Prints a list of all groups and datasets in the h5 data.
    """
    for group in get_group_names(h5_file):
        print(group)
        for dataset in get_dataset_names(group):
            print('\t', dataset)
        print()



def get_comment_str(h5_file: h5py.File) -> str:
    """
    Returns the cursor position in the h5 file.
    """

    comment_str = h5_file['comment'][()].decode('utf-8')

    return comment_str