"""
blob_detection.py - Module for detecting blobs in images

Description:
This module provides a BlobDetector class, which can be used to detect blobs in images. It contains
methods for detecting blobs in images, as well as performing basic statistical analysis on the
detected blobs.

Classes:
- ...

Usage:
... 


"""

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from skimage.feature import blob_log, blob_dog, blob_doh
from math import sqrt
from PIL import Image
import pandas as pd
import os
from tabulate import tabulate

# Import modules
from s04utils.modules.load import image
from charset_normalizer import detect



# ---------------------------------------------------------------------#
# --------------------- BLOB DETECTOR CLASS ---------------------------#
# ---------------------------------------------------------------------#

class BlobDetector:
    """
    Class for detecting blobs, e.g. single molecules, in images (numpy arrays).
    It uses the scikit-image blob detection functions, and provides additional
    methods for analyzing the detected blobs.
    
    See also: 
    https://scikit-image.org/docs/stable/auto_examples/features_detection/plot_blob.html
    """

    def __init__(self, blob_type='log', input_image=None, channel=None):
        """
        Initialize BlobDetector class.

        Parameters
        ----------
        image : numpy array or image object (modules.load.image)
            Image to detect blobs in.
        blob_type : str
            Type of blob detection to perform. Options are 'log' (Laplacian of Gaussian), 'dog'
            (Difference of Gaussian), and 'doh' (Determinant of Hessian).
        **kwargs : dict
            Additional keyword arguments for blob detection.

        Returns
        -------
        None.

        """

        # Save image
        if isinstance(input_image, image.Image) and channel is None:
            self.image = input_image.data['APD1'] + input_image.data['APD2']
            print('Warning: No channel specified, using sum of channel_0 and channel_1.')
        elif isinstance(input_image, image.Image) and channel is not None:
            self.image = input_image.data[channel]
        elif isinstance(input_image, np.ndarray):
            self.image = input_image

        self.image_object = input_image
        
        # Check if image is grayscale
        self.image = Image.fromarray(self.image.astype(np.uint8))

        # Initialize image data path
        self.image_path = input_image.path

        # Save blob type
        self.blob_type = blob_type

        # Initialize blob list
        self.blobs = None

        # Initialize blob properties
        self.blob_properties = None

        # Initialize blob statistics
        self.blob_statistics = None

        # Initialize blob statistics dataframe
        self.blob_statistics_df = None



    def detect_blobs(self, min_sigma=3, max_sigma=4, num_sigma=10, threshold=0.01):
        """
        Detect blobs in image using scikit-image blob detection functions.
        """
        if self.blob_type == 'log':
            return blob_log(self.image, min_sigma=min_sigma, max_sigma=max_sigma, num_sigma=num_sigma, threshold=threshold)
        elif self.blob_type == 'dog':
            return blob_dog(self.image, min_sigma=min_sigma, max_sigma=max_sigma, threshold=threshold)
        elif self.blob_type == 'doh':
            return blob_doh(self.image, min_sigma=min_sigma, max_sigma=max_sigma, num_sigma=num_sigma, threshold=threshold)
        else:
            print('Error: Invalid blob type. Options are "log", "dog", and "doh".')

            

    def compare_blob_detection(self, min_sigma=3, max_sigma=4, num_sigma=10, threshold=0.01):
        """
        Compare different blob detection methods.
        """
        fig, axes = plt.subplots(1, 3, figsize=(9, 3), sharex=True, sharey=True)
        ax = axes.ravel()

        blobs_log = blob_log(self.image, min_sigma=min_sigma, max_sigma=max_sigma, num_sigma=num_sigma, threshold=threshold)
        blobs_log[:, 2] = blobs_log[:, 2] * np.sqrt(2)

        blobs_dog = blob_dog(self.image, min_sigma=min_sigma, max_sigma=max_sigma, threshold=threshold)
        blobs_dog[:, 2] = blobs_dog[:, 2] * np.sqrt(2)

        blobs_doh = blob_doh(self.image, min_sigma=min_sigma, max_sigma=max_sigma, num_sigma=num_sigma, threshold=threshold)

        for blob in blobs_log:
            y, x, r = blob
            c = Circle((x, y), r, color='lime', linewidth=2, fill=False)
            ax[0].add_patch(c)

        for blob in blobs_dog:
            y, x, r = blob
            c = Circle((x, y), r, color='lime', linewidth=2, fill=False)
            ax[1].add_patch(c)

        for blob in blobs_doh:
            y, x, r = blob
            c = Circle((x, y), r, color='lime', linewidth=2, fill=False)
            ax[2].add_patch(c)

        ax[0].set_title('Laplacian of Gaussian')
        ax[1].set_title('Difference of Gaussian')
        ax[2].set_title('Determinant of Hessian')

        for a in ax:
            a.set_axis_off()

        plt.tight_layout()
        plt.show()



    def compare_blob_methods(self, min_sigma=3, max_sigma=4, num_sigma=10, threshold=0.01):

        blobs_log = blob_log(self.image, max_sigma=max_sigma, min_sigma=min_sigma, num_sigma=num_sigma, threshold=threshold)
        blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)

        blobs_dog = blob_dog(self.image, max_sigma=max_sigma, min_sigma=min_sigma, threshold=threshold)
        blobs_dog[:, 2] = blobs_dog[:, 2] * sqrt(2)

        blobs_doh = blob_doh(self.image, max_sigma=max_sigma, min_sigma=min_sigma, num_sigma=num_sigma, threshold=threshold)

        blobs_list = [blobs_log, blobs_dog, blobs_doh]
        colors = ['yellow', 'lime', 'red']
        titles = ['Laplacian of Gaussian', 'Difference of Gaussian',
                  'Determinant of Hessian']
        sequence = zip(blobs_list, colors, titles)

        fig, axes = plt.subplots(1, 3, figsize=(9, 3), sharex=True, sharey=True)
        ax = axes.ravel()

        for idx, (blobs, color, title) in enumerate(sequence):
            ax[idx].set_title(title)
            ax[idx].imshow(self.image, cmap='hot')
            for blob in blobs:
                y, x, r = blob
                c = Circle((x, y), r, color=color, linewidth=2, fill=False)
                ax[idx].add_patch(c)
            ax[idx].set_axis_off()

        plt.tight_layout()
        plt.show()



    def plot_blobs(self, blobs=None, min_sigma=3, max_sigma=4, num_sigma=10, threshold=0.01, vmin=0, figsize=(6,4)):
        """
        Plot blobs on image.
        """
        if blobs is None:
            blobs = self.detect_blobs(min_sigma=min_sigma, max_sigma=max_sigma, num_sigma=num_sigma, threshold=threshold)
        fig, ax = plt.subplots(1, 1, figsize=figsize, sharex=True, sharey=True)
        ax.set_title('Laplacian of Gaussian')
        im = ax.imshow(self.image, cmap='hot', vmin=vmin)
        for blob in blobs:
            y, x, r = blob
            c = Circle((x, y), 1.4*r, color='lime', linewidth=2, fill=False)
            ax.add_patch(c)
        ax.set_axis_off()
        fig.colorbar(im, ax=ax, pad=0.01)
        plt.tight_layout()
        plt.show()


    def compute_threshold(self):
        """
        Compute threshold for blob detection.
        """

        # Convert image to PIL Image object
        pil_img = self.image

        # Convert image to grayscale
        gray_img = pil_img.convert("L")

        # Convert grayscale image back to numpy array
        gray_arr = np.array(gray_img)

        # Scale values in gray_arr to range from 0 to 255
        scaled_arr = ((gray_arr - np.min(gray_arr)) / (np.max(gray_arr) - np.min(gray_arr))) * 255

        # Convert scaled_arr to uint8
        scaled_arr = scaled_arr.astype(np.uint8)

        # calculate threshold for blob detection
        threshold_mean = np.mean(scaled_arr)

        # Compute normalized threshold
        normalized_threshold_mean = threshold_mean / 255

        print('Threshold mean: {}'.format(threshold_mean))
        print('Normalized threshold mean: {}'.format(normalized_threshold_mean*0.1))

        return normalized_threshold_mean*0.1
    

    def extract_blob_pixels(self, blob_data, image_array):
        """
        Extract pixel values for each blob from source image.
        """

        # Initialize list to store blob pixels
        blobs = []

        for blob in blob_data:
            # Get blob coordinates and radius
            y, x, r = blob
            
            # Extract blob pixels from original image
            blob_pixels = image_array[int(y - r):int(y + r), int(x - r):int(x + r)]

            if blob_pixels.shape[0] == blob_pixels.shape[1]: 
                blobs.append(blob_pixels)
            else:
                pass

            # check blobs for entry with shape (0,0)
            for i, blob in enumerate(blobs):
                if blob.shape == (0,0):
                    print(i)
                    blobs.pop(i)
        
        return blobs
    

    def analyze_blob_int(self, blobs, verbose=False):
        """
        Analyze the average and maximum intensity of each blob.
        """

       # Create list to store average intensities
        avg_intensities = []
        max_intensities = []

        # Loop over blobs and plot in subplots
        for i, blob in enumerate(blobs):
            # Calculate average intensity
            avg_intensity = np.mean(blob)

            # Calculate max intensity
            max_intensity = np.max(blob)

            avg_intensities.append(avg_intensity)
            max_intensities.append(max_intensity)

        # Calculate histogram parameters
        r_avg_intensities = [round(intensity) for intensity in avg_intensities]
        r_max_intensities = [round(intensity) for intensity in max_intensities]

        if verbose:
            print('Average intensities:', r_avg_intensities)
            print('Max intensities:', r_max_intensities)

        return r_avg_intensities, r_max_intensities
    

    def create_blob_df(self, blobs, avg_int, max_int, verbose=False):
        """
        Create a dataframe with blob data.
        """

        blob_data = []
        for i, blob in enumerate(blobs):
            blob_data.append(['blob_{}'.format(i+1), avg_int[i], max_int[i]])


        # Convert blob_data to a dataframe
        blob_dataframe = pd.DataFrame(blob_data)

        # Set the column names
        blob_dataframe.columns = ['ID', 'Avg. intensity', 'Max. intensity']

        # Add a column that contains the file name for each blob (i.e. the file name of the image that was analyzed)
        blob_dataframe['File'] = os.path.basename(self.image_path)

        # Make 'File' the first column
        cols = blob_dataframe.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        blob_dataframe = blob_dataframe[cols]

        if verbose:
            # Print the table using the tabulate function
            print(tabulate(blob_data, headers=['ID', 'Avg.\nintensity', 'Max.\nintensity'], tablefmt='rst'))

        return blob_dataframe
