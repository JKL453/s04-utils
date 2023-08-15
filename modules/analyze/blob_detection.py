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

# Import modules
from modules.load import image



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
        
        # Check if image is grayscale
        self.image = img = Image.fromarray(self.image.astype(np.uint8))

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

    def plot_blobs(self, blobs=None, min_sigma=3, max_sigma=4, num_sigma=10, threshold=0.01):
        """
        Plot blobs on image.
        """
        if blobs is None:
            blobs = self.detect_blobs(min_sigma=min_sigma, max_sigma=max_sigma, num_sigma=num_sigma, threshold=threshold)
        fig, ax = plt.subplots(1, 1, figsize=(18, 6), sharex=True, sharey=True)
        ax.set_title('Laplacian of Gaussian')
        ax.imshow(self.image, cmap='hot', vmin=0)
        for blob in blobs:
            y, x, r = blob
            c = Circle((x, y), 1.4*r, color='lime', linewidth=2, fill=False)
            ax.add_patch(c)
        ax.set_axis_off()
        plt.tight_layout()
        plt.show()