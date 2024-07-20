"""
plot.py - Module for data visualization

This module provides functions for visualizing data using various plotting techniques.
It contains functions for creating line plots, scatter plots, bar plots, and other types
of plots to visualize data in different formats, such as time traces, images, and other
types of data.

Functions:
- plot_timetrace: Create a timetrace plot from data.

Usage:
from modules import plot


"""

# import statements
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns
import matplotlib.ticker as ticker
from IPython.core.display import set_matplotlib_formats
from s04utils.modules.load.BinnedTimestamps import BinnedTimestamps
from s04utils.modules.load import Timestamps

# ---------------------------------------------------------------------#
# --------------------- TIMESTAMPS PLOTTING ---------------------------#
# ---------------------------------------------------------------------#

def timetrace(timestamps:Timestamps, bin_width, save_path=None):
    '''
    Create a timetrace plot from timestamps data.
    '''
    timestamps0 = timestamps.data['detector_0'].to_numpy()
    timestamps1 = timestamps.data['detector_1'].to_numpy()
    
    timetrace_len = timestamps0[-1]
    timetrace_len_in_s = timetrace_len * 5e-9
    n_bins = timetrace_len_in_s/bin_width
    _, ax = plt.subplots(figsize=(8, 2))
    preview = sns.histplot(timestamps0, 
                           element="poly", 
                           fill=False, 
                           bins=int(np.floor(n_bins)), 
                           ax=ax,
                           linewidth=1,
                           color='#517BA1')
    
    sns.histplot(timestamps1, 
                           element="poly", 
                           fill=False, 
                           bins=int(np.floor(n_bins)), 
                           ax=ax,
                           linewidth=1,
                           color='#CA4B43')
    
    
    #plt.xlim(0)
    plt.ylim(0)
    preview.set(xlabel='time (s)', 
                ylabel='counts per ' + str(int(bin_width*1e3)) + ' ms',
                title=str(timestamps.file_name))
    # Set x-axis labels to seconds using FuncFormatter
    plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(timestamps.seconds_formatter))
    plt.minorticks_on()
    plt.grid(linewidth = 0.5, alpha = 0.3, which = 'major')
    set_matplotlib_formats('retina')
    
    # Save plot if save_path is provided
    if save_path is not None:
        base_path, ext = os.path.splitext(save_path)
       
        # Get the file name from timestamps.file_name
        file_name = os.path.basename(timestamps.file_name)

        # Append the file name to save_path
        save_path = os.path.join(base_path, file_name)

        # Replace the extension with .png
        base_path, _ = os.path.splitext(save_path)
        save_path = base_path + '.png'
        #if not os.path.exists(plots_path):
        #    os.makedirs(plots_path)

        # Save plot
        plt.savefig(save_path, dpi=600)
        plt.close('all')
    # Otherwise, show plot
    else:
        plt.show()


    





# ---------------------------------------------------------------------#
# ------------------------ IMAGE PLOTTING -----------------------------#
# ---------------------------------------------------------------------#

def image(image_object, cmap="hot", vmin=0, size="normal", title=True, save_path=None):
    '''
    Uses imshow() to display a preview of the loaded image file. 
    Always uses the 'preview' dataset from get_image_data().
    '''

    # get image preview from image data
    img_preview = np.flipud(image_object.get_img_data()['preview'])

    # create label for colorbar
    label = 'counts per ' + str(int(image_object.dwelltime*1e3)) + ' ms'

    # create image plot with chosen colormap
    if size == 'large':
        fig, ax = plt.subplots(figsize=(10, 10))
        im = ax.imshow(img_preview, cmap=cmap, vmin=vmin)
        fig.colorbar(im, fraction=0.03, pad=0.04, label=label)
    else:
        fig, ax = plt.subplots()
        im = ax.imshow(img_preview, cmap=cmap, vmin=vmin)
        cbar = fig.colorbar(im, label=label)
    
    # set major tick options
    plt.tick_params(axis='both', 
                  which='major',
                  labelsize=10, 
                  labelbottom=False,
                  bottom=False, 
                  top=True, 
                  labeltop=True,
                  length=5, 
                  width=1.1)
    
    # set minor tick options
    plt.tick_params(axis='both', 
                    which='minor',
                    labelsize=10, 
                    labelbottom=False,
                    bottom=False, 
                    top=True, 
                    labeltop=True,
                    direction='out')
    
    scaling_factor_x = (round(image_object.range_x*1e6))/image_object.pixel_x
    scaling_factor_y = (round(image_object.range_y*1e6))/image_object.pixel_y
    rounded_x_ticks = lambda x, _:'{:d}'.format(round(x*scaling_factor_x))
    rounded_y_ticks = lambda y, _:'{:d}'.format(round(y*scaling_factor_y))

    # compute x and y ticks
    x_ticks = np.linspace(0, image_object.pixel_x, 5)
    y_ticks = np.linspace(0, image_object.pixel_y, 5)

    # set x and y ticks
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)

    ax.xaxis.set_major_formatter(rounded_x_ticks)
    ax.yaxis.set_major_formatter(rounded_y_ticks)
    
    plt.minorticks_on()

    if title == True:
        plt.title(str(image_object.file_name), pad=40)
    else:
        pass

    plt.xlabel('x in µm')
    plt.ylabel('y in µm')

    set_matplotlib_formats('retina')

    ax.xaxis.set_label_position('top')

    # Save plot if save_path is provided
    if save_path is not None:
        base_path, ext = os.path.splitext(save_path)

        # Get the file name from timestamps.file_name
        file_name = os.path.basename(image_object.file_name)

        # Append the file name to save_path
        save_path = os.path.join(base_path, file_name)

        # Replace the extension with .png
        base_path, _ = os.path.splitext(save_path)
        save_path = base_path + '.png'
        #if not os.path.exists(plots_path):
        #    os.makedirs(plots_path)

        # Save plot
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
        plt.close('all')
    # Otherwise, show plot
    else:
        plt.show()


def plot_pol_image(image_object, threshold=0, m=1):
    """
    Plots an image in red/green/yellow colormap depending on the polarization of the image.
    """

    img = image_object
    
    img_data = img.get_img_data()

    vertical = np.flipud(img_data['APD1'])
    horizontal = np.flipud(img_data['APD2'])

    # threshold
    threshold = 0

    # Multiplicator for higher contrast in the resulting image
    m = 1

    # Compute the result image based on the two detectors
    result = np.zeros((img.pixel_x, img.pixel_y, 3))
    result[:, :, 0] = m *np.where(horizontal < threshold, 0, horizontal) * 255/np.max(horizontal)  # Red channel
    result[:, :, 1] = m * np.where(vertical < threshold, 0, vertical) * 255/np.max(vertical)  # Green channel
    result[:, :, 2] = m * np.logical_and(horizontal >= threshold, 
                                     vertical >= threshold) * 255/(np.max(horizontal)+np.max(vertical)*0.5)  # Yellow channel

    # Create a custom colormap for the colorbar
    #cmap = colors.ListedColormap(['red', 'yellow', 'green'])
    cmap = colors.ListedColormap([(0.8, 0.2, 0.3), (0.9, 0.9, 0.2), (0.3, 0.8, 0.3)])
        #['red', 'yellow', 'green'])
    bounds = [0, 1, 2, 3]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    # Plot the resulting image with a colorbar
    fig, ax = plt.subplots(figsize=(7, 7))
    im = ax.imshow(result.astype(np.uint8), cmap=cmap, norm=norm)
    #ax.set_axis_off()
    cbar = fig.colorbar(im, ticks=[0.5, 1.5, 2.5], orientation='vertical', shrink = 0.6, pad=0.04, aspect=30)
    cbar.ax.set_yticklabels(['horizontal', 'both','vertical'], rotation=-45)

    # set major tick options
    plt.tick_params(axis='both', 
                        which='major',
                        labelsize=10, 
                        labelbottom=False,
                        bottom=False, 
                        top=True, 
                        labeltop=True,
                        length=5, 
                        width=1.1)

    # set minor tick options
    plt.tick_params(axis='both', 
                        which='minor',
                        labelsize=10, 
                        labelbottom=False,
                        bottom=False, 
                        top=True, 
                        labeltop=True,
                        direction='out')

    range_x = '{:0.3e}'.format(img.range_x)
    range_y = '{:0.3e}'.format(img.range_y)

    scaling_factor_x = (img.range_x*1e6)/img.pixel_x
    scaling_factor_y = (img.range_y*1e6)/img.pixel_y
    rounded_x_ticks = lambda x, _:'{:d}'.format(round(x*scaling_factor_x))
    rounded_y_ticks = lambda y, _:'{:d}'.format(round(y*scaling_factor_y))

    # compute x and y ticks
    x_ticks = np.linspace(0, img.pixel_x, 5)
    y_ticks = np.linspace(0, img.pixel_y, 5)

    # set x and y ticks
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)

    ax.xaxis.set_major_formatter(rounded_x_ticks)
    ax.yaxis.set_major_formatter(rounded_y_ticks)

    ax.minorticks_on()
    plt.xlabel('x in µm')
    plt.ylabel('y in µm')
    ax.xaxis.set_label_position('top')
    plt.show()