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
import seaborn as sns
import matplotlib.ticker as ticker
from IPython.core.display import set_matplotlib_formats

# ---------------------------------------------------------------------#
# --------------------- TIMESTAMPS PLOTTING ---------------------------#
# ---------------------------------------------------------------------#

def timetrace(timestamps_object, bin_width=0.01, save_path=None):
    '''
    Create a timetrace plot from timestamps data.
    '''
    timestamps0 = timestamps_object.data['detector0'].to_numpy()
    timestamps1 = timestamps_object.data['detector1'].to_numpy()
    #timestamps = self.data['detector0'].to_numpy()
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
                title=str(timestamps_object.file_name))
    # Set x-axis labels to seconds using FuncFormatter
    plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(timestamps_object.seconds_formatter))
    plt.minorticks_on()
    plt.grid(linewidth = 0.5, alpha = 0.3, which = 'major')
    set_matplotlib_formats('retina')
    
    # Save plot if save_path is provided
    if save_path is not None:
        base_path, ext = os.path.splitext(save_path)
       
        # Get the file name from timestamps_object.file_name
        file_name = os.path.basename(timestamps_object.file_name)

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

def image(image_object, cmap="hot", size="normal", title=True, save_path=None):
        '''
        Uses imshow() to display a preview of the loaded image file. 
        Always uses the 'preview' dataset from get_image_data().
        '''
    
        # get image preview from image data
        img_preview = image_object.get_img_data()['preview']
    
        # create label for colorbar
        label = 'counts per ' + str(int(image_object.dwelltime*1e3)) + ' ms'
    
        # create image plot with chosen colormap
        if size == 'large':
            fig, ax = plt.subplots(figsize=(10, 10))
            im = ax.imshow(img_preview, cmap=cmap)
            fig.colorbar(im, fraction=0.03, pad=0.04, label=label)
        else:
            fig, ax = plt.subplots()
            im = ax.imshow(img_preview, cmap=cmap)
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

            # Get the file name from timestamps_object.file_name
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