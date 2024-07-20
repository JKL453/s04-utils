"""
image.py - Module for loading and processing image data

This module provides functions for loading and processing image data 
in various formats.
It contains functions for reading image files, converting image data 
to NumPy arrays, and performing basic image processing tasks such as 
resizing, cropping, and color conversion.

Functions:
- load_from_path: Load an image file from specified path and return 
   an image class object.

Classes:
- image: Class for loading and processing image data.

"""

# import statements
import os
import struct
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import pandas as pd
import pprint
from IPython.core.display import set_matplotlib_formats
from bokeh.plotting import figure, show
from bokeh.models import LinearColorMapper, ColorBar
from bokeh.colors import RGB

# loading function that returns an image class object
def load_from_path(path):
    '''
    Loads image data from a binary file and returns an image class object.
    '''
    return Image(path)


# ---------------------------------------------------------------------#
# ----------------------- IMAGE DATA CLASS ----------------------------#
# ---------------------------------------------------------------------#

class Image():
    '''
    This class loads image data from a binary file and provides methods
    for accessing the image data and header information.
    '''

    def __init__(self, path):
        '''
        Initialize the image class object.
        '''
        self.path = path
        self.file_name = self.path.split('/')[-1]
        self.header_size = 48
        self.bin_file = self.open_bin_file()
        self.bin_header = self.get_bin_header()
        self.bin_content = self.get_bin_content()
        self.dwelltime = self.get_img_header('dwelltime')
        self.pixel_x = self.get_img_header('pixel_x')
        self.pixel_y = self.get_img_header('pixel_y')
        self.offset_x = self.get_img_header('offset_x')
        self.offset_y = self.get_img_header('offset_y')
        self.range_x = self.get_img_header('range_x')
        self.range_y = self.get_img_header('range_y')
        self.data = self.get_img_data()
        self.header = self.get_img_header()
  
  
    def open_bin_file(self):
        '''
        Returns the binary file from path.
        '''
        return open(self.path, "rb")
    
  
    def get_bin_content(self):
        '''
        Returns the unparsed content of the opened binary file.
        '''
        return self.bin_file.read()
  
    def get_bin_header(self):
        '''
        Returns the unparsed content of the opened binary file.
        '''
        return self.bin_file.read(self.header_size)
  
    def get_img_header(self, param='all'):
        '''
        Reads the header information from the binary file and 
        returns a dictionary with the image characteristics.
        '''
        struct_format = '>3d2L2d'
        unpacked_list = list(struct.unpack(struct_format, self.bin_header))
        header_param = ['dwelltime', 
                        'offset_x', 
                        'offset_y', 
                        'pixel_x', 
                        'pixel_y', 
                        'range_x',
                        'range_y']
        header_dict = dict(zip(header_param, unpacked_list))
        
        if param == 'all':
            return header_dict
        else:
            return header_dict[param]
  
  
    def get_struct_format_str(self):
        '''
        Returns the format structure string for parsing the binary content 
        of the opened binary file. 
        The format string is generated for one array, size is given by
        number of pixels.
        '''
        bin_array_size = (self.pixel_x * self.pixel_y) + 2
        struct_format = '>' + str(bin_array_size) + 'L'
        return struct_format
    
    def get_img_data(self):
        '''
        Reads the arrays with the actual image data from the 
        binary file and returns a dictionary with image data of the detectors.
        Because of historical reasons, the binary file always contains
        five arrays of same size. Usually only three contain data.
        This method deletes the arrays that only contain zeros and only 
        returns the ones that contain actual image data.
        '''
    
        # unpack binary data to list, using struct format
        array_struct_format = self.get_struct_format_str()
        unpacked_list = list(struct.iter_unpack(array_struct_format, 
                                            self.bin_content))
        # convert tuples to lists
        unpacked_list = [list(tuple_element) for tuple_element in unpacked_list]
    
        # delete the first two entries (artifacts) in each of these lists
        # and convert lists with image data to array and reshape
        cleaned_list = []
        for list_item in unpacked_list:
            del list_item[0:2]
            cleaned_list.append(np.array(list_item).reshape(self.pixel_x, 
                                                            self.pixel_y))
    
        # create dictionary with image arrays
        array_names = ['preview', 'APD1', 'APD2', 'APD3', 'APD4']
        img_data_all_arrays = dict(zip(array_names, cleaned_list))
    
        # make copy of data image dict
        img_data_arrays_cleaned = img_data_all_arrays
  
        # delete arrays from dictionary that only contain zeros
        for key, value in list(img_data_arrays_cleaned.items()):
            if (not np.any(value)) == True:
              del img_data_arrays_cleaned[key]
    
        img_data_arrays_transformed = {}
        # correct for proper image orientation
        for key, value in list(img_data_arrays_cleaned.items()):
            if key == 'preview':
              img_data_arrays_transformed[key] = np.rot90(value)
            else:
              img_data_arrays_transformed[key]= np.flipud(value)
    
        # return cleaned image dict
        return img_data_arrays_transformed
    
  
    def preview(self, cmap='hot', size='normal', title=True, vmin=0):
        '''
        Uses imshow() to display a preview of the loaded image file. 
        Always uses the 'preview' dataset from get_image_data().
        '''
    
        # get image preview from image data
        img_preview = self.get_img_data()['preview']
    
        # create label for colorbar
        label = 'counts per ' + str(int(self.dwelltime*1e3)) + ' ms'
    
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
      
        scaling_factor_x = (self.range_x*1e6)/self.pixel_x
        scaling_factor_y = (self.range_y*1e6)/self.pixel_y
        rounded_x_ticks = lambda x, _:'{:d}'.format(round(x*scaling_factor_x))
        rounded_y_ticks = lambda y, _:'{:d}'.format(round(y*scaling_factor_y))
  
        ax.xaxis.set_major_formatter(rounded_x_ticks)
        ax.yaxis.set_major_formatter(rounded_y_ticks)
      
        plt.minorticks_on()
    
        if title == True:
            plt.title('Image preview', pad=40)
        else:
            pass
    
        plt.xlabel('x in µm')
        plt.ylabel('y in µm')
    
        set_matplotlib_formats('retina')
    
        ax.xaxis.set_label_position('top')
    
        plt.show()
  
        print('Data (arrays) in image file:')
  
        for key in list(self.get_img_data().keys()):
            if key == 'preview':
                indent_correction = ''
            else:
                indent_correction = '   '
            
            print('  - ' 
                  + indent_correction
                  + str(key) 
                  + ' : ' 
                  + str(self.get_img_data()[key].shape)
                  + ', '
                  + str(self.get_img_data()[key].dtype))
        print('\n')
  
  
    def info(self):
        '''
        Prints out file name, file size and the image header.
        '''
        file_name_str = 'file name: ' + os.path.basename(self.path)
        print(len(file_name_str)*'-')
        print(file_name_str)
        print(len(file_name_str)*'-')
        print('file size: ' + str(os.path.getsize(self.path)) + ' bytes')
        print('header:')
        pprint.pprint(self.header)
  
  
    def get_hot_palette(self):
        '''
        Converts the color palette hot from matplotlib to a color palette
        that can be used by Bokeh image plots.
        '''
        m_hot_rgb = (255 * mpl.cm.hot(range(256))).astype('int')
        hot_palette = [RGB(*tuple(rgb)).to_hex() for rgb in m_hot_rgb]
    
        return hot_palette
    
  
    def round_range_to_um(self, value):
        '''
        Converts the given self.range value to mikrometers, rounds it and 
        returns the result as float.
        '''
        return (value*1e6)
  
  
    def explore(self, data='preview'):
        '''
        Displays an interactive Bokeh image plot for exploratory data analysis.
        '''
        x_range_max = self.round_range_to_um(self.range_x)
        y_range_max = self.round_range_to_um(self.range_y)
        x_range_max = int(x_range_max)
        y_range_max = int(y_range_max)
    
        # create figure
        p = figure(
            x_range=(0,x_range_max), 
            y_range=(0,y_range_max), 
            frame_width=500, 
            frame_height=500,
            x_axis_location="above",
            tooltips=[("x", "$x"), ("y", "$y"), ("value", "@image")])
        
        # load image data
        img_data = self.data[data]
    
        # set color map
        color = LinearColorMapper(palette = self.get_hot_palette(),
                                  low = np.amin(img_data), 
                                  high = np.amax(img_data))
        p.image(image=[img_data], 
                x=0, 
                y=0, 
                dw=x_range_max, 
                dh=y_range_max, 
                color_mapper = color)
    
        # set axes
        p.xaxis.axis_label = 'x in µm'
        p.yaxis.axis_label = 'y in µm'
    
        # create color bar
        cb = ColorBar(color_mapper = color, 
                      location = (0,0),
                      major_tick_line_color = 'black', 
                      title='counts per '+ str(int(self.dwelltime*1e3)) + 'ms')
    
        p.add_layout(cb, 'right')
        show(p)