"""
report.py - Module for creating a simple report pdf

This module provides functions for loading and processing timestamps 
data from various sources.
It contains functions for reading timestamps data from CSV files, 
filtering and processing timestampss, and performing basic statistical 
analysis on timestamps data.

Functions:
- load_timetrace: Load timestamps data from a H5 file and return as a
    timestamps class object.

Classes:
- timestamps: Class for loading and processing timestamps data.

"""

# import statements

import os
import glob
import shutil
import numpy as np
from fpdf import FPDF
from modules.load import timestamps, image
from modules.visualize import plot

from tqdm import tqdm
from ipyfilechooser import FileChooser

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from PIL import Image

# ---------------------------------------------------------------------#
# ------------------------ CREATE REPORT ------------------------------#
# ---------------------------------------------------------------------#

PLOT_DIR = 'plots'

def create(path_to_data_dir, plot_type='timetrace'):
    """
    Creates a report for the data in the specified directory.
    """

    # Get path to data directory
    data_dir = os.path.dirname(path_to_data_dir)

    # Create the plots directory
    try:
        # Create the directory
        os.mkdir(f'{data_dir}/{PLOT_DIR}')
    except FileExistsError:
        # If directory already exists
        print('Directory already exists.')
        # Check if Overview.pdf already exists
        if os.path.exists(os.path.join(data_dir, 'Overview.pdf')):
            print('PDF report already exists.')
            # Stop execution
            return
        else:
            # Create the PDF
            create_pdf(f'{data_dir}/{PLOT_DIR}', plot_type=plot_type)
            # Stop execution
            return
    
    # Create the plots
    create_plots(path_to_data_dir, plot_type=plot_type)

    # Create the PDF
    create_pdf(f'{data_dir}/{PLOT_DIR}', plot_type=plot_type)



def create_plots(path, plot_type='timetrace'):
    """
    Creates the directory /plots in the current data directory 
    specified by path.
    Creates plots for all files in the data directory and saves 
    them in the /plots directory.
    """

    # Get path to data directory
    data_dir = os.path.dirname(path)

    # Get a list of all files in the directory
    files = os.listdir(data_dir)

    if plot_type == 'timetrace':
        # Iterate over the files
        for filename in tqdm(files, desc='Creating plots'):
            # Check if the file is a data file
            if filename.endswith(".h5"):
                # Create a timestamps object
                timestamps_object = timestamps.load_from_path(f'{data_dir}/{filename}')
                # Save visualization
                plot.timetrace(timestamps_object, bin_width=0.01, save_path=(f'{data_dir}/{PLOT_DIR}'))
                # clean up
                del timestamps_object

    elif plot_type == 'image':
        # Iterate over the files
        for filename in tqdm(files, desc='Creating plots'):
            # Check if the file is a data file
            if filename.endswith(".img"):
                # Create a image object
                image_object = image.load_from_path(f'{data_dir}/{filename}')
                # Save visualization
                plot.image(image_object, save_path=(f'{data_dir}/{PLOT_DIR}'))
                # clean up
                del image_object



def create_pdf(images_path, plot_type='timetrace'):
    """
    Creates a PDF document from the images in the specified directory.
    The images are arranged in a grid with the specified number of rows and columns.
    """

    # Check if pdf report already exists
    if os.path.exists(os.path.join(images_path, 'Overview.pdf')):
        print('PDF report already exists.')
        return

    # Create a new PDF document
    pdf_path = os.path.join(images_path, 'Overview.pdf')
    pdf = canvas.Canvas(pdf_path, pagesize=A4)

    # Get the page size
    page_width, page_height = A4

    # Set default values for the number of rows and columns
    images_per_column = 6
    images_per_row = 1
    image_width = 400 #page_width / images_per_row
    image_height = 100 #page_height / images_per_column
    

    if plot_type == 'timetrace':
        # Calculate the size of each image and the spacing between images
        images_per_column = 6
        images_per_row = 1
        image_width = 400 #page_width / images_per_row
        image_height = 100 #page_height / images_per_column
    
    if plot_type == 'image':
        # Calculate the size of each image and the spacing between images
        images_per_column = 4
        images_per_row = 2
        image_width = 270 #page_width / images_per_row
        image_height = 200 #page_height / images_per_column

    spacing_x = (page_width - image_width * images_per_row) / (images_per_row + 1)
    spacing_y = (page_height - image_height * images_per_column) / (images_per_column + 1)

    # Initialize the row and column counters
    row = 0
    col = 0
    

    # Get a list of image filenames sorted by file number
    image_filenames = sorted(os.listdir(images_path), key=lambda x: int(x.split('_')[-1].split('.')[0]))

    # Split the list of image filenames into sublists for each page
    pages = [image_filenames[i:i+images_per_row*images_per_column] for i in range(0, len(image_filenames), images_per_row*images_per_column)]
    
    # Iterate over the pages
    rows = []
    for page in pages:
        # Reverse the order of the rows for this page
        rows.append(list(reversed(page)))
    
    # Join the rows into a single list of images
    image_filenames = []
    for row in rows:
        image_filenames.extend(row)

    # Initialize the row and column counters
    row = 0
    col = 0

    # Iterate over the images in the directory
    for filename in tqdm(image_filenames, desc='  Creating PDF'):
        # Check if the file is an image
        if filename.endswith('.png'):
            # Get the full path to the image file
            image_path = os.path.join(images_path, filename)
            # Open the image file using PIL
            image = Image.open(image_path)
            # Resize the image to fit the page
            image = image.resize((int(image_width), int(image_height)), Image.ANTIALIAS)
            # Calculate the position of the image on the page
            x = spacing_x + col * (image_width + spacing_x)
            y = spacing_y + row * (image_height + spacing_y)
            # Add the image to the PDF document
            pdf.drawImage(image_path, x, y, width=image_width, height=image_height)

            # Increment the column counter
            col += 1
            # Check if we have reached the end of the row
            if col == images_per_row:
                # Reset the column counter and increment the row counter
                col = 0
                row += 1
            # Check if we have reached the end of the page
            if row == images_per_column:
                # Add a new page to the PDF document
                pdf.showPage()
                # Reset the row and column counters
                row = 0
                col = 0

    # Save the PDF document
    pdf.save()
    print('PDF report created.')



def sort_files(selected_path):
    """
    Sorts files in the specified directory into the "timestamps" and "images" directories.
    """

    # Replace fc.selected_path with the path to check
    path = selected_path

    # Check if there are any files with extensions ".h5" or ".img" in the path
    if not any(file.endswith((".h5", ".img")) for file in os.listdir(path)):
        print("No files with extensions '.h5' or '.img' found in the specified path")
    else:
        # Check if "timestamps" and "images" directories exist
        timestamps_path = os.path.join(path, "timestamps")
        images_path = os.path.join(path, "images")

        if os.path.isdir(timestamps_path) and os.path.isdir(images_path):
            # If both directories exist:
            print("Directories 'timestamps' and 'images' already exist")
        elif os.path.isdir(timestamps_path):
            # If only "timestamps" directory exists: create "images" directory
            os.mkdir(images_path)
            print("Missing 'images' directory has been created")
        elif os.path.isdir(images_path):
            # If only "images" directory exists: create "timestamps" directory
            os.mkdir(timestamps_path)
            print("Missing 'timestamps' directory has been created")
        else:
            # If both directories do not exist: create both directories
            os.mkdir(timestamps_path)
            os.mkdir(images_path)
            print("Missing 'timestamps' and 'images' directories have been created")

        # Search for files with extensions ".h5" and ".img"
        h5_files = glob.glob(os.path.join(path, "*.h5"))
        img_files = glob.glob(os.path.join(path, "*.img"))

        # Print the list of files found
        print("\n")
        print("H5 files:")
        # Print file names without the path
        for file in h5_files:
            print(os.path.basename(file))
        print("\n")
        print("IMG files:")
        for file in img_files:
            print(os.path.basename(file))
        
        print("\n")

        # Move all files with extensions ".h5" and ".img" from the source directories to the destination directory
        for file in os.listdir(path):
            if file.endswith(".h5"):
                shutil.move(os.path.join(path, file), os.path.join(timestamps_path, file))
            elif file.endswith(".img"):
                shutil.move(os.path.join(path, file), os.path.join(images_path, file))
            else:
                if os.path.isfile(os.path.join(path, file)):
                    print("Warning: File with invalid extension found: " + file)
                else:
                    pass
                
        # Print a message indicating that the files have been moved
        print("Files have been moved to the destination directory")
