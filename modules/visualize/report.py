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
import shutil
import glob
from fpdf import FPDF
from modules.load import timestamps
from modules.visualize import plot

from tqdm import tqdm

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from PIL import Image

# ---------------------------------------------------------------------#
# ------------------------ CREATE REPORT ------------------------------#
# ---------------------------------------------------------------------#

PLOT_DIR = 'plots'

def create(path_to_data_dir):
    """
    Creates a report for the data in the specified directory.
    """

    # Get path to data directory
    data_dir = os.path.dirname(path_to_data_dir)

    # Create the plots
    create_plots(path_to_data_dir)

    # Create the PDF
    create_pdf(f'{data_dir}/{PLOT_DIR}')



def create_plots(path):
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

    # Create the plots directory
    try:
        os.mkdir(f'{data_dir}/{PLOT_DIR}')
    except FileExistsError:
        print('Directory already exists.')
        # Stop execution
        return
    
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



def create_pdf(images_path, images_per_row=1, images_per_column=6):
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

    # Calculate the size of each image and the spacing between images
    image_width = 400 #page_width / images_per_row
    image_height = 100 #page_height / images_per_column
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
