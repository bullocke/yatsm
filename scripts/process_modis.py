import click
import numpy as np
import csv
import sys
import ftplib
from ftplib import FTP
import os
import time
from shutil import copyfile
from yatsm.cli import options 
from yatsm.config_parser import parse_config_file
from image_composites import *
from prep_modis import *

@click.command(short_help='Download and pre-process MODIS data in NRT')
@options.arg_config_file
@click.pass_context

def process_modis(config):
    """Master function (TODO: Add inputs)
    Inputs: TODO
	config: Configuration file for YATSM
    Outputs: TODO
    """
    wkdir=os.getcwd()
    #Change to directory to work in 
#    os.chdir(imagedirectory)
    cfg = parse_config_file(config)
    downloaddirectory = cfg['NRT']['download_directory'] #Where images get downloaded
    stackdirectory = cfg['NRT']['stack_directory'] #Where images get stacked
    outputdir = cfg['NRT']['output_directory'] #Where images get composited
    tile = cfg['NRT']['tile'] #The modis tile

    #Set up lists to store file names for each step in the process
    pre_list = []
    comp_list = []
    download_list=[]

    #Download the images
    download_list = get_modis("MOD09GA", tile, 2016, download_list) 
    download_list = get_modis("MYD09GA", tile, 2016, download_list)
    download_list = get_modis("MOD09GQ", tile, 2016, download_list)
    download_list = get_modis("MYD09GQ", tile, 2016, download_list)

    #Pre-process the images
    pre = preprocess(downloaddirectory, stackdirectory, "MOD*09*hdf")
    pre_list.append(pre)
    pre = preprocess(downloaddirectory, stackdirectory, "MYD*09*hdf")
    pre_list.append(pre)

    #Find pairs of images to composite, or find which days have only 1 image
    tocomp_list, myd_list, mod_list = find_comp_pairs(stackdirectory)

    #Composite the images based on view angle
    for images in tocomp_list:
	comp = composite(images, outputdir, tile)
	comp_list.append(comp)

    #if no images that day just copy original modis file
    for image in mod_list:
	im = copy_image(image, outputdir, tile)
	comp_list.append(im)

    for image in myd_list:
	im = copy_image(image, outputdir, tile)
	comp_list.append(im)

    #Save output to daily log

def find_comp_pairs(location):
    """!!!!Re-written from code by Chris Holden!!!!
       Finds matching sets of M[OY]D09GQ and M[OY]D09GA within location

    Args:
      location (str): directory of stored data
      pattern (str, optional): glob pattern to limit search

    Returns:
      pairs (list): list of tuples containing M[OY]D09GQ and M[OY]D09GA

    """
    pattern='M*tif'
    files = [os.path.join(location, f) for f in
             fnmatch.filter(os.listdir(location), pattern)]

    if len(files) < 2:
        raise IOError('Could not find any MODIS image pairs')

    # Parse out product and acquisition date
    products = []
    dates = []
    for f in files:
        s = os.path.basename(f).split('_')
        products.append(s[0])
        dates.append(s[1])

    products = np.array(products)
    dates = np.array(dates)

    # Retain dates if there are matching MOD09GA/MOD09GQ or MYD09GA/MYD09GQ
    pairs = []
    mod = []
    myd = []
    for d in np.unique(dates):
        i = np.where(dates == d)[0]
        prods = products[i]

        i_mod = np.core.defchararray.startswith(prods, 'MOD')
        i_myd = np.core.defchararray.startswith(prods, 'MYD')

        if i_mod.sum() == 1 and i_myd.sum() == 1:
            pairs.append((files[i[i_mod]], files[i[i_myd]]))
        if i_mod.sum() == 1 and i_myd.sum() == 0:
            mod.append(files[i[i_mod]])
        if i_mod.sum() == 0 and i_myd.sum() == 1:
            myd.append(files[i[i_myd]])


    return pairs, myd, mod



def composite(images, outputdir, tile):
    vza_band = 7 #TODO make this an option
    mask_band = 6
    s = os.path.basename(images[0]).split('_')
    date = s[1][1:]
    output = '%s/Composite_%s_%d.tif'  % (outputdir, tile, date) 
    image_composite(images, 'minBlue', 'None', False, output, 'Gtiff', False, vza_band, 1, 2, 3, 4, 5, False, mask_band, 0, True, False) 
    return output

def copy_image(image, outputdir, tile):
    vza_band = 7 #TODO make this an option
    mask_band = 6
    s = os.path.basename(images[0]).split('_')
    date = s[1][1:]
    output = '%s/Composite_%s_%d.tif'  % (outputdir, tile, date) 
    copyfile(image,output)
    return output
    
def download_files(ftp_con, sub, download_list):
    """!!Writen by Paulo Arevalo!!
    Finds the files in the current folder that meet the required criteria, e.g.
        last calculated daily product. The substring (sub variable) makes reference
         to the string we want to look for in the filenames, e.g. product type or
         tile information"""

    # Find the files we need to download
    files = ftp_con.nlst()  # Get files in the folder
    req_files = list(s for s in files if sub in s)  # Retrieve required files only
    download_list = []
    # Download the files
    for f in range(len(req_files)):
        if os.path.isfile(req_files[f]):
            pass
        else:
            retr_command = "RETR " + req_files[f]
            ftp_con.retrbinary(retr_command, open(req_files[f], 'wb').write)
            download_list.append(req_files[f])
    return download_list


def get_modis(product, sub, start_year, download_list):
    """!!Writen by Paulo Arevalo!!
	This function downloads all the files for a given MODIS product starting from the specified year until the
        current date that contain the specified substring (e.g h09v06)
     Args:
        product (str): Name of the MODIS product
        sub (str): Substring to look for in each filename
        start_year (int): Initial year of the period for which files will be downloaded
     """
    # Connect and go to the product folder
    ftp = ftp_connect("ladsweb.nascom.nasa.gov", "anonymous", passw, "/allData/6/" + product)
    #password could be @anonymous
    # Iterate over folder names, download all the files
    for y in range(start_year, year + 1):  # Getting data from 2015 until the current date
        ftp.cwd(str(y))
        print ftp.pwd()
        day_list = ftp.nlst()  # Get day folder list
        for d in day_list:
            ftp.cwd(str(d))
            downloaded = download_files(ftp, sub, download_list)
	    download_list.append(downloaded)
            ftp.cwd('..')
        ftp.pwd()
        ftp.cwd('..')

    ftp.quit()
    return download_list

def preprocess(downloaddirectory, stackdirectory, pattern):
    blockysize = 1
    pattern = imagename #TODO need to get this
    pairs = find_MODIS_pairs(downloaddirectory, pattern)
    output_names = get_output_names(pairs,stackdirectory)   

    for i, (p, o) in enumerate(zip(pairs, output_names)):
        logger.info('Stacking {i} / {n}: {p}'.format(
            i=i, n=len(pairs), p=os.path.basename(p[0])))
        try:
            create_stack(p, o,
                         ndv=ndv, compression=compression,
                         tiled=tiled,
                         blockxsize=blockxsize, blockysize=blockysize)
        except RuntimeError:
            if skip_error:
                logger.error('Could not preprocess pair: {p1} {p2}'.format(
                    p1=p[0], p2=p[1]))
                failed += 1
            else:
                raise

    if failed != 0:
        logger.error('Could not process {n} pairs'.format(n=failed))
    logger.info('Complete')


    return output_names

##Possible functions below::
def get_image_dates(config):
    """Retrieve dates of images to be processed
    Inputs: TODO

    Outputs: TODO
    """

    # Parse config
    cfg = parse_config_file(config)


def read_log():
    """!!Taken from http://www.gisremotesensing.com/2010/06/getting-modis-image-automatically-from.html"""
    logFileread=open(r"H:\MODIS_LST_NDVI\MOD11A2\lpdaac.txt",'r')
    logFileread.seek(0)
    logInfo=int(logFileread.read())
    logFileread.close()

def write_log():
    """!!Taken from http://www.gisremotesensing.com/2010/06/getting-modis-image-automatically-from.html"""
    logFwrite=open(r"H:\MODIS_LST_NDVI\MOD11A2\lpdaac.txt",'w') 
    logFwrite.seek(0) 
    logFwrite.write(dirInfo)
    logFwrite.close()

