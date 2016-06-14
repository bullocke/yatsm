""" Module for creating probability of change maps based on previous time series results
"""
from datetime import datetime
import logging
import os
import re
import click
import csv
import numpy as np
from osgeo import gdal, gdal_array
import patsy
from ..algorithms.ccdc_monitor import *
from yatsm.cli import options
from monitor_map import make_map, write_shapefile
from yatsm.config_parser import parse_config_file
from yatsm.utils import get_output_name, find_results, iter_records, write_output
gdal.AllRegister()
gdal.UseExceptions()

logger = logging.getLogger('yatsm')
logger_algo = logging.getLogger('yatsm_algo')

pattern = '*npz'
_result_record = 'yatsm_r*'

@click.command(short_help='Monitor for changes give up to 1 year of new images')
@options.arg_config_file
@options.arg_mon_csv
@options.arg_output
@options.opt_date_format
@options.opt_nodata
@options.opt_format
@click.option('--save', is_flag=True,
              help='Update YATSM .npz files for monitoring')
@click.option('--band', '-b', multiple=True, metavar='<band>', type=int,
              help='Bands to export for coefficient/prediction maps')
@click.option('--output_rast', '-o', metavar='<output_rast>', type=click.Choice(['ChangeScore', 'Probability']),
              help='Output Probability or ChangeScore?')
@click.pass_context


def monitor(ctx, config, output, mon_csv, gdal_frmt, date_frmt, ndv, band, output_rast, save):
    """Command line interface to handle monitoring of new imagery. This program will not
     pre-process the data, which is done in yatsm.process_modis. This program will calculate
     the change probabilities in time-sequential order for all images in input monitoring log.
     Currently, the output is written to shapefiles for tileing on Mapbox. """

    logger_algo.setLevel(logging.DEBUG)
    #Parse config and open input csv
    cfg = parse_config_file(config)
    done_csv = cfg['dataset']['input_file']
    read=csv.reader(open(done_csv,"rb"),delimiter=',')
    done_array = list(read)
    last = int(done_array[-1][0])
    veryfirst = int(done_array[1][0])

    #Read monitor csv
    read_mon=csv.reader(open(mon_csv,"rb"),delimiter=',')
    monitor_array = list(read_mon)

    if monitor_array is None: 
        logger.error('Incorrect path to monitor csv')
        raise click.Abort()

    if len(monitor_array) == 0:
        logger.error('Not new images to monitor')
        raise click.Abort()
    first = int(monitor_array[0][0])

    #Loop over each date in monitor list. Check again if the date is in input list
    num_monitor=len(monitor_array)
    for i in range(num_monitor):
        cur_image = monitor_array[i]
	date = int(cur_image[0])
	image_path = cur_image[1]
	if date <= last:
            logger.error('Previous results processed past image date. Skipping.')
            continue

        #Read the image as an array. 
        try:
            image_ds = gdal.Open(image_path, gdal.GA_ReadOnly)
        except:
            logger.error('Could not open new image for reading')
            raise click.Abort()


        #Do monitor
        logger.info('Doing image %s' % image_path)
        out = ccdc_monitor(cfg, date, image_ds, save)

        output_lp_today, output_hp_today, output_lp, output_hp, output_conf, output_conf_today, master = get_mon_outputs(cfg, date)
	#Write out the shapefiles
        if np.any(out['lowprob'] > 2016000):
            write_shapefile(out['lowprob'], output_lp_today,image_ds, gdal_frmt, 
     	    	            ndv, band_names=None)
	    if os.path.isfile(output_lp):
	        os.remove(output_lp)
            write_shapefile(out['lowprob'], output_lp,image_ds, gdal_frmt, 
     	    	            ndv, band_names=None)
        if np.any(out['highprob'] > 2016000):
            write_shapefile(out['highprob'], output_hp_today,image_ds, gdal_frmt, 
   	    	            ndv, band_names=None)
	    if os.path.isfile(output_hp):
	        os.remove(output_hp)
            write_shapefile(out['highprob'], output_hp,image_ds, gdal_frmt, 
     	    	            ndv, band_names=None)
        if np.any(out['confirmed_today'] > 2016000):
            write_shapefile(out['confirmed_today'], output_conf_today,image_ds, gdal_frmt, 
   	    	            ndv, band_names=None)
        if np.any(out['confirmed'] > 2016000):
	    if os.path.isfile(master):
	        os.remove(master)
            write_shapefile(out['confirmed'], master,image_ds, gdal_frmt, 
   	    	            ndv, band_names=None)
	#update image list
	out_log = [str(date),'Com',image_path]
	done_array.append(out_log)
	with open(done_csv, 'wb') as f:
	    writer = csv.writer(f)
	    writer.writerows(done_array)

	output_rast = None

def get_mon_outputs(cfg, date):
    "Get output shapefile names" 
    out_dir = cfg['NRT']['master_shapefile_dir']
    if not os.path.isdir(out_dir):
	os.mkdir(out_dir)
    date_path = '%s/%s' % (out_dir, date)
    if not os.path.isdir(date_path):
	os.mkdir(date_path)
    output_lp_today = '%s/lowprob.shp' % (date_path)
    output_hp_today = '%s/highprob.shp' % (date_path)
    output_lp = '%s/lowprob.shp' % (out_dir)
    output_hp = '%s/highprob.shp' % (out_dir)
    output_conf = '%s/confirmed.shp' % (date_path)
    output_conf_today = '%s/confirmed.shp' % (date_path)
    master = cfg['NRT']['master_shapefile'] 
    return output_lp_today, output_hp_today, output_lp, output_hp, output_conf, output_conf_today, master


def write_mon_output(raster, output, image_ds, gdal_frmt, ndv):
    """ Write raster to output file """

    logger.debug('Writing output to disk')

    driver = gdal.GetDriverByName(str(gdal_frmt))
    band_names=['change_date']

    if len(raster.shape) > 2:
        nband = raster.shape[2]
    else:
        nband = 1

    ds = driver.Create(
        output,
        image_ds.RasterXSize, image_ds.RasterYSize, nband,
        gdal_array.NumericTypeCodeToGDALTypeCode(raster.dtype.type)
    )


    if raster.ndim > 2:
        for b in range(nband):
            logger.debug('    writing band {b}'.format(b=b + 1))
            ds.GetRasterBand(b + 1).WriteArray(raster[:, :, b])
            ds.GetRasterBand(b + 1).SetNoDataValue(ndv)

                
    else:
        logger.debug('    writing band')
        ds.GetRasterBand(1).WriteArray(raster)
        ds.GetRasterBand(1).SetNoDataValue(ndv)

        if band_names is not None:
            ds.GetRasterBand(1).SetDescription(band_names[0])
            ds.GetRasterBand(1).SetMetadata({'band_1': band_names[0]})

    ds.SetProjection(image_ds.GetProjection())
    ds.SetGeoTransform(image_ds.GetGeoTransform())

    ds = None


