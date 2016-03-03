""" Command line interface for quick, basic, NRT classification of forest and non-forest. 
"""
from datetime import datetime as dt
import logging
import os


import click
import numpy as np
from osgeo import gdal

from yatsm.cli import options
from yatsm.utils import find_results, iter_records, write_output
from yatsm.cli.map import find_result_attributes
gdal.AllRegister()
gdal.UseExceptions()

logger = logging.getLogger('yatsm')

# Filters for results
_result_record = 'yatsm_*'

WARN_ON_EMPTY = False


@click.command(
    short_help='Map change found by YATSM algorithm over time period')
@click.argument('map_type', metavar='<map_type>',
                type=click.Choice(['first', 'last', 'num', 'class']))
@options.arg_date(var='start_date', metavar='<start_date>')
@options.arg_date(var='end_date', metavar='<end_date>')
@options.arg_output
@options.opt_rootdir
@options.opt_resultdir
@options.opt_exampleimg
@options.opt_date_format
@options.opt_nodata
@options.opt_format
@click.option('--monitor_date', 'monitor_date', metavar='<monitor_date>',
              help='Date to begin monitoring')
@click.option('--band', '-b', multiple=True, metavar='<band>', type=int,
              help='Bands to export for coefficient/prediction maps')
@click.option('--out_date', 'out_date_frmt', metavar='<format>',
              default='%Y%j', show_default=True, help='Output date format')
@click.option('--warn-on-empty', is_flag=True,
              help='Warn user when reading in empty results files')
@click.option('--magnitude', is_flag=True,
              help='Add magnitude of change as extra image '
                   '(pattern is [name]_mag[ext])')
@click.option('--detect', is_flag=True,
              help='Create map of dates change is detected, not date it happened ')
@click.pass_context
def monitor_map(ctx, map_type, start_date, end_date, monitor_date, output,
              root, result, image, date_frmt, ndv, gdal_frmt, out_date_frmt, band,
              warn_on_empty, magnitude, detect):
    """
    Examples: TODO
    """
    gdal_frmt = str(gdal_frmt)  # GDAL GetDriverByName doesn't work on Unicode

    frmt = '%Y%j'
    start_txt, end_txt = start_date.strftime(frmt), end_date.strftime(frmt)
    start_date, end_date = start_date.toordinal(), end_date.toordinal()

    try:
        image_ds = gdal.Open(image, gdal.GA_ReadOnly)
    except:
        logger.error('Could not open example image for reading')
        raise

    changemap = get_NRT_class(
            start_date, end_date, monitor_date, band, detect, result, image_ds,
            ndv=ndv, pattern=_result_record
        )
    band_names=['class']
    write_output(changemap, output, image_ds, gdal_frmt, ndv,
                     band_names=band_names)

    image_ds = None

# UTILITIES

def get_NRT_class(start, end, monitor, ndvi, detect, result_location, image_ds,
            ndv=-9999, pattern=_result_record):
    """ Output a raster with forest/non forest classes for time period specied. 

    Args:
        date (int): Ordinal date for prediction image
        result_location (str): Location of the results
        ndvi (list): Band of NDVI in results (indexed on 1). 
        image_ds (gdal.Dataset): Example dataset
        prefix (str, optional): Use coef/rmse with refit prefix (default: '')
        ndv (int, optional): NoDataValue (default: -9999)
        pattern (str, optional): filename pattern of saved record results

    Returns:
        tuple: A tuple (np.ndarray, list) containing the 3D numpy.ndarray of
            with the first band:
		1 = stable forest 2 = stable nonforest 3 = forest to nonforest
            second band:
		date of forest to nonforest
    """

    #TDODS: Make band not hard coded to 1

    # Find results
    records = find_results(result_location, pattern)
    prefix=''
    # Find result attributes to extract
    i_bands, i_coefs, use_rmse, coef_names, _, _ = find_result_attributes(
        records,ndvi, 'all', prefix=prefix)

    n_bands = len(i_bands)
    n_coefs = len(i_coefs)
    n_rmse = n_bands 
    ndvi = ndvi[0] - 1

    raster = np.zeros((image_ds.RasterYSize, image_ds.RasterXSize, n_bands),
                     dtype=np.int32) * int(ndv)
    raster[:,:]=1
    for a, rec in iter_records(records):

        # How many changes for unique values of px_changed?
        if n_coefs > 0:
                # Normalize intercept to mid-point in time segment
            rec['coef'][:, 0, :] += (
            (rec['start'] + rec['end'])
             / 2.0)[:, np.newaxis] * \
             rec['coef'][:, 1, :]

            indice = np.where((rec['start'] <= end))[0]
            pre_mon_indice = np.where((rec['end'] <= end))[0]
	    forest = np.where((rec['coef'][indice][:,0,ndvi]>8000))[0]

	    #Set forested pixels to two
            raster[rec['py'][indice][forest],rec['px'][indice][forest]] = 2

            #Overwrite with date of deforestation
            deforestation = np.where((rec['break'] >= end) & (rec['coef'][:,0,ndvi]>8000) & (rec['start'] <= end) \
                                & (rec['rmse'][:,ndvi]<600) & (rec['coef'][:,1,ndvi]<1))[0]
	    if np.shape(deforestation)[0] > 0:
                if detect:
                    dates = np.array([int(dt.fromordinal(_d).strftime('%Y%j'))
                                     for _d in rec['detect'][deforestation]])
		    for i, a in enumerate(dates):
			    raster[rec['py'][deforestation][i],rec['px'][deforestation][i]]=dates[i]                   
                else:
                    dates = np.array([int(dt.fromordinal(_d).strftime('%Y%j'))
                                     for _d in rec['break'][deforestation]])
		    for i, a in enumerate(dates):
			    raster[rec['py'][deforestation][i],rec['px'][deforestation][i]]=dates[i]                   

            #Overwrite if it contained nonforest before monitoring period? 
	    nonforest = np.where((rec['coef'][indice][:,0,ndvi]<8000) \
                                & (rec['rmse'][indice][:,ndvi]>300))[0]
            raster[rec['py'][indice][nonforest],rec['px'][indice][nonforest]] = 1 
	    px_all=[]
       
    return raster
