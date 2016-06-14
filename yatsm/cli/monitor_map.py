""" Command line interface for quick, basic, NRT classification of forest and non-forest. 
"""
from datetime import datetime as dt
import logging
import os


import numpy as np
from osgeo import gdal, osr, ogr
import click
from yatsm.config_parser import parse_config_file
from yatsm.cli import options
from yatsm.utils import find_results, iter_records, write_output
from yatsm.cli.map import find_result_attributes
gdal.AllRegister()
gdal.UseExceptions()

logger = logging.getLogger('yatsm')

# Filters for results

WARN_ON_EMPTY = False

@click.command(short_help='Create map of deforestation')
@options.arg_config_file
@options.arg_date(var='start_date', metavar='<start_date>')
@options.arg_date(var='end_date', metavar='<end_date>')
@options.arg_date(var='monitor_date', metavar='<monitor_date>')
@options.arg_output
@options.opt_date_format
@options.opt_exampleimg
@click.option('--detect', is_flag=True,
              help='Output date detected instead of change date')

@click.pass_context

def monitor_map(cfx, config, start_date, end_date, monitor_date, output,
               date_frmt, image, detect):
    """
    Examples: TODO
    """
    make_map(config, start_date, end_date, monitor_date, output,
               date_frmt, image, detect)



def make_map(config, start_date, end_date, monitor_date, output,
               date_frmt, image, detect):


    gdal_frmt = 'GTiff' #TODO
    gdal_frmt = str(gdal_frmt)  # GDAL GetDriverByName doesn't work on Unicode
    config = parse_config_file(config)
    frmt = '%Y%j' #TODO 
    ndv = 0 #TODO 
    start_date = start_date.toordinal()
    end_date = end_date.toordinal()
    monitor_date = monitor_date.toordinal()
    #Need to incorporate this into monitor script correctly
#    start_date = dt.strptime(str(start_date), frmt).toordinal()
#    end_date = dt.strptime(str(end_date), frmt).toordinal()
#    monitor_date = dt.strptime(str(monitor_date), frmt).toordinal()

    #start_date, end_date = start_date.toordinal(), end_date.toordinal()

    try:
        image_ds = gdal.Open(image, gdal.GA_ReadOnly)
    except:
        logger.error('Could not open example image for reading')
        raise

    changemap = get_NRT_class(
            config, start_date, end_date, monitor_date, detect,image_ds,
            ndv=ndv
        )
    band_names=['class']
    shapefile = True #TODO 
    if shapefile: #TODO
        write_shapefile(changemap, output,image_ds, gdal_frmt, 
	    	         ndv, band_names=band_names)
        #write_output(changemap, '/projectnb/landsat/datasets/MODIS/h12v09/monitor/2016.tif', image_ds, gdal_frmt, ndv,
         #            band_names=band_names)
    else:
        write_output(changemap, output, image_ds, gdal_frmt, ndv,
                     band_names=band_names)
    image_ds = None

# UTILITIES

def write_shapefile(changemap, output, image_ds, gdal_frmt, ndv, band_names):
    """ Write raster to output shapefile """
    from osgeo import gdal, gdal_array

    logger.debug('Writing output to disk')

    driver = gdal.GetDriverByName('MEM')

    nband = 1
    ds = driver.Create(
        output,
        image_ds.RasterXSize, image_ds.RasterYSize, nband,
        gdal_array.NumericTypeCodeToGDALTypeCode(changemap.dtype.type) #TODO issue
    )

    ds.GetRasterBand(1).WriteArray(changemap)
    ds.GetRasterBand(1).SetNoDataValue(ndv)

    ds.SetProjection(image_ds.GetProjection())
    ds.SetGeoTransform(image_ds.GetGeoTransform())

    srcband = ds.GetRasterBand(1)

#  create output datasource

    dst_srs = osr.SpatialReference()
    dst_srs.ImportFromProj4('+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +a=6371007.181 +b=6371007.181 +units=m +no_defs')

    dst_layername = output
    drv = ogr.GetDriverByName("ESRI Shapefile")
    dst_ds = drv.CreateDataSource( dst_layername )
    dst_layer = dst_ds.CreateLayer(dst_layername, srs = dst_srs )
    newField = ogr.FieldDefn('Date', ogr.OFTInteger)
    dst_layer.CreateField(newField)

    gdal.Polygonize( srcband, srcband, dst_layer, 0, ["8CONNECTED=8"], callback=None )	

    ds = None
    dst_layer = None 


def get_NRT_class(cfg, start, end, monitor,detect,image_ds,
            ndv=-9999):
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

    rmse_thresh = cfg['NRT']['rmse_threshold']
    ndvi_thresh = cfg['NRT']['ndvi_threshold']
    slope_thresh = cfg['NRT']['slope_threshold']
    ndvi = cfg['CCDCesque']['test_indices']
    pattern = 'yatsm_*' #TODO 
    result_location = cfg['dataset']['output']
    # Find results
    records = find_results(result_location, pattern)
    prefix=''
    # Find result attributes to extract
    i_bands, i_coefs, use_rmse, coef_names, _, _ = find_result_attributes(
        records,ndvi, 'all', prefix=prefix)

    n_bands = len(i_bands)
    n_coefs = len(i_coefs)
    n_rmse = n_bands 

    raster = np.zeros((image_ds.RasterYSize, image_ds.RasterXSize, n_bands),
                     dtype=np.int32) * int(ndv)
    stable = False #TODO 
    if stable: #TODO 
        raster[:,:]=1
    changemap = True #TODO: Prob or change? 
    for a, rec in iter_records(records):

        # How many changes for unique values of px_changed?
        if changemap & (n_coefs > 0):
                # Normalize intercept to mid-point in time segment
            rec['coef'][:, 0, :] += (
            (rec['start'] + rec['end'])
             / 2.0)[:, np.newaxis] * \
             rec['coef'][:, 1, :]

            indice = np.where((rec['start'] <= end))[0]
	    forest = np.where((rec['coef'][indice][:,0,ndvi]>ndvi_thresh))[0]

	    #Set forested pixels to two
	    if stable: #TODO 
                raster[rec['py'][indice][forest],rec['px'][indice][forest]] = 2
	    else:
                raster[rec['py'][indice][forest],rec['px'][indice][forest]] = 0

	    i_break = (rec['break'] >= end)
	    i_ndvi = (rec['coef'][:,0,ndvi]>ndvi_thresh)[:,0]
	    i_end = (rec['start'] <= end)
	    i_rmse = (rec['rmse'][:,ndvi]<rmse_thresh)[:,0]
	    i_slope = (rec['coef'][:,1,ndvi]<slope_thresh)[:,0]
            deforestation = np.where(np.logical_and.reduce((i_break, i_ndvi, i_end, i_rmse, i_slope)))[0]
            #Overwrite with date of deforestation. This needs to be faster
#	    if 
	    if np.shape(deforestation)[0] > 0:
                if changemap & detect:
                    dates = np.array([int(dt.fromordinal(_d).strftime('%Y%j'))
                                     for _d in rec['detect'][deforestation]])
		    for i, a in enumerate(dates):
			    raster[rec['py'][deforestation[i]],rec['px'][deforestation[i]]]=dates[i]
                elif changemap:
		    try: 
                        dates = np.array([int(dt.fromordinal(_d).strftime('%Y%j'))
                                         for _d in rec['break'][deforestation]])
		    except:
			continue 
			#import pdb; pdb.set_trace()
		    for i, a in enumerate(dates):
			    raster[rec['py'][deforestation][i],rec['px'][deforestation][i]]=dates[i]                   
                else:
                    raster[rec['py'][deforestation],
                           rec['px'][deforestation]] = rec['consec'][deforestation]

            #Overwrite if it contained nonforest before monitoring period? 
	    nonforest = np.where((rec['coef'][indice][:,0,ndvi]<ndvi_thresh) \
                                & (rec['rmse'][indice][:,ndvi]>rmse_thresh))[0]
	    if stable: 
                raster[rec['py'][indice][nonforest],rec['px'][indice][nonforest]] = 1 
	    else: 
                raster[rec['py'][indice][nonforest],rec['px'][indice][nonforest]] = 0 
    return raster[:,:,0]
