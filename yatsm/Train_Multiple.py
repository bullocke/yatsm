"""Train image classifier for YATSM training script based on multiple input images. ROIs can be across all images."""
from yatsm.cli.train import get_training_inputs
from datetime import datetime as dt
from itertools import izip
import logging
import os

import click
import matplotlib.pyplot as plt
import numpy as np
from osgeo import gdal

from sklearn.cross_validation import KFold, StratifiedKFold
from sklearn.externals import joblib

from yatsm.cli import options
from yatsm.config_parser import parse_config_file
from yatsm import classifiers
from yatsm.classifiers import diagnostics
from yatsm.errors import TrainingDataException
from yatsm import plots
from yatsm import reader
from yatsm import utils

logger = logging.getLogger('yatsm')

gdal.AllRegister()
gdal.UseExceptions()

rasterlist=[]


X=[]
Y=[]
num=[]

def get_train(shape, YATSMlist, prs):

    """ Returns X features and y labels specified in config file
    Args:
	shape (OGR vector): ROI shapefile. One column is ROI, one is PathRow
	YATSMlist (np array): List of YATSM result folders to use in training
        cfg (dict): YATSM configuration dictionary
	prs (array): Array of Path Rows used in training
 
    Returns:
        X (np.ndarray): matrix of feature inputs for each training data sample
        y (np.ndarray): array of labeled training data samples
        row (np.ndarray): row pixel locations of `y`
        col (np.ndarray): column pixel locations of `y`
        labels (np.ndarraY): label of `y` if found, else None
	"""

    driver = ogr.GetDriverByName("ESRI Shapefile")
    dataSource = driver.Open(shape, 0)
    layer = dataSource.GetLayer()

    rasterlist=get_rast_list(YATSMlist) 

    prlist=get_prs(layer)

    #Loop over path rows, creating a memory vector for each
    for i in prslist:
	#Open raster associated with path row
	input_value_raster=rasterlist[i]
        rast = gdal.Open(input_value_raster) 
        #Get srs stuff from raster list
        raster_srs = osr.SpatialReference()
        raster_srs.ImportFromWkt(rast.GetProjectionRef())
        #Create memory vectors layer for feautres
        driver = ogr.GetDriverByName('MEMORY')
        out_ds = driver.CreateDataSource('tmp')
        out_layer = out_ds.CreateLayer('out', geom_type=ogr.wkbPolygon, srs=raster_srs)
        vector_srs = layer.GetSpatialRef()
        coord_trans = osr.CoordinateTransformation(vector_srs, raster_srs)
        featureDefn = out_layer.GetLayerDefn()
        feature = ogr.Feature(featureDefn)
	#Loop over layer and add appropriate features to memory vector
	for feat in layer:
	    Date = feature.GetField('PRs')	
            if Date == i:
                geom = feat.GetGeometryRef()
                geom.Transform(coord_trans)
                feature.SetGeometry(geom)
                out_layer.CreateFeature(feat)
            else:
		continue
	#Now rasterize the memory vector to the extent of the example image
        memLayer = out_ds.GetLayer()
	trainingraster= rasterize_mem(rast, memlayer) #TODO 

        #Now that we have the memory vector for that file we can do the training
        x, y, out_row, out_col, labels = get_mult_training_inputs(raster, memlayer)


def rasterize_mem(raster, memlayer):
    """ Returns X features and y labels specified in config file
    Args:
	raster (GDAL raster): Input example raster
	memlayer (OGR Memory Array): memory layer with features for training
 
    Returns:
        trainraster (GDAL Raster): Raster with training classes burned in
	"""

    gt = raster.GetGeoTransform()
    ul_x, ul_y = gt[0], gt[3]
    ps_x, ps_y = gt[1], gt[5]
    xmin, xmax, ymin, ymax = memlayer.GetExtent()
    xoff = int((xmin - ul_x) / ps_x)
    yoff = int((ul_y - ymax) / ps_x)
    xcount = int((xmax - xmin) / ps_x) + 1
    ycount = int((ymax - ymin) / ps_x) + 1
    target_ds = gdal.GetDriverByName('MEM').Create('', xcount, ycount, gdal.GDT_Byte)
    target_ds.SetGeoTransform((xmin, ps_x, 0,
                                   ymax, 0, ps_y))

    raster_srs = osr.SpatialReference()
    raster_srs.ImportFromWkt(raster.GetProjectionRef())
    target_ds.SetProjection(raster_srs.ExportToWkt())
     # Rasterize zone polygon to raster
    gdal.RasterizeLayer(target_ds, [1], memlayer, options = ["ATTRIBUTE=Class"])    

    return target_ds

def get_prs(layer):

    """ Get the Path Rows we will be using for training
    Args:
	shape (OGR vector): ROI shapefile. One column is ROI, one is PathRow
 
    Returns:
        prs (np.ndarray): array with list of prs in ROIs """

    prs_all = []
    for feature in layer:
	try:
            Date = feature.GetField('PRs')
            prs_all.append(Date)
	except:
	    print "No attribute called PRs in %s" % feature
     prs=np.unique(prs_all)
     return prs

def get_mult_training_inputs(cfg,trainingraster, trainingvector, exit_on_missing=False):
    """ Returns X features and y labels specified in config file
    Args:
	TODO:
        cfg (dict): YATSM configuration dictionary
    Returns:
        X (np.ndarray): matrix of feature inputs for each training data sample
	TODO
    """
    # Find and parse training data
    roi = reader.read_image(trainingraster)
    logger.debug('Read in training data')
    if len(roi) == 2:
        logger.info('Found labels for ROIs -- including in output')
        labels = roi[1]
    else:
        roi = roi[0]
        labels = None

    # Determine start and end dates of training sample relevance
    try:
        training_start = dt.strptime(
            cfg['classification']['training_start'],
            cfg['classification']['training_date_format']).toordinal()
        training_end = dt.strptime(
            cfg['classification']['training_end'],
            cfg['classification']['training_date_format']).toordinal()
    except:
        logger.error('Failed to parse training data start or end dates')
        raise

    # Loop through samples in ROI extracting features
    mask_values = cfg['classification']['roi_mask_values']
    mask = ~np.in1d(roi, mask_values).reshape(roi.shape)
    row, col = np.where(mask)
    y = roi[row, col]

    X = []
    out_y = []
    out_row = []
    out_col = []

    _row_previous = None
    for _row, _col, _y in izip(row, col, y):
        # Load result
        if _row != _row_previous:
            output_name = utils.get_output_name(cfg['dataset'], _row)
            try:
                rec = np.load(output_name)['record']
                _row_previous = _row
            except:
                logger.error('Could not open saved result file %s' %
                             output_name)
                if exit_on_missing:
                    raise
                else:
                    continue

        # Find intersecting time segment
        i = np.where((rec['start'] < training_start) &
                     (rec['end'] > training_end) &
                     (rec['px'] == _col))[0]

        if i.size == 0:
            logger.debug('Could not find model for label %i at x/y %i/%i' %
                         (_y, _col, _row))
            continue
        elif i.size > 1:
            raise TrainingDataException(
                'Found more than one valid model for label %i at x/y %i/%i' %
                (_y, _col, _row))

        # Extract coefficients with intercept term rescaled
        coef = rec[i]['coef'][0, :]
        coef[0, :] = (coef[0, :] +
                      coef[1, :] * (rec[i]['start'] + rec[i]['end']) / 2.0)

        X.append(np.concatenate((coef.reshape(coef.size), rec[i]['rmse'][0])))
        out_y.append(_y)
        out_row.append(_row)
        out_col.append(_col)

    out_row = np.array(out_row)
    out_col = np.array(out_col)

    if labels is not None:
        labels = labels[out_row, out_col]

    return np.array(X), np.array(out_y), out_row, out_col, labels


def get_rast_list(YATSMlist):
        """ 
	Args:
	TODO

	Returns:
	TODO
        """


    rastlist=[]
    for yat in YATSMlist:
	cfg = parse_config_file(config)	
        rastlist.append(cfg['classification']['training_image']

    return rastlist





