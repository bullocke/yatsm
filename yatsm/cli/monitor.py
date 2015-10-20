""" Module for creating probability of change maps based on previous time series results
"""
import datetime as dt
import logging
import os
import re

import click
import numpy as np
from osgeo import gdal
import patsy

from yatsm.cli import options
from yatsm.config_parser import parse_config_file
from yatsm.utils import get_output_name, find_results, iter_records, write_output
from yatsm.regression import design_to_indices, design_coefs
from yatsm.regression.transforms import harm
from yatsm.cli.map import get_prediction 
gdal.AllRegister()
gdal.UseExceptions()

logger = logging.getLogger('yatsm')

result_location = '/projectnb/landsat/users/bullocke/yatsm_newest/yatsm_NRT/sample_data/YATSM/'
pattern = '*npz'
_result_record = 'yatsm_r*'

@click.command(short_help='Make map of YATSM output for a given date')
@options.arg_config_file
@options.arg_date()
@options.arg_output
@options.opt_rootdir
@options.opt_resultdir
@options.opt_exampleimg
@options.opt_date_format
@options.opt_nodata
@options.opt_format
@click.option('--band', '-b', multiple=True, metavar='<band>', type=int,
              help='Bands to export for coefficient/prediction maps')
@click.pass_context


def monitor(ctx, config, date, output,
        root, result, image, date_frmt, ndv, gdal_frmt, band):
    """
    Examples:
    > yatsm monitor -r YATSM -b 1 -b 2 -b 3 -b 4 -b 5 -b 6 -b 7 -b 8 -b 9 -i 2012/MOD_A2012001.gtif params.yaml 2012-01-01 2012001_Prob.tif \

    Notes:
        - Any notes?
    """

#    import pdb; pdb.set_trace() 
    dataset_config = parse_config_file(config)
    try:
        image_ds = gdal.Open(image, gdal.GA_ReadOnly)
    except:
        logger.error('Could not open new image for reading')
        raise

    #Open new mage as array
    image_ar = image_ds.ReadAsArray()
    date = date.toordinal()

    band_names = 'Probability'
    raster, probability = get_mon_prediction(
            date, result, image_ds, image_ar, dataset_config,
            band,
            ndv=ndv
        )
    write_mon_output(probability, output, image_ds,
                 gdal_frmt, ndv)


def get_mon_prob(linescores, linestatus, lineconsec, probweights, lineprob, vzaweight):
    threshold = 5
    mag = np.linalg.norm(np.abs(linescores), axis=2)
    for i in range(np.shape(mag)[1]):
	if mag[0,i] > threshold:
	    lineprob[0,i] = ((vzaweight[0,i]*1) + lineconsec[0,i,0])
            linestatus[0,i,0] = 1
	    #one means last one was change
	    lineconsec[0,i,0] += 1
	elif linestatus[0,i,0] == 0:
	    linestatus[0,i,0] = 0
	#If it's below threshold and it's negative one - this means the last one was below. 	
	elif linestatus[0,i,0] < 0:
	    lineprob[0,i,0] = 0
	    lineconsec[0,i,0] = 0
	    linestatus[0,i,0] = 0        
   
    return lineprob

def get_mon_scores(raster, image_ar, _px, _py, i_bands, rmse, temp):

    scores=np.zeros((np.shape(raster)[1],np.shape(raster)[2]))
    vzaweight=np.zeros((np.shape(raster)[1],1))

    for _band in i_bands: 
        for px in _px:
            py=_py[0]
	    #Do not continue if it is masked as cloud. Turn this off for testing purposes. 
#	    if image_ar[9,py,px] == 0:
 #               continue
            im_val=image_ar[_band,py,px]
            scores[px,_band]=(im_val.astype(float) - raster[py,px,_band])/rmse[0,px,_band]
	    vzaweight[px] = 3000.0 / image_ar[10,py,px]

    return scores, vzaweight


def write_mon_output(raster, output, image_ds, gdal_frmt, ndv):
    """ Write raster to output file """
    from osgeo import gdal, gdal_array

    logger.debug('Writing output to disk')

    driver = gdal.GetDriverByName(str(gdal_frmt))

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


def find_mon_result_attributes(results, bands, coefs, prefix=''):
    """ Returns attributes about the dataset from result files

    Args:
        results (list): Result filenames
        bands (list): Bands to describe for output
        coefs (list): Coefficients to describe for output
        prefix (str, optional): Search for coef/rmse results with given prefix
            (default: '')

    Returns:
        tuple: Tuple containing `list` of indices for output bands and output
            coefficients, `bool` for outputting RMSE, `list` of coefficient
            names, `str` design specification, and `OrderedDict` design_info
            (i_bands, i_coefs, use_rmse, design, design_info)

    """
    _coef = prefix + 'coef' if prefix else 'coef'
    _rmse = prefix + 'rmse' if prefix else 'rmse'

    n_bands, n_coefs = None, None
    design = None
    for r in results:
        try:
            _result = np.load(r)
            rec = _result['record']
            design = _result['design_matrix'].item()
            design_str = _result['design'].item()
        except:
            continue

        if not rec.dtype.names:
            continue

        if _coef not in rec.dtype.names or _rmse not in rec.dtype.names:
            logger.error('Could not find coefficients ({0}) and RMSE ({1}) '
                         'in record'.format(_coef, _rmse))
            if prefix:
                logger.error('Coefficients and RMSE not found with prefix %s. '
                             'Did you calculate them?' % prefix)
            raise click.Abort()

        try:
            n_coefs, n_bands = rec[_coef][0].shape
        except:
            continue
        else:
            break

    if n_coefs is None or n_bands is None:
        logger.error('Could not determine the number of coefficients or bands')
        raise click.Abort()
    if design is None:
        logger.error('Design matrix specification not found in results.')
        raise click.Abort()

    # How many bands does the user want?
    if bands == 'all':
        i_bands = range(0, n_bands)
    else:
        # NumPy index on 0; GDAL on 1 -- so subtract 1
        i_bands = [b - 1 for b in bands]
        if any([b > n_bands for b in i_bands]):
            logger.error('Bands specified exceed size of bands in results')
            raise click.Abort()

    # How many coefficients did the user want?
    use_rmse = False
    if coefs:
        if 'rmse' in coefs or 'all' in coefs:
            use_rmse = True
        i_coefs, coef_names = design_to_indices(design, coefs)
    else:
        i_coefs, coef_names = None, None

    logger.debug('Bands: {0}'.format(i_bands))
    if coefs:
        logger.debug('Coefficients: {0}'.format(i_coefs))

    return (i_bands, i_coefs, use_rmse, coef_names, design_str, design)

def find_mon_indices(record, date, after=False, before=False):
    """ Yield indices matching time segments for a given date

    Args:
      record (np.ndarray): Saved model result
      date (int): Ordinal date to use when finding matching segments
        non-disturbed time segment

    Yields:
      tuple: (int, np.ndarray) the QA value and indices of `record` containing
        indices matching criteria. If `before` or `after` are specified,
        indices will be yielded in order of least desirability to allow
        overwriting -- `before` indices, `after` indices, and intersecting
        indices.

    """

    #This needs to be the most RECENT model, right now it's the first. Ok for stable landcovers, but not ideal. 
    index = np.where((record['end'] <= date))[0]
    yield index


def get_mon_prediction(date, result_location, image_ds, image_ar, dataset_config,
		   bands='all', prefix='', ndv=-9999, pattern=_result_record):

    """ Get prediction for date of input image.

    Args:
        date (int): ordinal date for input (new) image
        result_location (str): Location of the results
        image_ds (gdal.Dataset): Example dataset
        ndv (int, optional): NoDataValue
        pattern (str, optional): filename pattern of saved record results

    Returns:
        np.ndarray: 2D numpy array containing the change probability map for the
            image specified """


    # Find results
    records = find_results(result_location, pattern)

    # Find result attributes to extract
    i_bands, _, _, _, design, design_info = find_mon_result_attributes(
        records, bands, None, prefix=prefix)
    n_bands = len(i_bands)
    n_i_bands = len(i_bands)

    # Create X matrix from date -- ignoring categorical variables
    if re.match(r'.*C\(.*\).*', design):
        logger.warning('Categorical variable found in design matrix not used'
                       ' in predicted image estimate')
    design = re.sub(r'[\+\-][\ ]+C\(.*\)', '', design)
    X = patsy.dmatrix(design, {'x': date}).squeeze()

    i_coef = []
    for k, v in design_info.iteritems():
        if not re.match('C\(.*\)', k):
            i_coef.append(v)
    i_coef = np.asarray(i_coef)

    logger.debug('Allocating memory')

    raster = np.ones((image_ds.RasterYSize, image_ds.RasterXSize, n_bands),
                     dtype=np.int16) * int(ndv)
    probability = np.ones((image_ds.RasterYSize, image_ds.RasterXSize, 1),
                     dtype=np.int16) * int(ndv)
    logger.debug('Processing results')

    temp=0
    csv=[]
    for z, rec in iter_records(records, warn_on_empty=True):
	#looping over indices
       temp+=1
       linescores = np.zeros((1, image_ds.RasterXSize, n_bands),
                     dtype=np.float32) * int(ndv)
       vzaweight = np.zeros((1, image_ds.RasterXSize, 1),
                     dtype=np.float32) * int(ndv)
       
       rmse = np.ones((1, image_ds.RasterXSize, n_bands),
                     dtype=np.float32) * int(ndv)

       lineconsec = np.zeros((1, image_ds.RasterXSize, 1),
                     dtype=np.float32) * int(ndv)

       linestatus = np.zeros((1, image_ds.RasterXSize, 1),
                     dtype=np.float32) * int(ndv)
       lineprob = np.zeros((1, image_ds.RasterXSize, 1),
                     dtype=np.float32) * int(ndv)
       for index in find_mon_indices(rec, date):
           if index.shape[0] == 0:
                continue

           # Calculate prediction
           _coef = rec['coef'].take(index, axis=0).\
                take(i_coef, axis=1).take(i_bands, axis=2)
           raster[rec['py'][index], rec['px'][index], :n_i_bands] = \
                np.tensordot(_coef, X, axes=(1, 0))

           #Get x and y location from records
	   _px = rec['px'][index]
	   _py = rec['py'][index]

	   #Calculate RMSE
	   rmse[:, rec['px'][index], :n_i_bands] = rec['rmse'][index][:, i_bands]

           #Calculate how many (if any) consective observations have been above threshold
	   lineconsec[:, rec['px'][index],0] = rec['consec'][index][:,]

	   #Calculate the 'status' of the pixel. Necessary since we are allowing one consecutive under the threshold
	   linestatus[:, rec['px'][index],0] = rec['status'][index][:,]

	   #calculate change score for each band 
	   linescores[:,:,:], vzaweight[:,:,:] = get_mon_scores(raster, image_ar, _px, _py, i_bands, rmse, temp)

	   #For testing purposes, save the scores
	   testname='%s' % rec['py'][index][0]
	   np.save(testname, linescores)

	   #Calculate current probability
	   probweights = 0
           probability[np.unique(rec['py']),:,:] = get_mon_prob(linescores, linestatus, lineconsec, probweights, lineprob, vzaweight) 
       csv.append(linescores)
    #resave 
       out = {}
       for k, v in z.iteritems():
           out[k] = v
       for q in range(np.shape(probability)[1]):
           indice=np.where(out['record']['px']==q)
           if np.shape(indice)[1] == 0:
	       continue
	   out['record']['consec'][indice] = lineconsec[0,q,0]
	   out['record']['status'][indice] = linestatus[0,q,0]
	   out['record']['probability'][indice] = lineprob[0,q,0]
       #filename = get_output_name(dataset_config['dataset'], _py[0])       
    #   np.savez(filename, **out) 
    return raster, probability

