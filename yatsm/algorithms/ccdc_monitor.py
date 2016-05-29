""" Module for creating probability of change maps based on previous time series results
"""
import datetime as dt
import logging
import os
import re
from datetime import datetime#
import click
import numpy as np
from osgeo import gdal
import patsy

from ..config_parser import parse_config_file
from ..utils import get_output_name, find_results, iter_records, write_output
from ..regression import design_to_indices, design_coefs
from ..regression.transforms import harm
from ..cli.map import get_prediction 
gdal.AllRegister()
gdal.UseExceptions()

logger = logging.getLogger('yatsm')
logger_algo = logging.getLogger('yatsm_algo')

pattern = '*npz'
_result_record = 'yatsm_r*'


def ccdc_monitor(cfg, date, image_ds, save):
    """Update previous CCDC results based on newly available imagery

    Args:
	TODO

    """
    logger_algo.setLevel(logging.DEBUG)
    #Open new mage as array
    image_ar = image_ds.ReadAsArray()
    band_names = 'Probability'
    outputrast = 'Probability'
    result = cfg['dataset']['output']


    #get prediction for image
    raster, probability, linescores_rast = do_monitor(
            date, result, image_ds, image_ar, cfg, save, outputrast
        )
    _outputrast = probability

    #Determine what to output
    if outputrast == "ChangeScore":
	_outputrast = linescores_rast*10

    return _outputrast



def get_mon_prob(lineprob,linescores, lineconsec, image_ar, py, px, date, linebreak, threshold, test_ind, cloudthreshold, green, swir, shadowthreshold, mask_band):
    """Get the probability of deforestation on given date
    inputs: TODO
    outputs: TODO
    """

    cloudprob=np.ones((1,np.shape(linescores)[1]))
    shadowprob=np.ones((1,np.shape(linescores)[1]))
    _px=px[0]
    #loop over pixels
    for i in range(np.shape(linescores)[1]):
	if lineconsec[0,i,0] >= 5: #TODO not hard coded
	    continue
	#First do the cloud test:
	if linescores[0,i,green] > cloudthreshold:
	    cloudprob[0,i]=0
	if image_ar[mask_band,py,i] < 1:
	    continue 
	#Now do the shadow test:
	if linescores[0,i,swir] < shadowthreshold:
	    continue
	#First check if it passes NDVI threshold
	if linescores[0,i,test_ind] < threshold:
	    lineconsec[0,i,0] += 1
	    if lineconsec[0,i,0] >= 3: #record date for low prob to confirmed
	        linebreak[0,i,0] = date
	#But if it's not change, and last one was...set it to -1 to not start over. 
    return lineprob[0,:,0], lineconsec[0,:,0], linebreak[0,:,0]

def get_mon_scores(raster, image_ar, _px, _py, i_bands, rmse, outputrast, mask_band, mask_values):

    """Return change scores based on model RMSE. Also returns VZA weight.

    Args:
	raster (arr): predicted image for date specified.
	image_ar (arr): array of input image. 
	_px: X location of pixel
	_py: Y location of pixel
	i_bands (list): List of bands to use in change detection
	rmse (arr): rmse for model being used for prediction.
	VZAweight: TODO: Add this as a variable

    Returns:
        scores(np.ndarray): 2D numpy array containing the change scores for the
            image specified 
	vzaweight(np.ndarray): 2D numpy array containing vza weights to use for image. 

	"""
    scores=np.zeros((np.shape(raster)[1],np.shape(raster)[2]))
    for _band in i_bands: 
        for px in _px:
            py=_py[0]
	    if image_ar[mask_band,py,px] == np.any(mask_values): #TODO get mask_band and value from config file
                continue
            im_val=image_ar[_band,py,px]
            scores[px,_band]=((im_val.astype(float) - raster[py,px,_band])/rmse[0,px,_band])
    return scores



def find_mon_result_attributes(results, bands, coefs, config, prefix=''):
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
    _coef = 'coef' 
    _rmse = 'rmse' 

    n_bands, n_coefs = None, None
    design = None
    for r in results:
        try:
            _result = np.load(r)
            rec = _result['record']
            if 'metadata' in _result.files:
                logger.debug('Finding X design info for version>=v0.5.4')
                md = _result['metadata'].item()
                design = md['YATSM']['design']
                design_str = md['YATSM']['design_matrix']
            else:
                logger.debug('Finding X design info for version<0.5.4')
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
    index = np.where(record['start'] <= date)[0]
    yield index


def get_prediction(index, rec, i_coef, n_i_bands, X, i_bands, raster): 
    """ Get prediction for date of input based on model coeffiecients. 
    Args:
	index (int): 
	rec (int): 
	i_coef (list): 
	n_i_bands(int): 
	X (int): 
	i_bands (int):
	raster (array): TODO  

    Yields:
	raster (array): 
	"""

    _coef = rec['coef'].take(index, axis=0).\
            take(i_coef, axis=1).take(i_bands, axis=2)
    return np.tensordot(_coef, X, axes=(1, 0))


def do_monitor(date, result_location, image_ds, image_ar, cfg, save, outputrast,
		   bands='all', prefix='', ndv=-9999, pattern=_result_record):

    """ Get change prediction for date of input image.

    Args:
        date (int): ordinal date for input (new) image
        result_location (str): Location of the results
        image_ds (gdal.Dataset): Example dataset
        ndv (int, optional): NoDataValue
        pattern (str, optional): filename pattern of saved record results

    Returns:
        np.ndarray: 2D numpy array containing the change probability map for the
            image specified
    """

    #Define parameters. This should be from the parameter file. 
    threshold = cfg['CCDCesque']['threshold']
    test_ind = cfg['CCDCesque']['test_indices']
    cloudthreshold = cfg['CCDCesque']['screening_crit']
    green = cfg['dataset']['green_band']
    swir = cfg['dataset']['swir1_band']
    shadowthreshold = 0 - (cfg['CCDCesque']['screening_crit'])
    mask_band = cfg['dataset']['mask_band']
    mask_values = cfg['dataset']['mask_values']


    # Find results
    records = find_results(result_location, pattern)

    # Find result attributes to extract
    i_bands, _, _, _, design, design_info = find_mon_result_attributes(
        records, bands, None, cfg, prefix=prefix)
    n_bands = len(i_bands)
    n_i_bands = len(i_bands)
    im_Y = image_ds.RasterYSize
    im_X = image_ds.RasterXSize
    raster = np.ones((im_Y, im_X, n_bands), dtype=np.float16) * int(ndv)
    probout = np.ones((im_Y, im_X, 1), dtype=np.float32) * int(ndv)
    breakout = np.ones((im_Y, im_X, 1), dtype=np.float32) * int(ndv)
    consecout = np.ones((im_Y, im_X, 1), dtype=np.float32) * int(ndv)
    linescores_rast = np.ones((im_Y, im_X, n_bands), dtype=np.float16) * int(ndv)

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
    logger.debug('Processing results')

    iter = 1
    for z, rec in iter_records(records, warn_on_empty=True):
       logger.info('Working on record %s' % iter)
       iter+=1	
	#looping over indices
       linescores = np.zeros((1, im_X, n_bands), dtype=np.float32) * int(ndv)
       rmse = np.ones((1, im_X, n_bands), dtype=np.float32) * int(ndv)
       lineconsec = np.zeros((1, im_X, 1), dtype=np.float32) * int(ndv)
       linebreak = np.zeros((1, im_X, 1), dtype=np.float32) * int(ndv)
       lineprob = np.zeros((1, im_X, 1), dtype=np.float32) * int(ndv)

       for index in find_mon_indices(rec, date):
           if index.shape[0] == 0:
                continue

	   #First, get prediction
           raster[rec['py'][index], 
		   rec['px'][index], 
		         :n_i_bands] = get_prediction(index, rec, i_coef, 
						n_i_bands, X, i_bands, raster) 
           #Get x and y location from records
	   _px = rec['px'][index]
	   _py = rec['py'][index]

	   #Calculate RMSE
	   rmse[:, rec['px'][index], :n_i_bands] = rec['rmse'][index][:, i_bands]

	   #Initation array to save break dates
	   linebreak[:, rec['px'][index],0] = rec['break'][index][:,]

	   #calculate change score for each band 
	   try:
	       linescores_rast[rec['py'][index], :, :n_i_bands] = get_mon_scores(raster, image_ar, _px, 
	     	   							         _py, i_bands, rmse, 
           									 outputrast, mask_band, mask_values)
	   #This is a hack TODO. 
	   except:
	       linescores_rast[rec['py'][index], :, :n_i_bands] = get_mon_scores(raster, image_ar, _px, 
	     	   							         _py, i_bands, rmse, 
									         outputrast, mask_band, mask_values)[0]
	       print "excepted" 
           py=np.unique(rec['py'])

	   #If you are actually monitoring, get the probability of change
	   if outputrast == 'Probability':
               #Calculate how many (if any) consective observations have been above threshold
               if not rec['consec'].dtype:
	           lineconsec[:, rec['px'][index],0] = 0
	       else:
	           lineconsec[:, rec['px'][index],0] = rec['consec'][index][:,]

	       out1, out2, out3 = get_mon_prob(lineprob, linescores, 
			                       lineconsec, image_ar, py, rec['py'][index], 
                                               date, linebreak, threshold, test_ind, 
					       cloudthreshold,  green, swir, shadowthreshold, mask_band ) 
	       probout[np.unique(rec['py']),:,0] = out1
	       consecout[np.unique(rec['py']),:,0] = out2
	       breakout[np.unique(rec['py']),:,0] = out3
	       out1 = out2 = out3 = None

       #Save #TODO get rid of all this - this is what it is slowing us down.
       #If the results should be saved, do so. 
       if save:
           out = {}
           for k, v in z.iteritems(): # Get meta data items from z
               out[k] = v
           for q in range(np.shape(probout)[1]): #TODO This whoie part needs to go 
	       indice=np.where(out['record']['px']==q)
               if np.shape(indice)[1] == 0:
	           continue
	       out['record']['consec'][indice] = consecout[np.unique(rec['py']),q,:] 
	       out['record']['break'][indice] = breakout[np.unique(rec['py']),q,:] 
	       out['record']['probability'][indice] = probout[np.unique(rec['py']),q,:] 
           #filename = get_output_name(cfg['dataset'], _py[0])       
           #np.savez(filename, **out)
    return raster, probout, linescores_rast

