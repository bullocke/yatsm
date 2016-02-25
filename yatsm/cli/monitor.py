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

from yatsm.cli import options
from yatsm.config_parser import parse_config_file
from yatsm.utils import get_output_name, find_results, iter_records, write_output
from yatsm.regression import design_to_indices, design_coefs
from yatsm.regression.transforms import harm
from yatsm.cli.map import get_prediction 
gdal.AllRegister()
gdal.UseExceptions()

logger = logging.getLogger('yatsm')

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
@click.option('--save', is_flag=True,
              help='Update YATSM .npz files for monitoring')
@click.option('--band', '-b', multiple=True, metavar='<band>', type=int,
              help='Bands to export for coefficient/prediction maps')
@click.option('--outputrast', '-o', metavar='<outputrast>', type=click.Choice(['ChangeScore', 'Probability']),
              help='Output Probability or ChangeScore?')
@click.pass_context


def monitor(ctx, config, date, output,
        root, result, image, date_frmt, ndv, gdal_frmt, band, outputrast, save):
    """
    Examples:
    > yatsm monitor -r YATSM -b 1 -b 2 -b 3 -b 4 -b 5 -b 6 -b 7 -b 8 -b 9 -i 2012/MOD_A2012001.gtif params.yaml 2012-01-01 2012001_Prob.tif \

    Notes:
        - Any notes?
    """

    try:
        image_ds = gdal.Open(image, gdal.GA_ReadOnly)
    except:
        logger.error('Could not open new image for reading')
        raise


    dataset_config = parse_config_file(config)

    #Open new mage as array
    image_ar = image_ds.ReadAsArray()
    date = date.toordinal()
    band_names = 'Probability'

    #get prediction for image
    raster, probability, linescores_rast = get_mon_predictions(
            date, result, image_ds, image_ar, dataset_config, save, outputrast,
            band,
            ndv=ndv
        )
    _outputrast = probability

    #Determine what to output
    if outputrast == "ChangeScore":
	_outputrast = linescores_rast*10
    write_mon_output(_outputrast, output, image_ds,
                 gdal_frmt, ndv)


def do_monitor(ctx, config, output,
        root, result, image, date_frmt, ndv, gdal_frmt, band, outputrast, save, num, lines):
    """
    Main function. Should be broken up by line. 
    """
    dataset_config = parse_config_file(config)


def get_dates():
    """
    Return array containing dates to be predicted and monitored
    """

def check_date():
    """
    Check to make sure the newest images are later than the previous ones. Return false if previous end dates are past images. 
    """

def get_x():
    """
    Organize input images and return x and y for images to monitor
    """

def get_y():
    """
    Read in a line and return image values in array form.
    """

def check_date():
    """
    Checks if date input is before or after end of previous run
    """
    return date_status

def get_mon_prob(linescores, lineconsec, image_ar, py, px, date, linebreak, threshold, test_ind, cloudthreshold, blue, swir, shadowthreshow):
  #  mag = np.linalg.norm(np.abs(linescores), axis=2)
    cloudprob=np.ones((1,np.shape(linescores)[1]))
    shadowprob=np.ones((1,np.shape(linescores)[1]))
    _px=px[0]
    #loop over pixels
    for i in range(np.shape(linescores)[1]):
	if lineprob[0,i,0] >= 100:
	    continue
	#First do the cloud test:
	if linescores[0,i,blue] > cloudthreshold:
	    cloudprob[0,i]=0
	if image_ar[9,py,i] < 1:
	    continue 
	#Now do the shadow test:
	if linescores[0,i,swir] < shadowthreshold:
	    shadowprob[0,i]=0
	#First check if it passes NDVI threshold
	if linescores[0,i,test_ind] < threshold:
	    lineprob[0,i,0] = lineprob[0,i,0] + (((cloudprob[0,i] * shadowprob[0,i]) * (lineconsec[0,i,0])) * 10)  
	#One means last one was change
	    lineconsec[0,i,0] += 1
	    if lineprob[0,i,0] >= 100:
	        linebreak[0,i,0] = date
	#But if it's not change, and last one was...set it to -1 to not start over. 
    return lineprob[0,:,0], lineconsec[0,:,0], linebreak[0,:,0]

def get_mon_scores(raster, image_ar, _px, _py, i_bands, rmse, temp, outputrast):

    """Return change scores based on model RMSE. Also returns VZA weight.

    Args:
	raster (arr): predicted image for date specified.
	image_ar (arr): array of input image. 
	_px: X location of pixel
	_py: Y location of pixel
	i_bands (list): List of bands to use in change detection
	rmse (arr): rmse for model being used for prediction.
	temp: TODO: Get rid of this
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
	    #Do not continue if it is masked as cloud. Turn this off for testing purposes. 
#	    if image_ar[9,py,px] == 0:
 #               continue
            im_val=image_ar[_band,py,px]
            scores[px,_band]=((im_val.astype(float) - raster[py,px,_band])/rmse[0,px,_band])
    return scores


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
    #TODO MOST URGENT FIX THIS!!!
    #index = np.where((record['end'] <= date))[0]
    #import pdb; pdb.set_trace()
    #TODO get last date. see if date input is before or after end. 
    #index = np.where((record['start'] >= 0) & (record['break'] == 0))[0]
    index = np.where((record['start'] >= 0))[0]
    #_, _index = np.unique(record['px'][index], return_index=True)
    #index = index[_index]
    yield index





def get_mon_predictions(date, result_location, image_ds, image_ar, dataset_config, save, outputrast,
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
        records, bands, None, dataset_config, prefix=prefix)
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
                     dtype=np.float16) * int(ndv)
    probout = np.ones((image_ds.RasterYSize, image_ds.RasterXSize, 1),
                     dtype=np.float32) * int(ndv)
    breakout = np.ones((image_ds.RasterYSize, image_ds.RasterXSize, 1),
                     dtype=np.float32) * int(ndv)
    consecout = np.ones((image_ds.RasterYSize, image_ds.RasterXSize, 1),
                     dtype=np.float32) * int(ndv)
    linescores_rast = np.ones((image_ds.RasterYSize, image_ds.RasterXSize, n_bands),
		     dtype=np.float16) * int(ndv)
    logger.debug('Processing results')

    temp=0
    csv=[]

    #Define parameters. This should be from the parameter file. 
    threshold = dataset_config['CCDCesque']['threshold']
    test_ind = dataset_config['CCDCesque']['test_indices']
    cloudthreshold = dataset_config['CCDCesque']['screening_crit']
    green = dataset_config['dataset']['green_band']
    swir = dataset_config['dataset']['swir1_band']
    shadowthreshold = 0 - (dataset_config['CCDCesque']['screening_crit'])

    for z, rec in iter_records(records, warn_on_empty=True):
	#looping over indices
       temp+=1
       linescores = np.zeros((1, image_ds.RasterXSize, n_bands),
                     dtype=np.float32) * int(ndv)
       
       rmse = np.ones((1, image_ds.RasterXSize, n_bands),
                     dtype=np.float32) * int(ndv)

       lineconsec = np.zeros((1, image_ds.RasterXSize, 1),
                     dtype=np.float32) * int(ndv)

       linebreak = np.zeros((1, image_ds.RasterXSize, 1),
                     dtype=np.float32) * int(ndv)
       lineprob = np.zeros((1, image_ds.RasterXSize, 1),
                     dtype=np.float32) * int(ndv)

       for index in find_mon_indices(rec, date):
           if index.shape[0] == 0:
                continue

	   #Test is 

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


	   #Initation array to save break dates
	   linebreak[:, rec['px'][index],0] = rec['break'][index][:,]

	   #calculate change score for each band 
	   try:
	       linescores_rast[rec['py'][index], :, :n_i_bands] = get_mon_scores(raster, image_ar, _px, 
	     	   							         _py, i_bands, rmse, temp, 
									         outputrast)
	   except:
	       import pdb; pdb.set_trace()
	       linescores_rast[rec['py'][index], :, :n_i_bands] = get_mon_scores(raster, image_ar, _px, 
	     	   							         _py, i_bands, rmse, temp, 
									         outputrast)[0]
	       print "excepted" 
           py=np.unique(rec['py'])

	   #For testing purposes, save the scores
	   testname='%s' % rec['py'][index][0]

	   #If you are actually monitoring, get the probability of change
	   if outputrast == 'Probability':
               #Calculate how many (if any) consective observations have been above threshold
               if not rec['consect'].dtype:
	           lineconsec[:, rec['px'][index],0] = 0
	       else:
	           lineconsec[:, rec['px'][index],0] = rec['consec'][index][:,]
               probout[np.unique(rec['py']),:,0], 
	       consecout[np.unique(rec['py']),:,0], 
	       breakout[np.unique(rec['py']),:,0] = get_mon_prob(linescores, 
			                                         lineconsec, image_ar, py, rec['py'][index], 
                                        		         date, linebreak, threshold, test_ind, 
								 cloudthreshold,  blue, swir, shadowthreshow) 

        #rec = nprfn.merge_arrays((rec, classified), flatten=True)

       if save:
           out = {}
           for k, v in z.iteritems():
               out[k] = v
           for q in range(np.shape(probout)[1]):
               fulltime=end-beginning
               print "%s took %s time to run" % (py, fulltime)	
               if np.shape(indice)[1] == 0:
	           continue
	       out['record']['consec'][indice] = consecout[np.unique(rec['py']),q,:] 
	       out['record']['break'][indice] = breakout[np.unique(rec['py']),q,:] 
	       out['record']['probability'][indice] = probout[np.unique(rec['py']),q,:] 
           filename = get_output_name(dataset_config['dataset'], _py[0])       
           np.savez(filename, **out)
    return raster, probout, linescores_rast

