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
from yatsm.utils import find_results, iter_records, write_output
from yatsm.regression import design_to_indices, design_coefs
from yatsm.regression.transforms import harm
from yatsm.cli.map import get_prediction 
gdal.AllRegister()
gdal.UseExceptions()

logger = logging.getLogger('yatsm')

result_location = '/projectnb/landsat/users/bullocke/yatsm_newest/yatsm_NRT/sample_data/YATSM/'
pattern = '*npz'


@click.command(short_help='Make map of YATSM output for a given date')
@options.arg_date()
@options.arg_output
@options.opt_rootdir
@options.opt_resultdir
@options.opt_exampleimg
@options.opt_date_format
@options.opt_nodata
@options.opt_format
@click.pass_context


def monitor(ctx, date, output,
        root, result, image, date_frmt, ndv, gdal_frmt):
    """
    Examples:
    > yatsm monitor 2013001 \

    Notes:
        - Any notes?
    """
    try:
        image_ds = gdal.Open(image, gdal.GA_ReadOnly)
    except:
        logger.error('Could not open new image for reading')
        raise

    #Open new mage as array
    image_ar = image_ds.ReadAsArray()

    date = date.toordinal()


    band_names = 'Probability'
    raster = get_mon_prediction(
            date, result, image_ds,
            band,
            ndv=ndv
        )
    print raster

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

    # How many coefficients and bands exist in the results?
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

    #This needs to be the most RECENT model, right now it's the first
    index = np.where((record['end'] <= date))[0]
    yield index


def get_mon_prediction(date, result_location, image_ds, 
		   bands='all', prefix='',ndv=-9999, pattern=_result_record):

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

    logger.debug('Processing results')
    for rec in iter_records(records, warn_on_empty=WARN_ON_EMPTY):
        for index in find_mon_indices(rec, date):
            if index.shape[0] == 0:
                continue

            # Calculate prediction
            _coef = rec['coef'].take(index, axis=0).\
                take(i_coef, axis=1).take(i_bands, axis=2)
            raster[rec['py'][index], rec['px'][index], :n_i_bands] = \
                np.tensordot(_coef, X, axes=(1, 0))

    return raster


def check_mask():
  #Check to see if pixels overlap with mask of monitoring location
    continue 
def update_status():
  #Save new entry on status - is it in the middle of a possible change? 
    continue 
def update_probability():
  #Save the current probability that the pixel is undergoing a chance - this can be sensor specific
    continue 
  
def check_refit():
  #Check if enough data exists to refit the model - may or may not be necessary
    continue 
  
def run_pixel():
  #Run YATSM on pixel if is within the mask (if there is a mask). This may reference yatsm.py, or may need unique functions
    continue 
def determine_prob():
  """Use various weighted inputs to determine the probability of change in near real time. This inputs include:
     -Magnitude of change
     -Type of change (as decided in inputs)
     -View angle for MODIS
     -Consecutive days
     -Cloud and shadow masks (similar to type of change)"""
    continue
