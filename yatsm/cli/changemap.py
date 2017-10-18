""" Command line interface for creating changemaps of YATSM algorithm output
"""
from datetime import datetime as dt
import logging
import os
import sys
import csv
import click
import numpy as np
from osgeo import gdal
import pandas as pd

from yatsm.cli import options
from yatsm.utils import find_results, iter_records, write_output
from yatsm.reader import get_image_attribute
from yatsm.cli.map import find_result_attributes
from yatsm.regression import design_to_indices, design_coefs
gdal.AllRegister()
gdal.UseExceptions()

logger = logging.getLogger('yatsm')

# Filters for results
_result_record = 'yatsm_*'

WARN_ON_EMPTY = False


@click.command(
    short_help='Map change found by YATSM algorithm over time period')
@click.argument('map_type', metavar='<map_type>',
                type=click.Choice(['numstatus','first', 'year', 'last', 'num', 
				     'status', 'class','lenstatus','omis', 'commis', 
				     'numlc', 'lencommis', 'forestchange', 'forest', 
			 	     'length', 'greatest','disturbance', 'aftercoef']))
@options.arg_date(var='start_date', metavar='<start_date>')
@options.arg_date(var='end_date', metavar='<end_date>')
@options.arg_output
@options.opt_rootdir
@options.opt_resultdir
@options.opt_exampleimg
@options.opt_date_format
@options.opt_nodata
@options.opt_format
@click.option('--coef', '-c', multiple=True, metavar='<coef>',
              type=click.Choice(design_coefs), default=('all', ),
              help='Coefficients to export for coefficient maps')
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
@click.option('--statusfmt', 'statusfmt', metavar='<statusfmt>',
              default='NoPP', show_default=True, help='Which postprocess status')
@click.option('--monitor', metavar='<monitor>', help='Date for first forest change')
@click.option('--forest', type=int, metavar='<forest>', help='Forest label')
@click.option('--distclass', type=int, metavar='<distclass>', help='Disturbance label')
@click.pass_context
def changemap(ctx, map_type, start_date, end_date, output,
              root, result, image, date_frmt, ndv, gdal_frmt, out_date_frmt, coef, band,
              warn_on_empty, magnitude, detect,statusfmt, monitor, forest, distclass):
    """
    Examples: TODO
    """
    gdal_frmt = str(gdal_frmt)  # GDAL GetDriverByName doesn't work on Unicode

    frmt = '%Y%j'
    start_txt, end_txt = start_date.strftime(frmt), end_date.strftime(frmt)
    start_date, end_date = start_date.toordinal(), end_date.toordinal()

    if statusfmt:
	if statusfmt == 'NoPP':
	    stat_label = 1
	elif statusfmt == 'Omis':
	    stat_label = 2
	elif statusfmt == 'Commis':
	    stat_label = 3
	elif statusfmt == 'Both':
	    stat_label = 4

    try:
        image_ds = gdal.Open(image, gdal.GA_ReadOnly)
    except:
        logger.error('Could not open example image for reading')
        raise

    if map_type in ('first', 'last'):
        changemap, magnitudemap, magnitude_indices = get_datechangemap(
            start_date, end_date, result, image_ds, detect,
            first=map_type == 'first', out_format=out_date_frmt,
            magnitude=magnitude,
            ndv=ndv, pattern=_result_record
        )

        band_names = ['ChangeDate_s{s}-e{e}'.format(s=start_txt, e=end_txt)]
        write_output(changemap, output, image_ds, gdal_frmt, ndv,
                     band_names=band_names)

        if magnitudemap is not None:
            band_names = (['Magnitude Index {}'.format(i) for
                           i in magnitude_indices])
            name, ext = os.path.splitext(output)
            output = name + '_mag' + ext
            write_output(magnitudemap, output, image_ds, gdal_frmt, ndv,
                         band_names=band_names)
	    sys.exit()

    elif map_type == 'disturbance':
        changemap  = get_distmap(
            start_date, end_date, result, image_ds, detect, distclass,
            forest, first=False, out_format=out_date_frmt,
            ndv=ndv, pattern=_result_record
        )

        band_names = ['ChangeDate_s{s}-e{e}'.format(s=start_txt, e=end_txt)]
        write_output(changemap, output, image_ds, gdal_frmt, ndv,
                     band_names=band_names)

	sys.exit()

    elif map_type == 'omis':
	map1, map2 = get_omis(start_date, end_date, result, image, image_ds,stat_label, 
	    ndv=ndv, pattern=_result_record)
	name, ext = os.path.splitext(output)
	output1 = name + '_class1' + ext
	output2 = name + '_class2' + ext
	band_names1 = ['Class1']
	band_names2 = ['Class2']
        write_output(map1, output1, image_ds, gdal_frmt, ndv,
                     band_names=band_names1)
        write_output(map2, output2, image_ds, gdal_frmt, ndv,
                     band_names=band_names2)
	sys.exit()

    elif map_type == 'aftercoef':
        changemap, band_names = get_aftercoefmap(
            start_date, end_date, result, image_ds, detect, band, coef,
            first=True, out_format=out_date_frmt,
            magnitude=magnitude,
            ndv=ndv, pattern=_result_record
        )
        write_output(changemap, output, image_ds, gdal_frmt, ndv,
                     band_names=band_names)
	sys.exit()
    elif map_type == 'commis':
	map1, map2 = get_commis(start_date, end_date, result, image, image_ds, 
	    ndv=ndv, pattern=_result_record)
	name, ext = os.path.splitext(output)
	output1 = name + '_class1' + ext
	output2 = name + '_class2' + ext
	band_names1 = ['Class1']
	band_names2 = ['Class2']
        write_output(map1, output1, image_ds, gdal_frmt, ndv,
                     band_names=band_names1)
        write_output(map2, output2, image_ds, gdal_frmt, ndv,
                     band_names=band_names2)
	sys.exit()

    elif map_type == 'forest':
	changemap = get_forest(start_date, end_date, result, image, image_ds,forest, 
	    ndv=ndv, pattern=_result_record)

    elif map_type == 'num':
        changemap = get_numchangemap(
            start_date, end_date, result, image_ds,
            ndv=ndv, pattern=_result_record
        )
    elif map_type == 'numlc':
        changemap = get_numlcchangemap(
            output, start_date, end_date, result, image_ds,
            ndv=ndv, pattern=_result_record
        )
	sys.exit()
    elif map_type == 'year':
        changemap = get_yearchangemap(
            stat_label, output, start_date, end_date, result, image_ds,
            ndv=ndv, pattern=_result_record
        )
	sys.exit()
    elif map_type == 'numstatus': #Note: HACK!!!
        changemap = get_numomissionmap(output, 
            start_date, end_date, result, image_ds,
            ndv=ndv, pattern=_result_record
        )
    elif map_type == 'status': 
        changemap  = get_changestatus(
            start_date, end_date, result, image_ds, stat_label,
            first=map_type == 'first',
            ndv=ndv, pattern=_result_record
        )
    elif map_type == 'lenstatus': 
        changemap  = get_lenstatus(
            output, start_date, end_date, result, image_ds, stat_label,
            first=map_type == 'first',
            ndv=ndv, pattern=_result_record
        )
    elif map_type == 'lencommis': 
        changemap  = get_lencommis(
            output, start_date, end_date, result, image, image_ds,
            ndv=ndv, pattern=_result_record
        )
    elif map_type == 'droughtclass': 
        changemap  = get_droughtclass(
            output, start_date, end_date, result, image, image_ds, stat_label,
            ndv=ndv, pattern=_result_record
        )
    elif map_type == 'length':
        changemap = get_length(
            start_date, end_date, result, image_ds, detect,
            out_format=out_date_frmt,
            ndv=ndv, pattern=_result_record
        )

    elif map_type == 'greatest':
        changemap = get_greatest(
            start_date, end_date, result, image_ds, bands='all',
            out_format=out_date_frmt,
            magnitude=magnitude,
            ndv=ndv, pattern=_result_record
        )
    band_names = [map_type]
    write_output(changemap, output, image_ds, gdal_frmt, ndv,
                 band_names=band_names)

    image_ds = None


# UTILITIES
def get_magnitude_indices(results):
    """ Finds indices of result containing magnitude of change information

    Args:
      results (iterable): list of result files to check within

    Returns:
      np.ndarray: indices containing magnitude change information from the
        tested indices

    """
    for result in results:
        try:
            rec = np.load(result)
        except (ValueError, AssertionError):
            logger.warning('Error reading {f}. May be corrupted'.format(
                f=result))
            continue

        # First search for record of `test_indices`
        if 'test_indices' in rec.files:
            logger.debug('Using `test_indices` information for magnitude')
            return rec['test_indices']

        # Fall back to using non-zero elements of 'record' record array
        rec_array = rec['record']
        if rec_array.dtype.names is None:
            # Empty record -- skip
            continue

        if 'magnitude' not in rec_array.dtype.names:
            logger.error('Cannot map magnitude of change')
            logger.error('Magnitude information not present in file {f} -- '
                'has it been calculated?'.format(f=result))
            logger.error('Version of result file: {v}'.format(
                v=rec['version'] if 'version' in rec.files else 'Unknown'))
            raise click.Abort()

        changed = np.where(rec_array['break'] != 0)[0]
        if changed.size == 0:
            continue

        logger.debug('Using non-zero elements of "magnitude" field in '
                     'changed records for magnitude indices')
        return np.nonzero(np.any(rec_array[changed]['magnitude'] != 0))[0]


# MAPPING FUNCTIONS
def get_datechangemap(start, end, result_location, image_ds, detect,
                      first=False,
                      out_format='%Y%j',
                      magnitude=False,
                      ndv=-9999, pattern=_result_record):
    """ Output raster with changemap

    Args:
      start (int): Ordinal date for start of map records
      end (int): Ordinal date for end of map records
      result_location (str): Location of results
      image_ds (gdal.Dataset): Example dataset
      first (bool): Use first change instead of last
      out_format (str, optional): Output date format
      magnitude (bool, optional): output magnitude of each change?
      ndv (int, optional): NoDataValue
      pattern (str, optional): filename pattern of saved record results

    Returns:
      tuple: A 2D np.ndarray array containing the changes between the
        start and end date. Also includes, if specified, a 3D np.ndarray of
        the magnitude of each change plus the indices of these magnitudes

    """
    # Find results
    records = find_results(result_location, pattern)

    logger.debug('Allocating memory...')
    datemap = np.ones((image_ds.RasterYSize, image_ds.RasterXSize),
                      dtype=np.int32) * int(ndv)
    # Determine what magnitude information to output if requested
    if magnitude:
        magnitude_indices = get_magnitude_indices(records)
        magnitudemap = np.ones((image_ds.RasterYSize, image_ds.RasterXSize,
                                magnitude_indices.size),
                               dtype=np.float32) * float(ndv)

    logger.debug('Processing results')
    for a, rec in iter_records(records):
	try:
            index = np.where((rec['break'] >= start) &
                         (rec['break'] <= end))[0]
	except:
	    continue
        if first:
            _, _index = np.unique(rec['px'][index], return_index=True)
            index = index[_index]
        if detect:
            if index.shape[0] != 0:
                if out_format != 'ordinal':
                    dates = np.array([int(dt.fromordinal(_d).strftime(out_format))
                                     for _d in rec['detect'][index]])
                    datemap[rec['py'][index], rec['px'][index]] = dates
                else:
                    datemap[rec['py'][index], rec['px'][index]] = \
                        rec['detect'][index]
                if magnitude:
                    magnitudemap[rec['py'][index], rec['px'][index], :] = \
                        rec[index]['magnitude'][:, magnitude_indices]
	else:
            if index.shape[0] != 0:
                if out_format != 'ordinal':
                    dates = np.array([int(dt.fromordinal(_d).strftime(out_format))
                                     for _d in rec['break'][index]])
                    datemap[rec['py'][index], rec['px'][index]] = dates
                else:
                    datemap[rec['py'][index], rec['px'][index]] = \
                        rec['break'][index]
                if magnitude:
                    magnitudemap[rec['py'][index], rec['px'][index], :] = \
                        rec[index]['magnitude'][:, magnitude_indices]
	

    if magnitude:
        return datemap, magnitudemap, magnitude_indices
    else:
        return datemap, None, None

def get_numchangemap(start, end, result_location, image_ds,
                     ndv=-9999, pattern=_result_record):
    """ Output raster with changemap

    Args:
      start (int): Ordinal date for start of map records
      end (int): Ordinal date for end of map records
      result_location (str): Location of results
      image_ds (gdal.Dataset): Example dataset
      ndv (int, optional): NoDataValue
      pattern (str, optional): filename pattern of saved record results

    Returns:
      np.ndarray: 2D numpy array containing the number of changes
        between the start and end date; list containing band names

    """
    # Find results
    records = find_results(result_location, pattern)

    logger.debug('Allocating memory...')
    raster = np.ones((image_ds.RasterYSize, image_ds.RasterXSize),
                     dtype=np.int32) * int(ndv)

    logger.debug('Processing results')
    for a, rec in iter_records(records):
        # X location of each changed model
        px_changed = rec['px'][np.where((rec['break'] >= start) & (rec['break'] <= end))]
        # Count occurrences of changed pixel locations
        bincount = np.bincount(px_changed)
        # How many changes for unique values of px_changed?
        n_change = bincount[np.nonzero(bincount)[0]]

        # Add in the values
        px = np.unique(px_changed)
        py = rec['py'][np.in1d(px, rec['px'])]
        raster[py, px] = n_change

    return raster


def get_yearchangemap(fmt, output,start, end, result_location, image_ds,
                     ndv=-9999, pattern=_result_record):
    """ Output raster with changemap

    Args:
      start (int): Ordinal date for start of map records
      end (int): Ordinal date for end of map records
      result_location (str): Location of results
      image_ds (gdal.Dataset): Example dataset
      ndv (int, optional): NoDataValue
      pattern (str, optional): filename pattern of saved record results

    Returns:
      np.ndarray: 2D numpy array containing the number of lc changes
        between the start and end date; list containing band names

    """
    # Find results
    records = find_results(result_location, pattern)

    logger.debug('Allocating memory...')
    raster = np.ones((image_ds.RasterYSize, image_ds.RasterXSize),
                     dtype=np.int32) * int(ndv)

    logger.debug('Processing results')
    out_count = 0
    for a, rec in iter_records(records):
        # X location of each changed model
        px_changed1 = np.logical_and(
			       np.logical_and(
 			          rec['break'] >= start, rec['break'] <= end),
				  rec['status'] == fmt)
	out_count += np.sum(px_changed1)
    np.save(output, out_count)
    sys.exit()

def get_length(start, end, result_location, image_ds, detect,
                      out_format='%Y%j',
                      ndv=-9999, pattern=_result_record):
    """ Output raster with length of model that has change

    Args:
      start (int): Ordinal date for start of map records
      end (int): Ordinal date for end of map records
      result_location (str): Location of results
      image_ds (gdal.Dataset): Example dataset
      first (bool): Use first change instead of last
      out_format (str, optional): Output date format
      ndv (int, optional): NoDataValue
      pattern (str, optional): filename pattern of saved record results

    Returns:
      tuple: A 2D np.ndarray with length of model that has changes between
	start and end date. 

    """
    # Find results
    records = find_results(result_location, pattern)

    logger.debug('Allocating memory...')
    datemap = np.ones((image_ds.RasterYSize, image_ds.RasterXSize),
                      dtype=np.int32) * int(ndv)
    # Determine what magnitude information to output if requested

    logger.debug('Processing results')
    for a, rec in iter_records(records):
        index = np.where((rec['break'] >= start) &
                         (rec['break'] <= end))[0]

        datemap[rec['py'][index], rec['px'][index]] = \
            rec['break'][index] - rec['start'][index] 

    else:
        return datemap

def get_greatest(start, end, result_location, image_ds, bands='all',
                      out_format='%Y%j',
                      magnitude=False,
                      ndv=-9999, pattern=_result_record):
    """ Output raster with changemap for greatest change based on magnitude of 
	test indices specified with 'band' option

    Args:
      start (int): Ordinal date for start of map records
      end (int): Ordinal date for end of map records
      result_location (str): Location of results
      image_ds (gdal.Dataset): Example dataset
      band (list): Bands to base magnitude on
      out_format (str, optional): Output date format
      ndv (int, optional): NoDataValue
      pattern (str, optional): filename pattern of saved record results

    Returns:
      tuple: A 2D np.ndarray array containing the changes between the
        start and end date. Also includes, if specified, a 3D np.ndarray of
        the magnitude of each change plus the indices of these magnitudes

    """
    # Find results
    records = find_results(result_location, pattern)

    logger.debug('Allocating memory...')
    datemap = np.ones((image_ds.RasterYSize, image_ds.RasterXSize),
                      dtype=np.int32) * int(ndv)

    #TODO
    bands = 4

    logger.debug('Processing results')
    for a, rec in iter_records(records):
        index = np.where((rec['break'] >= start) &
                         (rec['break'] <= end) & (rec['magnitude'][:,bands] > 0))[0]

	#This is first:
        uniq, _index = np.unique(rec['px'][index], return_index=True)


	#This will output the max magnitude
        dfs = pd.DataFrame({'A' : pd.Categorical(rec['px'][index], categories=uniq, ordered=True), 'B' : np.abs(rec['magnitude'][index][:,bands]), 'C' : index})
	new_ind = dfs.groupby('A').B.agg('idxmax')

	#Alternatively, do agg(max) to just get max magnitude
        index = np.array(dfs.C[new_ind])

	length = True 
        if index.shape[0] != 0:
	    if length:
                datemap[rec['py'][index], rec['px'][index]] = \
                    end - rec['break'][index]
	    else:
                if out_format != 'ordinal':
                    dates = np.array([int(dt.fromordinal(_d).strftime(out_format))
                                     for _d in rec['break'][index]])
                    datemap[rec['py'][index], rec['px'][index]] = dates
                else:
                    datemap[rec['py'][index], rec['px'][index]] = \
                        rec['break'][index]
	
    return datemap

def get_distmap(start, end, result_location, image_ds, detect, disturbance,
		      or_class,
                      first=False,
                      out_format='%Y%j',
                      ndv=-9999, pattern=_result_record):
    """ Output raster with date of disturbance class

    Args:
      start (int): Ordinal date for start of map records
      end (int): Ordinal date for end of map records
      result_location (str): Location of results
      image_ds (gdal.Dataset): Example dataset
      disturbance (int): disturbance class label
      or_class (int): original class ebfore disturbance
      first (bool): Use first change instead of last
      out_format (str, optional): Output date format
      magnitude (bool, optional): output magnitude of each change?
      ndv (int, optional): NoDataValue
      pattern (str, optional): filename pattern of saved record results

    Returns:
      tuple: A 2D np.ndarray array containing the changes between the
        start and end date. Also includes, if specified, a 3D np.ndarray of
        the magnitude of each change plus the indices of these magnitudes

    """
    # Find results
    records = find_results(result_location, pattern)

    logger.debug('Allocating memory...')
    datemap = np.ones((image_ds.RasterYSize, image_ds.RasterXSize),
                      dtype=np.int32) * int(ndv)
    # Determine what magnitude information to output if requested

    logger.debug('Processing results')
    for a, rec in iter_records(records):
	try:
	    index = np.where((rec['dclass'] == disturbance) & (rec['break'] > 0)
			     & (rec['class'] == or_class))[0]
	except:
	    continue
        if first:
            _, _index = np.unique(rec['px'][index], return_index=True)
            index = index[_index]
        if index.shape[0] != 0:
            if out_format != 'ordinal':
                dates = np.array([int(dt.fromordinal(_d).strftime(out_format))
                                 for _d in rec['break'][index]])
                datemap[rec['py'][index], rec['px'][index]] = dates
            else:
                datemap[rec['py'][index], rec['px'][index]] = \
                    rec['break'][index]

    return datemap

def get_aftercoefmap(start, end, result_location, image_ds, detect,
		      bands, coefs,
                      first=False,
                      out_format='%Y%j',
                      magnitude=False,
                      ndv=-9999, pattern=_result_record):
    """ Output raster with coef after a change

    Args:
      start (int): Ordinal date for start of map records
      end (int): Ordinal date for end of map records
      result_location (str): Location of results
      image_ds (gdal.Dataset): Example dataset
      first (bool): Use first change instead of last
      out_format (str, optional): Output date format
      magnitude (bool, optional): output magnitude of each change?
      ndv (int, optional): NoDataValue
      pattern (str, optional): filename pattern of saved record results

    Returns:

    """


    # Find results
    records = find_results(result_location, pattern)

    prefix = ''
    # Find result attributes to extract
    bands = 'all'
    i_bands, i_coefs, use_rmse, coef_names, _, _ = find_result_attributes(
        records, bands, coefs, prefix=prefix)

    # Process amplitude transform for seasonality coefficients
    amplitude = False #TODO
    if amplitude:
        harm_coefs = []
        for i, (c, n) in enumerate(zip(i_coefs, coef_names)):
            if re.match(r'harm\(x, [0-9]+\)\[0]', n):
                harm_coefs.append(c)
                coef_names[i] = re.sub(r'harm(.*)\[.*', r'amplitude\1', n)
        # Remove sin term from each harmonic pair
        i_coefs = [c for c in i_coefs if c - 1 not in harm_coefs]
        coef_names = [n for n in coef_names if 'harm' not in n]

    n_bands = len(i_bands)
    n_coefs = len(i_coefs)
    n_rmse = n_bands if use_rmse else 0

    # Setup output band names
    band_names = []
    for _c in coef_names:
        for _b in i_bands:
            band_names.append('B' + str(_b + 1) + '_' + _c.replace(' ', ''))
    if use_rmse is True:
        for _b in i_bands:
            band_names.append('B' + str(_b + 1) + '_RMSE')
    n_qa = 0
    n_out_bands = n_bands * n_coefs + n_rmse + n_qa

    _coef = prefix + 'coef' if prefix else 'coef'
    _rmse = prefix + 'rmse' if prefix else 'rmse'

    logger.debug('Allocating memory...')
    raster = np.ones((image_ds.RasterYSize, image_ds.RasterXSize, n_out_bands),
                     dtype=np.float32) * ndv

    logger.debug('Processing results')
    for rec in iter_records(records, warn_on_empty=WARN_ON_EMPTY):
	rec=rec[1]
	try:
            indices = np.where((rec['break'] >= start) &
                         (rec['break'] <= end))[0]
	except:
	    continue
	first = True #TODO
        if first:
            _, _indices = np.unique(rec['px'][indices], return_index=True)
            indices = indices[_indices]
        for index in indices:
	    try:
		if rec['px'][index] != rec['px'][index+1]:
		    continue
	    except:
		continue
	    index += 1
            if n_coefs > 0:
                # Normalize intercept to mid-point in time segment
                rec[_coef][index, 0, :] += (
                    (rec['start'][index] + rec['end'][index])
                        / 2.0)[np.newaxis] * \
                    rec[_coef][index, 1, :]

                # If we want amplitude, calculate it
                if amplitude:
                    for harm_coef in harm_coefs:
                        rec[_coef][index, harm_coef, :] = np.linalg.norm(
                            rec[_coef][index, harm_coef:harm_coef + 2, :],
                            axis=1)

                # Extract coefficients
                raster[rec['py'][index],
                       rec['px'][index], :n_coefs * n_bands] =\
                    np.reshape(rec[_coef][index][i_coefs, :][:, i_bands],
                               (index.size, n_coefs * n_bands))

            if use_rmse:
                raster[rec['py'][index], rec['px'][index],
                       n_coefs * n_bands:n_out_bands - n_qa] =\
                    rec[_rmse][index][:, i_bands]
    return raster, band_names

