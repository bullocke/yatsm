""" Command line interface for post-processing based on previously run results """
import logging
import os
import time

import click
import numpy as np
import patsy
import scipy.io as sio

from yatsm.cache import test_cache
from yatsm.cli import options
from yatsm.config_parser import parse_config_file
import yatsm._cyprep as cyprep
from yatsm.errors import TSLengthException
from yatsm.utils import (get_matlab_name, distribute_jobs, get_output_name, get_image_IDs,
                         csvfile_to_dataframe)
from yatsm.reader import get_image_attribute, read_line
from yatsm.regression.transforms import harm
from yatsm.algorithms import ccdc, postprocess
try:
    import yatsm.phenology as pheno
except ImportError:
    pheno = None
from yatsm.version import __version__

logger = logging.getLogger('yatsm')
logger_algo = logging.getLogger('yatsm_algo')


@click.command(short_help='Post-process YATSM/CCDC on an entire image line by line')
@options.arg_config_file
@options.arg_job_number
@options.arg_total_jobs
@click.option('--verbose-yatsm', is_flag=True,
              help='Show verbose debugging messages in YATSM algorithm')
@click.pass_context
def postprocess_results(ctx, config, job_number, total_jobs,
         verbose_yatsm):
    if verbose_yatsm:
        logger_algo.setLevel(logging.DEBUG)

    # Parse config
    cfg = parse_config_file(config)

    # Make sure output directory exists and is writable
    output_dir = cfg['dataset']['output']
    try:
        os.makedirs(output_dir)
    except OSError as e:
        # File exists
        if e.errno == 17:
            pass
        elif e.errno == 13:
            raise click.Abort('Cannot create output directory %s' % output_dir)

    if not os.access(output_dir, os.W_OK):
        raise click.Abort('Cannot write to output directory %s' % output_dir)

    # Test existence of cache directory
    read_cache, write_cache = test_cache(cfg['dataset'])

    logger.info('Job {i} of {n} - using config file {f}'.format(i=job_number,
                                                                n=total_jobs,
                                                                f=config))
    df = csvfile_to_dataframe(cfg['dataset']['input_file'],
                              cfg['dataset']['date_format'])
    df['image_ID'] = get_image_IDs(df['filename'])

    # Get attributes of one of the images
    nrow, ncol, nband, dtype = get_image_attribute(df['filename'][0])

    # Calculate the lines this job ID works on
    job_lines = distribute_jobs(job_number, total_jobs, nrow)
    logger.debug('Responsible for lines: {l}'.format(l=job_lines))

    # Calculate X feature input
    dates = np.asarray(df['date'])
    kws = {'x': dates}
    kws.update(df.to_dict())
    X = patsy.dmatrix(cfg['YATSM']['design_matrix'], kws)

    # Form YATSM class arguments
    fit_indices = np.arange(cfg['dataset']['n_bands'])
    if cfg['dataset']['mask_band'] is not None:
        fit_indices = fit_indices[:-1]

    md = cfg[cfg['YATSM']['algorithm']].copy()
    md.update({
        'algorithm': cfg['YATSM']['algorithm'],
        'design': cfg['YATSM']['design_matrix'],
        'design_matrix': X.design_info.column_name_indexes,
        'prediction': cfg['YATSM']['prediction']
    })

    # Create output metadata to save
    for line in job_lines:
        out = get_output_name(cfg['dataset'], line)
        mat_in = get_matlab_name(cfg['dataset'], line + 1)  

	if not os.path.isfile(mat_in):
	    continue #TODO warn the user
        

        #Read in matlab result. This will have to be scipy read
        rec_mat = sio.loadmat(mat_in)['rec_cg']
        if len(rec_mat[0]['t_start'][0]) == 0:
	    continue #is there a better way to check this? TODO

        Y = read_line(line, df['filename'], df['image_ID'], cfg['dataset'],
                      ncol, nband, dtype,
                      read_cache=read_cache, write_cache=write_cache,
                      validate_cache=False)
        output = []
        for col in np.arange(Y.shape[-1]):
            _Y = Y.take(col, axis=2)
            # Mask
            idx_mask = cfg['dataset']['mask_band'] - 1

            valid = cyprep.get_valid_mask(
                _Y,
                cfg['dataset']['min_values'],
                cfg['dataset']['max_values']).astype(bool)

            valid *= np.in1d(_Y.take(idx_mask, axis=0),
                                     cfg['dataset']['mask_values'],
                                     invert=True).astype(np.bool)

            _Y = np.delete(_Y, idx_mask, axis=0)[:, valid]
            _X = X[valid, :]
            _dates = dates[valid]
            cls = cfg['YATSM']['algorithm_cls']
            yatsm = cls(lm=cfg['YATSM']['prediction_object'],
                        **cfg[cfg['YATSM']['algorithm']])
            yatsm.px = col
            yatsm.py = line
	    yatsm.X = _X
	    yatsm.Y = _Y
	    yatsm.dates = _dates
            # Maybe screen something here #TODO

	    #TODO: Convert format
            yatsm.record = yatsm.convert_matlab(yatsm, rec_mat, col, line)


            if yatsm.record is None or len(yatsm.record) == 0:
                continue

            # Postprocess
            if cfg['YATSM']['commission_alpha']:
                yatsm.record = postprocess.commission_test(
                    yatsm, cfg['YATSM']['commission_alpha'])

	    #Check once if it's in config file, then if it is desired
	    if 'omission_alpha' in cfg['YATSM']:
		if cfg['YATSM']['omit_any']: 
		    omit_decision = 'ANY'
		else:
		    omit_decision = 'ALL' 
		if cfg['YATSM']['omission_alpha']:
                    yatsm.record = postprocess.omission_test(
                                   yatsm, cfg['YATSM']['omission_alpha'], omit_decision)

            if yatsm.record is not None:
            	output.extend(yatsm.record)

        logger.debug('    Saving YATSM output to %s' % out)
        np.savez(out,
                 version=__version__,
                 record=np.array(output),
                 **{k: v for k, v in md.iteritems()})


