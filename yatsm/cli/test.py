import click
import datetime as dt
import logging
import os
import re
from yatsm.cli import options
import click
import numpy as np
from osgeo import gdal
import patsy
@click.command(short_help='Train classifier on YATSM output')
#@options.arg_config_file
@click.argument('classifier_config', metavar='<classifier_config>', nargs=1,
                type=click.Path(exists=True, readable=True,
                                dir_okay=False, resolve_path=True))
@click.argument('model', metavar='<model>', nargs=1,
                type=click.Path(writable=True, dir_okay=False,
                                resolve_path=True))
@click.option('--kfold', 'n_fold', nargs=1, type=click.INT, default=3,
              help='Number of folds in cross validation (default: 3)')
@click.option('--seed', nargs=1, type=click.INT,
              help='Random number generator seed')
@click.option('--multipath', '-m', nargs=1, metavar='<YATSMlist>',
              help='CSV with list of YATSM config files')
@click.pass_context

def test(YATSMlist):
    print YATSMlist
    print shapefile
    print "ok"


test(YATSMlist)
