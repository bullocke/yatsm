""" Calculate 'dip' in time series using spline smoothing function

See:
    Melaas, EK, MA Friedl, and Z Zhu. 2013. Detecting interannual variation in
    deciduous broadleaf forest phenology using Landsat TM/ETM+ data. Remote
    Sensing of Environment 132: 176-185.

"""
from __future__ import division

from datetime import datetime as dt
import math

import numpy as np
import numpy.lib.recfunctions

# Grab `stats` package from R for smoothing spline
from rpy2 import robjects as ro
from rpy2.robjects.packages import importr
import rpy2.robjects.numpy2ri

rpy2.robjects.numpy2ri.activate()
Rstats = importr('stats')

from vegetation_indices import EVI



def CRAN_spline(x, y, spar=0.55):
    """ Return a prediction function for a smoothing spline from R

    Use `rpy2` package to fit a smoothing spline using "smooth.spline".

    Args:
      x (np.ndarray): independent variable
      y (np.ndarray): dependent variable
      spar (float): smoothing parameter

    Returns:
      callable: prediction function of smoothing spline that provides
        smoothed estimates of the dependent variable given an input
        independent variable array

    Example:
      Fit a smoothing spline for y ~ x and predict for days in year::

        pred_spl = CRAN_spline(x, y)
        y_smooth = pred_spl(np.arange(1, 366))

    """
    spl = Rstats.smooth_spline(x, y, spar=spar)

    return lambda _x: np.array(Rstats.predict_smooth_spline(spl, _x)[1])



class MinSpline(object):
    """ Calculate species transition based on characteristics of the time series fit with a smothing spline algorithm. 

    Attributes:
      self.pheno (np.ndarray): NumPy structured array containing phenology
        metrics. These metrics include:

        - spring_doy: the long term mean day of year of the start of spring

    Args:
      model (yatsm.YATSM): instance of `yatsm.YATSM` that has been run for
        change detection
      greenness_index (int, optional): index of model.Y containing greenness band
        (default: 0)

    """
    def __init__(self, model, greenness_index=0):
        self.model = model
        if greenness_index:
            self.greenness = model.Y[greenness_index, :]

        self.ordinal = model.X[:, 1].astype(np.uint32)
        self.yeardoy = ordinal2yeardoy(self.ordinal)

        # Mask based on unusual EVI values
        valid_greenness = np.where((self.greenness >= 700))[0]
        self.greenness = self.evi[valid_greenness]
        self.ordinal = self.ordinal[valid_greenness]
        self.yeardoy = self.yeardoy[valid_greenness, :]

        self.pheno = np.zeros(self.model.record.shape, dtype=[
            ('green_start', 'u2'),
            ('green_peak', 'u2'),
            ('green_dip', 'u2'),
            ('spline_green', 'f8', np.length(self.greenness)
        ])

    def _fit_record(self, evi, yeardoy, year_interval, q_min, q_max):
	#This is where i need to calculate the metric used to find species transition. 
        # Calculate year-to-year groupings for EVI normalization
        # Fit spline and predict EVI
        spl_pred = CRAN_spline(pad_doy, pad_evi_norm, spar=0.55)
	#Need beginning and end
        greenness_smooth = spl_pred(np.arange(1, 366))
        min_green = np.argmin(greenness_smooth)
        peak_evi = np.max(evi_smooth)
        evi_smooth_spring = evi_smooth[:peak_doy + 1]
        evi_smooth_autumn = evi_smooth[peak_doy + 1:]

        return firstpeak, transition

    def fit(self):
        """ Fit phenology metrics for each time segment within a YATSM model

        Args:
          year_interval (int, optional): number of years to group together when
            normalizing EVI to upper and lower percentiles of EVI within the
            group (default: 3)

        Returns:
          np.ndarray: updated copy of YATSM model instance with phenology
            added into yatsm.record structured array

        """
        for i, _record in enumerate(self.model.record):
            # Subset variables to range of current record
            rec_range = np.where((self.ordinal >= _record['start']) &
                                 (self.ordinal <= _record['end']))[0]
            if rec_range.size == 0:
                continue

            _greenness = self.greenness[rec_range]
            _yeardoy = self.yeardoy[rec_range, :]

            # Fit and save results
            _result = self._fit_record(_greenness, _yeardoy,
                                       year_interval)

            self.pheno[i]['green_start'] = _result[0]
            self.pheno[i]['green_peak'] = _result[1]
            self.pheno[i]['green_dip'] = _result[2]
            self.pheno[i]['spline_green'][:] = _result[3]

        return np.lib.recfunctions.merge_arrays(
            (self.model.record, self.pheno), flatten=True)
