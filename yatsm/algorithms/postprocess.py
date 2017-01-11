""" Result post-processing utilities

Includes comission and omission tests and robust linear model result
calculations
"""
import logging

import numpy as np
import numpy.lib.recfunctions as nprf
import scipy.stats
from scipy.optimize import minimize, minimize_scalar
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
from ..regression.diagnostics import rmse
from ..utils import date2index
import patsy
import re
from ..regression.transforms import harm
from datetime import datetime as dt
from statsmodels.formula.api import ols
import pandas
import sys
logger = logging.getLogger('yatsm')
plt.style.use('ggplot')

#import R package for MANOVA
from rpy2 import robjects as ro
from rpy2.robjects.packages import importr
import rpy2.robjects.numpy2ri

rpy2.robjects.numpy2ri.activate()
mtvnorm = importr('mvtnorm')

#Temporary, import core numpy values to test lstsq code
from numpy.core import (
    array, asarray, zeros, empty, empty_like, transpose, intc, single, double,
    csingle, cdouble, inexact, complexfloating, newaxis, ravel, all, Inf, dot,
    add, multiply, sqrt, maximum, fastCopyAndTranspose, sum, isfinite, size,
    finfo, errstate, geterrobj, longdouble, rollaxis, amin, amax, product, abs,
    broadcast, atleast_2d, intp, asanyarray, isscalar
    )
from numpy.lib import triu, asfarray
from numpy.linalg import lapack_lite, _umath_linalg
from numpy.matrixlib.defmatrix import matrix_power
from numpy.compat import asbytes

def do_manova(yatsm, m_1_start, m_1_end, m_2_start, m_2_end, m_r_start, m_r_end, models):
    """ Multivariate analysis of variance based on R's mvtnorm package

    Use MANOVA to test the difference between having two seperate
    models with a break in between and having a single model fit. 
    The result tests whether the break was unnecessary and the models
    can be merged. 

    """
    both = {}
    single = {}
    len_single = m_r_end[0] - m_1_start[0]
    for i_b, b in enumerate(yatsm.test_indices):
        band_both_residuals = []
        band_both_residuals.extend(lstsq(yatsm.X[m_1_start:m_1_end, :],
                                         yatsm.Y[b, m_1_start:m_1_end])[4])
        band_both_residuals.extend(lstsq(yatsm.X[m_2_start:m_2_end, :],
                                         yatsm.Y[b, m_2_start:m_2_end])[4])
	obs = len(band_both_residuals)
        #Remove observations from single model so they are the same length
	num_obs_to_remove = len_single - obs
        band_both_residuals.extend(lstsq(yatsm.X[m_r_start:m_r_end, :],
                                        yatsm.Y[b, m_r_start:m_r_end])[4][num_obs_to_remove:])
        both[str(i_b)] = ro.FloatVector(band_both_residuals)

    #np.cov()
    factor1 = [1] * int(.5 * len(band_both_residuals))
    factor2 = [2] * int(.5 * len(band_both_residuals))
    factor1.extend(factor2)
    both["fact"] = np.array(factor1)
    r_both = ro.DataFrame(both)
    r_factor = ro.IntVector(factor1)      
    ro.globalenv["both"] = r_both
    ro.globalenv["m_factor"] = r_factor
    ro.r("fit <- manova(cbind(X2,X2,X3,X4) ~ factor(fact),data=both)")
    f_stat = ro.r('summary(fit, tol=0)$stats[1,3]')
    _plot = False
    if _plot:
	xnew=[]
	xnew.extend(yatsm.X[:,1])
	xnew.extend(yatsm.X[:,1] + (xnew[-1] - xnew[0]))
	col=np.zeros(len(band_both_residuals))
	col=col.astype(str)
	ind = len(band_both_residuals) / 2
	col[0:ind]='red'
	col[ind:]='blue'
	plt.scatter(xnew[:-4],band_both_residuals,c=col)
	plt.ylim(-1000,1000)
	plt.show()
    #Accept null and merge when Pillais is small
    pillai = ro.r('summary(fit, tol=0)$stats[1,2]')
    f_stat2 = ro.r('summary(fit, tol=0, test="Roy")$stats[1,3]')
    roy = ro.r('summary(fit, tol=0, test="Roy")$stats[1,2]')
    f_stat3 = ro.r('summary(fit, tol=0, test="Wilks")$stats[1,3]')
    wilks = ro.r('summary(fit, tol=0, test="Wilks")$stats[1,2]')
    f_stat4 = ro.r('summary(fit, tol=0, test="Hotelling-Lawley")$stats[1,3]')
    hotelling = ro.r('summary(fit, tol=0, test="Wilks")$stats[1,2]')
    _break = dt.fromordinal(yatsm.X[m_1_end][:,1])
    #print "Break date of model: %s" % _break
    print 'F(Pillais) Stat: %s' % f_stat
    print 'Pillais Trace: %s' % pillai
    print 'F(Roy) Stat: %s' % f_stat2
    print 'Roy: %s' % roy
    print 'F(Wilks) Stat: %s' % f_stat3
    print 'Wilks: %s' % wilks
    print 'F(Hotelling) Stat: %s' % f_stat4
    print 'Hotelling: %s' % hotelling
    #stats
    #(summary(fit, tol=0, tol=0)$stats)[1,]
#    fit = ro.r("manova(both ~ single)")
    reject = True
    return reject

def get_mult_r(yatsm,indices=[1,2,3,4,5]):
    """Get multiple correlation coefficients for each test indice based
    on multiple regressions. Let each band = n,
  
    dependent variable: n 
    independent variables: test_indices - n
    multiple regression: n ~ test_indices
    correlation coefficient: rsquared ^ (1/2)

    """

    columns=["blue","green","red","nir","swir1","swir2","thermal"]
    col_ar = np.array(columns)
    col_ind = np.array(indices)
    test_bands = col_ar[col_ind]
    mult_correlation_coefficients = []
    #Create pandas dataframe to make multiple regression easier
    y_df = pandas.DataFrame(np.swapaxes(yatsm.Y,0,1),columns=columns)
    #Note: hard coded at the moment, DEFINITELY not best way to do this
    model_green = ols("green ~ red+nir+swir1+swir2", data=y_df).fit()
    model_red = ols("red ~ green+nir+swir1+swir2", data=y_df).fit()
    model_nir = ols("nir ~ green+red+swir1+swir2", data=y_df).fit()
    model_swir1 = ols("swir1 ~ green+red+nir+swir2", data=y_df).fit()
    model_swir2 = ols("swir2 ~ green+red+nir+swir1", data=y_df).fit()
    mult_correlation_coefficients.append(1 - model_green.rsquared**5)
    mult_correlation_coefficients.append(1 - model_red.rsquared**5)
    mult_correlation_coefficients.append(1 - model_nir.rsquared**5)
    mult_correlation_coefficients.append(1 - model_swir1.rsquared**5)
    mult_correlation_coefficients.append(1 - model_swir2.rsquared**5)
    return mult_correlation_coefficients


def do_chowda(yatsm, m_1_start, m_1_end, m_2_start, m_2_end, m_r_start, m_r_end, models, behavior, k, n, F_crit):
    """ Merge adjacent records based on Boston's version of Chow Tests for nested models


    Use Chow Test to find false positive, spurious, or unnecessary breaks
    in the timeseries by comparing the effectiveness of two separate
    adjacent models with one single model that spans the entire time
    period.

    Chow test is described:

    .. math::
        \\frac{[RSS_r - (RSS_1 + RSS_2)] / k}{(RSS_1 + RSS_2) / (n - 2k)}

    where:

        - :math:`RSS_r` is the RSS of the combined, or, restricted model
        - :math:`RSS_1` is the RSS of the first model
        - :math:`RSS_2` is the RSS of the second model
        - :math:`k` is the number of model parameters
        - :math:`n` is the number of total observations

    Because we look for change in multiple bands, the RSS used to compare
    the unrestricted versus restricted models is the mean RSS
    values from all ``model.test_indices``.
    """
    F_stats = []
    # Allocate memory outside of loop
    m_1_rss = np.zeros(yatsm.test_indices.size)
    m_2_rss = np.zeros(yatsm.test_indices.size)
    m_r_rss = np.zeros(yatsm.test_indices.size)

    for i_b, b in enumerate(yatsm.test_indices):
        m_1_rss[i_b] = np.linalg.lstsq(yatsm.X[m_1_start:m_1_end, :],
                                       yatsm.Y[b, m_1_start:m_1_end])[1]
        m_2_rss[i_b] = np.linalg.lstsq(yatsm.X[m_2_start:m_2_end, :],
                                       yatsm.Y[b, m_2_start:m_2_end])[1]
        m_r_rss[i_b] = np.linalg.lstsq(yatsm.X[m_r_start:m_r_end, :],
                                       yatsm.Y[b, m_r_start:m_r_end])[1]
        F_band = (((m_r_rss[i_b] - (m_1_rss[i_b] + m_2_rss[i_b])) / k) / ((m_1_rss[i_b] + m_2_rss[i_b]) / (n - 2 * k)))
        F_stats.append(F_band)


    #Weight all bands by atleast .5 plus 1/2 correlation coefficient
    mult_r_squared = np.array(get_mult_r(yatsm))
    mult_r_squared = .5 + (mult_r_squared / 2)
        # Collapse RSS across all test indices for F statistic
    if behavior == 'collapse':
        F = (
            ((m_r_rss.mean() - (m_1_rss.mean() + m_2_rss.mean())) / k) /
            ((m_1_rss.mean() + m_2_rss.mean()) / (n - 2 * k))
             )
        if F > F_crit:
   	    reject = True
	else:
	    reject = False

    elif behavior.lower() == 'mode':
        F_over = len(np.where(np.array(F_stats) > F_crit)[0])
        if F_over > (len(yatsm.test_indices) / float(2)): #TODO: There has to be a better way to get the mode
  	    reject = True
	else:
	    reject = False	    

    elif behavior == 'weighted_mean':
        F2 = (
            ((np.average(m_r_rss, weights=mult_r_squared) - (np.average(m_1_rss, weights=mult_r_squared) + np.average(m_2_rss,weights=mult_r_squared))) / k) /
            ((np.average(m_1_rss,weights=mult_r_squared) + np.average(m_2_rss,weights=mult_r_squared)) / (n - 2 * k))
             )
        if F2 > F_crit:
   	    reject = True
	else:
	    reject = False
	

    return reject

# POST-PROCESSING
def commission_test(yatsm, alpha=0.10,behavior="collapse"):
    """ Master function for testing whether to merge adjacent models due to incorrect
    changes detected by CCDC. 

    Args:
        yatsm (YATSM model): fitted YATSM model to check for commission errors
        alpha (float): significance level for F-statistic (default: 0.10)

    Returns:
        np.ndarray: updated copy of ``yatsm.record`` with spurious models
            combined into unified model

    """
    if yatsm.record.size == 1:
        return yatsm.record

    k = yatsm.record[0]['coef'].shape[0]


    models = []
    merged = False
    for i in range(len(yatsm.record) - 1):
        if merged:
            m_1 = models[-1]
        else:
            m_1 = yatsm.record[i]
        m_2 = yatsm.record[i + 1]

        # Unrestricted model starts/ends
        m_1_start = date2index(yatsm.dates, m_1['start'])
        m_1_end = date2index(yatsm.dates, m_1['end'])
        m_2_start = date2index(yatsm.dates, m_2['start'])
        m_2_end = date2index(yatsm.dates, m_2['end'])
        # Restricted start/end
        m_r_start = m_1_start
        m_r_end = m_2_end


        # Need enough obs to fit models (n > k)
        if (m_1_end - m_1_start) <= (k + 2) or (m_2_end - m_2_start) <= (k + 2):
            logger.debug('Too few obs (n <= k) to merge segment')
            merged = False
            if i == 0:
                models.append(m_1)
            models.append(m_2)
            continue

        n = m_r_end - m_r_start
        F_crit = scipy.stats.f.ppf(1 - alpha, k, n - 2 * k)
	F_stats = []
	#Parameterize this
	commission_method = 'CHOW'
	behavior = "weighted_mean"
        if commission_method == 'MANOVA':
	    reject = do_manova(yatsm, m_1_start, m_1_end, m_2_start, m_2_end, m_r_start, m_r_end, models)

	elif commission_method == 'CHOW':
	    reject = do_chowda(yatsm, m_1_start, m_1_end, m_2_start, m_2_end, m_r_start, m_r_end, models, behavior, k,n, F_crit)
	
        if reject:
            # Reject H0 and retain change
            # Only add in previous model if first model
            if i == 0:
                models.append(m_1)
            models.append(m_2)
            merged = False
        else:
	    if not yatsm.n_series:
		yatsm.n_series = np.shape(np.array(m_1)['coef'])[1]
		yatsm.n_features = np.shape(np.array(m_1)['coef'])[0]
            # Fail to reject H0 -- ignore change and merge
            m_new = np.copy(yatsm.record_template)[0]

            # Remove last previously added model from list to merge
            if i != 0:
                del models[-1]

            m_new['start'] = m_1['start']
            m_new['end'] = m_2['end']
            m_new['break'] = m_2['break']
	    m_new['status'] = 3 

            # Re-fit models and copy over attributes
            yatsm.models = yatsm.fit_models(yatsm.X[m_r_start:m_r_end, :],
                             yatsm.Y[:, m_r_start:m_r_end],bands=yatsm.test_indices)
            for i_m, _m in enumerate(yatsm.models):
                m_new['coef'][:, i_m] = _m.coef
                m_new['rmse'][i_m] = _m.rmse

           # if 'magnitude' in yatsm.record.dtype.names:
                # Preserve magnitude from 2nd model that was merged
           #     m_new['magnitude'] = m_2['magnitude']

            models.append(m_new)

#    	    import pdb; pdb.set_trace()
            merged = True

    return np.array(models)


def omission_test(model, crit=0.01, behavior='ANY', dates=None, ylim=None, indices=None, output=None):
    """ Add omitted breakpoint into records based on residual stationarity

    Uses recursive residuals within a CUMSUM test to check if each model
    has omitted a "structural change" (e.g., land cover change). Returns
    an array of True or False for each timeseries segment record depending
    on result from `statsmodels.stats.diagnostic.breaks_cusumolsresid`.

    Args:
        crit (float, optional): Critical p-value for rejection of null
            hypothesis that data contain no structural change
        behavior (str, optional): Method for dealing with multiple
            `test_indices`. `ANY` will return True if any one test index
            rejects the null hypothesis. `ALL` will only return True if ALL
            test indices reject the null hypothesis.
        indices (np.ndarray, optional): Array indices to test. User provided
            indices must be a subset of `model.test_indices`.

    Returns:
        np.ndarray: Array of True or False for each record where
            True indicates omitted break point

    """
    if behavior.lower() not in ['any', 'all', 'mode','mean']:
        raise ValueError('`behavior` must be "any" or "all"')

    if not indices:
        indices = model.test_indices

    if not np.all(np.in1d(indices, model.test_indices)):
        raise ValueError('`indices` must be a subset of '
                         '`model.test_indices`')

    #if not model.ran:
    #    return np.empty(0, dtype=bool)

    omission = np.zeros((model.record.size, len(indices)), dtype=bool)
    models = []
    for i, r in enumerate(model.record):
        # Skip if no model fit
        if r['start'] == 0 or r['end'] == 0:
            continue
        # Find matching X and Y in data
        index = np.where(
            (model.dates >= min(r['start'], r['end'])) &
            (model.dates <= max(r['end'], r['start'])))[0]

	#Continue if there are not enough observations. TODO: Fix. 
	if len(index) < 2 * model.min_obs:
	    models.append(r)
	    continue

        # Grab matching X and Y
        _X = model.X[index, :]
        _Y = model.Y[:, index]
	breakindex = np.zeros(len(indices))

	test = []
        for i_b, b in enumerate(indices):
            # Create OLS regression
            ols = sm.OLS(_Y[b, :], _X).fit()
            # Perform CUMSUM test on residuals
            _test = sm.stats.diagnostic.breaks_cusumolsresid(
                ols.resid, _X.shape[1])
	    test.append(_test[1])
	    saveols = True
	    if output: 
		if saveols:
		    fig, ax1 = plt.subplots(figsize=(7.5,2))
		    design = re.sub(r'[\+\-][\ ]+C\(.*\)', '', "1 + x + harm(x, 1) + harm(x, 2) + harm(x,3)")
		    mx = np.arange(r['start'], r['end'], 1)
		    mX = patsy.dmatrix(design, {'x': mx}).T
		    test2=np.dot(ols.params,mX)
		    mx_date = np.array([dt.fromordinal(int(_x)) for _x in mx])
		    mx_date2 = np.array([dt.fromordinal(int(_x)) for _x in _X[:,1]])
		    fig, ax1 = plt.subplots(figsize=(7.5,2))
		    ax1.plot(_X[:,1],_Y[b,:],'.',linewidth=2,color='black')
		    plt.ylim(ylim) 
		    out6 = output + '_nots.png'
		    plt.xlim([mx_date2[0],mx_date2[-1]])
		    plt.tight_layout()
		    plt.savefig(out6, dpi=300)

		    fig, ax1 = plt.subplots(figsize=(7.5,2))
		    ax1.plot(_X[:,1],_Y[b,:],'.',linewidth=2,color='c')
		    ax1.plot(mx_date,test2)
		    plt.ylim(ylim) 
		    out1 = output + '_ols.png'
		    plt.xlim([mx_date2[0],mx_date2[-1]])
		    plt.tight_layout()
		    plt.savefig(out1, dpi=300)
		    fig, ax1 = plt.subplots(figsize=(7.5,2))
		    ax1.plot(mx_date2,ols.resid,'.',linewidth=2,color='b')
		    plt.ylim([-1000,1500]) 
		    plt.tight_layout()
		    out2 = output + '_resid.png'
		    plt.savefig(out2, dpi=300)
	        cusum_plot_x = []
	        for itera in _X[:,1]: 
		    cusum_plot_x.append(dt.fromordinal(int(itera)))
		nobs = len(ols.resid)
		nobssigma2 = (ols.resid**2).sum()
		nobssigma2 = nobssigma2 / (nobs - _X.shape[1]) * nobs
	        cusum_plot_y = np.cumsum(ols.resid) / np.sqrt(nobssigma2)
		#cusum_plot_y = ols.resid
		fig, ax1 = plt.subplots(figsize=(7.5,2))
	        ax1.plot(cusum_plot_x, cusum_plot_y)
		ax1.axhline(1.63, color='blue', linestyle='dashed')
		ax1.axhline(-1.63, color='blue', linestyle='dashed')
		from datetime import date
		d=date(year=1982, month=3, day=1)
		d1=date(year=2014, month=3, day=1)
		plt.xlim([mx_date2[0],d1])
		plt.tight_layout()
		out3 = output + '_cusum.png'
	        plt.savefig(out3,dpi=300)
            if test[i_b] < crit:
                omission[i, i_b] = True
            else:
                omission[i, i_b] = False
        # Collapse band answers according to `behavior`
	if (behavior.lower() == 'mean') and (np.mean(test) < crit):
	    redo_models = True
        elif (behavior.lower() == 'any') and (np.any(omission[i,:])):
	    redo_models = True
        elif (behavior.lower() == 'all') and (np.all(omission[i,:])):
	    redo_models = True
        elif (behavior.lower() == 'mode') and scipy.stats.mode(omission[i,:])[0] == 1:
	    redo_models = True
        else:
	    redo_models = False
	    models.append(r)

	#If model break was found, refit the models:
        if redo_models: 
	    for ind in np.nonzero(omission[i,:])[0]:
		#Number of observations
		nobs = _X[:,1].shape[0]
		#Maximum index that will leave remaining model with enough observations
		max_ind = nobs - model.min_obs  
		params=(_Y[b,:], _X, nobs, model.min_obs)
		#Optimize bounded break detection
		opt_break = minimize_scalar(opt_breakdetection, bounds=(model.min_obs,max_ind), method='bounded', args=(params,), options={'disp': True, 'maxiter': 40})
		breakindex[ind] = int(opt_break.x)

            _breakindex = np.min(breakindex[breakindex > 0])


     	    breakdate = _X[:,1][_breakindex]

            m_new_1 = np.copy(model.record_template)[0]
            m_new_2 = np.copy(model.record_template)[0]

            m_new_1['start'] = _X[:,1][0]
            m_new_1['end'] = _X[:,1][_breakindex - 1]
            m_new_1['break'] = _X[:,1][_breakindex - 1]
	    if r['status'] == 2:
	        m_new_1['status'] = 4
	        m_new_2['status'] = 4
	    else: 
	        m_new_1['status'] = 2
	        m_new_2['status'] = 2

            m_new_2['start'] = breakdate
            m_new_2['end'] = _X[:,1][-1]
            m_new_2['break'] = 0

            #organize indices
            m_1_start = 0
            m_1_end = _breakindex - 1 
            m_2_start = _breakindex
            m_2_end = len(_X[:,1]) - 1

            # Re-fit models and copy over attributes
            m1 = model.fit_models(_X[m_1_start:m_1_end, :],
                                  _Y[:, m_1_start:m_1_end])
            m2 = model.fit_models(_X[m_2_start:m_2_end, :],
                                  _Y[:, m_2_start:m_2_end])


            for i_m, _m in enumerate(m1):
                m_new_1['coef'][:, i_m] = _m.coef
                m_new_1['rmse'][i_m] = _m.rmse

            for i_m, _m in enumerate(m2):
                m_new_2['coef'][:, i_m] = _m.coef
                m_new_2['rmse'][i_m] = _m.rmse

            models.append(m_new_1)
            models.append(m_new_2)


    return np.array(models)



def refit_record(model, prefix, estimator,
                 fitopt=None, keep_regularized=False):
    """ Refit YATSM model segments with a new estimator and update record

    YATSM class model must be ran and contain at least one record before this
    function is called.

    Args:
        model (YATSM model): YATSM model to refit
        prefix (str): prefix for refitted coefficient and RMSE (don't include
            underscore as it will be added)
        estimator (object): instance of a scikit-learn compatible estimator
            object
        fitopt (dict, optional): dict of options for the ``fit`` method of the
            ``estimator`` provided (default: None)
        keep_regularized (bool, optional): do not use features with coefficient
            estimates that are fit to 0 (i.e., if using L1 regularization)

    Returns:
        np.array: updated model.record NumPy structured array with refitted
            coefficients and RMSE

    """
    if not model:
        return None

    fitopt = fitopt or {}

    refit_coef = prefix + '_coef'
    refit_rmse = prefix + '_rmse'

    # Create new array for robust coefficients and RMSE
    n_coef, n_series = model.record[0]['coef'].shape
    refit = np.zeros(model.record.shape[0], dtype=[
        (refit_coef, 'float32', (n_coef, n_series)),
        (refit_rmse, 'float32', (n_series)),
    ])

    for i_rec, rec in enumerate(model.record):
        # Find matching X and Y in data
        # start/end dates are considered in case ran backward
        index = np.where((model.dates >= min(rec['start'], rec['end'])) &
                         (model.dates <= max(rec['start'], rec['end'])))[0]

        X = model.X.take(index, axis=0)
        Y = model.Y.take(index, axis=1)

        # Refit each band
        for i_y, y in enumerate(Y):
            if keep_regularized:
                # Find nonzero in case of regularized regression
                nonzero = np.nonzero(rec['coef'][:, i_y])[0]
                if nonzero.size == 0:
                    refit[i_rec][refit_rmse][:] = rec['rmse']
                    continue
            else:
                nonzero = np.arange(n_series)

            # Fit
            estimator.fit(X[:, nonzero], y, **fitopt)
            # Store updated coefficients
            refit[i_rec][refit_coef][nonzero, i_y] = estimator.coef_
            refit[i_rec][refit_coef][0, i_y] += getattr(
                estimator, 'intercept_', 0.0)

            # Update RMSE
            refit[i_rec][refit_rmse][i_y] = rmse(
                y, estimator.predict(X[:, nonzero]))

    # Merge
    refit = nprf.merge_arrays((model.record, refit), flatten=True)

    return refit

def find_breakpoints(endog, exog, nbreaks, trim=0.15):
    """
    Find the specified number of breakpoints in a sample.
    Author: Chad Fulton
    Source: https://gist.github.com/ChadFulton/6426282

    References
    ----------
    Bai, Jushan, and Pierre Perron. 1998.
    "Estimating and Testing Linear Models with Multiple Structural Changes."
    Econometrica 66 (1) (January 1): 47-78.

    Bai, Jushan, and Pierre Perron. 2003.
    "Computation and Analysis of Multiple Structural Change Models."
    Journal of Applied Econometrics 18 (1): 1-22.

    Parameters
    ----------
    endog : array-like
        The endogenous variable.
    exog : array-like
        The exogenous matrix.
    nbreaks : integer
        The number of breakpoints to select.
    trim : float or int, optional
        If a float, the minimum percentage of observations in each regime,
        if an integer, the minimum number of observations in each regime.
    
    Returns
    -------
    breakpoints : iterable
        The (zero-indexed) indices of the breakpoints. The k-th breakpoints is
        defined to be the last observation in the (k-1)th regime.
    ssr : float
        The sum of squared residuals from the model with the selected breaks.
    """
    nobs = len(endog)
    if trim < 1:
        trim = int(np.floor(nobs*trim))

    if nobs < 2*trim:
        raise InvalidRegimeError

    # TODO add test to make sure trim is consistent with the number of breaks

    # This is how many calculations will be performed
    ncalcs = (
        ( nobs * (nobs + 1) / 2 ) -
        ( (trim-1)*nobs - (trim-2)*(trim-1)/2 ) -
        ( (trim**2)*nbreaks*(nbreaks+1)/2 ) -
        ( nobs*(trim-1) - nbreaks*trim*(trim-1) -
          (trim-1)**2 - trim*(trim-1)/2 )
    )
    # Estimate the sum of squared errors for each possible segment
    # TODO Important - change this to compute via recursive OLS, will
    #      be much faster!
    results = {}
    for i in range(nobs):
        for j in range(i, nobs):
            length = j - i + 1
            # Current segment too small
            if length < trim:
                continue
            # First segment too small
            if i > 0 and i < trim:
                continue
            # Not enough room for other segments
            if (i // trim) + ((nobs - j - 1) // trim) < nbreaks:
                continue
            
            res = sm.OLS(endog[i:j+1], exog[i:j+1]).fit()

            # Change from zero-index to one-index
            results[(i+1,j+1)] = res.ssr
    assert(len(results) == ncalcs)

    # Dynamic Programming approach to select global minimia
    optimal = []
    for nbreak in range(1, nbreaks):
        optimal.append({})
        for end in range((nbreak+1)*trim, nobs-(nbreaks-nbreak)*trim+1):
            min_ssr = np.Inf
            optimal_breakpoints = None
            for breakpoint in range(nbreak*trim, end-trim+1):
                ssr = optimal[-2][breakpoint][0] if nbreak > 1 else results[(1, breakpoint)]
                ssr += results[(breakpoint+1, end)]
                if ssr < min_ssr:
                    min_ssr = ssr
                    if nbreak > 1:
                        optimal_breakpoints = optimal[-2][breakpoint][1] + (breakpoint,)
                    else:
                        optimal_breakpoints = (breakpoint,)
            optimal[-1][end] = (min_ssr, optimal_breakpoints)

    final_breaks = optimal[-1].keys() if nbreaks > 1 else range(trim, nobs-trim)

    min_ssr = np.Inf
    breakpoints = None
    for breakpoint in final_breaks:
        ssr = optimal[-1][breakpoint][0] if nbreaks > 1 else results[(1, breakpoint)]
        ssr += results[(breakpoint+1, nobs)]
        if ssr < min_ssr:
            min_ssr = ssr
            if nbreaks > 1:
                breakpoints = optimal[-1][breakpoint][1] + (breakpoint,)
            else:
                breakpoints = (breakpoint,)

    # Breakpoints are one-indexed, so change them to be zero-indexed
    breakpoints = tuple(np.array(breakpoints)-1)

    return breakpoints, min_ssr

def opt_breakdetection(i,params):
    """
    Function to determine optimum break date in time series given that there is 1 break. 
    Function minimum corresponds to index of best possible break date. 
    """
    x = params[0]
    y = params[1]
    nobs = params[2]
    trim = params[3]
    m1 = sm.OLS(x[0:i], y[0:i]).fit()
    m1 = m1.ssr
    m2 = sm.OLS(x[i:nobs], y[i:nobs]).fit()
    m2 = m2.ssr
    combined = m1 + m2
    return combined

#The following functions come from numpy linalg

def _makearray(a):
    new = np.asarray(a)
    wrap = getattr(a, "__array_prepare__", new.__array_wrap__)
    return new, wrap

def _assertRank2(*arrays):
    for a in arrays:
        if len(a.shape) != 2:
            raise LinAlgError('%d-dimensional array given. Array must be '
                    'two-dimensional' % len(a.shape))
def _realType(t, default=double):
    return _real_types_map.get(t, default)

def _complexType(t, default=cdouble):
    return _complex_types_map.get(t, default)

def _linalgRealType(t):
    """Cast the type t to either double or cdouble."""
    return double

def _commonType(*arrays):
    # in lite version, use higher precision (always double or cdouble)
    result_type = single
    is_complex = False
    for a in arrays:
        if issubclass(a.dtype.type, inexact):
            if isComplexType(a.dtype.type):
                is_complex = True
            rt = _realType(a.dtype.type, default=None)
            if rt is None:
                # unsupported inexact scalar
                raise TypeError("array type %s is unsupported in linalg" %
                        (a.dtype.name,))
        else:
            rt = double
        if rt is double:
            result_type = double
    if is_complex:
        t = cdouble
        result_type = _complex_types_map[result_type]
    else:
        t = double
    return t, result_type

def _fastCopyAndTranspose(type, *arrays):
    cast_arrays = ()
    for a in arrays:
        if a.dtype.type is type:
            cast_arrays = cast_arrays + (_fastCT(a),)
        else:
            cast_arrays = cast_arrays + (_fastCT(a.astype(type)),)
    if len(cast_arrays) == 1:
        return cast_arrays[0]
    else:
        return cast_arrays

def _to_native_byte_order(*arrays):
    ret = []
    for arr in arrays:
        if arr.dtype.byteorder not in ('=', '|'):
            ret.append(asarray(arr, dtype=arr.dtype.newbyteorder('=')))
        else:
            ret.append(arr)
    if len(ret) == 1:
        return ret[0]
    else:
        return ret

def isComplexType(t):
    return issubclass(t, complexfloating)

_real_types_map = {single : single,
                   double : double,
                   csingle : single,
                   cdouble : double}

_complex_types_map = {single : csingle,
                      double : cdouble,
                      csingle : csingle,
                      cdouble : cdouble}

_fastCT = fastCopyAndTranspose

fortran_int = intc

def lstsq(a, b, rcond=-1):
    """
    Taken directly from numpy's linalg.lstsq function.
    Modified to return full array of residuals. 
    Return the least-squares solution to a linear matrix equation.
    Solves the equation `a x = b` by computing a vector `x` that
    minimizes the Euclidean 2-norm `|| b - a x ||^2`.  The equation may
    be under-, well-, or over- determined (i.e., the number of
    linearly independent rows of `a` can be less than, equal to, or
    greater than its number of linearly independent columns).  If `a`
    is square and of full rank, then `x` (but for round-off error) is
    the "exact" solution of the equation.
    Parameters
    ----------
    a : (M, N) array_like
        "Coefficient" matrix.
    b : {(M,), (M, K)} array_like
        Ordinate or "dependent variable" values. If `b` is two-dimensional,
        the least-squares solution is calculated for each of the `K` columns
        of `b`.
    rcond : float, optional
        Cut-off ratio for small singular values of `a`.
        Singular values are set to zero if they are smaller than `rcond`
        times the largest singular value of `a`.
    Returns
    -------
    x : {(N,), (N, K)} ndarray
        Least-squares solution. If `b` is two-dimensional,
        the solutions are in the `K` columns of `x`.
    sumresiduals : {(), (1,), (K,)} ndarray
        Sums of residuals; squared Euclidean 2-norm for each column in
        ``b - a*x``.
        If the rank of `a` is < N or M <= N, this is an empty array.
        If `b` is 1-dimensional, this is a (1,) shape array.
        Otherwise the shape is (K,).
    rank : int
        Rank of matrix `a`.
    s : (min(M, N),) ndarray
        Singular values of `a`.
    residuals: {(N,)} ndarray
	Residuals from least-squares solution. 
    Raises
    ------
    LinAlgError
        If computation does not converge.
    Notes
    -----
    If `b` is a matrix, then all array results are returned as matrices.
    Examples
    --------
    Fit a line, ``y = mx + c``, through some noisy data-points:
    >>> x = np.array([0, 1, 2, 3])
    >>> y = np.array([-1, 0.2, 0.9, 2.1])
    By examining the coefficients, we see that the line should have a
    gradient of roughly 1 and cut the y-axis at, more or less, -1.
    We can rewrite the line equation as ``y = Ap``, where ``A = [[x 1]]``
    and ``p = [[m], [c]]``.  Now use `lstsq` to solve for `p`:
    >>> A = np.vstack([x, np.ones(len(x))]).T
    >>> A
    array([[ 0.,  1.],
           [ 1.,  1.],
           [ 2.,  1.],
           [ 3.,  1.]])
    >>> m, c = np.linalg.lstsq(A, y)[0]
    >>> print(m, c)
    1.0 -0.95
    Plot the data along with the fitted line:
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(x, y, 'o', label='Original data', markersize=10)
    >>> plt.plot(x, m*x + c, 'r', label='Fitted line')
    >>> plt.legend()
    >>> plt.show()
    """
    import math
    a, _ = _makearray(a)
    b, wrap = _makearray(b)
    is_1d = len(b.shape) == 1
    if is_1d:
        b = b[:, np.newaxis]
    _assertRank2(a, b)
    m  = a.shape[0]
    n  = a.shape[1]
    n_rhs = b.shape[1]
    ldb = max(n, m)
    if m != b.shape[0]:
        raise LinAlgError('Incompatible dimensions')
    t, result_t = _commonType(a, b)
    result_real_t = _realType(result_t)
    real_t = _linalgRealType(t)
    bstar = zeros((ldb, n_rhs), t)
    bstar[:b.shape[0], :n_rhs] = b.copy()
    a, bstar = _fastCopyAndTranspose(t, a, bstar)
    a, bstar = _to_native_byte_order(a, bstar)
    s = zeros((min(m, n),), real_t)
    nlvl = max( 0, int( math.log( float(min(m, n))/2. ) ) + 1 )
    iwork = zeros((3*min(m, n)*nlvl+11*min(m, n),), fortran_int)
    if isComplexType(t):
        lapack_routine = lapack_lite.zgelsd
        lwork = 1
        rwork = zeros((lwork,), real_t)
        work = zeros((lwork,), t)
        results = lapack_routine(m, n, n_rhs, a, m, bstar, ldb, s, rcond,
                                 0, work, -1, rwork, iwork, 0)
        lwork = int(abs(work[0]))
        rwork = zeros((lwork,), real_t)
        a_real = zeros((m, n), real_t)
        bstar_real = zeros((ldb, n_rhs,), real_t)
        results = lapack_lite.dgelsd(m, n, n_rhs, a_real, m,
                                     bstar_real, ldb, s, rcond,
                                     0, rwork, -1, iwork, 0)
        lrwork = int(rwork[0])
        work = zeros((lwork,), t)
        rwork = zeros((lrwork,), real_t)
        results = lapack_routine(m, n, n_rhs, a, m, bstar, ldb, s, rcond,
                                 0, work, lwork, rwork, iwork, 0)
    else:
        lapack_routine = lapack_lite.dgelsd
        lwork = 1
        work = zeros((lwork,), t)
        results = lapack_routine(m, n, n_rhs, a, m, bstar, ldb, s, rcond,
                                 0, work, -1, iwork, 0)
        lwork = int(work[0])
        work = zeros((lwork,), t)
        results = lapack_routine(m, n, n_rhs, a, m, bstar, ldb, s, rcond,
                                 0, work, lwork, iwork, 0)
    if results['info'] > 0:
        raise LinAlgError('SVD did not converge in Linear Least Squares')
    resids = array([], result_real_t)
    if is_1d:
        x = array(ravel(bstar)[:n], dtype=result_t, copy=True)
        if results['rank'] == n and m > n:
            if isComplexType(t):
                resids = array([sum(abs(ravel(bstar)[n:])**2)],
                               dtype=result_real_t)
            else:
                resids = array([sum((ravel(bstar)[n:])**2)],
                               dtype=result_real_t)
    else:
        x = array(transpose(bstar)[:n,:], dtype=result_t, copy=True)
        if results['rank'] == n and m > n:
            if isComplexType(t):
                resids = sum(abs(transpose(bstar)[n:,:])**2, axis=0).astype(
                    result_real_t, copy=False)
            else:
                resids = sum((transpose(bstar)[n:,:])**2, axis=0).astype(
                    result_real_t, copy=False)

    st = s[:min(n, m)].astype(result_real_t, copy=True)
    return wrap(x), wrap(resids), results['rank'], st, bstar[0]
