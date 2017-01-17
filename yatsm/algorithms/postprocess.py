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
import spm1d
logger = logging.getLogger('yatsm')
plt.style.use('ggplot')


def do_postprocess.py(yatsm):
    """Master function for postprocessing"""
    #TODO
    return yatsm

def do_chowda(yatsm, m_1_start, m_1_end, 
	      m_2_start, m_2_end, m_r_start, 
	      m_r_end, models, behavior, 
	      k, n, F_crit):
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
        F_band = (((m_r_rss[i_b] - (m_1_rss[i_b] + m_2_rss[i_b])) / k) 
		 / ((m_1_rss[i_b] + m_2_rss[i_b]) / (n - 2 * k)))
        F_stats.append(F_band)

    #Get weights for the mean based on average r^2 across bands
    weights = get_weights(yatsm)

    behavior = 'weighted_fmean' #TODO: Parameterize?  

    if behavior == 'collapse':
	""" Collapse: Take the mean (un-weighted) for each variable in the 
	formula across bands. """
        F = (
            ((m_r_rss.mean() - (m_1_rss.mean() + m_2_rss.mean())) / k) /
            ((m_1_rss.mean() + m_2_rss.mean()) / (n - 2 * k))
             )
        if F > F_crit:
   	    reject = True
	else:
	    reject = False

    elif behavior.lower() == 'mode':
	""" Mode: Do what the majority of bands do """
        F_over = len(np.where(np.array(F_stats) > F_crit)[0])
        if F_over > (len(yatsm.test_indices) / float(2)): 
  	    reject = True
	else:
	    reject = False	    

    elif behavior == 'weighted_mean':
	""" Weighted mean: take the weighted mean across bands for 
	each variable within the formula """
        F2 = (
	     ((w_av(m_r_ss, weights) - (w_av(m_1_rss, weights) 
	     + w_av(m_2_rss, weights))) / k ) / 
	     ((w_av(m_1_rss, weights) + w_av(m_2_rss, weights)) 
	     / (n - 2 * k)) 
	     )
        if F2 > F_crit:
   	    reject = True
	else:
	    reject = False

    elif behavior == 'weighted_fmean':
	""" Calculate F-statistic for each band and take the 
	weighted mean of the statistic """
	F = w_av(F_stats, weights)
        if F > F_crit:
   	    reject = True
	else:
	    reject = False

    return reject

def w_av(data, weights):
    """ Return the weighted average """
    return np.average(data, weights=weights)

def get_weights(yatsm):
    """ Get weights based on average coefficient of 
    determination between bands """
    weights = 1 - 
	      (np.sum((np.corrcoef(yatsm.Y[yatsm.test_indices]))**2,axis=0) / 
	      len(yatsm.test_indices))
    return weights

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

	#Calculate critical value for test
        F_crit = scipy.stats.f.ppf(1 - alpha, k, n - 2 * k)
	F_stats = []

	#Parameterize this
	commission_method = 'CHOW'

	if commission_method == 'CHOW':
	    reject = do_chowda(
			yatsm, m_1_start, m_1_end, 
			m_2_start, m_2_end, m_r_start, 
			m_r_end, models, behavior, k,n, 
			F_crit)

        elif commission_method == 'MANOVA':
	    #Ignore. Testing purposes only
	    reject = do_manova(
			yatsm, m_1_start, m_1_end, 
			m_2_start, m_2_end, m_r_start, 
			m_r_end, models)

	elif commission_method == "HOTELLING":
	    #Ignore. Testing purposes only
	    reject = do_hotelling(
			yatsm, m_1_start, m_1_end, 
			m_2_start, m_2_end, m_r_start, 
			m_r_end, models)
	
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
            yatsm.models = yatsm.fit_models(
				yatsm.X[m_r_start:m_r_end, :],
                                yatsm.Y[:, m_r_start:m_r_end],
				bands=yatsm.test_indices)

            for i_m, _m in enumerate(yatsm.models):
                m_new['coef'][:, i_m] = _m.coef
                m_new['rmse'][i_m] = _m.rmse

            models.append(m_new)

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
        raise ValueError('`behavior` must be "any" , "mode", 
			 "mean", or "all"')

    if not indices:
        indices = model.test_indices

    if not np.all(np.in1d(indices, model.test_indices)):
        raise ValueError('`indices` must be a subset of '
                         '`model.test_indices`')

    omission = np.zeros((model.record.size, len(indices)), 
		        dtype=bool)
    models = []
    for i, r in enumerate(model.record):
        # Skip if no model fit
        if r['start'] == 0 or r['end'] == 0:
            continue
        # Find matching X and Y in data
        index = np.where(
            (model.dates >= min(r['start'], r['end'])) &
            (model.dates <= max(r['end'], r['start'])))[0]

	#Continue if there are not enough observations. 
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
	    #Save OLS and CUSUM outputs, if desired
	    saveols = False
	    if output: 
		if saveols:
		    do_saveols(r, ols, _X, _Y, ylim, output) 
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
		#Optimize bounded break detection using scipy's minimize_scalar
		opt_break = minimize_scalar(
				opt_breakdetection, bounds=(model.min_obs,max_ind), 
				method='bounded', args=(params,), options={'disp': True, 
				'maxiter': 50})
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


