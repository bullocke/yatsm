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
import matplotlib.pyplot as plt
from ..regression.diagnostics import rmse
from ..utils import date2index

#logger = logging.getLogger('yatsm')


# POST-PROCESSING
def commission_test(yatsm, alpha=0.10):
    """ Merge adjacent records based on Chow Tests for nested models

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

    # Allocate memory outside of loop
    m_1_rss = np.zeros(yatsm.test_indices.size)
    m_2_rss = np.zeros(yatsm.test_indices.size)
    m_r_rss = np.zeros(yatsm.test_indices.size)

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
        if (m_1_end - m_1_start) <= k or (m_2_end - m_2_start) <= k:
            logger.debug('Too few obs (n <= k) to merge segment')
            merged = False
            if i == 0:
                models.append(m_1)
            models.append(m_2)
            continue

        n = m_r_end - m_r_start
        F_crit = scipy.stats.f.ppf(1 - alpha, k, n - 2 * k)
        for i_b, b in enumerate(yatsm.test_indices):
            m_1_rss[i_b] = np.linalg.lstsq(yatsm.X[m_1_start:m_1_end, :],
                                           yatsm.Y[b, m_1_start:m_1_end])[1]
            m_2_rss[i_b] = np.linalg.lstsq(yatsm.X[m_2_start:m_2_end, :],
                                           yatsm.Y[b, m_2_start:m_2_end])[1]
            m_r_rss[i_b] = np.linalg.lstsq(yatsm.X[m_r_start:m_r_end, :],
                                           yatsm.Y[b, m_r_start:m_r_end])[1]

        # Collapse RSS across all test indices for F statistic
        F = (
            ((m_r_rss.mean() - (m_1_rss.mean() + m_2_rss.mean())) / k) /
            ((m_1_rss.mean() + m_2_rss.mean()) / (n - 2 * k))
        )
        if F > F_crit:
            # Reject H0 and retain change
            # Only add in previous model if first model
            if i == 0:
                models.append(m_1)
            models.append(m_2)
            merged = False
        else:
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
                             yatsm.Y[:, m_r_start:m_r_end])
            for i_m, _m in enumerate(yatsm.models):
                m_new['coef'][:, i_m] = _m.coef
                m_new['rmse'][i_m] = _m.rmse

            if 'magnitude' in yatsm.record.dtype.names:
                # Preserve magnitude from 2nd model that was merged
                m_new['magnitude'] = m_2['magnitude']

            models.append(m_new)

            merged = True

    return np.array(models)


def omission_test(model, crit=0.01, behavior='ANY', indices=None):
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
    if behavior.lower() not in ['any', 'all']:
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

#	combine_bands = True
#	if combine_bands: 
#	    residuals = np.zeros((len(_X[:,1]),len(indices)))
#	    scaledresiduals = np.zeros((len(_X[:,1]),len(indices)))
#	    for i_b, b in enumerate(indices):
#                ols = sm.OLS(_Y[b, :], _X).fit()
#                residuals[:,i_b] = ols.resid
#                scaledresiduals[:,i_b] = residuals[:,i_b] / np.sqrt(((residuals[:,i_b]) ** 2).mean())
#	    resid_combined = np.linalg.norm(scaledresiduals,axis=1)
#            test = sm.stats.diagnostic.breaks_cusumolsresid(
#                resid_combined, _X.shape[1])
#	    test_stats = sm.stats.diagnostic.recursive_olsresiduals(
#                ols, _X.shape[1])
#            redo_models = True

	tst = []
        for i_b, b in enumerate(indices):
            # Create OLS regression
            ols = sm.OLS(_Y[b, :], _X).fit()
            # Perform CUMSUM test on residuals
            test = sm.stats.diagnostic.breaks_cusumolsresid(
                ols.resid, _X.shape[1])
	    test_stats = sm.stats.diagnostic.recursive_olsresiduals(
                ols, _X.shape[1])
	    tst.append(test_stats)
            if test[1] < crit:
	 #       print 'missed break' 
                omission[i, i_b] = True
            else:
	#	print 'did not miss break' 
                omission[i, i_b] = False
        # Collapse band answers according to `behavior`
        if (behavior.lower() == 'any') and (np.any(omission[i,:])):
	    redo_models = True
        elif (behavior.lower() == 'all') and (np.all(omission[i,:])):
	    redo_models = True
        else:
	    redo_models = False
	    models.append(r)

        if redo_models: 
	    for ind in np.nonzero(omission[i,:])[0]:
		nobs = _X[:,1].shape[0]
		max_ind = nobs - model.min_obs  
		params=(_Y[b,:], _X, nobs, model.min_obs)
		opt_break = minimize_scalar(opt_breakdetection, bounds=(model.min_obs,max_ind), method='bounded', args=(params,), options={'disp': True, 'maxiter': 100})
		breakindex[ind] = int(opt_break.x)

		#breakindex[ind] = opt_breakdetection(_Y[b,:], _X, model.min_obs)
#		breakindex[ind] = find_breakpoints(_Y[b, :], _X, 1,trim=model.min_obs)[0][0]
#	    import pdb; pdb.set_trace()
            _breakindex = np.min(breakindex[breakindex > 0])


     	    breakdate = _X[:,1][_breakindex]

            m_new_1 = np.copy(model.record_template)[0]
            m_new_2 = np.copy(model.record_template)[0]

            m_new_1['start'] = _X[:,1][0]
            m_new_1['end'] = _X[:,1][_breakindex - 1]
            m_new_1['break'] = _X[:,1][_breakindex - 1]
	    if r['status'] == 3:
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

