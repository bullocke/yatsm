""" Result post-processing utilities
This code represents the commission test for testing for false breaks
between two CCDC/YATSM models
"""
import logging

import numpy as np
import scipy.stats
from ..utils import date2index

logger = logging.getLogger('yatsm')

def do_chowda(yatsm, m_1_start, m_1_end,
          m_2_start, m_2_end, m_r_start,
          m_r_end, models, behavior,
          k, n, F_crit):
    """

    Merge adjacent records based on BU's version of Chow Tests for nested models.
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

    The restricted model corresponds to the model using the pooled observations
    spanning full test period. The model is restricted in that the coefficients are
    assumed to be equal for the entirety of the time period. To test the null
    hypothesis that the restrictions on the model are true (and there should not
    be two seperate groups, or in our case a model break), we calculate the Chow
    Test statistic, which follows an F-distribution. Accepting the null hypothesis
    therefore signifies the restrictions are valid, and we merge the models.

    Because we look for change in multiple bands, the Chow Test statistic must
    be collapsed across test bands.

    """
    F_stats = []
    # Allocate memory outside of loop
    m_1_rss = np.zeros(yatsm.test_indices.size)
    m_2_rss = np.zeros(yatsm.test_indices.size)
    m_r_rss = np.zeros(yatsm.test_indices.size)

    # Get sum of squared residuals: run np.linalg.listsq for:
    ## m_1_rss: the first model
    ## m_2_rss: the second model
    ## m_r_rss: a single model spanning entire time period

    # Since we are using multiple bands, iterate over test indices
    for i_b, b in enumerate(yatsm.test_indices):
        m_1_rss[i_b] = np.linalg.lstsq(yatsm.X[m_1_start:m_1_end, :],
                                       yatsm.Y[b, m_1_start:m_1_end])[1]
        m_2_rss[i_b] = np.linalg.lstsq(yatsm.X[m_2_start:m_2_end, :],
                                       yatsm.Y[b, m_2_start:m_2_end])[1]
        m_r_rss[i_b] = np.linalg.lstsq(yatsm.X[m_r_start:m_r_end, :],
                                       yatsm.Y[b, m_r_start:m_r_end])[1]
    # Calculate F stat for band
        F_band = (((m_r_rss[i_b] - (m_1_rss[i_b] + m_2_rss[i_b])) / k)
         / ((m_1_rss[i_b] + m_2_rss[i_b]) / (n - 2 * k)))
        F_stats.append(F_band)

    #Get weights for the mean based on average r^2 across bands
    weights = get_weights(yatsm)

    # How to collapse test statistic across bands? There are multiple possible ways, but
    # for testing we calculated the weighted means of each of the 3 rss across bands,
    # and used the means to calculate the Chow test statistic

    F2 = (
         ((w_av(m_r_rss, weights) - (w_av(m_1_rss, weights)
         + w_av(m_2_rss, weights))) / k ) /
         ((w_av(m_1_rss, weights) + w_av(m_2_rss, weights))
         / (n - 2 * k))
         )

    if F2 > F_crit:
        reject = True
    else:
        reject = False

    return reject

def w_av(data, weights):
    """ Return the weighted average """
    return np.average(data, weights=weights)

def get_weights(yatsm):
    """ Get weights based on average coefficient of determination between bands.
    Basically this is used to determine which bands are least correlated to the other
    bands based on the reflectance time series  """
    weights = 1 - (np.sum((np.corrcoef(yatsm.Y[yatsm.test_indices]))**2,axis=0) /
              len(yatsm.test_indices))
    return weights

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
    # Only continue if there are more than one model for this pixel time series
    if yatsm.record.size == 1:
        return yatsm.record

    # k = number of regression coefficients
    k = yatsm.record[0]['coef'].shape[0]

    models = []
    merged = False

    # Iterate over all the models for the pixel inthe time series
    for i in range(len(yatsm.record) - 1):
        # Need to find the first model
        if merged:
            m_1 = models[-1]
        else:
            m_1 = yatsm.record[i]

        # Find second model to test if it should be merged with the first
        m_2 = yatsm.record[i + 1]

        # Unrestricted model starts/ends
        # Start and end index of model 1 and model 2
        m_1_start = date2index(yatsm.dates, m_1['start'])
        m_1_end = date2index(yatsm.dates, m_1['end'])
        m_2_start = date2index(yatsm.dates, m_2['start'])
        m_2_end = date2index(yatsm.dates, m_2['end'])

        # Restricted start/end
        # Index of start of model 1 and end of model 2: The entire time period
        m_r_start = m_1_start
        m_r_end = m_2_end

        # Need enough obs to fit models (n > k)
        # For regression model n needs to be higher than # of model coefficients
        if (m_1_end - m_1_start) <= (k + 2) or (m_2_end - m_2_start) <= (k + 2):
            logger.debug('Too few obs (n <= k) to merge segment')
            merged = False
            if i == 0:
                models.append(m_1)
            models.append(m_2)
            continue

        # n = length of entire time period
        n = m_r_end - m_r_start

        #Calculate critical value for test based on coefficients k, length n, and alpha value
        F_crit = scipy.stats.f.ppf(1 - alpha, k, n - 2 * k)

        # Leaving open the option of other types of commisison tests, right now just Chow
        commission_method = 'CHOW'

        if commission_method == 'CHOW':
            # Reject and keep the change?
            reject = do_chowda(
                yatsm, m_1_start, m_1_end,
                m_2_start, m_2_end, m_r_start,
                m_r_end, models, behavior, k,n,
                F_crit)

            if reject:
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

            # Re-fit models and copy over attributes
            yatsm.models = yatsm.fit_models(
                                yatsm.X[m_r_start:m_r_end, :],
                                yatsm.Y[:, m_r_start:m_r_end])

            # Add coefficients and RMSE to records
            for i_m, _m in enumerate(yatsm.models):
                m_new['coef'][:, i_m] = _m.coef
                m_new['rmse'][i_m] = _m.rmse

            # add new model to list of models for pixel time series
            models.append(m_new)

            merged = True

    return np.array(models)



