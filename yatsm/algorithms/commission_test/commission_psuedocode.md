# Psuedo-code for commission test for YATSM/CCDC

## Psuedo-code without documentation

```python
function do_chowda:
    m_1_rss = empty numpy array the length of number of test bands
    m_2_rss = empty numpy array the length of number of test bands
    m_r_rss = empty numpy array the length of number test bands

    for band in every test band:
        m_1_rss[band] = residual sum of squares for time period of m_1_rss
        m_2_rss[band] = residual sum of squares for time period of m_2_rss
        m_r_rss[band] = residual sum of squares for time period of m_r_rss

    weights = perform function "get_weights"

    F = F statistic using weighted means from function "w_av" for each rss within formula
    if F > F_crit:
        reject = True
        dont merge
    else:
        merge

    return reject

function w_av:
    return weighted average based on weights variable

function get_weights:
    return band weights

function commission_test:
    if number of models = 1:
        return model

    k = number of regression coefficients
    models = empty list
    merged = False

    for every model containig a break and subsequent model in time series:
        if the previous models were merged:
            m_1 = previous model
        else: 
            m_1 = current model in loop
        m_2 = next time sequential model in time series

        m_1_start = start of model 1
        m_1_end = end of model 1
        m_2_start = start of model 1
        m_2_end = end of model 2 
        m_r_start = start of restricted model
        m_r_end = end of restricted model

        if either model periods <= (k + 2):
            continue

        n = number of observations covering both models
        F_crit = F test critical value

        reject = perform function "do_chowda"

        if reject:
            if these are the first 2 models tested for pixel:
                append m_1 to models
            append m_2 to models
        else:
            m_new = empty time series record template

            if these arent the first two models were testing:
                delete last model in models

            m_new['start'] = start index of the first model
            m_new['end'] = end of the second model
            m_new['break'] = break date (if any) for m_2

            re-estimate regression model parameters based on new start/end of segments

            for each model in the time series:
                copy coefficients and rmse to model record

            merged = True

    return models as numpy array
```
## Psuedo-code with documentation

### function do_chowda:

Function for performing the modified Chow Test [(Chow 1960)](http://www.jstor.org/stable/1910133?seq=1#page_scan_tab_contents) within the CCDC Commission Test for false breaks in a time series. The Chow Test is used to find false positive, spurious, or unnecessary breaks in the timeseries by comparing the effectiveness of two separate adjacent models with one single model that spans the entire time period.

Chow test is described:

![equation](https://raw.githubusercontent.com/bullocke/yatsm/postprocess/yatsm/algorithms/commission_test/equation.png)

where:
- RSS_r: the RSS of the combined, or, restricted model
- RSS_1: is the RSS of the first model
- RSS_2: is the RSS of the second model
- k: is the number of model parameters
- n: is the number of total observations

The restricted model corresponds to the model using the pooled observations
spanning full test period. The model is restricted in that the coefficients are
assumed to be equal for the entirety of the time period. To test the null
hypothesis that the restrictions on the model are true (and there should not
be two seperate groups, or in our case a model break), we calculate the Chow
Test statistic, which follows an F-distribution. Accepting the null hypothesis
therefore signifies the restrictions are valid, and we merge the models.

Because we look for change in multiple bands, the Chow Test statistic must
be collapsed across test bands.

First, create containers for the the residual sum of squares for the 2 models to test (m_1_rss, m_2_rss) and the restricted model (m_r_rss)

```python
m_1_rss = empty numpy array the length of number of test bands
m_2_rss = empty numpy array the length of number of test bands
m_r_rss = empty numpy array the length of number test bands
```

Next, loop every each band and calculate the residual sum of squares for the 3 model periods.
```python
for band in every test band:
    m_1_rss[band] = residual sum of squares for time period of m_1_rss
    m_2_rss[band] = residual sum of squares for time period of m_2_rss
    m_r_rss[band] = residual sum of squares for time period of m_r_rss
```

To collapse the statistic across bands, we need to get the correlation weights using the function 'get_weights'
```python
weights = perform function "get_weights"
```

How to collapse test statistic across bands? There are various ways of doing it, but in testing we used the band weights to calculate the weighted average of each RSS, and used the means to calculate the Chow Test Statistic. If the test statistic exceeds the critical value, we reject the null hypothesis that the model restrictions are true and retain that there are two statistically seperate groups of data and the break is
confirmed. 
```python
F = F statistic using weighted means from function "w_av" for each rss within formula
if F > F_crit:
    reject = True
    dont merge
else:
    merge
```

Return whether the null hypothesis was rejected or not.
```python
return reject
```

### function w_av:

Function for returning a weighted averaged based on a weights variable

```python
return weighted average based on weights variable
```

### function get_weights:

Function for calculating the weights for each band. the weights are based on the the Pearson product-moment correlation coefficients using the original cloud-masked spectral data. The weights are defined for band b as: 1 - r<sub>b</sub>, with r being the correlation coefficient.

```python
return weights
```

### function commission_test:
Master function for testing whether to merge adjacent models due to incorrect changes detected by CCDC.

Check if there are multiple models to possibly merge
```python
if number of models = 1:
    return model
```

Create variables:
- k: number of regression coefficients.
- models: an empty list to hold the resulting models are running the Chow Test. 
- merged: Flag to whether the test models have been merged. 

```python
k = number of regression coefficients
models = empty list
merged = False
```

Loop over every model in time series that has a break

```python
for every model containing a break and subsequent model in time series:
    if the previous models were merged:
        m_1 = previous model
    else: 
        m_1 = current model in loop
    m_2 = next time sequential model in time series
```

Determine indices of data for the beginning and end of each time series model:
m_1, m_2, and restricted model
```python
    m_1_start = start of model 1
    m_1_end = end of model 1
    m_2_start = start of model 1
    m_2_end = end of model 2 
    m_r_start = start of restricted model
    m_r_end = end of restricted model
```

Check if there are enough observations to fit least squares regression
```python
    if either model periods <= (k + 2):
        continue
```

Calculate two variables, n and F_crit
```python
    n = number of observations covering both models
    F_crit = F test critical value
```

Perform the Chow Test. If the null hypothesis is rejected, retain the change, otherwise the models are merged. If the models are merged, new models corresponding to the CCDC regression parameters must be fit for each spectral band for the pooled data period. Either the two original models, or the new merged model must be added to the 'models' list. Model 2 (in the case they were not merged), or the new merged model are then used as model 1 for the next iteration (assuming there are more
models in the time series). 
```python
    reject = perform function "do_chowda"

    if reject:
        if these are the first 2 models tested for pixel:
            append m_1 to models
        append m_2 to models
    else:
        m_new = empty time series record template

        if these arent the first two models were testing:
            delete last model in models

        m_new['start'] = start index of the first model
        m_new['end'] = end of the second model
        m_new['break'] = break date (if any) for m_2

        re-estimate regression model parameters based on new start/end of segments

        for each model in the time series:
            copy coefficients and rmse to model record

        merged = True
```
Once weve tested all of the models in the time series, the updated list of models can be returned.
```python
return models as numpy array
```
