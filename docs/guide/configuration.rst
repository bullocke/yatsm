.. _guide_model_config:

===================
Model Configuration
===================

The issue tracker on Github is being used to track additions to this
documentation section. Please see
`ticket 37 <https://github.com/ceholden/yatsm/issues/37>`_.


Configuration File
------------------

The batch running script uses an `YAML
file <https://en.wikipedia.org/wiki/YAML>`_ to
parameterize the run. The YAML file uses several sections:

1. ``dataset`` describes dataset attributes common to all analysis
2. ``YATSM`` describes model parameters common to all analysis and declares what change detection algorithm should be run
3. ``CCDCesque`` describes model parameters specific the the 'CCDCesque' algorithm implementation. 
4. ``Regression estimators`` locations of pre-pickled scikit-learn regression models
5. ``classification`` describes classification training data inputs
6. ``phenology`` describes phenology fitting parameters

The following tables describes the meanings of the parameter and values used
in the configuration file used in YATSM. Any parameters left blank will be
interpreted as ``None`` (e.g., ``cache_line_dir =``).

Version
-------

This states the version of YATSM that corresponds to the configuration file being used. 


Dataset Parameters
------------------

These parameters generally describe the format of the input dataset. 

+----------------------------+-------------+-----------------------------------------------------------------------------------------------------------------------------------------+
| Parameter                  | Data Type   | Explanation                                                                                                                             |
+============================+=============+=========================================================================================================================================+
| ``input_file``          | ``str``     | CSV file containing sorted image list. See notes below for addition details.                                                                                     |
+----------------------------+-------------+-----------------------------------------------------------------------------------------------------------------------------------------+
| ``date_format``            | ``str``    | Format of dates in input file. Default: "%Y&j"                                                                                  |
+----------------------------+-------------+-----------------------------------------------------------------------------------------------------------------------------------------+
| ``output``            | ``str``    | Location to store YATSM results                                                                                  |
+----------------------------+-------------+-----------------------------------------------------------------------------------------------------------------------------------------+
| ``output_prefix``            | ``str``    | Prefix for saved result files. Default: "yatsm_r"                                                                                 |
+----------------------------+-------------+-----------------------------------------------------------------------------------------------------------------------------------------+
| ``n_bands``            | ``int``    | Total number of bands in input files. Default: "8"                                                                                  |
+----------------------------+-------------+-----------------------------------------------------------------------------------------------------------------------------------------+
| ``mask_band``            | ``int``    | Index of mask band (indexed on 1). Default: "8"                                                                                  |
+----------------------------+-------------+-----------------------------------------------------------------------------------------------------------------------------------------+
| ``mask_values``            | ``list``    | Values corresponding to masked data in the mask band. Default: "[2, 3, 4, 255]"                                                                                  |
+----------------------------+-------------+-----------------------------------------------------------------------------------------------------------------------------------------+
| ``min_values``            | ``int/list``    | Minimum value allowed. Integer for one band or list for each band. Default: "0"                                                                                  |
+----------------------------+-------------+-----------------------------------------------------------------------------------------------------------------------------------------+
| ``max_values``            | ``int/list``    | Maximum value allowed. Integer for one band or list for each band. Default: "10000"                                                                                   |
+----------------------------+-------------+-----------------------------------------------------------------------------------------------------------------------------------------+
| ``green_band``            | ``int``    | Index of green band for multi-temporal cloud masking (indexed on 1). Default: "2"                                                                                  |
+----------------------------+-------------+-----------------------------------------------------------------------------------------------------------------------------------------+
| ``swir_band``            | ``int``    | Index of first short-wave infrared band for multi-temporal cloud masking (indexed on 1). Default: "5"                                                                                    |
+----------------------------+-------------+-----------------------------------------------------------------------------------------------------------------------------------------+
| ``use_bip_reader``            | ``bool``    | Use BIP reader for reading data. Default: "true"                                                                                  |
+----------------------------+-------------+-----------------------------------------------------------------------------------------------------------------------------------------+
| ``cache_line_dir``            | ``str``    | Directory location for caching dataset lines.                                                                                 |
+----------------------------+-------------+-----------------------------------------------------------------------------------------------------------------------------------------+

**Note**: you can use ``scripts/gen_date_file.sh`` to generate the CSV
file for ``input_file``.

YATSM Parameters
----------------

These parameters common to all timeseries analysis models within the YATSM package. 


+----------------------------+-------------+-----------------------------------------------------------------------------------------------------------------------------------------+
| Parameter                  | Data Type   | Explanation                                                                                                                             |
+============================+=============+=========================================================================================================================================+
| ``algorithm``          | ``str``     | Time series algorithm to use. Currently, the only one implemented is 'CCDCesque'                                                                                     |
+----------------------------+-------------+-----------------------------------------------------------------------------------------------------------------------------------------+
| ``prediction``          | ``str``     | Regression technique used for model prediction.                                                                                      |
+----------------------------+-------------+-----------------------------------------------------------------------------------------------------------------------------------------+
| ``design_matrix``          | ``str``     | Patsy-formed design matrix for regression formula.                                                                                     |
+----------------------------+-------------+-----------------------------------------------------------------------------------------------------------------------------------------+
| ``reverse``          | ``bool``     | Run model in reverse.                                                                                      |
+----------------------------+-------------+-----------------------------------------------------------------------------------------------------------------------------------------+
| ``common_alpha``          | ``bool``     | Alpha value for Chow commission test. Leave blank to ignore test.                                                                                      |
+----------------------------+-------------+-----------------------------------------------------------------------------------------------------------------------------------------+
| ``prefix``          | ``list``     | Prefix to use for the coefficients after model refitting.                                                                                      |
+----------------------------+-------------+-----------------------------------------------------------------------------------------------------------------------------------------+
| ``prediction``          | ``list``     | Prediction to use when model refitting.                                                                                      |
+----------------------------+-------------+-----------------------------------------------------------------------------------------------------------------------------------------+
| ``stay_regularized``          | ``list``     |                                                                                      |
+----------------------------+-------------+-----------------------------------------------------------------------------------------------------------------------------------------+

CCDCesque
---------

Parameters for CCDCesque algorithm in YATSM. 

+----------------------------+-------------+-----------------------------------------------------------------------------------------------------------------------------------------+
| Parameter                  | Data Type   | Explanation                                                                                                                             |
+============================+=============+=========================================================================================================================================+
| ``consecutive``          | ``int``     | Consecutive observations above threshold to be flagged as change.                                                                                      |
+----------------------------+-------------+-----------------------------------------------------------------------------------------------------------------------------------------+
| ``threshold``          | ``int``     | Threshold on the RMSE-normalized residuals to be considered change.                                                                                      |
+----------------------------+-------------+-----------------------------------------------------------------------------------------------------------------------------------------+
| ``min_obs``          | ``int``     | Minimum observations needed to start a model.                                                                                      |
+----------------------------+-------------+-----------------------------------------------------------------------------------------------------------------------------------------+
| ``min_rmse``          | ``int``     | Minimum RMSE to use in change detection. If model RMSE is too low this value will be used.                                                                                      |
+----------------------------+-------------+-----------------------------------------------------------------------------------------------------------------------------------------+
| ``test_indices``          | ``list``     | Indices of bands to be used for change detection (Indexed on 0).                                                                                      |
+----------------------------+-------------+-----------------------------------------------------------------------------------------------------------------------------------------+
| ``retrain_time``          | ``int``     | Number of days before the model is retrained.                                                                                      |
+----------------------------+-------------+-----------------------------------------------------------------------------------------------------------------------------------------+
| ``screening``          | ``str``     | Screening method for multi-temporal cloud and cloud shadow detection.                                                                                      |
+----------------------------+-------------+-----------------------------------------------------------------------------------------------------------------------------------------+
| ``screening_crit``          | ``int``     | Multi-temporal screening threshold. Values above this will be removed from analysis.                                                                                      |
+----------------------------+-------------+-----------------------------------------------------------------------------------------------------------------------------------------+
| ``slope_test``          | ``bool``     | Test to ensure models do not begin with a large slope.                                                                                      |
+----------------------------+-------------+-----------------------------------------------------------------------------------------------------------------------------------------+
| ``remove_noise``          | ``bool``     | Remove observations that are above the threshold but not flagged as change.                                                                                      |
+----------------------------+-------------+-----------------------------------------------------------------------------------------------------------------------------------------+
| ``dynamic_rmse``          | ``bool``     | Have the RMSE be a function of the time of year.                                                                                      |
+----------------------------+-------------+-----------------------------------------------------------------------------------------------------------------------------------------+

Regression estimators
---------------------

Locations of pre-pickled scikit-learn regression estimators. These estimators are referenced in the perferences in the above sections. 

Phenology
---------

The option for long term mean phenology calculation is an optional addition to `YATSM`. As such, visit :ref:`the phenology guide page <guide_phenology>` for configuration options.

Classification
--------------

The scripts included in YATSM which perform classification utilize a
configuration INI file that specify which algorithm will be used and the
parameters for said algorithm. The configuration details specified along
with the dataset and YATSM algorithm options deal with the training
data, not the algorithm details. These training data configuration
options include:

+----------------------------+-------------+-----------------------------------------------------------------------------------------------------------------------------------------+
| Parameter                  | Data Type   | Explanation                                                                                                                             |
+============================+=============+=========================================================================================================================================+
| ``training_data``          | ``str``     | Training data raster image containing labeled pixels                                                                                    |
+----------------------------+-------------+-----------------------------------------------------------------------------------------------------------------------------------------+
| ``mask_values``            | ``list``    | Values within the training data image to mask or ignore                                                                                 |
+----------------------------+-------------+-----------------------------------------------------------------------------------------------------------------------------------------+
| ``training_start``         | ``str``     | Earliest date that training data are applicable. Training data labels will be paired with models that begin at least before this date   |
+----------------------------+-------------+-----------------------------------------------------------------------------------------------------------------------------------------+
| ``training_end``           | ``str``     | Latest date that training data are applicable. Training data labels will be paired with models that end at least after this date        |
+----------------------------+-------------+-----------------------------------------------------------------------------------------------------------------------------------------+
| ``training_date_format``   | ``str``     | Format specification that maps ``training_start`` and ``training_end`` to a Python datetime object (e.g., ``%Y-%m-%d``)                 |
+----------------------------+-------------+-----------------------------------------------------------------------------------------------------------------------------------------+
| ``cache_xy``               | ``str``     | Filename used for caching paired X features and y training labels                                                                       |
+----------------------------+-------------+-----------------------------------------------------------------------------------------------------------------------------------------+

Example
-------

An example template of the parameter file is located within
``examples/p013r030/p013r030.yaml``:

.. literalinclude:: ../../examples/p013r030/p013r030.yaml
   :language: yaml
