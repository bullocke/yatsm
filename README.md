# Near-Real-Time version of Yet Another Timeseries Model (YATSM)

## About
This is a version of Chris Holden's Yet Another Timeseries Model (YATSM) algorithm optimized for the use with daily MODIS data. At the moment, YATSM and the near-real-time branch of YATSM have slightly diverged. The most up-to-date YATSM repository, managed by Ceholden,  can be found [here](https://www.github.com/ceholden/yatsm). The full documentation for YATSM is available [here](http://ceholden.github.io/yatsm/).

## Installation


It is strongly encouraged that you install YATSM into an isolated environment, either using [`virtualenv`](https://virtualenv.pypa.io/en/latest/) for `pip` installs or a separate environment using [`conda`](http://conda.pydata.org/docs/), to avoid dependency conflicts with other software.

This package requires an installation of [`GDAL`](http://gdal.org/), including the Python bindings. Note that [`GDAL`](http://gdal.org/) version 2.0 is not yet tested (it probably works, but I haven't tried GDAL 2.x), but recent 1.x versions (likely 1.8+) should work. [`GDAL`](http://gdal.org/) is not installable solely via `pip` and needs to be installed prior to following the `pip` instructions. If you follow the instructions for [`conda`](http://conda.pydata.org/docs/), you will not need to install `GDAL` on your own because [`conda`](http://conda.pydata.org/docs/) packages a compiled copy of the `GDAL` library (yet another reason to use [`conda`](http://conda.pydata.org/docs/)!).

### pip
The basic dependencies for YATSM are included in the `requirements.txt` file which is  by PIP as follows:

``` bash
    pip install -r requirements.txt
```

Additional dependencies are required for some timeseries analysis algorithms or for accelerating the computation in YATSM. These requirements are separate from the common base installation requirements so that YATSM may be more modular:

* Long term mean phenological calculations from Melaas *et al.*, 2013
    * Requires the R statistical software environment and the `rpy2` Python to R interface
    * `pip install -r requirements/pheno.txt`
* Computation acceleration
    * GLMNET Fortran wrapper for accelerating Elastic Net or Lasso regularized regression
    * Numba for speeding up computation through just in time compilation (JIT)
    * `pip install -r requirements/accel.txt`

A complete installation of YATSM, including acceleration dependencies and additional timeseries analysis dependencies, may be installed using the `requirements/all.txt` file:

``` bash
    pip install -r requirements/all.txt
```

Finally, install YATSM:

``` bash
    #Install YATSM:
    pip install .
```

### Conda
Requirements for YATSM may also be installed using [`conda`](http://conda.pydata.org/docs/), Python's cross-platform and platform agnostic binary package manager from [ContinuumIO](http://continuum.io/). [`conda`](http://conda.pydata.org/docs/) makes installation of Python packages, especially scientific packages, a breeze because it includes compiled library dependencies that remove the need for a compiler or pre-installed libraries.

Installation instructions for `conda` are available on their docs site [conda.pydata.org](http://conda.pydata.org/docs/get-started.html)

Since [`conda`](http://conda.pydata.org/docs/) makes installation so easy, installation through [`conda`](http://conda.pydata.org/docs/) will install all non-developer dependencies. Install YATSM using [`conda`](http://conda.pydata.org/docs/) into an isolated environment by using the `environment.yaml` file as follows:

``` bash
    # Install
    conda env create -n yatsm -f environment.yaml
    # Activate
    source activate yatsm
```

And as with pip, you need to instal YATSM:

``` bash
    #Install YATSM:
    pip install .
```
