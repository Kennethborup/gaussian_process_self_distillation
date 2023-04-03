<div align="center">

  [![Paper][paper-shield]][paper-url]
  [![MIT License][license-shield]][license-url]
  [![Lint][lint-shield]][lint-url]
  [![Build][build-shield]][build-url]

  <h1 align="center">Gaussian Process Self-Distillation (GPSD)</h1>

  <p align="center">
    Self-distillation for Gaussian Process Regression and Gaussian Process Classification.
  </p>
  <a href="https://scholar.google.com/citations?user=KsFyMREAAAAJ&hl=en">Read the Paper»</a>
</div>



## About The Project

This is the official implementation of the paper [Self-Distillation for Gaussian Processes][paper-url] by [Kenneth Borup][linkedin-url] and Lars N. Andersen.

```bibtex
@article{borup2023GPSD,
  title={Self-Distillation for Gaussian Processes},
  author={Borup, Kenneth and Andersen, Lars N.},
  year={2023},
}
```

The package is designed to be used in a similar way to scikit-learn. The package includes two classes for Gaussian Process Regression and two classes for Gaussian Process Classification.


## Getting Started

To get a local copy up and running follow the simple example under <a  href="#installation">Installation</a>.
For usage examples, see <a  href="#usage">Usage</a> and be careful of training times for different methods (see <a  href="#training-speed">Training speed</a>). For more details on the methods, see the [paper][paper-url].

### Dependencies

Currently the implementation is reliant on the following dependencies:

* scikit-learn
* torch
* numpy
* scipy

They are all installed by default when installing the package.

### Installation<a name="installation"></a>
The package can be installed using pip from the github repository or by cloning the repository and installing the package locally.

Directly from github using pip
```sh
pip install git+https://github.com/kennethborup/gaussian_process_self_distillation.git
```

or clone the repository and install locally
```sh
git clone https://github.com/kennethborup/gaussian_process_self_distillation.git
cd gaussian_process_self_distillation
pip install .
```

Verify the installation by running the following command in your terminal window:
```sh
python -c "import gpsd; print(f'{gpsd.__version__} by {gpsd.__author__}')"
```

## Usage<a name="usage"></a>

The package is designed to be used in a similar way to scikit-learn. The package includes four classes:

- `DataCentricGPR` for **Data-Centric** Self-Distillation for Gaussian Process **Regression**
- `DistributionCentricGPR` for **Distribution-Centric** Self-Distillation for Gaussian Process **Regression**
- `DataCentricGPC` for **Data-Centric** Self-Distillation for Gaussian Process **Classification**
- `DistributionCentricGPC` for **Distribution-Centric** Self-Distillation for Gaussian Process **Classification**

Each method has `.fit`, and `.predict` methods that are similar to the scikit-learn API - see the [paper][paper-url] for more details on each method.

### Examples
Given some data `X` and `y`, the following examples fit different GPSD models using a predefined `kernel` from `sklearn` and predict on the same data.
I.e. first run
```python
import GPSD
from sklearn.gaussian_process.kernels import RBF

X, y = # some data
kernel = 1.0 * RBF()
```
then run one of the following examples.

**Data-Centric GP Regression**
```python
model = gpsd.DataCentricGPR(
    kernel=kernel,
    num_distillations=5,
    alphas=1e-2, # If a single value is given, it is used for all distillation steps
    optimize_mode="first", # only perform hyperparameter optimization in the first distillation step
    fit_mode="efficient", # use efficient computation of distillation steps (alternative is "naive")
)
model.fit(X, y)
y_pred = model.predict(X)
```

**Distribution-Centric GP Regression**
```python
model = gpsd.DistributionCentricGPR(
    kernel=kernel,
    alphas=[1e-2]*5, # Must be a list of the amount of distillation steps
)
model.fit(X, y)
y_pred = model.predict(X)
```

**Data-Centric GP Classification**
```python
model = gpsd.DataCentricGPC(
    kernel=kernel,
    num_distillations=5,
    optimize_mode="first", # only perform hyperparameter optimization in the first distillation step
)
model.fit(X, y)
y_pred = model.predict(X)
```

**Distribution-Centric GP Classification**
```python
model = gpsd.DistributionCentricGPC(
    kernel=kernel,
    num_distillations=5,
    fit_mode="approx", # use approximate computation of distillation steps (alternative is "exact")
)
model.fit(X, y)
y_pred = model.predict(X)
```

## Training speed<a name="training-speed"></a>

Note, the training speed of different methods vary a lot (especially for large number of distillation steps). The following plots are training times for each of the methods on a simulated dataset, when fitted on a Mac M1 Pro CPU. The training time is presented relative to an ordinary fit of a GPR/GPC model using `sklearn` . The training time is measured across 30 replications, and the mean and 10%/90% quantiles are plotted.

![Training Speed][training-speed-image]


## Citation

If you use this code in your research, please cite the following paper:

```bibtex
@article{borup2023GPSD,
  title={Self-Distillation for Gaussian Processes},
  author={Borup, Kenneth and Andersen, Lars N.},
  year={2023},
}
```

## License

Distributed under the MIT License. See `LICENSE.txt` for more information.


<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[paper-shield]: https://img.shields.io/badge/ArXiv-Paper-red?style=for-the-badge
[paper-url]: https://scholar.google.com/citations?user=KsFyMREAAAAJ&hl=en
[training-speed-image]: figures/training_time_relative.png
[build-shield]: https://img.shields.io/github/actions/workflow/status/kennethborup/gaussian_process_self_distillation/build.yml?style=for-the-badge
[build-url]: https://github.com/Kennethborup/gaussian_process_self_distillation
[lint-shield]: https://img.shields.io/github/checks-status/kennethborup/gaussian_process_self_distillation/main?style=for-the-badge
[lint-url]: https://github.com/Kennethborup/gaussian_process_self_distillation
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=for-the-badge
[license-url]: https://github.com/othneildrew/Best-README-Template/blob/master/LICENSE.txt
[linkedin-url]: https://www.linkedin.com/in/borupkenneth/
