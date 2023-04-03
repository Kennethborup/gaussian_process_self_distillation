import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor


class DistributionCentricGPR(GaussianProcessRegressor):
    """Distribution Centric Gaussian Process Regressor

    Note, this is a wrapper around the sklearn GaussianProcessRegressor and
    requires the exact same (incl. hyperparameters) kernel for all distillation steps.

    Parameters
    ----------
    alphas : array-like, shape (n_samples,)
        The alphas for the distribution centric GPR.

    **kwargs : dict
        The usual kwargs for the GaussianProcessRegressor.
    """

    def __init__(
        self,
        alphas,
        **kwargs,
    ):
        self.alphas = np.asarray(alphas)
        assert np.all(self.alphas > 0), "alphas must be positive"
        self.num_distillations = len(self.alphas)
        self.alpha = 1 / (np.sum(1 / self.alphas))
        super().__init__(alpha=self.alpha, **kwargs)
