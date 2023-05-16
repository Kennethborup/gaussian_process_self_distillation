import inspect

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor


def clean_kwargs(func, kwargs):
    """Checks if the kwargs are valid for the function"""
    cleaned_kwargs = {}
    sig = inspect.signature(func)
    if not any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values()):
        for key, val in kwargs.items():
            if key in sig.parameters.keys():
                cleaned_kwargs[key] = val
    return cleaned_kwargs


class DistributionCentricGPR(GaussianProcessRegressor):
    """Distribution Centric Gaussian Process Regressor

    Note, this is a wrapper around the sklearn GaussianProcessRegressor and
    requires the exact same (incl. hyperparameters) kernel for all distillation steps.

    Parameters
    ----------
    alpha : float
        The alpha for the distribution centric GPR (i.e. the precomputed alpha for all steps).

    alphas : array-like, shape (num_distillations,)
        The alphas for the distribution centric GPR (i.e. list of alphas for each step).

    **kwargs : dict
        The usual kwargs for the GaussianProcessRegressor.
    """

    def __init__(
        self,
        kernel=None,
        alpha=[1e-10],
        alphas=None,
        optimizer="fmin_l_bfgs_b",
        n_restarts_optimizer=0,
        normalize_y=False,
        copy_X_train=True,
        random_state=None,
        **kwargs,
    ):
        if alphas is not None:
            self.alphas = np.asarray(alphas)
            assert np.all(self.alphas > 0), "alphas must be positive"
            self.num_distillations = len(self.alphas)
            self.alpha = 1 / (np.sum(1 / self.alphas))
        else:
            self.alphas = None
            self.alpha = alpha
            self.num_distillations = None
        super().__init__(
            kernel=kernel,
            alpha=self.alpha,
            optimizer=optimizer,
            n_restarts_optimizer=n_restarts_optimizer,
            normalize_y=normalize_y,
            copy_X_train=copy_X_train,
            random_state=random_state,
        )
