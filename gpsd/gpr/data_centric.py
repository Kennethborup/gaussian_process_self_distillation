import warnings
from operator import itemgetter

import numpy as np
from scipy.linalg import cholesky, solve_triangular
from sklearn.base import clone
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import ConstantKernel as C
from sklearn.preprocessing._data import _handle_zeros_in_scale
from sklearn.utils import check_random_state

GPR_CHOLESKY_LOWER = True


class DataCentricGPR(GaussianProcessRegressor):
    """Data Centric Gaussian Process Regressor

    Note, this is a wrapper around the sklearn GaussianProcessRegressor
    and requires the same kernel for all distillation steps. The kernel
    can be hyperparameter optimized on the first step against the true
    targets and then fixed for the rest of the distillation steps, or it
    can be optimized on all steps against the previous distillation
    predictions. The latter is only possible for the naive (and slow)
    mode.

    Parameters
    ----------
    kernel : Kernel object, default=None
        The kernel specifying the covariance function of the GP. If None is
        passed, the kernel "1.0 * RBF(1.0)" is used as default. Note that
        the kernel's hyperparameters are optimized during fitting.

    alphas : float or array-like, shape (n_samples,), default=1e-10
        The alphas for the data centric GPR. If a single value is passed,
        it is used for all distillation steps. If an array is passed, it
        must have the same length as num_distillations.

    optimizer : string or callable, default="fmin_l_bfgs_b"
        Can either be one of the internally supported optimizers for
        optimizing the kernel's parameters, specified by a string, or an
        externally defined optimizer passed as a callable.

    n_restarts_optimizer : int, default=0
        The number of restarts of the optimizer for finding the kernel's
        parameters which maximize the log-marginal likelihood. The first
        run of the optimizer is performed from the kernel's initial
        parameters, the remaining ones (if any) from thetas sampled
        log-uniform randomly from the space of allowed theta-values.
        If greater than 0, all bounds must be finite. Note that n_restarts_optimizer=0
        implies that one run is performed.

    normalize_y : bool, default=False
        Whether the target values y are normalized, i.e., the mean of the
        observed target values become zero. This parameter should be set
        to True if the target values' mean is expected to differ
        considerable from zero. When enabled, the normalization effectively
        modifies the GP's prior based on the data, which contradicts the
        likelihood principle. Use of this parameter is therefore not
        recommended.

    copy_X_train : bool, default=True
        If True, a persistent copy of the training data is stored in the
        object. Otherwise, just a reference to the training data is
        stored, which might cause predictions to change if the data is
        modified externally.

    random_state : int, default=None
        If an integer is given, it fixes the seed. Defaults to the global numpy random
        number generator.

    num_distillations : int, default=2
        The number of distillation steps. Must be at least 1, which corresponds
        to the original GPR and no distillation.

    fit_mode : str, default="efficient"
        The mode for fitting the GPR. Can be either "efficient" or "naive".
        In efficient mode the GPR is updated from the SVD decomposition, while
        in naive mode the GPR is updated by fitting a new GPR on the previous
        predictions. The efficient mode is much faster, but only works for
        optimize_mode=None or optimize_mode="first". The naive mode is much
        slower, but works for all optimize_mode options.

    optimize_mode : str, default=None
        The mode for optimizing the kernel hyperparameters. Can be either
        None, "first" or "all". If None, the kernel hyperparameters are
        not optimized. If "first", the kernel hyperparameters are optimized
        on the first distillation step against the true targets. If "all",
        the kernel hyperparameters are optimized on all distillation steps
        against the previous distillation predictions. Note that the efficient
        mode only works for optimize_mode=None or optimize_mode="first".
        You need to choose compatible fit_mode and optimize_mode options.
    """

    def __init__(
        self,
        kernel=None,
        alphas=1e-10,
        optimizer="fmin_l_bfgs_b",
        n_restarts_optimizer=0,
        normalize_y=False,
        copy_X_train=True,
        random_state=None,
        num_distillations=2,
        fit_mode="efficient",
        optimize_mode=None,
        compute_L_=True,
    ):
        self.kernel_orig = (
            kernel.clone_with_theta(kernel.theta) if kernel is not None else kernel
        )
        self.num_distillations = num_distillations
        if self.num_distillations < 1:
            raise ValueError("Number of distillation steps must be at least 1.")

        self.alphas = alphas

        self.fit_mode = fit_mode
        if self.fit_mode == "efficient" and self.num_distillations == 1:
            # self.fit_mode = "naive"
            warnings.warn(
                "Efficient mode is not necessary for 1 distillation step. Consider switching to naive mode."
            )

        self.optimize_mode = optimize_mode
        if self.optimize_mode is None:
            if optimizer is not None:
                # Currently we do not support hyperparameter optimization as the log-likelihood is computed on the previous distillation predictions
                # optimizer = None
                raise NotImplementedError(
                    "Optimizer should be set to None for self-distillation and optimize_mode=None."
                )
        elif self.optimize_mode == "first":
            # Perform optimization on first step against true targets
            if optimizer is None:
                # optimizer = "fmin_l_bfgs_b"
                raise NotImplementedError(
                    "Optimizer should be set to fmin_l_bfgs_b for the first step optimization."
                )
        elif self.optimize_mode == "all":
            # Perform optimzation on all steps (first step against true targets, rest against previous distillation predictions)
            if fit_mode == "efficient":
                raise NotImplementedError(
                    "optimize_mode='all' is not implemented for efficient mode. Use either fit_mode='naive' or optimize_mode='first' or None."
                )
            if optimizer is None:
                # optimizer = "fmin_l_bfgs_b"
                raise NotImplementedError(
                    "Optimizer should be set to fmin_l_bfgs_b for all steps."
                )
        else:
            raise NotImplementedError(
                f"optimize_mode must be either None, 'first', or 'all', not {self.optimize_mode}"
            )

        self.optimizer = optimizer
        self.main_kwargs = {
            "optimizer": self.optimizer,
            "n_restarts_optimizer": n_restarts_optimizer,
            "normalize_y": normalize_y,
            "copy_X_train": copy_X_train,
            "random_state": random_state,
        }
        super().__init__(
            kernel,
            alpha=self.alphas[0]
            if isinstance(self.alphas, (list, tuple))
            else self.alphas,
            **self.main_kwargs,
        )
        self.compute_L_ = compute_L_

    # def fit(self, X, y):
    #     self._main_fit(X, y)

    def fit(self, X, y):
        # Check formatting of alphas
        self.alphas_ = (
            self.alphas if isinstance(self.alphas, (list, tuple)) else [self.alphas]
        )
        if self.num_distillations != len(self.alphas_):
            if len(self.alphas_) == 1:
                self.alphas_ = self.alphas_ * self.num_distillations
            else:
                raise ValueError(
                    "Number of alphas must be equal to num_distillations or a single value."
                )

        if self.fit_mode == "naive":
            for i in range(
                self.num_distillations
            ):  # Perform self-distillation iterations
                if self.optimize_mode == "first" and i > 0:
                    self.main_kwargs.update({"optimizer": None})
                    super().__init__(
                        kernel=self.kernel_.clone_with_theta(self.kernel_.theta),
                        alpha=self.alphas_[i],
                        **self.main_kwargs,
                    )  # Initialize GPR
                else:
                    super().__init__(
                        kernel=self.kernel.clone_with_theta(self.kernel.theta),
                        alpha=self.alphas_[i],
                        **self.main_kwargs,
                    )  # Initialize GPR
                super().fit(X, y)  # Train a GPR on current data

                # Make predictions on training data and add them as new training data
                y_pred = super().predict(X)
                y = np.copy(y_pred.ravel())
        elif self.fit_mode == "efficient":
            super().__init__(
                kernel=self.kernel.clone_with_theta(self.kernel.theta),
                alpha=self.alphas_[0],
                **self.main_kwargs,
            )  # Initialize GPR
            self._main_fit(X, y)  # Train a GPR on current data
            for i in range(
                1, self.num_distillations
            ):  # Perform self-distillation iterations
                a_step_ = self.d_ / (self.d_ + self.alphas_[i])
                A_step_ = np.diag(self.a_)

                self.a_ *= a_step_
                self.A_ = np.diag(self.a_)
        return self

    def _main_fit(self, X, y):
        """Fit Gaussian process regression model.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or list of object
            Feature vectors or other representations of training data.
        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values.
        Returns
        -------
        self : object
            GaussianProcessRegressor class instance.
        """
        # self._validate_params()

        if self.kernel is None:  # Use an RBF kernel as default
            self.kernel_ = C(1.0, constant_value_bounds="fixed") * RBF(
                1.0, length_scale_bounds="fixed"
            )
        else:
            self.kernel_ = clone(self.kernel)

        self._rng = check_random_state(self.random_state)

        if self.kernel_.requires_vector_input:
            dtype, ensure_2d = "numeric", True
        else:
            dtype, ensure_2d = None, False
        X, y = self._validate_data(
            X,
            y,
            multi_output=True,
            y_numeric=True,
            ensure_2d=ensure_2d,
            dtype=dtype,
        )

        # Normalize target value
        if self.normalize_y:
            self._y_train_mean = np.mean(y, axis=0)
            self._y_train_std = _handle_zeros_in_scale(np.std(y, axis=0), copy=False)

            # Remove mean and make unit variance
            y = (y - self._y_train_mean) / self._y_train_std

        else:
            shape_y_stats = (y.shape[1],) if y.ndim == 2 else 1
            self._y_train_mean = np.zeros(shape=shape_y_stats)
            self._y_train_std = np.ones(shape=shape_y_stats)

        if np.iterable(self.alpha) and self.alpha.shape[0] != y.shape[0]:
            if self.alpha.shape[0] == 1:
                self.alpha = self.alpha[0]
            else:
                raise ValueError(
                    "alpha must be a scalar or an array with same number of "
                    f"entries as y. ({self.alpha.shape[0]} != {y.shape[0]})"
                )

        self.X_train_ = np.copy(X) if self.copy_X_train else X
        self.y_train_ = np.copy(y) if self.copy_X_train else y

        if self.optimizer is not None and self.kernel_.n_dims > 0:
            # Choose hyperparameters based on maximizing the log-marginal
            # likelihood (potentially starting from several initial values)
            def obj_func(theta, eval_gradient=True):
                if eval_gradient:
                    lml, grad = self.log_marginal_likelihood(
                        theta, eval_gradient=True, clone_kernel=False
                    )
                    return -lml, -grad
                else:
                    return -self.log_marginal_likelihood(theta, clone_kernel=False)

            # First optimize starting from theta specified in kernel
            optima = [
                (
                    self._constrained_optimization(
                        obj_func, self.kernel_.theta, self.kernel_.bounds
                    )
                )
            ]

            # Additional runs are performed from log-uniform chosen initial
            # theta
            if self.n_restarts_optimizer > 0:
                if not np.isfinite(self.kernel_.bounds).all():
                    raise ValueError(
                        "Multiple optimizer restarts (n_restarts_optimizer>0) "
                        "requires that all bounds are finite."
                    )
                bounds = self.kernel_.bounds
                for iteration in range(self.n_restarts_optimizer):
                    theta_initial = self._rng.uniform(bounds[:, 0], bounds[:, 1])
                    optima.append(
                        self._constrained_optimization(obj_func, theta_initial, bounds)
                    )
            # Select result from run with minimal (negative) log-marginal
            # likelihood
            lml_values = list(map(itemgetter(1), optima))
            self.kernel_.theta = optima[np.argmin(lml_values)][0]
            self.kernel_._check_bounds_params()

            self.log_marginal_likelihood_value_ = -np.min(lml_values)
        else:
            self.log_marginal_likelihood_value_ = self.log_marginal_likelihood(
                self.kernel_.theta, clone_kernel=False
            )

        # Precompute quantities required for predictions which are independent
        # of actual query points
        # Alg. 2.1, page 19, line 2 -> L = cholesky(K + sigma^2 I)
        K = self.kernel_(self.X_train_)
        K[np.diag_indices_from(K)] += self.alpha

        ###############################################################
        # Our implementation differs from here on
        ###############################################################
        try:
            self.V_, self.d_, _ = np.linalg.svd(
                K, full_matrices=True, compute_uv=True, hermitian=True
            )
            self.D_ = np.diag(self.d_)
        except np.linalg.LinAlgError as exc:
            exc.args = f"The kernel, is not decomposable with SVD..." + exc.args
            raise

        self.a_ = self.d_ / (self.d_ + self.alpha)
        self.A_ = np.diag(self.a_)
        self.VT_y_ = np.matmul(self.V_.T, self.y_train_)

        # Needed for e.g. the variance of the predictions in the current implementation
        if self.compute_L_:
            try:
                self.L_ = cholesky(K, lower=GPR_CHOLESKY_LOWER, check_finite=False)
            except np.linalg.LinAlgError as exc:
                exc.args = (
                    f"The kernel, {self.kernel_}, is not returning a positive "
                    "definite matrix. Try gradually increasing the 'alpha' "
                    "parameter of your GaussianProcessRegressor estimator.",
                ) + exc.args
                raise

        return self

    def predict(self, X, return_std=False, return_cov=False):
        if self.fit_mode == "naive":
            y_pred = super().predict(X, return_std, return_cov)
        elif self.fit_mode == "efficient":
            y_pred = self._efficient_predict(X, return_std, return_cov)
        else:
            raise ValueError(f"Unknown mode {self.fit_mode}")
        return y_pred

    def _efficient_predict(self, X, return_std=False, return_cov=False):
        """Predict using the Gaussian process regression model.
        We can also predict based on an unfitted model by using the GP prior.
        In addition to the mean of the predictive distribution, optionally also
        returns its standard deviation (`return_std=True`) or covariance
        (`return_cov=True`). Note that at most one of the two can be requested.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or list of object
            Query points where the GP is evaluated.
        return_std : bool, default=False
            If True, the standard-deviation of the predictive distribution at
            the query points is returned along with the mean.
        return_cov : bool, default=False
            If True, the covariance of the joint predictive distribution at
            the query points is returned along with the mean.
        Returns
        -------
        y_mean : ndarray of shape (n_samples,) or (n_samples, n_targets)
            Mean of predictive distribution a query points.
        y_std : ndarray of shape (n_samples,) or (n_samples, n_targets), optional
            Standard deviation of predictive distribution at query points.
            Only returned when `return_std` is True.
        y_cov : ndarray of shape (n_samples, n_samples) or \
                (n_samples, n_samples, n_targets), optional
            Covariance of joint predictive distribution a query points.
            Only returned when `return_cov` is True.
        """
        if return_std and return_cov:
            raise RuntimeError(
                "At most one of return_std or return_cov can be requested."
            )

        if self.kernel is None or self.kernel.requires_vector_input:
            dtype, ensure_2d = "numeric", True
        else:
            dtype, ensure_2d = None, False

        X = self._validate_data(X, ensure_2d=ensure_2d, dtype=dtype, reset=False)

        if not hasattr(self, "X_train_"):  # Unfitted;predict based on GP prior
            if self.kernel is None:
                kernel = C(1.0, constant_value_bounds="fixed") * RBF(
                    1.0, length_scale_bounds="fixed"
                )
            else:
                kernel = self.kernel
            y_mean = np.zeros(X.shape[0])
            if return_cov:
                y_cov = kernel(X)
                return y_mean, y_cov
            elif return_std:
                y_var = kernel.diag(X)
                return y_mean, np.sqrt(y_var)
            else:
                return y_mean
        else:  # Predict based on GP posterior
            # Alg 2.1, page 19, line 4 -> f*_bar = K(X_test, X_train) . alpha
            K_trans = self.kernel_(X, self.X_train_)
            y_mean = K_trans @ self.V_ @ np.diag((1 / self.d_) * self.a_) @ self.VT_y_

            # undo normalisation
            y_mean = self._y_train_std * y_mean + self._y_train_mean

            # if y_mean has shape (n_samples, 1), reshape to (n_samples,)
            if y_mean.ndim > 1 and y_mean.shape[1] == 1:
                y_mean = np.squeeze(y_mean, axis=1)

            # Alg 2.1, page 19, line 5 -> v = L \ K(X_test, X_train)^T
            V = solve_triangular(
                self.L_, K_trans.T, lower=GPR_CHOLESKY_LOWER, check_finite=False
            )

            if return_cov:
                # Alg 2.1, page 19, line 6 -> K(X_test, X_test) - v^T. v
                y_cov = self.kernel_(X) - V.T @ V

                # undo normalisation
                y_cov = np.outer(y_cov, self._y_train_std ** 2).reshape(
                    *y_cov.shape, -1
                )
                # if y_cov has shape (n_samples, n_samples, 1), reshape to
                # (n_samples, n_samples)
                if y_cov.shape[2] == 1:
                    y_cov = np.squeeze(y_cov, axis=2)

                return y_mean, y_cov
            elif return_std:
                # Compute variance of predictive distribution
                # Use einsum to avoid explicitly forming the large matrix
                # V^T @ V just to extract its diagonal afterward.
                y_var = self.kernel_.diag(X).copy()
                y_var -= np.einsum("ij,ji->i", V.T, V)

                # Check if any of the variances is negative because of
                # numerical issues. If yes: set the variance to 0.
                y_var_negative = y_var < 0
                if np.any(y_var_negative):
                    warnings.warn(
                        "Predicted variances smaller than 0. "
                        "Setting those variances to 0."
                    )
                    y_var[y_var_negative] = 0.0

                # undo normalisation
                y_var = np.outer(y_var, self._y_train_std ** 2).reshape(
                    *y_var.shape, -1
                )

                # if y_var has shape (n_samples, 1), reshape to (n_samples,)
                if y_var.shape[1] == 1:
                    y_var = np.squeeze(y_var, axis=1)

                return y_mean, np.sqrt(y_var)
            else:
                return y_mean
