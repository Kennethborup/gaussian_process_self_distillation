import warnings
from operator import itemgetter

import numpy as np
import torch
from scipy.linalg import cho_solve, cholesky, solve
from scipy.special import erf, expit
from sklearn.base import clone
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process._gpc import _BinaryGaussianProcessClassifierLaplace
from sklearn.gaussian_process.kernels import RBF, CompoundKernel
from sklearn.gaussian_process.kernels import ConstantKernel as C
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted
from torch.distributions import ContinuousBernoulli as CB


class ContinuousBernoulli:
    def _log_normalizing_derivative(self, logits, order=1):
        """Computes derivatives wrt. to logits of order up to 3"""
        # ensure logits are a numpy array
        logits = np.asarray(logits)
        non_zero_mask = logits != 0

        # Compute the derivatives
        derivative = np.zeros_like(logits)
        if order == 1:
            derivative[~non_zero_mask] = 0
            derivative[non_zero_mask] = 1 / logits[non_zero_mask] - 1 / np.sinh(
                logits[non_zero_mask]
            )
        elif order == 2:
            derivative[~non_zero_mask] = 1.0 / 6
            non_zero_logits = logits[non_zero_mask]
            derivative[non_zero_mask] = -1 / non_zero_logits**2 + 1 / (
                np.tanh(non_zero_logits) * np.sinh(non_zero_logits)
            )
        elif order == 3:
            derivative[~non_zero_mask] = 0
            non_zero_logits = logits[non_zero_mask]
            derivative[non_zero_mask] = 2 / non_zero_logits**3 - (
                1 + np.cosh(non_zero_logits) ** 2
            ) / (np.sinh(non_zero_logits) ** 3)
        else:
            raise ValueError("Only derivatives of order 1, 2, and 3 are supported")

        return derivative

    def _log_normalizing_constant(self, lambds=None, logits=None, log=True):
        assert (lambds is not None) != (logits is not None)
        if lambds is not None:
            lambds = torch.tensor(lambds)
            cb = CB(probs=lambds)
        else:
            logits = torch.tensor(logits)
            cb = CB(logits=logits)
        constant = cb._cont_bern_log_norm()
        constant = constant.exp() if not log else constant
        return constant.numpy()


# Values required for approximating the logistic sigmoid by
# error functions. coefs are obtained via:
# x = np.array([0, 0.6, 2, 3.5, 4.5, np.inf])
# b = logistic(x)
# A = (erf(np.dot(x, self.lambdas)) + 1) / 2
# coefs = lstsq(A, b)[0]
LAMBDAS = np.array([0.41, 0.4, 0.37, 0.44, 0.39])[:, np.newaxis]
COEFS = np.array(
    [-1854.8214151, 3516.89893646, 221.29346712, 128.12323805, -2010.49422654]
)[:, np.newaxis]


class _ContinuousBinaryGaussianProcessClassifierLaplace(
    _BinaryGaussianProcessClassifierLaplace
):
    def __init__(
        self,
        kernel=None,
        *,
        optimizer="fmin_l_bfgs_b",
        n_restarts_optimizer=0,
        max_iter_predict=100,
        warm_start=False,
        copy_X_train=True,
        random_state=None,
    ):
        self.kernel = kernel
        self.optimizer = optimizer
        self.n_restarts_optimizer = n_restarts_optimizer
        self.max_iter_predict = max_iter_predict
        self.warm_start = warm_start
        self.copy_X_train = copy_X_train
        self.random_state = random_state
        self.cont_bernoulli = ContinuousBernoulli()

    def fit(self, X, y):
        """Fit Gaussian process classification model.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or list of object
            Feature vectors or other representations of training data.
        y : array-like of shape (n_samples,)
            Target values, must be binary.
        Returns
        -------
        self : returns an instance of self.
        """
        if self.kernel is None:  # Use an RBF kernel as default
            self.kernel_ = C(1.0, constant_value_bounds="fixed") * RBF(
                1.0, length_scale_bounds="fixed"
            )
        else:
            self.kernel_ = clone(self.kernel)

        self.rng = check_random_state(self.random_state)

        self.X_train_ = np.copy(X) if self.copy_X_train else X

        # Encode class labels and check that it is a binary classification
        # problem
        # label_encoder = LabelEncoder()
        if y.ndim != 1:
            self.y_train_ = y
            self.classes_ = np.arange(y.ndim)
        else:
            self.y_train_ = y
            self.classes_ = np.array([0, 1])

        if self.classes_.size > 2:
            raise ValueError(
                "%s supports only binary classification. y contains classes %s"
                % (self.__class__.__name__, self.classes_)
            )
        elif self.classes_.size == 1:
            raise ValueError(
                "{0:s} requires 2 classes; got {1:d} class".format(
                    self.__class__.__name__, self.classes_.size
                )
            )

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
                self._constrained_optimization(
                    obj_func, self.kernel_.theta, self.kernel_.bounds
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
                    theta_initial = np.exp(self.rng.uniform(bounds[:, 0], bounds[:, 1]))
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
                self.kernel_.theta
            )

        # Precompute quantities required for predictions which are independent
        # of actual query points
        K = self.kernel_(self.X_train_)

        _, (self.pi_, self.W_sr_, self.L_, _, _) = self._posterior_mode(
            K, return_temporaries=True
        )

        return self

    def _posterior_mode(self, K, return_temporaries=False):
        """Mode-finding for binary Laplace GPC and fixed kernel.
        This approximates the posterior of the latent function values for given
        inputs and target observations with a Gaussian approximation and uses
        Newton's iteration to find the mode of this approximation.
        """
        # Based on Algorithm 3.1 of GPML

        # If warm_start are enabled, we reuse the last solution for the
        # posterior mode as initialization; otherwise, we initialize with 0
        if (
            self.warm_start
            and hasattr(self, "f_cached")
            and self.f_cached.shape == self.y_train_.shape
        ):
            f = self.f_cached
        else:
            f = np.zeros_like(self.y_train_, dtype=np.float64)

        # Use Newton's iteration method to find mode of Laplace approximation
        log_marginal_likelihood = -np.inf
        for _ in range(self.max_iter_predict):
            # Line 4
            pi = expit(f)
            W = pi * (1 - pi)
            # Continuous Bernoulli likelihood
            C_hessian = self.cont_bernoulli._log_normalizing_derivative(
                logits=f, order=2
            )
            W -= C_hessian
            # Line 5
            W_sr = np.sqrt(W)
            W_sr_K = W_sr[:, np.newaxis] * K
            B = np.eye(W.shape[0]) + W_sr_K * W_sr
            L = cholesky(B, lower=True)
            # Line 6
            C_grad = self.cont_bernoulli._log_normalizing_derivative(logits=f, order=1)
            b = W * f + (self.y_train_ - pi) + C_grad
            # Line 7
            a = b - W_sr * cho_solve((L, True), W_sr_K.dot(b))
            # Line 8
            f = K.dot(a)

            # Line 10: Compute log marginal likelihood in loop and use as
            #          convergence criterion
            # lml = (
            #     -0.5 * a.T.dot(f) # f^T K^-1 f / 2
            #     - np.log1p(np.exp(-(self.y_train_ * 2 - 1) * f)).sum() # log(p(y|f)
            #     - np.log(np.diag(L)).sum() # log(|B|)/2
            # )
            lml = (
                -np.log1p(np.exp(-(self.y_train_ * 2 - 1) * f)).sum()  # log(p(y|f)
                + self.cont_bernoulli._log_normalizing_constant(logits=f).sum()
                - 0.5 * a.T.dot(f)  # f^T K^-1 f / 2
                - np.log(np.diag(L)).sum()  # log(|B|)/2 (see eq. 3.32 in GPML)
            )
            # Check if we have converged (log marginal likelihood does
            # not decrease)
            # XXX: more complex convergence criterion
            if lml - log_marginal_likelihood < 1e-10:
                break
            log_marginal_likelihood = lml

        self.f_cached = f  # Remember solution for later warm-starts
        if return_temporaries:
            return log_marginal_likelihood, (pi, W_sr, L, b, a)
        else:
            return log_marginal_likelihood

    def log_marginal_likelihood(
        self, theta=None, eval_gradient=False, clone_kernel=True
    ):
        """Returns log-marginal likelihood of theta for training data.
        Parameters
        ----------
        theta : array-like of shape (n_kernel_params,), default=None
            Kernel hyperparameters for which the log-marginal likelihood is
            evaluated. If None, the precomputed log_marginal_likelihood
            of ``self.kernel_.theta`` is returned.
        eval_gradient : bool, default=False
            If True, the gradient of the log-marginal likelihood with respect
            to the kernel hyperparameters at position theta is returned
            additionally. If True, theta must not be None.
        clone_kernel : bool, default=True
            If True, the kernel attribute is copied. If False, the kernel
            attribute is modified, but may result in a performance improvement.
        Returns
        -------
        log_likelihood : float
            Log-marginal likelihood of theta for training data.
        log_likelihood_gradient : ndarray of shape (n_kernel_params,), \
                optional
            Gradient of the log-marginal likelihood with respect to the kernel
            hyperparameters at position theta.
            Only returned when `eval_gradient` is True.
        """
        # FIXME: The derivative of the log marginal likelihood computed here
        # is likely incorrect. The current implementation is based on the usual
        # log-marginal likelihood (GPML Algorithm 5.1), but some of the terms
        # (e.g. W) are computed according to the continous Bernoulli likelihood.
        # Adapt the algorithm to the continuous Bernoulli likelihood to get the
        # correct derivative.
        # warnings.warn(
        #     "Note, the provided gradient is likely incorrect. Consult source code for details."
        # )

        if theta is None:
            if eval_gradient:
                raise ValueError("Gradient can only be evaluated for theta!=None")
            return self.log_marginal_likelihood_value_

        if clone_kernel:
            kernel = self.kernel_.clone_with_theta(theta)
        else:
            kernel = self.kernel_
            kernel.theta = theta

        if eval_gradient:
            K, K_gradient = kernel(self.X_train_, eval_gradient=True)
        else:
            K = kernel(self.X_train_)

        # Compute log-marginal-likelihood Z and also store some temporaries
        # which can be reused for computing Z's gradient
        Z, (pi, W_sr, L, b, a) = self._posterior_mode(K, return_temporaries=True)

        if not eval_gradient:
            return Z

        # Compute gradient based on Algorithm 5.1 of GPML
        d_Z = np.empty(theta.shape[0])
        # XXX: Get rid of the np.diag() in the next line
        R = W_sr[:, np.newaxis] * cho_solve((L, True), np.diag(W_sr))  # Line 7
        C = solve(L, W_sr[:, np.newaxis] * K)  # Line 8
        # Line 9: (use einsum to compute np.diag(C.T.dot(C))))
        s_2 = (
            -0.5
            * (np.diag(K) - np.einsum("ij, ij -> j", C, C))
            * (
                pi * (1 - pi) * (1 - 2 * pi)
                + self.cont_bernoulli._log_normalizing_derivative(
                    self.f_cached, order=3
                )
            )
        )  # third derivative

        for j in range(d_Z.shape[0]):
            C = K_gradient[:, :, j]  # Line 11
            # Line 12: (R.T.ravel().dot(C.ravel()) = np.trace(R.dot(C)))
            s_1 = 0.5 * a.T.dot(C).dot(a) - 0.5 * R.T.ravel().dot(C.ravel())

            b = C.dot(
                self.y_train_
                - pi
                + self.cont_bernoulli._log_normalizing_derivative(
                    self.f_cached, order=1
                ),
            )  # Line 13
            s_3 = b - K.dot(R.dot(b))  # Line 14

            d_Z[j] = s_1 + s_2.T.dot(s_3)  # Line 15

        return Z, d_Z

    def predict(self, X):
        """Perform classification on an array of test vectors X.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or list of object
            Query points where the GP is evaluated for classification.
        Returns
        -------
        C : ndarray of shape (n_samples,)
            Predicted target values for X, values are from ``classes_``
        """
        check_is_fitted(self)

        # As discussed on Section 3.4.2 of GPML, for making hard binary
        # decisions, it is enough to compute the MAP of the posterior and
        # pass it through the link function
        K_star = self.kernel_(self.X_train_, X)  # K_star =k(x_star)
        f_star = K_star.T.dot(
            self.y_train_
            - self.pi_
            + self.cont_bernoulli._log_normalizing_derivative(self.f_cached, order=1)
        )  # Algorithm 3.2,Line 4

        return np.where(f_star > 0, self.classes_[1], self.classes_[0])

    def predict_proba(self, X):
        """Return probability estimates for the test vector X.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or list of object
            Query points where the GP is evaluated for classification.
        Returns
        -------
        C : array-like of shape (n_samples, n_classes)
            Returns the probability of the samples for each class in
            the model. The columns correspond to the classes in sorted
            order, as they appear in the attribute ``classes_``.
        """
        check_is_fitted(self)

        # Based on Algorithm 3.2 of GPML
        K_star = self.kernel_(self.X_train_, X)  # K_star =k(x_star)
        f_star = K_star.T.dot(
            self.y_train_
            - self.pi_
            + self.cont_bernoulli._log_normalizing_derivative(self.f_cached, order=1)
        )  # Line 4
        v = solve(self.L_, self.W_sr_[:, np.newaxis] * K_star)  # Line 5
        # Line 6 (compute np.diag(v.T.dot(v)) via einsum)
        var_f_star = self.kernel_.diag(X) - np.einsum("ij,ij->j", v, v)

        # Line 7:
        # Approximate \int log(z) * N(z | f_star, var_f_star)
        # Approximation is due to Williams & Barber, "Bayesian Classification
        # with Gaussian Processes", Appendix A: Approximate the logistic
        # sigmoid by a linear combination of 5 error functions.
        # For information on how this integral can be computed see
        # blitiri.blogspot.de/2012/11/gaussian-integral-of-error-function.html
        alpha = 1 / (2 * var_f_star)
        gamma = LAMBDAS * f_star
        integrals = (
            np.sqrt(np.pi / alpha)
            * erf(gamma * np.sqrt(alpha / (alpha + LAMBDAS**2)))
            / (2 * np.sqrt(var_f_star * 2 * np.pi))
        )
        pi_star = (COEFS * integrals).sum(axis=0) + 0.5 * COEFS.sum()

        return np.vstack((1 - pi_star, pi_star)).T


class DataCentricGPC(GaussianProcessClassifier):
    def __init__(
        self,
        kernel=None,
        optimizer="fmin_l_bfgs_b",
        n_restarts_optimizer=0,
        max_iter_predict=100,
        warm_start=False,
        copy_X_train=True,
        random_state=None,
        multi_class="one_vs_rest",
        n_jobs=None,
        num_distillations=2,
        optimize_mode="first",
    ):
        self.num_distillations = num_distillations
        assert self.num_distillations > 0, "num_distillations must be greater than 0"

        self.optimize_mode = optimize_mode
        assert self.optimize_mode in ["first", "all", None]
        self.kernel_orig = (
            kernel.clone_with_theta(kernel.theta) if kernel is not None else kernel
        )

        super().__init__(
            kernel=kernel,
            optimizer=optimizer,
            n_restarts_optimizer=n_restarts_optimizer,
            max_iter_predict=max_iter_predict,
            warm_start=warm_start,
            copy_X_train=copy_X_train,
            random_state=random_state,
            multi_class=multi_class,
            n_jobs=n_jobs,
        )

    def _classic_fit(self, X, y):
        return super().fit(X, y)

    def _continuous_fit(self, X, y):
        # Fit Gaussian process classification model with continuous labels
        # self._validate_params()

        if isinstance(self.kernel, CompoundKernel):
            raise ValueError("kernel cannot be a CompoundKernel")

        if self.kernel is None or self.kernel.requires_vector_input:
            X, y = self._validate_data(
                X, y, multi_output=False, ensure_2d=True, dtype="numeric"
            )
        else:
            X, y = self._validate_data(
                X, y, multi_output=False, ensure_2d=False, dtype=None
            )

        # Only change in this code is the use of _ContinuousBinaryGaussianProcessClassifierLaplace
        # instead of _BinaryGaussianProcessClassifierLaplace
        self.base_estimator_ = _ContinuousBinaryGaussianProcessClassifierLaplace(
            kernel=self.kernel,
            optimizer=self.optimizer,
            n_restarts_optimizer=self.n_restarts_optimizer,
            max_iter_predict=self.max_iter_predict,
            warm_start=self.warm_start,
            copy_X_train=self.copy_X_train,
            random_state=self.random_state,
        )

        if y.ndim != 1:
            self.classes_ = np.arange(y.ndim)
        else:
            self.classes_ = np.array([0, 1])

        # self.classes_ = np.unique(y)
        self.n_classes_ = self.classes_.size

        if self.n_classes_ == 1:
            raise ValueError(
                "GaussianProcessClassifier requires 2 or more "
                "distinct classes; got %d class (only class %s "
                "is present)" % (self.n_classes_, self.classes_[0])
            )
        if self.n_classes_ > 2:
            if self.multi_class == "one_vs_rest":
                self.base_estimator_ = OneVsRestClassifier(
                    self.base_estimator_, n_jobs=self.n_jobs
                )
            elif self.multi_class == "one_vs_one":
                self.base_estimator_ = OneVsOneClassifier(
                    self.base_estimator_, n_jobs=self.n_jobs
                )
            else:
                raise ValueError("Unknown multi-class mode %s" % self.multi_class)

        self.base_estimator_.fit(X, y)

        if self.n_classes_ > 2:
            self.log_marginal_likelihood_value_ = np.mean(
                [
                    estimator.log_marginal_likelihood()
                    for estimator in self.base_estimator_.estimators_
                ]
            )
        else:
            self.log_marginal_likelihood_value_ = (
                self.base_estimator_.log_marginal_likelihood()
            )

        return self

    def fit(self, X, y):
        self.main_kwargs = {
            "n_restarts_optimizer": self.n_restarts_optimizer,
            "max_iter_predict": self.max_iter_predict,
            "warm_start": self.warm_start,
            "copy_X_train": self.copy_X_train,
            "random_state": self.random_state,
            "multi_class": self.multi_class,
            "n_jobs": self.n_jobs,
        }
        # Optimize on first iteration with an ordinary GPC
        super().__init__(
            kernel=self.kernel.clone_with_theta(self.kernel.theta),
            optimizer=self.optimizer,
            **self.main_kwargs,
        )
        self._classic_fit(X, y)

        # Make predictions on training data and add them as new training data
        # Note predict_proba calls self.base_estimator_.predict_proba which
        # is _BinaryGaussianProcessClassifierLaplace here
        y_pred = self.predict_proba(X)
        y = np.copy(y_pred)[:, 1]

        for i in range(1, self.num_distillations):
            # print(f"  Fitting iteration {i+1} of {self.num_distillations}...")
            if self.optimize_mode == "first" or self.optimize_mode is None:
                # Optimized on first, then use the same kernel (i.e. fitted kernel_)
                super().__init__(
                    kernel=self.kernel_.clone_with_theta(self.kernel_.theta),
                    optimizer=None,
                    **self.main_kwargs,
                )
            elif self.optimize_mode == "all":
                # Optimized on first, but refitted on each iteration
                super().__init__(
                    kernel=self.kernel.clone_with_theta(self.kernel.theta),
                    optimizer=self.optimizer,
                    **self.main_kwargs,
                )
            else:
                raise ValueError(f"Unknown optimize_mode {self.optimize_mode}")
            self._continuous_fit(X, y)

            # Make predictions on training data and add them as new training data
            # Note predict_proba calls self.base_estimator_.predict_proba which
            # is _ContinuousBinaryGaussianProcessClassifierLaplace here
            y_pred = self.predict_proba(X)
            y = np.copy(y_pred)[:, 1]
        return self
