import numpy as np
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import ConstantKernel as C


class DistributionCentricGPC(GaussianProcessClassifier):
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
        fit_mode="approx",
        num_distillations=2,
    ):
        self.fit_mode = fit_mode
        self.num_distillations = num_distillations
        assert self.num_distillations > 0, "num_distillations must be greater than 0"

        if self.fit_mode == "approx":
            distill_kernel = C(
                self.num_distillations, constant_value_bounds="fixed"
            ) * kernel.clone_with_theta(kernel.theta)
        elif self.fit_mode == "exact":
            raise NotImplementedError("Exact distillation not implemented yet")
            # distill_kernel = kernel.clone_with_theta(kernel.theta)

        super().__init__(
            kernel=distill_kernel,
            optimizer=optimizer,
            n_restarts_optimizer=n_restarts_optimizer,
            max_iter_predict=max_iter_predict,
            warm_start=warm_start,
            copy_X_train=copy_X_train,
            random_state=random_state,
            multi_class=multi_class,
            n_jobs=n_jobs,
        )
