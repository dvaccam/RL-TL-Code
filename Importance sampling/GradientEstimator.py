import numpy as np

class GradientEstimator:
    def __init__(self, baseline_type):
        self.baseline_type = baseline_type


    def estimate_gradient(self, dataset, policy, weights=None, Q=None, V=None):
        state_idx = dataset['fsi']
        action_idx = dataset['ai']
        grads = policy.log_gradient_matrix[state_idx, action_idx]
        if self.baseline_type == 0:
            grad = (grads.T * Q).T
            if weights is not None:
                grad = (grad.T * weights).T
            grad = grad.mean(axis=0)
        if self.baseline_type == 1:
            grad = (grads.T * Q).T
            baseline = V
            grad = grad - (grads.T * baseline).T
            if weights is not None:
                grad = (grad.T * weights).T
            grad = grad.mean(axis=0)
        return grad