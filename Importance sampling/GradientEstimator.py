import numpy as np

class GradientEstimator:
    def __init__(self, gamma, baseline_type):
        self.gamma = gamma
        self.baseline_type = baseline_type
        self.use_source = False


    # Source info set only through here, no more at estimation time
    def add_sources(self):
        self.use_source = True



    def clean_sources(self):
        self.use_source = False



    # Weights are for the sources; log_gradient, Q and V are for [target, source]
    def estimate_gradient(self, dataset, log_gradient, Q, V=None, source_weights=None):
        if source_weights is not None:
            target_size = dataset['fsi'].shape[0]
            weights = np.hstack((np.ones(target_size, dtype=np.float64), source_weights))

        if self.baseline_type == 0:
            gradient = log_gradient*Q.reshape((-1,1)).copy()
            if source_weights is not None:
                gradient = gradient*weights.reshape((-1,1))
            gradient = gradient.mean(axis=0)/(1. - self.gamma)
        if self.baseline_type == 1:
            gradient = log_gradient*(Q-V).reshape((-1,1))
            if source_weights is not None:
                #gradient = gradient*weights.reshape((-1,1))
                #gradient = gradient.mean(axis=0) / (1. - self.gamma)
                gradient = np.average(gradient, axis=0, weights=weights)/(1. - self.gamma) #TRY ALSO DIRECT TRANSFER I. E. WITHOUT WEIGHTS
            else:
                gradient = gradient.mean(axis=0)/(1. - self.gamma)
        return gradient