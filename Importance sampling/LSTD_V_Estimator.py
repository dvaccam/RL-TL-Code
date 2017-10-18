import numpy as np
import scipy as sp


class LSTD_V_Estimator:
    def __init__(self, n_kernels_pos, n_kernels_vel, eps, fit_bias, gamma, lam, min_pos, max_pos, min_vel, max_vel):
        self.n_kernels_pos = n_kernels_pos
        self.n_kernels_vel = n_kernels_vel
        self.eps = eps
        self.fit_bias = fit_bias
        self.gamma = gamma
        self.lam = lam
        self.min_pos = min_pos
        self.max_pos = max_pos
        self.min_vel = min_vel
        self.max_vel = max_vel
        self.pos_kernels = np.linspace(min_pos, max_pos, n_kernels_pos)
        self.vel_kernels = np.linspace(min_vel, max_vel, n_kernels_vel)
        self.idx_grid = np.stack(np.meshgrid(np.arange(self.pos_kernels.shape[0]),
                                             np.arange(self.vel_kernels.shape[0]),
                                             indexing='ij')).reshape((-1, 2))
        self.source_phi_s = None
        self.source_phi_ns = None
        self.source_rewards = None



    def map_to_feature_space(self, s):
        if s.ndim == 2:
            dists_p = (s[:, 0].reshape((-1, 1)) - self.pos_kernels.reshape((1, -1))) / (self.max_pos - self.min_pos)
            dists_v = (s[:, 1].reshape((-1, 1)) - self.vel_kernels.reshape((1, -1))) / (self.max_vel - self.min_vel)
            dists = dists_p[:, self.idx_grid[:, 0]] ** 2 + dists_v[:, self.idx_grid[:, 1]] ** 2
            phi = np.exp(-dists / self.eps ** 2)
            if self.fit_bias:
                phi = np.hstack((phi, np.ones((dists.shape[0], 1), dtype=np.float64)))
        else:
            dists_p = (s[0] - self.pos_kernels) / (self.max_pos - self.min_pos)
            dists_v = (s[1] - self.vel_kernels) / (self.max_vel - self.min_vel)
            dists = dists_p[self.idx_grid[:, 0]] ** 2 + dists_v[self.idx_grid[:, 1]]
            phi = np.exp(-dists / self.eps ** 2)
            if self.fit_bias:
                phi = np.append(phi, 1.)
        return phi



    def add_sources(self, source_datasets):
        if self.source_phi_s is None and self.source_phi_ns is None and self.source_rewards is not None:
            self.source_phi_s = self.map_to_feature_space(np.vstack([sd['fs'] for sd in source_datasets]))
            self.source_phi_ns = self.map_to_feature_space(np.vstack([sd['ns'] for sd in source_datasets]))
            self.source_rewards = np.hstack([sd['r'] for sd in source_datasets])
        else:
            self.source_phi_s = np.vstack([self.source_phi_s] +
                                          self.map_to_feature_space(np.vstack([sd['fs'] for sd in source_datasets])))
            self.source_phi_ns = np.vstack([self.source_phi_ns] +
                                           self.map_to_feature_space(np.vstack([sd['ns'] for sd in source_datasets])))
            self.source_rewards = np.hstack([self.source_rewards] +
                                            np.hstack([sd['r'] for sd in source_datasets]))



    def clean_sources(self):
        self.source_phi_s = self.source_phi_ns = self.source_rewards = None



    def fit(self, dataset, source_weights=None, predict=False):
        first_states = dataset['fs']
        next_states = dataset['ns']
        phi_s = self.map_to_feature_space(first_states)
        phi_ns = self.map_to_feature_space(next_states)
        rewards = dataset['r']

        if source_weights is not None:
            phi_s = np.vstack((phi_s, self.source_phi_s))
            phi_ns = np.vstack((phi_ns, self.source_phi_ns))
            rewards = np.hstack((rewards, self.source_rewards))
            weights = np.hstack((np.ones(first_states.shape[0], dtype=np.float64), source_weights))

        delta_phi = phi_s - self.gamma * phi_ns
        if source_weights is not None:
            delta_phi *= weights.reshape((-1, 1))
        A = phi_s.T.dot(delta_phi)
        b = phi_s * rewards.reshape((-1,1))
        if source_weights is not None:
            b *= weights.reshape((-1, 1))
        b = b.sum(axis=0)
        self.theta = sp.linalg.pinv2(A).dot(b)
        if predict:
            return phi_s.dot(self.theta)



    def predict(self, s, phi_s):
        if phi_s is None:
            phi_s = self.map_to_feature_space(s)
        return phi_s.dot(self.theta)



    def calculate_theta(self, task, policy):
        task.env.set_policy(policy, self.gamma)
        phi = self.map_to_feature_space(task.env.state_reps)
        P_pi = np.transpose(task.env.transition_matrix, axes=(2, 0, 1)).copy()
        P_pi = (P_pi * policy.choice_matrix).sum(axis=2).T
        delta_phi = phi - self.gamma*P_pi.dot(phi)
        D = np.diag(task.env.delta_distr.flatten())
        A = phi.T.dot(D.dot(delta_phi))
        R = (task.env.R * policy.choice_matrix).sum(axis=1)
        b = phi.T.dot(D.dot(R))
        self.theta = np.linalg.pinv(A).dot(b)