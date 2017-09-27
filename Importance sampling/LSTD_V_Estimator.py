import numpy as np


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



    def fit(self, dataset, weights_d=None, weights_p=None, weights_r=None,
            phi_source=None, phi_ns_source=None, predict=False):
        n_feats = int((self.n_kernels_pos * self.n_kernels_vel) + self.fit_bias)
        A = np.zeros((n_feats, n_feats), dtype=np.float64)
        b = np.zeros(n_feats, dtype=np.float64)

        if phi_source is not None and phi_ns_source is not None:
            source_size = phi_source.shape[0]
            first_states = dataset['fs'][:-source_size]
            next_states = dataset['ns'][:-source_size]
            phi = np.vstack((self.map_to_feature_space(first_states), phi_source))
            phi_ns = np.vstack((self.map_to_feature_space(next_states), phi_ns_source))
        else:
            first_states = dataset['fs']
            next_states = dataset['ns']
            phi = self.map_to_feature_space(first_states)
            phi_ns = self.map_to_feature_space(next_states)

        rewards = dataset['r']

        if self.lam == 0:
            delta_phi = phi - self.gamma * phi_ns
            if weights_d is not None and weights_p is not None:
                delta_phi *= (weights_d * weights_p).reshape((-1, 1))
            A = phi.T.dot(delta_phi)
            b = phi * rewards.reshape((-1,1))
            if weights_r is not None:
                b *= (weights_d * weights_r).reshape((-1, 1))
            b = b.sum(axis=0)
        else: #TODO: Optimize
            if weights_d is None:
                weights_d = np.ones(dataset['fs'].shape[0], dtype=np.float64)
            if weights_p is None:
                weights_p = np.ones(dataset['fs'].shape[0], dtype=np.float64)
            if weights_r is None:
                weights_r = np.ones(dataset['fs'].shape[0], dtype=np.float64)
            for t in range(first_states.shape[0]):
                if t == 0:
                    z = phi[t].copy()
                    w_z = 1.
                else:
                    z = self.lam * self.gamma * z + phi[t] #
                    # w_z *= ratio of probs of going from phi[t-1] to phi[t] if lambda != 0. else 1.
                A += w_z * z.reshape((-1, 1)).dot(weights_d[t]*(phi[t] - self.gamma * weights_p[t] * phi_ns[t]).reshape((1, -1)))
                b += w_z * z * weights_r[t] * rewards[t]
        self.theta = np.linalg.pinv(A).dot(b)
        if predict:
            return phi.dot(self.theta)



    def fit2(self, dataset):
        phi = self.map_to_feature_space(dataset['fs'])
        phi_next = self.map_to_feature_space(dataset['ns'])
        if self.lam != 0:
            exps = np.tri(dataset['fs'].shape[0], k=-1, dtype=np.float64).cumsum(axis=0)
            gl = (np.power(self.gamma * self.lam, exps) * np.tri(dataset['fs'].shape[0], dtype=np.float64))
        else:
            gl = np.eye(dataset['fs'].shape[0], dtype=np.float64)
        z = ((phi.T.reshape(phi.shape[1], 1, -1)) * gl).sum(axis=2).T
        r = dataset['r']
        b = (z.T * r).T.sum(axis=0)
        A = (phi - self.gamma * phi_next)
        A = np.array([z[i].reshape((-1, 1)).dot(A[i].reshape((1, -1))) for i in range(A.shape[0])]).sum(axis=0)
        self.theta = np.linalg.pinv(A).dot(b)



    def predict(self, s, phi_source=None):
        if phi_source is not None:
            source_size = phi_source.shape[0]
            phi_s = np.vstack((self.map_to_feature_space(s[:-source_size]), phi_source))
        else:
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