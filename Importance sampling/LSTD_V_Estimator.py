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



    def fit(self, dataset):
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



    def fit_slow(self, dataset):
        n_feats = self.pos_kernels + self.vel_kernels + self.fit_bias
        A = np.zeros((n_feats, n_feats), dtype=np.float64)
        b = np.zeros(n_feats, dtype=np.float64)

        first_states = dataset['fs']
        next_states = dataset['ns']
        rewards = dataset['r']
        phi = self.map_to_feature_space(first_states)
        phi_ns = self.map_to_feature_space(next_states)
        for t in range(first_states.shape[0]):
            if t == 0:
                z = phi.copy()
            else:
                z = self.lam * self.gamma * z + phi
            A += z.reshape((-1, 1)).dot((phi - self.gamma * phi_ns).reshape((1, -1)))
            b += z * rewards[t]
        self.theta = np.linalg.pinv(A).dot(b)



    def transform(self, s):
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