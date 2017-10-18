import numpy as np
import scipy as sp
import time

class LSTD_Q_Estimator:

    def __init__(self, n_kernels_pos, n_kernels_vel, n_kernels_act, eps, fit_bias, gamma, lam, min_pos, max_pos, min_vel,
                 max_vel, min_act, max_act):
        self.n_kernels_pos = n_kernels_pos
        self.n_kernels_vel = n_kernels_vel
        self.n_kernels_act = n_kernels_act
        self.eps = eps
        self.fit_bias = fit_bias
        self.gamma = gamma
        self.min_pos = min_pos
        self.max_pos = max_pos
        self.min_vel = min_vel
        self.max_vel = max_vel
        self.min_act = min_act
        self.max_act = max_act
        self.pos_kernels = np.linspace(min_pos, max_pos, n_kernels_pos)
        self.vel_kernels = np.linspace(min_vel, max_vel, n_kernels_vel)
        if n_kernels_act != 0:
            self.act_kernels = np.linspace(min_act, max_act, n_kernels_act)
        else:
            self.act_kernels = np.array([0.])
        self.idx_cube = np.stack(np.meshgrid(np.arange(self.pos_kernels.shape[0]),
                                            np.arange(self.vel_kernels.shape[0]),
                                            np.arange(self.act_kernels.shape[0]),
                                            indexing='ij'))
        self.idx_cube = np.transpose(self.idx_cube, axes=(1, 2, 3, 0)).reshape((-1, 3))
        self.source_phi_sa = None
        self.source_phi_nsa = None
        self.source_rewards = None



    def map_to_feature_space(self, s, a):
        if s.ndim == 2:
            dists_p = (s[:, 0].reshape((-1, 1)) - self.pos_kernels.reshape((1, -1))) / (self.max_pos - self.min_pos)
            dists_v = (s[:, 1].reshape((-1, 1)) - self.vel_kernels.reshape((1, -1))) / (self.max_vel - self.min_vel)
            if self.n_kernels_act != 0:
                dists_a = (a.reshape((-1, 1)) - self.act_kernels.reshape((1, -1))) / (self.max_act - self.min_act)
            else:
                dists_a = np.zeros((a.shape[0], 1), dtype=np.float64)
            dists = dists_p[:, self.idx_cube[:, 0]] ** 2 + dists_v[:, self.idx_cube[:, 1]] ** 2 + dists_a[:, self.idx_cube[:, 2]] ** 2
            phi = np.exp(-dists / self.eps ** 2)
            if self.fit_bias:
                phi = np.hstack((phi, np.ones((dists.shape[0], 1), dtype=np.float64)))
        else:
            dists_p = (s[0] - self.pos_kernels) / (self.max_pos - self.min_pos)
            dists_v = (s[1] - self.vel_kernels) / (self.max_vel - self.min_vel)
            if self.n_kernels_act != 0:
                dists_a = (a - self.act_kernels) / (self.max_act - self.min_act)
            else:
                dists_a = np.array([0.])
            dists = dists_p[self.idx_cube[:, 0]] ** 2 + dists_v[self.idx_cube[:, 1]] ** 2 + dists_a[self.idx_cube[:, 2]] ** 2
            phi = np.exp(-dists / self.eps ** 2)
            if self.fit_bias:
                phi = np.append(phi, 1.)
        '''p1 = np.array([source_task.env.position_bins[0]])
        p3 = np.array([source_task.env.position_bins[-1]])
        p2 = np.array([(p3 + p1) / 2]).flatten()
        v1 = np.array([source_task.env.velocity_bins[0]])
        v3 = np.array([source_task.env.velocity_bins[-1]])
        v2 = np.array([(v3 + v1) / 2]).flatten()
        epsp = rescale_state(1.8 * 0.35) ** 2
        epsv = rescale_state(.14 * 0.35) ** 2
        if s.ndim == 2:
            dp1 = s[:, 0] - p1
            dp2 = s[:, 0] - p2
            dp3 = s[:, 0] - p3
            dv1 = s[:, 1] - v1
            dv2 = s[:, 1] - v2
            dv3 = s[:, 1] - v3
            d = np.stack((np.exp(-dp1**2/epsp - dv1**2/epsv), np.exp(-dp1**2/epsp - dv2**2/epsv), np.exp(-dp1**2/epsp - dv3**2/epsv),
                  np.exp(-dp2 ** 2 / epsp - dv1 ** 2 / epsv), np.exp(-dp2**2/epsp - dv3**2/epsv),
                  np.exp(-dp3 ** 2 / epsp - dv2 ** 2 / epsv),np.exp(-dp3 ** 2 / epsp - dv3 ** 2 / epsv))).T
            phi = np.hstack((d, np.ones((d.shape[0], 1), dtype=np.float64)))
        else:
            dp1 = np.abs(s[0] - p1)
            dp2 = np.abs(s[0] - p2)
            dp3 = np.abs(s[0] - p3)
            dv1 = np.abs(s[1] - v1)
            dv2 = np.abs(s[1] - v2)
            dv3 = np.abs(s[1] - v3)
            d = np.array([np.exp(-dp1**2/epsp - dv1**2/epsv), np.exp(-dp1**2/epsp - dv2**2/epsv), np.exp(-dp1**2/epsp - dv3**2/epsv),
                  np.exp(-dp2 ** 2 / epsp - dv1 ** 2 / epsv), np.exp(-dp2**2/epsp - dv3**2/epsv),
                  np.exp(-dp3 ** 2 / epsp - dv2 ** 2 / epsv),np.exp(-dp3 ** 2 / epsp - dv3 ** 2 / epsv)]).flatten()
            phi = np.append(d, 1.)'''
        return phi



    def add_sources(self, source_datasets):
        if self.source_phi_sa is None and self.source_phi_nsa is None and self.source_rewards is None:
            self.source_phi_sa = self.map_to_feature_space(np.vstack([sd['fs'] for sd in source_datasets]),
                                                           np.hstack([sd['a'] for sd in source_datasets]))
            self.source_phi_nsa = self.map_to_feature_space(np.vstack([sd['ns'] for sd in source_datasets]),
                                                            np.hstack([sd['na'] for sd in source_datasets]))
            self.source_rewards = np.hstack([sd['r'] for sd in source_datasets])
        else:
            self.source_phi_sa = np.vstack([self.source_phi_sa] +
                                           self.map_to_feature_space(np.vstack([sd['fs'] for sd in source_datasets]),
                                                                     np.hstack([sd['a'] for sd in source_datasets])))
            self.source_phi_nsa = np.vstack([self.source_phi_nsa] +
                                            self.map_to_feature_space(np.vstack([sd['ns'] for sd in source_datasets]),
                                                                      np.hstack([sd['na'] for sd in source_datasets])))
            self.source_rewards = np.hstack([self.source_rewards] +
                                            np.hstack([sd['r'] for sd in source_datasets]))



    def clean_sources(self):
        self.source_phi_sa = self.source_phi_nsa = self.source_rewards = None



    def fit(self, dataset, source_weights=None, predict=False): #Direct transfer works here because reward function is the same
        first_states = dataset['fs']
        actions = dataset['a']
        next_states = dataset['ns']
        next_actions = dataset['na']
        phi_sa = self.map_to_feature_space(first_states, actions)
        phi_nsa = self.map_to_feature_space(next_states, next_actions)
        rewards = dataset['r']

        if source_weights is not None:
            phi_sa = np.vstack((phi_sa, self.source_phi_sa))
            phi_nsa = np.vstack((phi_nsa, self.source_phi_nsa))
            rewards = np.hstack((rewards, self.source_rewards))
            weights = np.hstack((np.ones(first_states.shape[0], dtype=np.float64), source_weights))

        delta_phi = phi_sa - self.gamma * phi_nsa
        if source_weights is not None:
            delta_phi *= weights.reshape((-1,1))
        A = phi_sa.T.dot(delta_phi)
        b = phi_sa * rewards.reshape((-1,1))
        if source_weights is not None:
            b *= weights.reshape((-1,1))
        b = b.sum(axis=0)
        self.theta = sp.linalg.pinv2(A).dot(b)
        if predict:
            return phi_sa.dot(self.theta)



    def predict(self, s, a, phi_sa):
        if phi_sa is None:
            phi_sa = self.map_to_feature_space(s, a)
        return phi_sa.dot(self.theta)



    def calculate_theta(self, task, policy):
        task.env.set_policy(policy, self.gamma)
        idx_grid = np.dstack(np.meshgrid(np.arange(task.env.state_reps.shape[0]),
                                         np.arange(task.env.action_reps.shape[0]),
                                         indexing='ij')).reshape(-1, 2)
        phi = self.map_to_feature_space(task.env.state_reps[idx_grid[:, 0]],
                                        task.env.action_reps[idx_grid[:, 1]].reshape((-1, 1)))
        delta_phi = np.zeros_like(phi, dtype=np.float64)
        for i in range(delta_phi.shape[0]):
            p = task.env.transition_matrix[idx_grid[i, 0], idx_grid[i, 1]]
            dseta_pi = p[idx_grid[:, 0]] * policy.choice_matrix[idx_grid[:, 0], idx_grid[:, 1]]
            delta_phi[i] = phi[i] - self.gamma * dseta_pi.dot(phi)
        del p, dseta_pi
        D = np.diag(task.env.dseta_distr.flatten())
        A = phi.T.dot(D.dot(delta_phi))
        R = task.env.R.flatten()
        b = phi.T.dot(D.dot(R))
        self.theta = np.linalg.pinv(A).dot(b)