import numpy as np

class LSTD_Q_Estimator:
    
    def __init__(self, n_kernels_pos, n_kernels_vel, n_kernels_act, eps, fit_bias, gamma, lam, min_pos, max_pos, min_vel,
                 max_vel, min_act, max_act):
        self.n_kernels_pos = n_kernels_pos
        self.n_kernels_vel = n_kernels_vel
        self.n_kernels_act = n_kernels_act
        self.eps = eps
        self.fit_bias = fit_bias
        self.gamma = gamma
        self.lam = lam
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
        
        
    
    def fit(self, dataset):
        phi = self.map_to_feature_space(dataset['fs'], dataset['a'])
        phi_next = self.map_to_feature_space(dataset['ns'], dataset['na'])
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
        n_feats = self.pos_kernels + self.vel_kernels + self.act_kernels + self.fit_bias
        A = np.zeros((n_feats, n_feats), dtype=np.float64)
        b = np.zeros(n_feats, dtype=np.float64)

        first_states = dataset['fs']
        actions = dataset['a']
        next_states = dataset['ns']
        rewards = dataset['r']
        next_actions = dataset['na']
        phi = self.map_to_feature_space(first_states, actions)
        phi_ns = self.map_to_feature_space(next_states, next_actions)
        for t in range(first_states.shape[0]):
            if t == 0:
                z = phi.copy()
            else:
                z = self.lam * self.gamma * z + phi
            A += z.reshape((-1, 1)).dot((phi - self.gamma * phi_ns).reshape((1, -1)))
            b += z * rewards[t]
        self.theta = np.linalg.pinv(A).dot(b)



    def transform(self, s, a):
        phi_s = self.map_to_feature_space(s, a)
        return phi_s.dot(self.theta)



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