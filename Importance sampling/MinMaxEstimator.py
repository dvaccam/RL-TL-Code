import numpy as np
from scipy.optimize import minimize

class MinMaxWeightsEstimator():
    def __init__(self, gamma):
        self.gamma = gamma
        
        
    def set_sources(self, source_samples, source_tasks, source_policies, Qs):
        self.source_samples = source_samples
        self.source_tasks = source_tasks
        self.source_policies = source_policies
        self.m = len(source_tasks)
        self.a_max = source_policies[0].factory.action_reps[-1]
        self.R1 = - 0.1 * (2. ** self.a_max)
        self.R2 = 100.
        self.R = np.max(np.abs([self.R1, self.R2]))

        self.M_P_s_prime = np.zeros((self.m, source_tasks[0].env.V.shape[0]), dtype=np.float64)
        self.L_P_eps_s_prime = (1./source_tasks[0].env.position_noise + 1./source_tasks[0].env.velocity_noise)*self.a_max/(np.sqrt(2.*np.pi))  # TODO: check, maybe a tighter bound (e. g. 1/2)
        self.L_Q = np.zeros(self.m, dtype=np.float64)
        self.M_Q_sa = np.zeros((self.m,) + source_tasks[0].env.Q.shape, dtype=np.float64)
        self.L_zeta = np.zeros((self.m,) + source_tasks[0].env.Q.shape, dtype=np.float64)
        self.L_P_eps_theta = np.zeros(self.m, dtype=np.float64)
        self.M_P_a = np.zeros((self.m, source_tasks[0].env.Q.shape[1]), dtype=np.float64)
        self.L_P_eps_a = np.abs(source_policies[0].factory.action_reps)*(1./source_tasks[0].env.position_noise + 1./source_tasks[0].env.velocity_noise)/(np.sqrt(2.*np.pi))
        self.L_delta = np.zeros((self.m, source_tasks[0].env.V.shape[0]), dtype=np.float64)
        self.L_eta = np.zeros((self.m,) + source_tasks[0].env.Q.shape + (2,), dtype=np.float64)
        self.L_J = np.zeros((self.m, 2), dtype=np.float64)
        self.ns = np.zeros(self.m, dtype=np.int64)
        for i in range(self.m):
            self.M_P_s_prime[i] = source_tasks[i].env.transition_matrix.max(axis=(0, 1))

            self.M_P_a[i] = source_tasks[i].env.transition_matrix.max(axis=(0, 2))

            self.ns[i] = self.source_samples[i]['fs'].shape[0]
            
            self.source_samples[i]['eta_j'] = source_policies[i].log_gradient_matrix[source_samples[i]['fsi'], source_samples[i]['ai']]*Qs[i].reshape((-1,1))



    def estimate_weights(self, target_policy, target_eps, target_size, Qs):
        w_idx = np.hstack((0., self.ns)).cumsum().astype(np.int64)
        n = self.ns.sum() + target_size
        w0 = np.ones(n - target_size, dtype=np.float64)
        l_bounds = np.zeros_like(w0)
        u_bounds = np.zeros_like(w0)
        for i in range(self.m):
            self.L_Q[i] = ((0. + self.gamma*np.abs(self.source_tasks[i].env.Q)*(self.M_P_s_prime[i].reshape((-1,1))*np.abs(self.source_policies[i].choice_matrix - target_policy.choice_matrix) +
                                                                      target_policy.choice_matrix*min(self.L_P_eps_s_prime*np.abs(self.source_tasks[i].env.power - target_eps), 1.))) / (1. - self.gamma)).sum()

            self.M_Q_sa[i] = np.minimum(np.full_like(self.source_tasks[i].env.Q, self.R/(1. - self.gamma)),
                                        np.maximum(np.abs(self.source_tasks[i].env.Q + self.L_Q[i]), np.abs(self.source_tasks[i].env.Q - self.L_Q[i])))

            self.L_P_eps_theta[i] = (self.M_P_a[i] * np.abs(self.source_policies[i].choice_matrix - target_policy.choice_matrix).max(axis=0) + target_policy.choice_matrix.max(axis=0) * self.L_P_eps_a * np.abs(self.source_tasks[i].env.power - target_eps)).sum()
            self.L_delta[i] = (0. + self.gamma * self.L_P_eps_theta[i]) * self.source_tasks[i].env.P_inf

            self.L_zeta[i] = self.source_tasks[i].env.delta_distr.reshape((-1,1))*np.abs(self.source_policies[i].choice_matrix - target_policy.choice_matrix) + target_policy.choice_matrix*self.L_delta[i].reshape((-1,1))


            self.L_eta[i] = np.abs(target_policy.log_gradient_matrix)*np.minimum(self.L_Q[i],(self.R2 - self.R1)/(1. - self.gamma)) + np.abs(self.source_tasks[i].env.Q.reshape((self.source_tasks[i].env.Q.shape) + (-1,))*(self.source_policies[i].log_gradient_matrix - target_policy.log_gradient_matrix))

            self.L_J[i] = (np.abs(target_policy.log_gradient_matrix)*(self.M_Q_sa[i]*np.minimum(np.ones_like(self.L_zeta[i]),self.L_zeta[i])).reshape((self.M_Q_sa[i].shape) + (1,)) + self.source_tasks[i].env.dseta_distr.reshape((self.source_tasks[i].env.dseta_distr.shape) + (1,))*self.L_eta[i]).sum(axis=(0,1))

            self.source_samples[i]['eta_1'] = target_policy.log_gradient_matrix[self.source_samples[i]['fsi'], self.source_samples[i]['ai']]*Qs[i].reshape((-1,1))

            l_bounds[w_idx[i]:w_idx[i+1]] = np.maximum(np.zeros(self.ns[i], dtype=np.float64), w0 - np.minimum(np.ones_like(self.L_zeta[i]),self.L_zeta[i])[self.source_samples[i]['fsi'], self.source_samples[i]['ai']]/self.source_tasks[i].env.dseta_distr[self.source_samples[i]['fsi'], self.source_samples[i]['ai']])
            u_bounds[w_idx[i]:w_idx[i + 1]] = w0 + np.minimum(np.ones_like(self.L_zeta[i]),self.L_zeta[i])[self.source_samples[i]['fsi'], self.source_samples[i]['ai']]/self.source_tasks[i].env.dseta_distr[self.source_samples[i]['fsi'], self.source_samples[i]['ai']]


        def g(w):
            bias = (self.L_J*self.ns.reshape((-1,1))).sum(axis=0)/n
            vari = 0.
            for j in range(self.m):
                bias += (self.source_samples[j]['eta_j'] - w[w_idx[j]:w_idx[j+1]].reshape((-1,1))*self.source_samples[j]['eta_1']).sum(axis=0)/n
                vari += (((w[w_idx[j]:w_idx[j+1]].reshape((-1,1))*self.source_samples[j]['eta_1'])**2).sum(axis=0) - self.ns[j]*((w[w_idx[j]:w_idx[j+1]].reshape((-1,1))*self.source_samples[j]['eta_1']).sum(axis=0))**2)/n**2
            bias = (bias**2).sum()
            vari = vari.sum()
            return bias + vari

        def grad_g(w):
            bias = (self.L_J * self.ns.reshape((-1, 1))).sum(axis=0) / n
            for j in range(self.m):
                bias += (self.source_samples[j]['eta_j'] - w[w_idx[j]:w_idx[j + 1]].reshape((-1, 1)) * self.source_samples[j]['eta_1']).sum(axis=0) / n
            grad = np.zeros((w.shape) + (2,), dtype=np.float64)
            for j in range(self.m):
                grad[w_idx[j]:w_idx[j+1]] = 2*self.source_samples[j]['eta_1']*(-bias + self.source_samples[j]['eta_1']*w[w_idx[j]:w_idx[j+1]].reshape((-1,1)) - self.ns[j]*(self.source_samples[j]['eta_1']*w[w_idx[j]:w_idx[j+1]].reshape((-1,1))).sum())/n**2
            grad = grad.sum(axis=1)
            return grad

        def hess_g(w):
            hess = np.zeros([w.shape[0]]*2 + [2], dtype=np.float64)
            for j1 in range(self.m):
                for j2 in range(self.m):
                    hess[w_idx[j1]:w_idx[j1+1], w_idx[j2]:w_idx[j2+1]] = -self.source_samples[j1]['eta_1'].dot((1. + (j1 == j2)*self.ns[j1])*self.source_samples[j2]['eta_1'].T)
            return hess

        bounds = np.ones((l_bounds.size + u_bounds.size,), dtype=np.float64)
        bounds[0::2] = l_bounds
        bounds[1::2] = u_bounds
        bounds = tuple(map(tuple, bounds.reshape((-1,2))))
        res = minimize(g, w0, jac=grad_g, hess=hess_g, bounds=bounds)

        return res.x