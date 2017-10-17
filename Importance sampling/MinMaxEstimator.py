import numpy as np
import math
from scipy.optimize import minimize, fixed_point, root
from scipy.special import erf
import gym

class MinMaxWeightsEstimator():
    def __init__(self, gamma):
        self.gamma = gamma
        
        
    def set_sources(self, source_samples, source_tasks, source_policies, Qs):
        source1 = gym.make('MountainCarContinuous-v0', min_position=-10., max_position=10., min_action=-1.,
                         max_action=1., power=0., seed=None, model='S', discrete=True, n_position_bins=source_tasks[0].env.position_bins.shape[0],
                         n_velocity_bins=source_tasks[0].env.velocity_bins.shape[0], n_action_bins=source_tasks[0].env.action_bins.shape[0], position_noise=0.025, velocity_noise=0.025)
        source2 = gym.make('MountainCarContinuous-v0', min_position=-10., max_position=10., min_action=-1.,
                              max_action=1., power=0.5, seed=None, model='S', discrete=True,
                              n_position_bins=source_tasks[0].env.position_bins.shape[0],
                              n_velocity_bins=source_tasks[0].env.velocity_bins.shape[0],
                              n_action_bins=source_tasks[0].env.action_bins.shape[0], position_noise=0.025,
                              velocity_noise=0.025)
        self.source_samples = source_samples
        self.source_tasks = source_tasks
        self.source_policies = source_policies
        self.m = len(source_tasks)
        self.a_max = source_policies[0].factory.action_reps[-1]
        self.R1 = - 0.1 * (2. ** self.a_max)
        self.R2 = 100.
        self.R = np.max(np.abs([self.R1, self.R2]))

        self.M_P_s_prime = np.zeros((self.m, source_tasks[0].env.V.shape[0]), dtype=np.float64)
        self.L_P_eps_s_a_s_prime = np.zeros((self.m,) + source_tasks[0].env.transition_matrix.shape, dtype=np.float64)
        self.L_P_eps_s_prime = np.zeros((self.m, source_tasks[0].env.V.shape[0]), dtype=np.float64)
        self.L_P_eps_theta_s_s_prime = np.zeros((self.m, source_tasks[0].env.V.shape[0], source_tasks[0].env.V.shape[0]), dtype=np.float64)
        self.L_Q = np.zeros(self.m, dtype=np.float64)
        self.M_Q_sa = np.zeros((self.m,) + source_tasks[0].env.Q.shape, dtype=np.float64)
        self.L_zeta = np.zeros((self.m,) + source_tasks[0].env.Q.shape, dtype=np.float64)
        self.L_P_eps_theta_s = np.zeros((self.m, source_tasks[0].env.V.shape[0]), dtype=np.float64)
        self.M_P_a = np.zeros((self.m, source_tasks[0].env.Q.shape[1]), dtype=np.float64)
        self.L_P_eps_a = np.abs(source_policies[0].factory.action_reps)*(1./source_tasks[0].env.position_noise + 1./source_tasks[0].env.velocity_noise)/(np.sqrt(2.*np.pi))
        self.L_delta = np.zeros((self.m, source_tasks[0].env.V.shape[0]), dtype=np.float64)
        self.L_eta = np.zeros((self.m,) + source_tasks[0].env.Q.shape + (2,), dtype=np.float64)
        self.L_J = np.zeros((self.m, 2), dtype=np.float64)
        self.ns = np.zeros(self.m, dtype=np.int64)
        for i in range(self.m):
            self.M_P_s_prime[i] = source_tasks[i].env.transition_matrix.max(axis=(0, 1))

            for s in range(source_tasks[0].env.transition_matrix.shape[0]):
                mu_max_pos = np.zeros(source_tasks[i].env.position_reps.shape[0], dtype=np.float64)
                mu_max_vel = np.zeros(source_tasks[i].env.velocity_reps.shape[0], dtype=np.float64)
                for p_p in range(source_tasks[i].env.position_reps.shape[0]):
                    if p_p != 0 and p_p != source_tasks[i].env.position_reps.shape[0] - 1:
                        p1 = source_tasks[i].env.position_bins[p_p]
                        p2 = source_tasks[i].env.position_bins[p_p + 1]
                        def f(x):
                            return (np.log(p2 - x) - np.log(p1 - x) + (p1 ** 2 - p2 ** 2) / (2. * source_tasks[i].env.position_noise ** 2)) / ((p1 - p2) / source_tasks[i].env.position_noise ** 2) - x
                        mu_max_pos[p_p] = root(f, p1 - 0.1).x
                for v_p in range(source_tasks[i].env.velocity_reps.shape[0]):
                    if v_p != 0 and v_p != source_tasks[i].env.velocity_reps.shape[0] - 1:
                        v1 = source_tasks[i].env.velocity_bins[v_p]
                        v2 = source_tasks[i].env.velocity_bins[v_p + 1]
                        def f(x):
                            return (np.log(v2 - x) - np.log(v1 - x) + (v1 ** 2 - v2 ** 2) / (2. * source_tasks[i].env.velocity_noise ** 2)) / ((v1 - v2) / source_tasks[i].env.velocity_noise ** 2) - x
                        mu_max_vel[v_p] = root(f, v1 - 0.01).x
                for a in range(source_tasks[0].env.transition_matrix.shape[1]):
                    M_e_pos = np.zeros(source_tasks[i].env.position_reps.shape[0], dtype=np.float64)
                    M_e_vel = np.zeros(source_tasks[i].env.velocity_reps.shape[0], dtype=np.float64)
                    M_P_pos = np.zeros(source_tasks[i].env.position_reps.shape[0], dtype=np.float64)
                    M_P_vel = np.zeros(source_tasks[i].env.velocity_reps.shape[0], dtype=np.float64)
                    mu1 = source1.env.clean_step(source_tasks[i].env.state_reps[s], source_tasks[i].env.action_reps[a:a+1])
                    mu2 = source2.env.clean_step(source_tasks[i].env.state_reps[s], source_tasks[i].env.action_reps[a:a+1])
                    for p_p in range(source_tasks[i].env.position_reps.shape[0]):
                        if p_p == 0 or p_p == source_tasks[i].env.position_reps.shape[0] - 1:
                            if p_p == 0:
                                eps_max = (source_tasks[i].env.position_bins[1] - source_tasks[i].env.state_reps[s].sum() + source_tasks[i].env.rescale(0.0025) * math.cos(3 * source_tasks[i].env.inverse_transform(source_tasks[i].env.state_reps[s][0])))/ source_tasks[i].env.action_reps[a]
                                if eps_max < 0 or eps_max > 0.5:
                                    M_e_pos[p_p] = max(np.exp(-(source_tasks[i].env.position_bins[1] - mu1[0])**2/(2.*source_tasks[i].env.position_noise**2)),
                                                       np.exp(-(source_tasks[i].env.position_bins[1] - mu2[0])**2/(2.*source_tasks[i].env.position_noise**2)))
                                else:
                                    M_e_pos[p_p] = max(np.exp(-(source_tasks[i].env.position_bins[-2] - mu1[0]) ** 2 / (2. * source_tasks[i].env.position_noise ** 2)),
                                                       np.exp(-(source_tasks[i].env.position_bins[-2] - mu2[0]) ** 2/(2. * source_tasks[i].env.position_noise ** 2)))
                                M_P_pos[p_p] = max((1. - erf((source_tasks[i].env.position_bins[-2] - mu1[0]) / (np.sqrt(2.) * source_tasks[i].env.position_noise))) / 2.,
                                                    (1. - erf((source_tasks[i].env.position_bins[-2] - mu2[0]) / (np.sqrt(2.) * source_tasks[i].env.position_noise))) / 2.)
                                M_P_pos[p_p] = max((1. + erf((source_tasks[i].env.position_bins[1] - mu1[0])/(np.sqrt(2.)*source_tasks[i].env.position_noise)))/2.,
                                                   (1. + erf((source_tasks[i].env.position_bins[1] - mu2[0])/(np.sqrt(2.) * source_tasks[i].env.position_noise))) / 2.)
                            else:
                                M_e_pos[p_p] = 1.
                                M_P_pos[p_p] = 0.5
                        else:
                            p1 = source_tasks[i].env.position_bins[p_p]
                            p2 = source_tasks[i].env.position_bins[p_p + 1]
                            pm = (p1 + p2) / 2.
                            mu_max = mu_max_pos[p_p]
                            if mu_max < p1:
                                mu_max1 = mu_max
                                mu_max2 = 2*pm - mu_max
                            else:
                                mu_max2 = mu_max
                                mu_max1 = 2 * pm - mu_max
                            eps_max1 = (mu_max1 - source_tasks[i].env.state_reps[s].sum() + source_tasks[i].env.rescale(0.0025) * math.cos(3 * source_tasks[i].env.inverse_transform(source_tasks[i].env.state_reps[s][0])))/ source_tasks[i].env.action_reps[a]
                            eps_max2 = (mu_max2 - source_tasks[i].env.state_reps[s].sum() + source_tasks[i].env.rescale(0.0025) * math.cos(3 * source_tasks[i].env.inverse_transform(source_tasks[i].env.state_reps[s][0]))) / source_tasks[i].env.action_reps[a]
                            eps_max = (pm - source_tasks[i].env.state_reps[s].sum() + source_tasks[i].env.rescale(0.0025) * math.cos(3 * source_tasks[i].env.inverse_transform(source_tasks[i].env.state_reps[s][0]))) / source_tasks[i].env.action_reps[a]
                            if (eps_max1 < 0 or eps_max1 > 0.5) and (eps_max2 < 0 or eps_max2 > 0.5):
                                M_e_pos[p_p] = max(np.exp(-(p1 - mu1[0]) ** 2 / (2 * source_tasks[i].env.position_noise ** 2)) - np.exp(-(p2 - mu1[0]) ** 2 / (2 * source_tasks[i].env.position_noise ** 2)),
                                                   np.exp(-(p1 - mu2[0]) ** 2 / (2 * source_tasks[i].env.position_noise ** 2)) - np.exp(-(p2 - mu2[0]) ** 2 / (2 * source_tasks[i].env.position_noise ** 2)))
                            else:
                                M_e_pos[p_p] = np.abs(np.exp(-(p1 - mu_max) ** 2 / (2 * source_tasks[i].env.position_noise ** 2)) - np.exp(-(p2 - mu_max) ** 2 / (2 * source_tasks[i].env.position_noise ** 2)))

                            if eps_max < 0 or eps_max > 0.5:
                                M_P_pos[p_p] = max((erf((p2 - mu1[0])/(np.sqrt(2.)*source_tasks[i].env.position_noise)) - erf((p1 - mu1[0])/(np.sqrt(2.)*source_tasks[i].env.position_noise)))/2.,
                                                   (erf((p2 - mu2[0])/(np.sqrt(2.)*source_tasks[i].env.position_noise)) - erf((p1 - mu2[0])/(np.sqrt(2.)*source_tasks[i].env.position_noise)))/2.)
                            else:
                                M_P_pos[p_p] = (erf((p2 - pm) / (np.sqrt(2.) * source_tasks[i].env.position_noise)) - erf((p1 - pm) / (np.sqrt(2.) * source_tasks[i].env.position_noise))) / 2.
                    for v_p in range(source_tasks[i].env.velocity_reps.shape[0]):
                        aux1 = source_tasks[i].env.state_reps[s][1] + source_tasks[i].env.action_reps[a]*0. - source_tasks[i].env.rescale(0.0025) * math.cos(3 * source_tasks[i].env.inverse_transform(source_tasks[i].env.state_reps[s][0]))
                        aux2 = source_tasks[i].env.state_reps[s][1] + source_tasks[i].env.action_reps[a] * 0.5 - source_tasks[i].env.rescale(0.0025) * math.cos(3 * source_tasks[i].env.inverse_transform(source_tasks[i].env.state_reps[s][0]))
                        aux1, aux2 = np.clip([aux1, aux2], source_tasks[i].env.min_speed, source_tasks[i].env.max_speed)
                        if v_p == 0 or v_p == source_tasks[i].env.velocity_reps.shape[0] - 1:
                            eps_max = (source_tasks[i].env.velocity_reps[v_p] - source_tasks[i].env.state_reps[s][1] + source_tasks[i].env.rescale(0.0025) * math.cos(3 * source_tasks[i].env.inverse_transform(source_tasks[i].env.state_reps[s][0]))) / source_tasks[i].env.action_reps[a]
                            if (eps_max < 0 or eps_max > 0.5) and (not (aux1 < 0 and mu1[0] == source_tasks[i].env.position_bins[1]) and not (aux2 < 0 and mu2[0] == source_tasks[i].env.position_bins[1])):
                                if v_p == 0:
                                    M_e_vel[v_p] = max(np.exp(-(source_tasks[i].env.velocity_bins[1] - mu1[1]) ** 2 / (2. * source_tasks[i].env.velocity_noise ** 2)),
                                                       np.exp(-(source_tasks[i].env.velocity_bins[1] - mu1[1]) ** 2 / (2. * source_tasks[i].env.velocity_noise ** 2)))
                                    M_P_vel[v_p] = max((1. + erf((source_tasks[i].env.velocity_bins[1] - mu1[1]) / (np.sqrt(2.) * source_tasks[i].env.velocity_noise))) / 2.,
                                                       (1. + erf((source_tasks[i].env.velocity_bins[1] - mu2[1]) / (np.sqrt(2.) * source_tasks[i].env.velocity_noise))) / 2.)
                                else:
                                    M_e_vel[v_p] = max(np.exp(-(source_tasks[i].env.velocity_bins[-2] - mu1[1]) ** 2 / (2. * source_tasks[i].env.velocity_noise ** 2)),
                                                       np.exp(-(source_tasks[i].env.velocity_bins[-2] - mu1[1]) ** 2 / (2. * source_tasks[i].env.velocity_noise ** 2)))
                                    M_P_vel[v_p] = max((1. + erf((source_tasks[i].env.velocity_bins[-2] - mu1[1]) / (np.sqrt(2.) * source_tasks[i].env.velocity_noise))) / 2.,
                                                       (1. + erf((source_tasks[i].env.velocity_bins[-2] - mu2[1]) / (np.sqrt(2.) * source_tasks[i].env.velocity_noise))) / 2.)
                            else:
                                M_e_vel[v_p] = 1.
                                M_P_vel[v_p] = 0.5
                        else:
                            v1 = source_tasks[i].env.velocity_bins[v_p]
                            v2 = source_tasks[i].env.velocity_bins[v_p + 1]
                            vm = (v1 + v2) / 2.
                            mu_max = mu_max_vel[v_p]
                            if mu_max < v1:
                                mu_max1 = mu_max
                                mu_max2 = 2 * vm - mu_max
                            else:
                                mu_max2 = mu_max
                                mu_max1 = 2 * vm - mu_max
                            eps_max1 = (mu_max1 - source_tasks[i].env.state_reps[s][1] + source_tasks[i].env.rescale(0.0025) * math.cos(3 * source_tasks[i].env.inverse_transform(source_tasks[i].env.state_reps[s][0]))) / source_tasks[i].env.action_reps[a]
                            eps_max2 = (mu_max2 - source_tasks[i].env.state_reps[s][1] + source_tasks[i].env.rescale(0.0025) * math.cos(3 * source_tasks[i].env.inverse_transform(source_tasks[i].env.state_reps[s][0]))) / source_tasks[i].env.action_reps[a]
                            eps_max = (vm - source_tasks[i].env.state_reps[s][1] + source_tasks[i].env.rescale(0.0025) * math.cos(3 * source_tasks[i].env.inverse_transform(source_tasks[i].env.state_reps[s][0]))) / source_tasks[i].env.action_reps[a]
                            if (eps_max1 < 0 or eps_max1 > 0.5) and (eps_max2 < 0 or eps_max2 > 0.5) and (not (aux1 < 0 and mu1[0] == source_tasks[i].env.position_bins[1]) and not (aux2 < 0 and mu2[0] == source_tasks[i].env.position_bins[1])):
                                M_e_vel[v_p] = max(np.exp(-(v1 - mu1[1]) ** 2 / (2 * source_tasks[i].env.velocity_noise ** 2)) - np.exp(-(v2 - mu1[1]) ** 2 / (2 * source_tasks[i].env.velocity_noise ** 2)),
                                                   np.exp(-(v1 - mu2[1]) ** 2 / (2 * source_tasks[i].env.velocity_noise ** 2)) - np.exp(-(v2 - mu2[1]) ** 2 / (2 * source_tasks[i].env.velocity_noise ** 2)))
                            else:
                                M_e_vel[v_p] = np.abs(np.exp(-(v1 - mu_max) ** 2 / (2 * source_tasks[i].env.velocity_noise ** 2)) - np.exp(-(v2 - mu_max) ** 2 / (2 * source_tasks[i].env.velocity_noise ** 2)))
                            if (eps_max < 0 or eps_max > 0.5) and (not (aux1 < 0 and mu1[0] == source_tasks[i].env.position_bins[1]) and not (aux2 < 0 and mu2[0] == source_tasks[i].env.position_bins[1])):
                                M_P_vel[v_p] = max((erf((v2 - mu1[1]) / (np.sqrt(2.) * source_tasks[i].env.velocity_noise)) - erf((v1 - mu1[1]) / (np.sqrt(2.) * source_tasks[i].env.velocity_noise))) / 2.,
                                                   (erf((v2 - mu2[1]) / (np.sqrt(2.) * source_tasks[i].env.velocity_noise)) - erf((v1 - mu2[1]) / (np.sqrt(2.) * source_tasks[i].env.velocity_noise))) / 2.)
                            else:
                                M_P_vel[v_p] = (erf((v2 - vm) / (np.sqrt(2.) * source_tasks[i].env.velocity_noise)) - erf((v1 - vm) / (np.sqrt(2.) * source_tasks[i].env.velocity_noise))) / 2.
                    self.L_P_eps_s_a_s_prime[i, s, a] = ((np.abs(source_tasks[i].env.action_reps[a])/(np.sqrt(2.*np.pi)*source_tasks[i].env.velocity_noise))*M_e_vel.reshape((-1,1)).dot(M_P_pos.reshape((1,-1))) + (np.abs(source_tasks[i].env.action_reps[a])/(np.sqrt(2.*np.pi)*source_tasks[i].env.position_noise))*M_P_vel.reshape((-1,1)).dot(M_e_pos.reshape((1,-1)))).flatten()

            self.L_P_eps_s_prime[i] = self.L_P_eps_s_a_s_prime[i].max(axis=(0,1))

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
            self.L_Q[i] = ((0. +
                            self.gamma*np.abs(self.source_tasks[i].env.Q)*(self.M_P_s_prime[i].reshape((-1,1))*np.abs(self.source_policies[i].choice_matrix - target_policy.choice_matrix) +
                                                                           target_policy.choice_matrix*np.minimum(self.L_P_eps_s_prime[i]*np.abs(self.source_tasks[i].env.power - target_eps), np.ones_like(self.L_P_eps_s_prime[i])).reshape((-1,1)))) / (1. - self.gamma)).sum()

            self.M_Q_sa[i] = np.minimum(np.full_like(self.source_tasks[i].env.Q, self.R/(1. - self.gamma)),
                                        np.maximum(np.abs(self.source_tasks[i].env.Q + self.L_Q[i]), np.abs(self.source_tasks[i].env.Q - self.L_Q[i])))

            self.L_P_eps_theta_s_s_prime[i] = (self.source_tasks[i].env.transition_matrix*np.abs(self.source_policies[i].choice_matrix - target_policy.choice_matrix).reshape((target_policy.choice_matrix.shape) + (1,)) +
                                               target_policy.choice_matrix.reshape(target_policy.choice_matrix.shape + (1,))*self.L_P_eps_s_a_s_prime[i]*np.abs(self.source_tasks[i].env.power - target_eps)).sum(axis=1)
            self.L_P_eps_theta_s[i] = self.L_P_eps_theta_s_s_prime[i].max(axis=0)
            self.L_delta[i] = (1. - self.gamma)*self.source_tasks[i].env.P_pi_inv.T.dot(0. + self.gamma * np.minimum(self.L_P_eps_theta_s[i], np.ones_like(self.L_P_eps_theta_s[i])))

            self.L_zeta[i] = self.source_tasks[i].env.delta_distr.reshape((-1,1))*np.abs(self.source_policies[i].choice_matrix - target_policy.choice_matrix) + target_policy.choice_matrix*np.minimum(self.L_delta[i], np.ones_like(self.L_delta[i])).reshape((-1,1))

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