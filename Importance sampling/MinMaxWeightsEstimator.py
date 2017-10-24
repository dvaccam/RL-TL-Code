import numpy as np
import math
from scipy.optimize import minimize, root
from scipy.special import erf
import gym

class MinMaxWeightsEstimator():
    def __init__(self, gamma):
        self.gamma = gamma



    def set_flags(self, for_gradient, for_LSTDQ, for_LSTDV):
        self.for_gradient = for_gradient
        self.for_LSTDQ = for_LSTDQ
        self.for_LSTDV = for_LSTDV
        


    def add_sources(self, source_samples, source_tasks, source_policies, all_phi_Q, all_phi_V):
        min_power = 0.
        max_power = 0.5
        min_source = gym.make('MountainCarContinuous-v0', min_position=-10., max_position=10., min_action=-1., max_action=1.,
                              power=min_power, seed=None, model='S', discrete=True,
                              n_position_bins=source_tasks[0].env.position_bins.shape[0],
                              n_velocity_bins=source_tasks[0].env.velocity_bins.shape[0],
                              n_action_bins=source_tasks[0].env.action_bins.shape[0],
                              position_noise=0.025, velocity_noise=0.025)
        max_source = gym.make('MountainCarContinuous-v0', min_position=-10., max_position=10., min_action=-1., max_action=1.,
                              power=max_power, seed=None, model='S', discrete=True,
                              n_position_bins=source_tasks[0].env.position_bins.shape[0],
                              n_velocity_bins=source_tasks[0].env.velocity_bins.shape[0],
                              n_action_bins=source_tasks[0].env.action_bins.shape[0],
                              position_noise=0.025, velocity_noise=0.025)

        self.source_samples = source_samples
        self.source_tasks = source_tasks
        self.source_policies = source_policies
        self.m = len(source_tasks)

        if self.for_LSTDQ:
            self.n_features_q = all_phi_Q.shape[1]
        if self.for_LSTDV:
            self.n_features_v = all_phi_V.shape[1]

        self.L_P_eps_s_a_s_prime = np.zeros((self.m,) + source_tasks[0].env.transition_matrix.shape, dtype=np.float64)
        self.L_P_eps_s_prime = np.zeros((self.m, source_tasks[0].env.V.shape[0]), dtype=np.float64)
        self.delta_P_eps_theta_s_s_prime = np.zeros((self.m, source_tasks[0].env.V.shape[0], source_tasks[0].env.V.shape[0]), dtype=np.float64)
        self.delta_P_eps_theta_s = np.zeros((self.m, source_tasks[0].env.V.shape[0]), dtype=np.float64)
        self.delta_zeta = np.zeros((self.m,) + source_tasks[0].env.Q.shape, dtype=np.float64)
        self.delta_delta = np.zeros((self.m, source_tasks[0].env.V.shape[0]), dtype=np.float64)
        self.source_sizes = np.zeros(self.m, dtype=np.int64)

        if self.for_gradient:
            self.delta_J = np.zeros((self.m, 2), dtype=np.float64)

        if self.for_LSTDQ or self.for_LSTDV:
            self.M_P_eps_s_a_s_prime = np.zeros((self.m,) + source_tasks[0].env.transition_matrix.shape, dtype=np.float64)

        if self.for_LSTDQ:
            self.delta_d_q = np.zeros((self.m,) + source_tasks[0].env.Q.shape + source_tasks[0].env.Q.shape, dtype=np.float64)
            self.delta_A_q = np.zeros((self.m, self.n_features_q, self.n_features_q), dtype=np.float64)
            self.delta_b_q = np.zeros((self.m, self.n_features_q), dtype=np.float64)
            aux = np.swapaxes(np.multiply.outer(all_phi_Q, all_phi_Q), 1, 2)[np.arange(source_tasks[0].env.Q.size),
                                                                             np.arange(source_tasks[0].env.Q.size)]
            aux2 = np.swapaxes(np.multiply.outer(all_phi_Q, -self.gamma * all_phi_Q), 1, 2)
            self.var_phi_q = (aux[:, None, :, :] - aux2).reshape(source_tasks[0].env.Q.shape + source_tasks[0].env.Q.shape +
                                                                 (self.n_features_q, self.n_features_q))
            self.rho_q = all_phi_Q.reshape(source_tasks[0].env.Q.shape + (self.n_features_q,))[:, :, None, :] *\
                         self.source_tasks[0].env.r[:, :, :,None]

        if self.for_LSTDV:
            self.delta_d_v = np.zeros((self.m,) + source_tasks[0].env.Q.shape + source_tasks[0].env.V.shape,
                                      dtype=np.float64)
            self.delta_A_v = np.zeros((self.m, self.n_features_v, self.n_features_v), dtype=np.float64)
            self.delta_b_v = np.zeros((self.m, self.n_features_v), dtype=np.float64)
            aux = np.swapaxes(np.multiply.outer(all_phi_V, all_phi_V), 1, 2)[np.arange(source_tasks[0].env.V.size),
                                                                             np.arange(source_tasks[0].env.V.size)]
            aux2 = np.swapaxes(np.multiply.outer(all_phi_V, -self.gamma * all_phi_V), 1, 2)
            self.var_phi_v = aux[:, None, :, :] - aux2
            self.rho_v = all_phi_V[:, None, None, :] * self.source_tasks[0].env.r[:, :, :, None]

        for j in range(self.m):
            for s in range(source_tasks[0].env.transition_matrix.shape[0]):
                mu_peak_pos_e = np.zeros(source_tasks[j].env.position_reps.shape[0], dtype=np.float64)
                mu_peak_vel_e = np.zeros(source_tasks[j].env.velocity_reps.shape[0], dtype=np.float64)
                for pos_prime in range(source_tasks[j].env.position_reps.shape[0]):
                    if pos_prime != 0 and pos_prime != source_tasks[j].env.position_reps.shape[0] - 1:
                        p1 = source_tasks[j].env.position_bins[pos_prime]
                        p2 = source_tasks[j].env.position_bins[pos_prime + 1]
                        def f(x):
                            return (np.log(p2 - x) - np.log(p1 - x) + (p1 ** 2 - p2 ** 2) / (2. * source_tasks[j].env.position_noise ** 2)) / ((p1 - p2) / source_tasks[j].env.position_noise ** 2) - x
                        mu_peak_pos_e[pos_prime] = root(f, p1 - 0.1).x
                for vel_prime in range(source_tasks[j].env.velocity_reps.shape[0]):
                    if vel_prime != 0 and vel_prime != source_tasks[j].env.velocity_reps.shape[0] - 1:
                        v1 = source_tasks[j].env.velocity_bins[vel_prime]
                        v2 = source_tasks[j].env.velocity_bins[vel_prime + 1]
                        def f(x):
                            return (np.log(v2 - x) - np.log(v1 - x) + (v1 ** 2 - v2 ** 2) / (2. * source_tasks[j].env.velocity_noise ** 2)) / ((v1 - v2) / source_tasks[j].env.velocity_noise ** 2) - x
                        mu_peak_vel_e[vel_prime] = root(f, v1 - 0.01).x
                for a in range(source_tasks[0].env.transition_matrix.shape[1]):
                    M_e_pos = np.zeros(source_tasks[j].env.position_reps.shape[0], dtype=np.float64)
                    M_e_vel = np.zeros(source_tasks[j].env.velocity_reps.shape[0], dtype=np.float64)
                    M_P_pos = np.zeros(source_tasks[j].env.position_reps.shape[0], dtype=np.float64)
                    M_P_vel = np.zeros(source_tasks[j].env.velocity_reps.shape[0], dtype=np.float64)
                    mu_min_source = min_source.env.clean_step(source_tasks[j].env.state_reps[s],
                                                              source_tasks[j].env.action_reps[a:a + 1])
                    mu_max_source = max_source.env.clean_step(source_tasks[j].env.state_reps[s],
                                                              source_tasks[j].env.action_reps[a:a + 1])
                    for pos_prime in range(source_tasks[j].env.position_reps.shape[0]):
                        if pos_prime == 0:
                            eps_peak =\
                                (source_tasks[j].env.position_bins[1] - source_tasks[j].env.state_reps[s].sum() +
                                 source_tasks[j].env.rescale(0.0025) * math.cos(3 * source_tasks[j].env.inverse_transform(source_tasks[j].env.state_reps[s][0]))) / \
                                source_tasks[j].env.action_reps[a]
                            if eps_peak < min_power or eps_peak > max_power:
                                M_e_pos[pos_prime] = max(np.exp(-(source_tasks[j].env.position_bins[1] - mu_min_source[0]) ** 2 /
                                                                (2. * source_tasks[j].env.position_noise ** 2)),
                                                         np.exp(-(source_tasks[j].env.position_bins[1] - mu_max_source[0]) ** 2 /
                                                                (2. * source_tasks[j].env.position_noise ** 2)))
                            else:
                                M_e_pos[pos_prime] = 1.
                            eps_peak =\
                                (source_tasks[j].env.position_bins[0] - source_tasks[j].env.state_reps[s].sum() +
                                 source_tasks[j].env.rescale(0.0025) * math.cos(3 * source_tasks[j].env.inverse_transform(source_tasks[j].env.state_reps[s][0]))) / \
                                source_tasks[j].env.action_reps[a]
                            if eps_peak < min_power or eps_peak > max_power:
                                M_P_pos[pos_prime] = max((1. + erf((source_tasks[j].env.position_bins[1] - mu_min_source[0]) /
                                                                   (np.sqrt(2.) * source_tasks[j].env.position_noise))) / 2.,
                                                         (1. + erf((source_tasks[j].env.position_bins[1] - mu_max_source[0]) /
                                                                   (np.sqrt(2.) * source_tasks[j].env.position_noise))) / 2.)
                            else:
                                M_P_pos[pos_prime] = (1. + erf((source_tasks[j].env.position_bins[1] - source_tasks[j].env.position_bins[0]) /
                                                               (np.sqrt(2.) * source_tasks[j].env.position_noise))) / 2.
                        elif pos_prime == source_tasks[j].env.position_reps.shape[0] - 1:
                            eps_peak = \
                                (source_tasks[j].env.position_bins[-2] - source_tasks[j].env.state_reps[s].sum() +
                                 source_tasks[j].env.rescale(0.0025) * math.cos(3 * source_tasks[j].env.inverse_transform(source_tasks[j].env.state_reps[s][0]))) / \
                                source_tasks[j].env.action_reps[a]
                            if eps_peak < min_power or eps_peak > max_power:
                                M_e_pos[pos_prime] = max(np.exp(-(source_tasks[j].env.position_bins[-2] - mu_min_source[0]) ** 2 /
                                                                (2. * source_tasks[j].env.position_noise ** 2)),
                                                         np.exp(-(source_tasks[j].env.position_bins[-2] - mu_max_source[0]) ** 2 /
                                                                (2. * source_tasks[j].env.position_noise ** 2)))
                            else:
                                M_e_pos[pos_prime] = 1.
                            eps_peak =\
                                (source_tasks[j].env.position_bins[-1] - source_tasks[j].env.state_reps[s].sum() +
                                 source_tasks[j].env.rescale(0.0025) * math.cos(3 * source_tasks[j].env.inverse_transform(source_tasks[j].env.state_reps[s][0]))) / \
                                source_tasks[j].env.action_reps[a]
                            if eps_peak < min_power or eps_peak > max_power:
                                M_P_pos[pos_prime] = max((1. - erf((source_tasks[j].env.position_bins[-2] - mu_min_source[0]) /
                                                                   (np.sqrt(2.) * source_tasks[j].env.position_noise))) / 2.,
                                                         (1. - erf((source_tasks[j].env.position_bins[-2] - mu_max_source[0]) /
                                                                   (np.sqrt(2.) * source_tasks[j].env.position_noise))) / 2.)
                            else:
                                M_P_pos[pos_prime] = (1. - erf((source_tasks[j].env.position_bins[-2] - source_tasks[j].env.position_bins[-1]) /
                                                               (np.sqrt(2.) * source_tasks[j].env.position_noise))) / 2.
                        else:
                            p1 = source_tasks[j].env.position_bins[pos_prime]
                            p2 = source_tasks[j].env.position_bins[pos_prime + 1]
                            pm = (p1 + p2) / 2.
                            mu_peak = mu_peak_pos_e[pos_prime]
                            if mu_peak < p1:
                                mu_peak1 = mu_peak
                                mu_peak2 = 2 * pm - mu_peak
                            else:
                                mu_peak2 = mu_peak
                                mu_peak1 = 2 * pm - mu_peak
                            eps_peak1 =\
                                (mu_peak1 - source_tasks[j].env.state_reps[s].sum() +
                                 source_tasks[j].env.rescale(0.0025) * math.cos(3 * source_tasks[j].env.inverse_transform(source_tasks[j].env.state_reps[s][0]))) / \
                                source_tasks[j].env.action_reps[a]
                            eps_peak2 =\
                                (mu_peak2 - source_tasks[j].env.state_reps[s].sum() +
                                 source_tasks[j].env.rescale(0.0025) * math.cos(3 * source_tasks[j].env.inverse_transform(source_tasks[j].env.state_reps[s][0]))) / \
                                source_tasks[j].env.action_reps[a]
                            if (eps_peak1 < min_power or eps_peak1 > max_power) and (eps_peak2 < min_power or eps_peak2 > max_power):
                                M_e_pos[pos_prime] = max(np.abs(np.exp(-(p1 - mu_min_source[0]) ** 2 /
                                                                       (2 * source_tasks[j].env.position_noise ** 2)) -
                                                                np.exp(-(p2 - mu_min_source[0]) ** 2 /
                                                                       (2 * source_tasks[j].env.position_noise ** 2))),
                                                         np.abs(np.exp(-(p1 - mu_max_source[0]) ** 2 /
                                                                       (2 * source_tasks[j].env.position_noise ** 2))
                                                                - np.exp(-(p2 - mu_max_source[0]) ** 2 /
                                                                         (2 * source_tasks[j].env.position_noise ** 2))))
                            else:
                                M_e_pos[pos_prime] =\
                                    np.abs(np.exp(-(p1 - mu_peak) ** 2 / (2 * source_tasks[j].env.position_noise ** 2)) -
                                           np.exp(-(p2 - mu_peak) ** 2 / (2 * source_tasks[j].env.position_noise ** 2)))
                            eps_peak =\
                                (pm - source_tasks[j].env.state_reps[s].sum() +
                                 source_tasks[j].env.rescale(0.0025) * math.cos(3 * source_tasks[j].env.inverse_transform(source_tasks[j].env.state_reps[s][0]))) / \
                                source_tasks[j].env.action_reps[a]
                            if eps_peak < min_power or eps_peak > max_power:
                                M_P_pos[pos_prime] =\
                                    max((erf((p2 - mu_min_source[0]) /(np.sqrt(2.) * source_tasks[j].env.position_noise)) -
                                         erf((p1 - mu_min_source[0]) / (np.sqrt(2.) * source_tasks[j].env.position_noise))) / 2.,
                                        (erf((p2 - mu_max_source[0]) / (np.sqrt(2.) * source_tasks[j].env.position_noise)) -
                                         erf((p1 - mu_max_source[0]) / (np.sqrt(2.) * source_tasks[j].env.position_noise))) / 2.)
                            else:
                                M_P_pos[pos_prime] = (erf((p2 - pm) / (np.sqrt(2.) * source_tasks[j].env.position_noise)) -
                                                      erf((p1 - pm) / (np.sqrt(2.) * source_tasks[j].env.position_noise))) / 2.
                    for vel_prime in range(source_tasks[j].env.velocity_reps.shape[0]):
                        real_vel_min_source =\
                            source_tasks[j].env.state_reps[s][1] + source_tasks[j].env.action_reps[a] * 0. -\
                            source_tasks[j].env.rescale(0.0025) * math.cos(3 * source_tasks[j].env.inverse_transform(source_tasks[j].env.state_reps[s][0]))
                        real_vel_max_source =\
                            source_tasks[j].env.state_reps[s][1] + source_tasks[j].env.action_reps[a] * 0.5 - \
                            source_tasks[j].env.rescale(0.0025) * math.cos(3 * source_tasks[j].env.inverse_transform(source_tasks[j].env.state_reps[s][0]))
                        real_vel_min_source, real_vel_max_source = np.clip([real_vel_min_source, real_vel_max_source], source_tasks[j].env.min_speed, source_tasks[j].env.max_speed)
                        if vel_prime == 0:
                            eps_peak =\
                                (source_tasks[j].env.velocity_bins[1] - source_tasks[j].env.state_reps[s][1] +
                                 source_tasks[j].env.rescale(0.0025) * math.cos(3 * source_tasks[j].env.inverse_transform(source_tasks[j].env.state_reps[s][0]))) / \
                                source_tasks[j].env.action_reps[a]
                            if (eps_peak < min_power or eps_peak > max_power) and\
                                (not (real_vel_min_source < 0 and mu_min_source[0] <= source_tasks[j].env.position_bins[1]) and
                                 not (real_vel_max_source < 0 and mu_max_source[0] <= source_tasks[j].env.position_bins[1])):
                                M_e_vel[vel_prime] = max(np.exp(-(source_tasks[j].env.velocity_bins[1] - mu_min_source[1]) ** 2 /
                                                                (2. * source_tasks[j].env.velocity_noise ** 2)),
                                                         np.exp(-(source_tasks[j].env.velocity_bins[1] - mu_max_source[1]) ** 2 /
                                                                (2. * source_tasks[j].env.velocity_noise ** 2)))
                            else:
                                M_e_vel[vel_prime] = 1.
                            eps_peak  =\
                                (source_tasks[j].env.velocity_bins[0] - source_tasks[j].env.state_reps[s][1] +
                                 source_tasks[j].env.rescale(0.0025) * math.cos(3 * source_tasks[j].env.inverse_transform(source_tasks[j].env.state_reps[s][0]))) / \
                                source_tasks[j].env.action_reps[a]
                            if (eps_peak < min_power or eps_peak > max_power) and \
                                (not (real_vel_min_source < 0 and mu_min_source[0] <= source_tasks[j].env.position_bins[1]) and
                                 not (real_vel_max_source < 0 and mu_max_source[0] <= source_tasks[j].env.position_bins[1])):
                                M_P_vel[vel_prime] = max((1. + erf((source_tasks[j].env.velocity_bins[1] - mu_min_source[1]) /
                                                                   (np.sqrt(2.) * source_tasks[j].env.velocity_noise))) / 2.,
                                                         (1. + erf((source_tasks[j].env.velocity_bins[1] - mu_max_source[1]) /
                                                                   (np.sqrt(2.) * source_tasks[j].env.velocity_noise))) / 2.)
                            else:
                                M_P_vel[vel_prime] = (1. + erf((source_tasks[j].env.velocity_bins[1] - source_tasks[j].env.velocity_bins[0]) /
                                                               (np.sqrt(2.) * source_tasks[j].env.velocity_noise))) / 2.
                        elif vel_prime == source_tasks[j].env.velocity_reps.shape[0] - 1:
                            eps_peak = \
                                (source_tasks[j].env.velocity_bins[-2] - source_tasks[j].env.state_reps[s][1] +
                                 source_tasks[j].env.rescale(0.0025) * math.cos(3 * source_tasks[j].env.inverse_transform(source_tasks[j].env.state_reps[s][0]))) / \
                                source_tasks[j].env.action_reps[a]
                            if (eps_peak < min_power or eps_peak > max_power) and \
                                (not (real_vel_min_source < 0 and mu_min_source[0] <= source_tasks[j].env.position_bins[1]) and
                                 not (real_vel_max_source < 0 and mu_max_source[0] <= source_tasks[j].env.position_bins[1])):
                                M_e_vel[vel_prime] = max(
                                    np.exp(-(source_tasks[j].env.velocity_bins[-2] - mu_min_source[1]) ** 2 /
                                           (2. * source_tasks[j].env.velocity_noise ** 2)),
                                    np.exp(-(source_tasks[j].env.velocity_bins[-2] - mu_max_source[1]) ** 2 /
                                           (2. * source_tasks[j].env.velocity_noise ** 2)))
                            else:
                                M_e_vel[vel_prime] = 1.
                            eps_peak = \
                                (source_tasks[j].env.velocity_bins[-1] - source_tasks[j].env.state_reps[s][1] +
                                 source_tasks[j].env.rescale(0.0025) * math.cos(3 * source_tasks[j].env.inverse_transform(source_tasks[j].env.state_reps[s][0]))) / \
                                source_tasks[j].env.action_reps[a]
                            if (eps_peak < min_power or eps_peak > max_power) and \
                                (not (real_vel_min_source < 0 and mu_min_source[0] <= source_tasks[j].env.position_bins[1]) and
                                 not (real_vel_max_source < 0 and mu_max_source[0] <= source_tasks[j].env.position_bins[1])):
                                M_P_vel[vel_prime] = max((1. - erf((source_tasks[j].env.velocity_bins[-2] - mu_min_source[1]) /
                                                                   (np.sqrt(2.) * source_tasks[j].env.velocity_noise))) / 2.,
                                                         (1. - erf((source_tasks[j].env.velocity_bins[-2] - mu_max_source[1]) /
                                                                   (np.sqrt(2.) * source_tasks[j].env.velocity_noise))) / 2.)
                            else:
                                M_P_vel[vel_prime] = (1. - erf((source_tasks[j].env.velocity_bins[-2] - source_tasks[j].env.velocity_bins[-1]) /
                                                               (np.sqrt(2.) * source_tasks[j].env.velocity_noise))) / 2.
                        else:
                            v1 = source_tasks[j].env.velocity_bins[vel_prime]
                            v2 = source_tasks[j].env.velocity_bins[vel_prime + 1]
                            vm = (v1 + v2) / 2.
                            mu_peak = mu_peak_vel_e[vel_prime]
                            if mu_peak < v1:
                                mu_peak1 = mu_peak
                                mu_peak2 = 2 * vm - mu_peak
                            else:
                                mu_peak2 = mu_peak
                                mu_peak1 = 2 * vm - mu_peak
                            eps_peak1 =\
                                (mu_peak1 - source_tasks[j].env.state_reps[s][1] +
                                 source_tasks[j].env.rescale(0.0025) * math.cos(3 * source_tasks[j].env.inverse_transform(source_tasks[j].env.state_reps[s][0]))) / \
                                source_tasks[j].env.action_reps[a]
                            eps_peak2 =\
                                (mu_peak2 - source_tasks[j].env.state_reps[s][1] +
                                 source_tasks[j].env.rescale(0.0025) * math.cos(3 * source_tasks[j].env.inverse_transform(source_tasks[j].env.state_reps[s][0]))) / \
                                source_tasks[j].env.action_reps[a]
                            if (eps_peak1 < min_power or eps_peak1 > max_power) and (eps_peak2 < min_power or eps_peak2 > max_power) and\
                                (not (real_vel_min_source < 0 and mu_min_source[0] == source_tasks[j].env.position_bins[1]) and
                                 not (real_vel_max_source < 0 and mu_max_source[0] == source_tasks[j].env.position_bins[1])):
                                M_e_vel[vel_prime] =\
                                    max(np.abs(np.exp(-(v1 - mu_min_source[1]) ** 2 / (2 * source_tasks[j].env.velocity_noise ** 2)) -
                                               np.exp(-(v2 - mu_min_source[1]) ** 2 / (2 * source_tasks[j].env.velocity_noise ** 2))),
                                        np.abs(np.exp(-(v1 - mu_max_source[1]) ** 2 / (2 * source_tasks[j].env.velocity_noise ** 2)) -
                                               np.exp(-(v2 - mu_max_source[1]) ** 2 / (2 * source_tasks[j].env.velocity_noise ** 2))))
                            else:
                                M_e_vel[vel_prime] =\
                                    np.abs(np.exp(-(v1 - mu_peak) ** 2 / (2 * source_tasks[j].env.velocity_noise ** 2)) -
                                           np.exp(-(v2 - mu_peak) ** 2 / (2 * source_tasks[j].env.velocity_noise ** 2)))
                            eps_peak =\
                                (vm - source_tasks[j].env.state_reps[s][1] +
                                 source_tasks[j].env.rescale(0.0025) * math.cos(3 * source_tasks[j].env.inverse_transform(source_tasks[j].env.state_reps[s][0]))) / \
                                source_tasks[j].env.action_reps[a]
                            if (eps_peak < min_power or eps_peak > max_power) and\
                                (not (real_vel_min_source < 0 and mu_min_source[0] == source_tasks[j].env.position_bins[1]) and
                                 not (real_vel_max_source < 0 and mu_max_source[0] == source_tasks[j].env.position_bins[1])):
                                M_P_vel[vel_prime] =\
                                    max((erf((v2 - mu_min_source[1]) / (np.sqrt(2.) * source_tasks[j].env.velocity_noise)) -
                                         erf((v1 - mu_min_source[1]) / (np.sqrt(2.) * source_tasks[j].env.velocity_noise))) / 2.,
                                        (erf((v2 - mu_max_source[1]) / (np.sqrt(2.) * source_tasks[j].env.velocity_noise)) -
                                         erf((v1 - mu_max_source[1]) / (np.sqrt(2.) * source_tasks[j].env.velocity_noise))) / 2.)
                            else:
                                M_P_vel[vel_prime] =\
                                    (erf((v2 - vm) / (np.sqrt(2.) * source_tasks[j].env.velocity_noise)) -
                                     erf((v1 - vm) / (np.sqrt(2.) * source_tasks[j].env.velocity_noise))) / 2.
                    self.L_P_eps_s_a_s_prime[j, s, a] =\
                        ((np.abs(source_tasks[j].env.action_reps[a]) / (np.sqrt(2. * np.pi) * source_tasks[j].env.velocity_noise)) * M_e_vel.reshape((-1, 1)).dot(M_P_pos.reshape((1, -1))) +
                         (np.abs(source_tasks[j].env.action_reps[a]) / (np.sqrt(2. * np.pi) * source_tasks[j].env.position_noise)) * M_P_vel.reshape((-1, 1)).dot(M_e_pos.reshape((1, -1)))).flatten()

            self.L_P_eps_s_prime[j] = self.L_P_eps_s_a_s_prime[j].max(axis=(0, 1))

            self.source_sizes[j] = self.source_samples[j]['fs'].shape[0]

            if self.for_gradient:
                self.source_samples[j]['eta_j'] =\
                    source_policies[j].log_gradient_matrix[source_samples[j]['fsi'], source_samples[j]['ai']] *\
                    (source_tasks[j].env.Q[source_samples[j]['fsi'], source_samples[j]['ai']] -
                     source_tasks[j].env.V[source_samples[j]['fsi']]).reshape((-1, 1))

            if self.for_LSTDQ:
                self.source_samples[j]['var_phi_q'] = \
                    self.var_phi_q[self.source_samples[j]['fsi'], self.source_samples[j]['ai'],
                                 self.source_samples[j]['nsi'], self.source_samples[j]['nai']]

                self.source_samples[j]['rho_q'] = \
                    self.rho_q[self.source_samples[j]['fsi'], self.source_samples[j]['ai'],
                             self.source_samples[j]['nsi']]

            if self.for_LSTDV:
                self.source_samples[j]['var_phi_v'] = \
                    self.var_phi_v[self.source_samples[j]['fsi'], self.source_samples[j]['nsi']]

                self.source_samples[j]['rho_v'] = \
                    self.rho_v[self.source_samples[j]['fsi'], self.source_samples[j]['ai'],
                               self.source_samples[j]['nsi']]

        if self.for_gradient:
            self.l_bounds_grad = np.zeros(self.source_sizes.sum(), dtype=np.float64)
            self.u_bounds_grad = np.zeros(self.source_sizes.sum(), dtype=np.float64)
        if self.for_LSTDQ:
            self.l_bounds_lstdq = np.zeros(self.source_sizes.sum(), dtype=np.float64)
            self.u_bounds_lstdq = np.zeros(self.source_sizes.sum(), dtype=np.float64)
        if self.for_LSTDV:
            self.l_bounds_lstdv = np.zeros(self.source_sizes.sum(), dtype=np.float64)
            self.u_bounds_lstdv = np.zeros(self.source_sizes.sum(), dtype=np.float64)



    def clean_sources(self):
        pass



    def prepare_lstd(self, target_policy, target_power):
        w_idx = np.hstack((0., self.source_sizes)).cumsum().astype(np.int64)
        for i in range(self.m):
            self.delta_P_eps_theta_s_s_prime[i] = \
                (self.source_tasks[i].env.transition_matrix * np.abs(self.source_policies[i].choice_matrix - target_policy.choice_matrix)[:,:,None] +
                 target_policy.choice_matrix.reshape(target_policy.choice_matrix.shape + (1,)) *
                 self.L_P_eps_s_a_s_prime[i] * np.abs(self.source_tasks[i].env.power - target_power)).sum(axis=1)
            self.delta_P_eps_theta_s[i] = self.delta_P_eps_theta_s_s_prime[i].max(axis=0)
            self.delta_delta[i] =\
                (1. - self.gamma) * self.source_tasks[i].env.P_pi_inv.T.dot(0. +self.gamma * np.minimum(self.delta_P_eps_theta_s[i],
                                                                                                        np.ones_like(self.delta_P_eps_theta_s[i])))
            self.delta_zeta[i] =\
                self.source_tasks[i].env.delta_distr[:,None] * np.abs(self.source_policies[i].choice_matrix -target_policy.choice_matrix) + \
                target_policy.choice_matrix * np.minimum(self.delta_delta[i], np.ones_like(self.delta_delta[i]))[:,None]

            if self.for_LSTDV or self.for_LSTDQ:
                self.M_P_eps_s_a_s_prime[i] = np.minimum(np.ones_like(self.M_P_eps_s_a_s_prime[i]),
                                                         self.source_tasks[i].env.transition_matrix +
                                                         self.L_P_eps_s_a_s_prime*np.abs(self.source_tasks[i].env.power - target_power))
            if self.for_LSTDQ:
                self.delta_d_q[i] = self.source_tasks[i].env.zeta_distr[:, :, None, None] *\
                                    self.source_tasks[i].env.transition_matrix[:, :, :, None] * \
                                    np.abs(target_policy.choice_matrix - self.source_policies[i].choice_matrix)[None,None, :, :] + \
                                    target_policy.choice_matrix[None, None, :, :] * self.source_tasks[i].env.zeta_distr[:,:, None, None] * \
                                    np.minimum(self.L_P_eps_s_a_s_prime * np.abs(self.source_tasks[i].env.power - target_power),
                                               np.ones_like(self.L_P_eps_s_a_s_prime))[:, :, :, None] + \
                                    target_policy.choice_matrix[None, None, :, :] * self.M_P_eps_s_a_s_prime[i][:, :, :,None] * \
                                    np.minimum(self.delta_zeta[i], np.ones_like(self.delta_zeta[i]))[:, :, None, None]

                self.delta_A_q[i] = (self.var_phi_q * np.minimum(self.delta_d_q[i], np.ones_like(self.delta_d_q[i]))[:, :, :, :, None, None]).sum(axis=(0, 1, 2, 3))
                self.delta_b_q[i] = (self.rho_q[:,:,:,None,:] * np.minimum(self.delta_d_q[i], np.ones_like(self.delta_d_q[i]))[:, :, :, :,None]).sum(axis=(0, 1, 2, 3))
                source_d_distr = self.source_tasks[i].env.zeta_distr[self.source_samples[i]['fsi'], self.source_samples[i]['ai']] * \
                                 self.source_tasks[i].env.transition_matrix[self.source_samples[i]['fsi'], self.source_samples[i]['ai'],
                                                                            self.source_samples[i]['nsi']] * \
                                 self.source_policies[self.source_samples[i]['nsi'], self.source_samples[i]['nai']]
                self.l_bounds_lstdq[w_idx[i]:w_idx[i + 1]] = np.maximum(np.zeros(self.source_sizes[i], dtype=np.float64),
                                                                        np.ones(self.source_sizes[i], dtype=np.float64) -
                                                                        np.minimum(np.ones_like(self.delta_d_q[i]),
                                                                                   self.delta_d_q[i])[self.source_samples[i]['fsi'],
                                                                                                      self.source_samples[i]['ai'],
                                                                                                      self.source_samples[i]['nsi'],
                                                                                                      self.source_samples[i]['nai']] /
                                                                        source_d_distr)
                self.u_bounds_lstdq[w_idx[i]:w_idx[i + 1]] = np.ones(self.source_sizes[i], dtype=np.float64) +\
                                                             np.minimum(np.ones_like(self.delta_d_q[i]),
                                                                        self.delta_d_q[i])[self.source_samples[i]['fsi'],
                                                                                           self.source_samples[i]['ai'],
                                                                                           self.source_samples[i]['nsi'],
                                                                                           self.source_samples[i]['nai']] / \
                                                             source_d_distr
            if self.for_LSTDV:
                self.delta_d_v[i] = self.source_tasks[i].env.zeta_distr[:,:, None] * \
                                    np.minimum(self.L_P_eps_s_a_s_prime * np.abs(self.source_tasks[i].env.power - target_power),
                                               np.ones_like(self.L_P_eps_s_a_s_prime)) + \
                                    self.M_P_eps_s_a_s_prime[i] * np.minimum(self.delta_zeta[i], np.ones_like(self.delta_zeta[i]))[:, :, None]

                self.delta_A_v[i] = (self.var_phi_v[:,None,:,:,:] * np.minimum(self.delta_d_v[i], np.ones_like(self.delta_d_v[i]))[:, :, :, None, None]).sum(axis=(0, 1, 2))
                self.delta_b_v[i] = np.abs(self.rho_v * np.minimum(self.delta_d_v[i], np.ones_like(self.delta_d_v[i]))[:, :, :,None]).sum(axis=(0, 1, 2))
                source_d_distr = self.source_tasks[i].env.zeta_distr[self.source_samples[i]['fsi'], self.source_samples[i]['ai']] * \
                                 self.source_tasks[i].env.transition_matrix[self.source_samples[i]['fsi'], self.source_samples[i]['ai'],
                                                                            self.source_samples[i]['nsi']]
                self.l_bounds_lstdv[w_idx[i]:w_idx[i + 1]] = np.maximum(np.zeros(self.source_sizes[i], dtype=np.float64),
                                                                        np.ones(self.source_sizes[i], dtype=np.float64) -
                                                                        np.minimum(np.ones_like(self.delta_d_v[i]),
                                                                                   self.delta_d_v[i])[self.source_samples[i]['fsi'],
                                                                                                      self.source_samples[i]['ai'],
                                                                                                      self.source_samples[i]['nsi']] /
                                                                        source_d_distr)
                self.u_bounds_lstdv[w_idx[i]:w_idx[i + 1]] = np.ones(self.source_sizes[i], dtype=np.float64) +\
                                                             np.minimum(np.ones_like(self.delta_d_v[i]),
                                                                        self.delta_d_v[i])[self.source_samples[i]['fsi'],
                                                                                           self.source_samples[i]['ai'],
                                                                                           self.source_samples[i]['nsi']] / \
                                                             source_d_distr



    def prepare_gradient(self, target_policy, target_power, all_target_Q, target_V):
        w_idx = np.hstack((0., self.source_sizes)).cumsum().astype(np.int64)
        for i in range(self.m):
            if not self.for_LSTDQ and not self.for_LSTDV:
                self.delta_P_eps_theta_s_s_prime[i] = \
                    (self.source_tasks[i].env.transition_matrix * np.abs(self.source_policies[i].choice_matrix - target_policy.choice_matrix)[:, :, None] +
                     target_policy.choice_matrix.reshape(target_policy.choice_matrix.shape + (1,)) *
                     self.L_P_eps_s_a_s_prime[i] * np.abs(self.source_tasks[i].env.power - target_power)).sum(axis=1)
                self.delta_P_eps_theta_s[i] = self.delta_P_eps_theta_s_s_prime[i].max(axis=0)
                self.delta_delta[i] = \
                    (1. - self.gamma) * self.source_tasks[i].env.P_pi_inv.T.dot(0. + self.gamma * np.minimum(self.delta_P_eps_theta_s[i],
                                                                                                             np.ones_like(self.delta_P_eps_theta_s[i])))
                self.delta_zeta[i] = \
                    self.source_tasks[i].env.delta_distr[:, None] * np.abs(self.source_policies[i].choice_matrix - target_policy.choice_matrix) + \
                    target_policy.choice_matrix * np.minimum(self.delta_delta[i], np.ones_like(self.delta_delta[i]))[:, None]

            self.delta_J[i] =\
                (np.abs(target_policy.log_gradient_matrix * (all_target_Q * np.minimum(np.ones_like(self.delta_zeta[i]), self.delta_zeta[i]))[:,:,None]) +
                 self.source_tasks[i].env.zeta_distr[:,:,None] * np.abs(target_policy.log_gradient_matrix * all_target_Q[:,:,None] -
                                                                        self.source_policies[i].log_gradient_matrix * self.source_tasks[i].env.Q[:,:,None]))\
                    .sum(axis=(0, 1)) / (1. - self.gamma)

            self.source_samples[i]['eta_1'] =\
                target_policy.log_gradient_matrix[self.source_samples[i]['fsi'], self.source_samples[i]['ai']] *\
                (all_target_Q[self.source_samples[i]['fsi'], self.source_samples[i]['ai']] -
                 target_V[w_idx[i]:w_idx[i + 1]])[:,None]

            self.l_bounds_grad[w_idx[i]:w_idx[i + 1]] = np.maximum(np.zeros(self.source_sizes[i], dtype=np.float64),
                                                                   np.ones(self.source_sizes[i], dtype=np.float64) -
                                                                   np.minimum(np.ones_like(self.delta_zeta[i]),self.delta_zeta[i])[self.source_samples[i]['fsi'],
                                                                                                                                   self.source_samples[i]['ai']] /
                                                                   self.source_tasks[i].env.zeta_distr[self.source_samples[i]['fsi'],
                                                                                                       self.source_samples[i]['ai']])
            self.u_bounds_grad[w_idx[i]:w_idx[i + 1]] = np.ones(self.source_sizes[i], dtype=np.float64) +\
                                                        np.minimum(np.ones_like(self.delta_zeta[i]),
                                                                   self.delta_zeta[i])[self.source_samples[i]['fsi'], self.source_samples[i]['ai']] /\
                                                        self.source_tasks[i].env.zeta_distr[self.source_samples[i]['fsi'],
                                                                                            self.source_samples[i]['ai']]



    def estimate_weights_gradient(self, target_size):
        w_idx = np.hstack((0., self.source_sizes)).cumsum().astype(np.int64)
        n = self.source_sizes.sum() + target_size
        w0 = np.ones(n - target_size, dtype=np.float64)
        
        def g(w):
            bias = 0.
            vari = 0.
            for j in range(self.m):
                bias += (self.source_samples[j]['eta_j'] - w[w_idx[j]:w_idx[j+1]].reshape((-1,1))*self.source_samples[j]['eta_1']).sum(axis=0)/(n*(1. - self.gamma))
                vari += (((w[w_idx[j]:w_idx[j+1]].reshape((-1,1))*self.source_samples[j]['eta_1'])**2).sum(axis=0) -
                         ((w[w_idx[j]:w_idx[j+1]].reshape((-1,1))*self.source_samples[j]['eta_1']).sum(axis=0)) ** 2 / self.source_sizes[j]) / (n * (1. - self.gamma)) ** 2
            bias = np.abs(bias) + (self.delta_J * self.source_sizes.reshape((-1, 1))).sum(axis=0) / n
            bias = (bias**2).sum()
            vari = vari.sum()
            return bias + vari

        def grad_g(w):
            bias = 0.
            for j in range(self.m):
                bias += (self.source_samples[j]['eta_j'] - w[w_idx[j]:w_idx[j + 1]].reshape((-1, 1)) * self.source_samples[j]['eta_1']).sum(axis=0) / (n*(1. - self.gamma))
            sum_etas = bias
            bias = np.abs(bias) + (self.delta_J * self.source_sizes.reshape((-1, 1))).sum(axis=0) / n
            grad = np.zeros(w.shape + (2,), dtype=np.float64)
            for j in range(self.m):
                grad[w_idx[j]:w_idx[j+1]] = 2*self.source_samples[j]['eta_1']*(np.power(-1., sum_etas > 0) * bias / (1. - self.gamma) + self.source_samples[j]['eta_1'] *w[w_idx[j]:w_idx[j+1]].reshape((-1,1)) / (1. - self.gamma) ** 2 - (self.source_samples[j]['eta_1']*w[w_idx[j]:w_idx[j+1]].reshape((-1,1))).sum() / (self.source_sizes[j] * (1. - self.gamma) ** 2)) / n ** 2
            grad = grad.sum(axis=1)
            return grad

        bounds = np.ones((self.l_bounds_grad.size + self.u_bounds_grad.size,), dtype=np.float64)
        bounds[0::2] = self.l_bounds_grad
        bounds[1::2] = self.u_bounds_grad
        bounds = tuple(map(tuple, bounds.reshape((-1,2))))
        res = minimize(g, w0, jac=grad_g, bounds=bounds)

        return res.x



    def estimate_weights_lstdq(self, target_size):
        w_idx = np.hstack((0., self.source_sizes)).cumsum().astype(np.int64)
        n = self.source_sizes.sum() + target_size
        w0 = np.ones(n - target_size, dtype=np.float64)
        
        def g(w):
            bias_A = 0.
            bias_b = 0.
            vari_A = 0.
            vari_b = 0.
            for j in range(self.m):
                bias_A += (1. - w[w_idx[j]:w_idx[j+1]])[:,None,None]*self.source_samples[j]['var_phi_q']/n
                vari_A += (((w[w_idx[j]:w_idx[j + 1]][:,None,None]*self.source_samples[j]['var_phi_q'])**2).sum(axis=0) -
                          (w[w_idx[j]:w_idx[j + 1]][:,None,None]*self.source_samples[j]['var_phi_q']).sum(axis=0) ** 2 / self.source_sizes[j])\
                          / n ** 2
                bias_b += (1. - w[w_idx[j]:w_idx[j + 1]])[:, None, None] * self.source_samples[j]['rho_q'] / n
                vari_b += (((w[w_idx[j]:w_idx[j + 1]][:, None, None] * self.source_samples[j]['rho_q']) ** 2).sum(axis=0) -
                           (w[w_idx[j]:w_idx[j + 1]][:, None, None] * self.source_samples[j]['rho_q']).sum(axis=0) ** 2 /self.source_sizes[j]) \
                          / n ** 2
            bias_A = np.abs(bias_A) + (self.delta_A_q * self.source_sizes[:,None,None]).sum(axis=0) / n
            bias_A = (bias_A ** 2).sum()
            vari_A = vari_A.sum()
            bias_b = np.abs(bias_b) + (self.delta_b_q * self.source_sizes[:, None]).sum(axis=0) / n
            bias_b = (bias_b ** 2).sum()
            vari_b = vari_b.sum()
            return bias_A + vari_A + bias_b + vari_b

        def grad_g(w):
            bias_A = 0.
            bias_b = 0.
            for j in range(self.m):
                bias_A += (1. - w[w_idx[j]:w_idx[j + 1]])[:, None, None] * self.source_samples[j]['var_phi_q'] / n
                bias_b += (1. - w[w_idx[j]:w_idx[j + 1]])[:, None, None] * self.source_samples[j]['rho_q'] / n
            sum_etas_A = bias_A
            bias_A = np.abs(bias_A) + (self.delta_A_q * self.source_sizes[:,None,None]).sum(axis=0) / n
            sum_etas_b = bias_b
            bias_b = np.abs(bias_A) + (self.delta_b_q * self.source_sizes[:,None]).sum(axis=0) / n
            grad_A = np.zeros(w.shape + (self.n_features_q,self.n_features_q), dtype=np.float64)
            grad_b = np.zeros(w.shape + (self.n_features_q,), dtype=np.float64)
            for j in range(self.m):
                grad_A[w_idx[j]:w_idx[j + 1]] = 2 * self.source_samples[j]['var_phi_q'] *\
                                              (np.power(-1., sum_etas_A > 0) * bias_A +
                                               self.source_samples[j]['var_phi_q'] * w[w_idx[j]:w_idx[j + 1]][:,None,None] -
                                               (self.source_samples[j]['var_phi_q'] * w[w_idx[j]:w_idx[j + 1]][:,None,None]).sum() / self.source_sizes[j]) /\
                                              n ** 2
                grad_b[w_idx[j]:w_idx[j + 1]] = 2 * self.source_samples[j]['rho_q'] *\
                                              (np.power(-1., sum_etas_b > 0) * bias_b +
                                               self.source_samples[j]['rho_q'] * w[w_idx[j]:w_idx[j + 1]][:,None,None] -
                                               (self.source_samples[j]['rho_q'] * w[w_idx[j]:w_idx[j + 1]][:,None,None]).sum() / self.source_sizes[j]) /\
                                              n ** 2
            grad = grad_A.sum(axis=(1,2)) + grad_b.sum(axis=1)
            return grad

        bounds = np.ones((self.l_bounds_lstdq.size + self.u_bounds_lstdq.size,), dtype=np.float64)
        bounds[0::2] = self.l_bounds_lstdq
        bounds[1::2] = self.u_bounds_lstdq
        bounds = tuple(map(tuple, bounds.reshape((-1, 2))))
        res = minimize(g, w0, jac=grad_g, bounds=bounds)

        return res.x



    def estimate_weights_lstdv(self, target_size):
        w_idx = np.hstack((0., self.source_sizes)).cumsum().astype(np.int64)
        n = self.source_sizes.sum() + target_size
        w0 = np.ones(n - target_size, dtype=np.float64)

        def g(w):
            bias_A = 0.
            bias_b = 0.
            vari_A = 0.
            vari_b = 0.
            for j in range(self.m):
                bias_A += (1. - w[w_idx[j]:w_idx[j + 1]])[:, None, None] * self.source_samples[j]['var_phi_v'] / n
                vari_A += (((w[w_idx[j]:w_idx[j + 1]][:, None, None] * self.source_samples[j]['var_phi_v']) ** 2).sum(axis=0) -
                           (w[w_idx[j]:w_idx[j + 1]][:, None, None] * self.source_samples[j]['var_phi_v']).sum(axis=0) ** 2 / self.source_sizes[j]) \
                          / n ** 2
                bias_b += (1. - w[w_idx[j]:w_idx[j + 1]])[:, None, None] * self.source_samples[j]['rho_v'] / n
                vari_b += (((w[w_idx[j]:w_idx[j + 1]][:, None, None] * self.source_samples[j]['rho_v']) ** 2).sum(axis=0) -
                           (w[w_idx[j]:w_idx[j + 1]][:, None, None] * self.source_samples[j]['rho_v']).sum(axis=0) ** 2 / self.source_sizes[j]) \
                          / n ** 2
            bias_A = np.abs(bias_A) + (self.delta_A_v * self.source_sizes[:, None, None]).sum(axis=0) / n
            bias_A = (bias_A ** 2).sum()
            vari_A = vari_A.sum()
            bias_b = np.abs(bias_b) + (self.delta_b_v * self.source_sizes[:, None]).sum(axis=0) / n
            bias_b = (bias_b ** 2).sum()
            vari_b = vari_b.sum()
            return bias_A + vari_A + bias_b + vari_b

        def grad_g(w):
            bias_A = 0.
            bias_b = 0.
            for j in range(self.m):
                bias_A += (1. - w[w_idx[j]:w_idx[j + 1]])[:, None, None] * self.source_samples[j]['var_phi_v'] / n
                bias_b += (1. - w[w_idx[j]:w_idx[j + 1]])[:, None, None] * self.source_samples[j]['rho_v'] / n
            sum_etas_A = bias_A
            bias_A = np.abs(bias_A) + (self.delta_A_v * self.source_sizes[:, None, None]).sum(axis=0) / n
            sum_etas_b = bias_b
            bias_b = np.abs(bias_A) + (self.delta_b_v * self.source_sizes[:, None]).sum(axis=0) / n
            grad_A = np.zeros(w.shape + (self.n_features_v, self.n_features_v), dtype=np.float64)
            grad_b = np.zeros(w.shape + (self.n_features_v,), dtype=np.float64)
            for j in range(self.m):
                grad_A[w_idx[j]:w_idx[j + 1]] = 2 * self.source_samples[j]['var_phi_v'] * \
                                                (np.power(-1., sum_etas_A > 0) * bias_A +
                                                 self.source_samples[j]['var_phi_v'] * w[w_idx[j]:w_idx[j + 1]][:, None, None] -
                                                 (self.source_samples[j]['var_phi_v'] * w[w_idx[j]:w_idx[j + 1]][:, None, None]).sum() / self.source_sizes[j]) / \
                                                n ** 2
                grad_b[w_idx[j]:w_idx[j + 1]] = 2 * self.source_samples[j]['rho_v'] * \
                                                (np.power(-1., sum_etas_b > 0) * bias_b +
                                                 self.source_samples[j]['rho_v'] * w[w_idx[j]:w_idx[j + 1]][:, None, None] -
                                                 (self.source_samples[j]['rho_v'] * w[w_idx[j]:w_idx[j + 1]][:, None, None]).sum() / self.source_sizes[j]) / \
                                                n ** 2
            grad = grad_A.sum(axis=(1, 2)) + grad_b.sum(axis=1)
            return grad

        bounds = np.ones((self.l_bounds_lstdv.size + self.u_bounds_lstdv.size,), dtype=np.float64)
        bounds[0::2] = self.l_bounds_lstdv
        bounds[1::2] = self.u_bounds_lstdv
        bounds = tuple(map(tuple, bounds.reshape((-1, 2))))
        res = minimize(g, w0, jac=grad_g, bounds=bounds)

        return res.x