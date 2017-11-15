import numpy as np
import math
from scipy.optimize import minimize, root
from scipy.special import erf
import gym
import time

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
            self.all_phi_Q = all_phi_Q
            self.n_features_q = all_phi_Q.shape[1]
        if self.for_LSTDV:
            self.all_phi_V = all_phi_V
            self.n_features_v = all_phi_V.shape[1]

        self.L_P_eps_s_a_s_prime = np.zeros((self.m,) + source_tasks[0].env.transition_matrix.shape, dtype=np.float64)
        self.L_P_eps_s_prime = np.zeros((self.m, source_tasks[0].env.V.shape[0]), dtype=np.float64)
        self.delta_P_eps_theta_s_s_prime = np.zeros((self.m, source_tasks[0].env.V.shape[0], source_tasks[0].env.V.shape[0]), dtype=np.float64)
        self.delta_P_eps_theta_s = np.zeros((self.m, source_tasks[0].env.V.shape[0]), dtype=np.float64)
        self.delta_zeta = np.zeros((self.m,) + source_tasks[0].env.Q.shape, dtype=np.float64)
        self.delta_delta = np.zeros((self.m, source_tasks[0].env.V.shape[0]), dtype=np.float64)
        self.source_sizes = np.zeros(self.m, dtype=np.int64)

        if self.for_gradient:
            self.reduced_source_sizes_grad = np.zeros(self.m, dtype=np.int64)
            self.delta_J = np.zeros((self.m, 2), dtype=np.float64)

        if self.for_LSTDQ or self.for_LSTDV:
            self.reduced_source_sizes_q = np.zeros(self.m, dtype=np.int64)
            self.reduced_source_sizes_v = np.zeros(self.m, dtype=np.int64)
            self.M_P_eps_s_a_s_prime = np.zeros((self.m,) + source_tasks[0].env.transition_matrix.shape, dtype=np.float64)

        if self.for_LSTDQ:
            self.zeta_rep = np.zeros((self.m,) + source_tasks[0].env.Q.shape + source_tasks[0].env.Q.shape, dtype=np.float64)
            self.p_rep = np.zeros((self.m,) + source_tasks[0].env.Q.shape + source_tasks[0].env.Q.shape, dtype=np.float64)
            self.L_p_rep = np.zeros((self.m,) + source_tasks[0].env.Q.shape + source_tasks[0].env.Q.shape, dtype=np.float64)
            self.M_p_rep = np.zeros((self.m,) + source_tasks[0].env.Q.shape + source_tasks[0].env.Q.shape, dtype=np.float64)
            self.delta_d_q = np.zeros((self.m,) + source_tasks[0].env.Q.shape + source_tasks[0].env.Q.shape, dtype=np.float64)
            self.delta_d_q_b = np.zeros((self.m,) + source_tasks[0].env.Q.shape + source_tasks[0].env.V.shape, dtype=np.float64)
            self.delta_A_q = np.zeros((self.m, self.n_features_q, self.n_features_q), dtype=np.float64)
            self.delta_b_q = np.zeros((self.m, self.n_features_q), dtype=np.float64)

        if self.for_LSTDV:
            self.delta_d_v = np.zeros((self.m,) + source_tasks[0].env.Q.shape + source_tasks[0].env.V.shape, dtype=np.float64)
            self.delta_A_v = np.zeros((self.m, self.n_features_v, self.n_features_v), dtype=np.float64)
            self.delta_b_v = np.zeros((self.m, self.n_features_v), dtype=np.float64)

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

            state_idx = source_samples[j]['fsi']
            action_idx = source_samples[j]['ai']
            next_state_idx = source_samples[j]['nsi']
            next_action_idx = source_samples[j]['nai']
            sorted_idx = np.lexsort((next_action_idx, next_state_idx, action_idx, state_idx))
            state_sorted = state_idx[sorted_idx]
            action_sorted = action_idx[sorted_idx]
            next_state_sorted = next_state_idx[sorted_idx]
            next_action_sorted = next_action_idx[sorted_idx]
            state_groups = np.ones(state_idx.shape[0], dtype=bool)
            action_groups = np.ones(state_idx.shape[0], dtype=bool)
            next_state_groups = np.ones(state_idx.shape[0], dtype=bool)
            next_action_groups = np.ones(state_idx.shape[0], dtype=bool)
            state_groups[1:] = state_sorted[1:] != state_sorted[:-1]
            action_groups[1:] = action_sorted[1:] != action_sorted[:-1]
            next_state_groups[1:] = next_state_sorted[1:] != next_state_sorted[:-1]
            next_action_groups[1:] = next_action_sorted[1:] != next_action_sorted[:-1]

            if self.for_gradient:
                groups = np.logical_or(state_groups, action_groups)
                groups = np.arange(groups.shape[0])[groups]
                self.source_samples[j]['idx_s_grad'] = sorted_idx
                self.source_samples[j]['grps_grad'] = groups
                self.source_samples[j]['grp_szs_grad'] = np.diff(np.append(groups, self.source_sizes[j]))
                self.reduced_source_sizes_grad[j] = groups.size
                self.source_samples[j]['eta_j'] =\
                    np.add.reduceat(source_policies[j].log_gradient_matrix[state_sorted, action_sorted] *\
                                    (source_tasks[j].env.Q[state_sorted, action_sorted] -source_tasks[j].env.V[state_sorted])[:,None], groups, axis=0)

            if self.for_LSTDQ:
                groups = np.logical_or.reduce((state_groups, action_groups, next_state_groups, next_action_groups))
                groups = np.arange(groups.shape[0])[groups]
                self.source_samples[j]['idx_s_q'] = sorted_idx
                self.source_samples[j]['grps_q'] = groups
                self.source_samples[j]['grp_szs_q'] = np.diff(np.append(groups, self.source_sizes[j]))
                self.reduced_source_sizes_q[j] = groups.size

                all_phi_Q_rsp = all_phi_Q.reshape(source_tasks[0].env.Q.shape + (self.n_features_q,))
                self.source_samples[j]['var_phi_q'] = np.zeros((self.reduced_source_sizes_q[j], self.n_features_q, self.n_features_q),
                                                               dtype=np.float64)
                for k in range(self.n_features_q):
                    self.source_samples[j]['var_phi_q'][:,k] =\
                        np.add.reduceat(all_phi_Q_rsp[state_sorted,action_sorted][:,k][:,None]*\
                                        (all_phi_Q_rsp[state_sorted,action_sorted] -
                                         self.gamma*all_phi_Q_rsp[next_state_sorted,next_action_sorted]), groups, axis=0)
                self.source_samples[j]['var_phi_q_sq'] = self.source_samples[j]['var_phi_q']**2

                self.source_samples[j]['rho_q'] = \
                    np.add.reduceat(all_phi_Q_rsp[state_sorted,action_sorted]*source_samples[j]['r'][sorted_idx,None], groups, axis=0)

            if self.for_LSTDV:
                groups = np.logical_or.reduce((state_groups, action_groups, next_state_groups))
                groups = np.arange(groups.shape[0])[groups]
                self.source_samples[j]['idx_s_v'] = sorted_idx
                self.source_samples[j]['grps_v'] = groups
                self.source_samples[j]['grp_szs_v'] = np.diff(np.append(groups, self.source_sizes[j]))
                self.reduced_source_sizes_v[j] = groups.size
                
                self.source_samples[j]['var_phi_v'] = np.zeros((self.reduced_source_sizes_v[j], self.n_features_v, self.n_features_v),
                                                               dtype=np.float64)
                for k in range(self.n_features_v):
                    self.source_samples[j]['var_phi_v'][:,k] =\
                        np.add.reduceat(all_phi_V[state_sorted][:,k][:,None]*\
                                        (all_phi_V[state_sorted] - self.gamma*all_phi_V[next_state_sorted]), groups, axis=0)
                self.source_samples[j]['var_phi_v_sq'] = self.source_samples[j]['var_phi_v'] ** 2

                self.source_samples[j]['rho_v'] = \
                    np.add.reduceat(all_phi_V[state_sorted]*source_samples[j]['r'][sorted_idx,None], groups, axis=0)

        if self.for_gradient:
            self.l_bounds_grad = np.zeros(self.reduced_source_sizes_grad.sum(), dtype=np.float64)
            self.u_bounds_grad = np.zeros(self.reduced_source_sizes_grad.sum(), dtype=np.float64)
        if self.for_LSTDQ:
            self.l_bounds_lstdq = np.zeros(self.reduced_source_sizes_q.sum(), dtype=np.float64)
            self.u_bounds_lstdq = np.zeros(self.reduced_source_sizes_q.sum(), dtype=np.float64)
        if self.for_LSTDV:
            self.l_bounds_lstdv = np.zeros(self.reduced_source_sizes_v.sum(), dtype=np.float64)
            self.u_bounds_lstdv = np.zeros(self.reduced_source_sizes_v.sum(), dtype=np.float64)



    def clean_sources(self):
        pass



    def prepare_lstd(self, target_policy, target_power):
        w_idx = np.hstack((0., self.source_sizes)).cumsum().astype(np.int64)
        for i in range(self.m):
            self.delta_P_eps_theta_s_s_prime[i] = \
                (self.source_tasks[i].env.transition_matrix * np.abs(self.source_policies[i].choice_matrix - target_policy.choice_matrix)[:,:,None] +
                 target_policy.choice_matrix[:,:,None] *
                 self.L_P_eps_s_a_s_prime[i] * np.abs(self.source_tasks[i].env.power - target_power)).sum(axis=1)
            self.delta_P_eps_theta_s[i] = self.delta_P_eps_theta_s_s_prime[i].max(axis=0)
            self.delta_delta[i] =\
                (1. - self.gamma) * self.source_tasks[i].env.P_pi_inv.T.dot(0. +self.gamma * np.clip(self.delta_P_eps_theta_s[i],0.,1.))
            self.delta_zeta[i] =\
                self.source_tasks[i].env.delta_distr[:,None] * np.abs(self.source_policies[i].choice_matrix -target_policy.choice_matrix) + \
                target_policy.choice_matrix * np.clip(self.delta_delta[i],0., 1.)[:,None]

            if self.for_LSTDV or self.for_LSTDQ:
                self.M_P_eps_s_a_s_prime[i] = np.clip(self.source_tasks[i].env.transition_matrix +
                                                      self.L_P_eps_s_a_s_prime[i]*np.abs(self.source_tasks[i].env.power - target_power), 0., 1.)
            if self.for_LSTDQ:
                reduced_w_idx = np.hstack((0., self.reduced_source_sizes_q)).cumsum().astype(np.int64)
                #self.delta_d_q[i] = self.source_tasks[i].env.zeta_distr[:, :, None, None] * \
                #                    self.source_tasks[i].env.transition_matrix[:, :, :, None] * \
                #                    np.abs(target_policy.choice_matrix - self.source_policies[i].choice_matrix)[None, None, :, :] + \
                #                    target_policy.choice_matrix[None, None, :, :] * self.source_tasks[i].env.zeta_distr[:, :, None, None] * \
                #                    np.clip(self.L_P_eps_s_a_s_prime[i] * np.abs(self.source_tasks[i].env.power - target_power), 0., 1.)[:, :, :, None] + \
                #                    target_policy.choice_matrix[None, None, :, :] * self.M_P_eps_s_a_s_prime[i][:, :, :, None] * \
                #                    np.clip(self.delta_zeta[i], 0., 1.)[:, :, None, None]
                self.delta_d_q[i] = self.source_tasks[i].env.zeta_distr[:, :, None, None] *\
                                    self.source_tasks[i].env.transition_matrix[:, :, :, None] * \
                                    np.abs(target_policy.choice_matrix - self.source_policies[i].choice_matrix)[None,None, :, :]
                self.delta_d_q[i] += target_policy.choice_matrix[None, None, :, :] * self.source_tasks[i].env.zeta_distr[:,:, None, None] * \
                                     np.clip(self.L_P_eps_s_a_s_prime[i] * np.abs(self.source_tasks[i].env.power - target_power), 0., 1.)[:, :, :, None]
                self.delta_d_q[i] += target_policy.choice_matrix[None, None, :, :] * self.M_P_eps_s_a_s_prime[i][:, :, :,None] * \
                                     np.clip(self.delta_zeta[i], 0., 1.)[:, :, None, None]
                self.delta_d_q_b[i] = self.source_tasks[i].env.zeta_distr[:,:, None] * \
                                      np.clip(self.L_P_eps_s_a_s_prime[i] * np.abs(self.source_tasks[i].env.power - target_power), 0., 1.) + \
                                      self.M_P_eps_s_a_s_prime[i] * np.clip(self.delta_zeta[i], 0., 1.)[:, :, None]
                #self.delta_d_q_b[i] = self.source_tasks[i].env.zeta_distr[:, :, None] * \
                #                      np.clip(self.L_P_eps_s_a_s_prime[i] * np.abs(self.source_tasks[i].env.power - target_power), 0., 1.)
                #self.delta_d_q_b[i] += self.M_P_eps_s_a_s_prime[i] * np.clip(self.delta_zeta[i], 0., 1.)[:, :, None]
                self.delta_A_q[i] = self.all_phi_Q.T.dot(np.clip(self.delta_d_q[i], 0., 1.).reshape((self.all_phi_Q.shape[0],
                                                                                                      self.all_phi_Q.shape[0],1)).sum(axis=1))
                #self.delta_A_q[i] = self.all_phi_Q.T.dot((np.abs(self.all_phi_Q[:,None,:] - self.gamma*self.all_phi_Q[None,:,:])*
                #                                          np.clip(self.delta_d_q[i], 0., 1.).reshape((self.all_phi_Q.shape[0],
                #                                                                                      self.all_phi_Q.shape[0],1))).sum(axis=1))
                #for k in range(self.n_features_q):
                #    aux = np.abs(self.all_phi_Q[:,None,k] - self.gamma*self.all_phi_Q[None,:,k])
                #    aux *= np.clip(self.delta_d_q[i], 0., 1.).reshape((self.all_phi_Q.shape[0], self.all_phi_Q.shape[0]))
                #    aux = aux.sum(axis=1)
                #    self.delta_A_q[i,:,k] = self.all_phi_Q.T.dot(aux)

                self.delta_b_q[i] = self.all_phi_Q.T.dot(np.abs(self.source_tasks[i].env.r*np.clip(self.delta_d_q_b[i], 0., 1.)).sum(axis=2).flatten())
                source_d_distr = (self.source_tasks[i].env.zeta_distr[self.source_samples[i]['fsi'], self.source_samples[i]['ai']] *
                                  self.source_tasks[i].env.transition_matrix[self.source_samples[i]['fsi'], self.source_samples[i]['ai'],
                                                                             self.source_samples[i]['nsi']] *
                                  self.source_policies[i].choice_matrix[self.source_samples[i]['nsi'], self.source_samples[i]['nai']])[self.source_samples[i]['idx_s_q']][self.source_samples[i]['grps_q']]
                self.l_bounds_lstdq[reduced_w_idx[i]:reduced_w_idx[i + 1]] = np.clip(np.ones(self.reduced_source_sizes_q[i], dtype=np.float64) -
                                                                                     np.clip(self.delta_d_q[i], 0., 1.)[self.source_samples[i]['fsi'],
                                                                                                                        self.source_samples[i]['ai'],
                                                                                                                        self.source_samples[i]['nsi'],
                                                                                                                        self.source_samples[i]['nai']][self.source_samples[i]['idx_s_q']][self.source_samples[i]['grps_q']] /
                                                                                     source_d_distr, 0., 1.)
                self.u_bounds_lstdq[reduced_w_idx[i]:reduced_w_idx[i + 1]] = np.ones(self.reduced_source_sizes_q[i], dtype=np.float64) +\
                                                                             np.clip(self.delta_d_q[i], 0., 1.)[self.source_samples[i]['fsi'],
                                                                                                                self.source_samples[i]['ai'],
                                                                                                                self.source_samples[i]['nsi'],
                                                                                                                self.source_samples[i]['nai']][self.source_samples[i]['idx_s_q']][self.source_samples[i]['grps_q']] / \
                                                                             source_d_distr
                
            if self.for_LSTDV:
                reduced_w_idx = np.hstack((0., self.reduced_source_sizes_v)).cumsum().astype(np.int64)
                self.delta_d_v[i] = self.source_tasks[i].env.zeta_distr[:, :, None] * \
                                    np.clip(self.L_P_eps_s_a_s_prime[i] * np.abs(self.source_tasks[i].env.power - target_power), 0., 1.) + \
                                    self.M_P_eps_s_a_s_prime[i] * np.clip(self.delta_zeta[i], 0., 1.)[:, :, None]
                self.delta_d_v[i] = self.source_tasks[i].env.zeta_distr[:,:, None] * \
                                    np.clip(self.L_P_eps_s_a_s_prime[i] * np.abs(self.source_tasks[i].env.power - target_power),0., 1.)
                self.delta_d_v[i] += self.M_P_eps_s_a_s_prime[i] * np.clip(self.delta_zeta[i], 0., 1.)[:, :, None]
                self.delta_A_v[i] = self.all_phi_V.T.dot((np.abs(self.all_phi_V[:, None, :] - self.gamma * self.all_phi_V[None, :, :]) *
                                                                np.clip(self.delta_d_v[i], 0., 1.).sum(axis=1)[:,:,None]).sum(axis=1))
                #for k in range(self.n_features_v):
                #    self.delta_A_v[i,:,k] = self.all_phi_V.T.dot((np.abs(self.all_phi_V[:,None,k] - self.gamma*self.all_phi_V[None,:,k])*
                #                                                  np.clip(self.delta_d_v[i], 0., 1.).sum(axis=1)).sum(axis=1))

                self.delta_b_v[i] = self.all_phi_V.T.dot(np.abs(self.source_tasks[i].env.r*np.clip(self.delta_d_v[i], 0., 1.)).sum(axis=(1,2)))
                source_d_distr = (self.source_tasks[i].env.zeta_distr[self.source_samples[i]['fsi'], self.source_samples[i]['ai']] *
                                  self.source_tasks[i].env.transition_matrix[self.source_samples[i]['fsi'], self.source_samples[i]['ai'],
                                                                             self.source_samples[i]['nsi']][self.source_samples[i]['idx_s_v']])[self.source_samples[i]['grps_v']]
                self.l_bounds_lstdv[reduced_w_idx[i]:reduced_w_idx[i + 1]] = np.clip(np.ones(self.reduced_source_sizes_v[i], dtype=np.float64) -
                                                                                     np.clip(self.delta_d_v[i], 0., 1.)[self.source_samples[i]['fsi'],
                                                                                                                        self.source_samples[i]['ai'],
                                                                                                                        self.source_samples[i]['nsi']][self.source_samples[i]['idx_s_v']][self.source_samples[i]['grps_v']] /
                                                                                     source_d_distr, 0., 1.)
                self.u_bounds_lstdv[reduced_w_idx[i]:reduced_w_idx[i + 1]] = np.ones(self.reduced_source_sizes_v[i], dtype=np.float64) +\
                                                                             np.clip(self.delta_d_v[i], 0., 1.)[self.source_samples[i]['fsi'],
                                                                                                                self.source_samples[i]['ai'],
                                                                                                                self.source_samples[i]['nsi']][self.source_samples[i]['idx_s_v']][self.source_samples[i]['grps_v']] / \
                                                                             source_d_distr



    def prepare_gradient(self, target_policy, target_power, all_target_Q, target_V):
        w_idx = np.hstack((0., self.source_sizes)).cumsum().astype(np.int64)
        reduced_w_idx = np.hstack((0., self.reduced_source_sizes_grad)).cumsum().astype(np.int64)
        for i in range(self.m):
            if not self.for_LSTDQ and not self.for_LSTDV:
                self.delta_P_eps_theta_s_s_prime[i] = \
                    (self.source_tasks[i].env.transition_matrix * np.abs(self.source_policies[i].choice_matrix - target_policy.choice_matrix)[:, :, None] +
                     target_policy.choice_matrix[:, :, None] *
                     self.L_P_eps_s_a_s_prime[i] * np.abs(self.source_tasks[i].env.power - target_power)).sum(axis=1)
                self.delta_P_eps_theta_s[i] = self.delta_P_eps_theta_s_s_prime[i].max(axis=0)
                self.delta_delta[i] = \
                    (1. - self.gamma) * self.source_tasks[i].env.P_pi_inv.T.dot(0. + self.gamma * np.clip(self.delta_P_eps_theta_s[i], 0., 1.))
                self.delta_zeta[i] = \
                    self.source_tasks[i].env.delta_distr[:, None] * np.abs(self.source_policies[i].choice_matrix - target_policy.choice_matrix) + \
                    target_policy.choice_matrix * np.clip(self.delta_delta[i], 0., 1.)[:, None]

            self.delta_J[i] =\
                (np.abs(target_policy.log_gradient_matrix * (all_target_Q * np.clip(self.delta_zeta[i], 0., 1.))[:,:,None]) +
                 self.source_tasks[i].env.zeta_distr[:,:,None] * np.abs(target_policy.log_gradient_matrix * all_target_Q[:,:,None] -
                                                                        self.source_policies[i].log_gradient_matrix * self.source_tasks[i].env.Q[:,:,None]))\
                    .sum(axis=(0, 1)) / (1. - self.gamma)

            self.source_samples[i]['eta_1'] =\
                np.add.reduceat((target_policy.log_gradient_matrix[self.source_samples[i]['fsi'], self.source_samples[i]['ai']] *
                                 (all_target_Q[self.source_samples[i]['fsi'], self.source_samples[i]['ai']] -
                                  target_V[w_idx[i]:w_idx[i + 1]])[:,None])[self.source_samples[i]['idx_s_grad']], self.source_samples[i]['grps_grad'],
                                axis=0)

            self.l_bounds_grad[reduced_w_idx[i]:reduced_w_idx[i + 1]] = np.clip(np.ones(self.reduced_source_sizes_grad[i], dtype=np.float64) -
                                                                                np.clip(self.delta_zeta[i], 0., 1.)[self.source_samples[i]['fsi'],
                                                                                                                    self.source_samples[i]['ai']][self.source_samples[i]['idx_s_grad']][self.source_samples[i]['grps_grad']] /
                                                                                self.source_tasks[i].env.zeta_distr[self.source_samples[i]['fsi'],
                                                                                                                    self.source_samples[i]['ai']][self.source_samples[i]['idx_s_grad']][self.source_samples[i]['grps_grad']],
                                                                                0., 1.)
            self.u_bounds_grad[reduced_w_idx[i]:reduced_w_idx[i + 1]] = np.ones(self.reduced_source_sizes_grad[i], dtype=np.float64) +\
                                                                        np.clip(self.delta_zeta[i], 0., 1.)[self.source_samples[i]['fsi'],
                                                                                                            self.source_samples[i]['ai']][self.source_samples[i]['idx_s_grad']][self.source_samples[i]['grps_grad']] /\
                                                                        self.source_tasks[i].env.zeta_distr[self.source_samples[i]['fsi'],
                                                                                                            self.source_samples[i]['ai']][self.source_samples[i]['idx_s_grad']][self.source_samples[i]['grps_grad']]



    def estimate_weights_gradient(self, target_size):
        reduced_w_idx = np.hstack((0., self.reduced_source_sizes_grad)).cumsum().astype(np.int64)
        n = self.source_sizes.sum() + target_size
        w0 = np.ones(self.reduced_source_sizes_grad.sum(), dtype=np.float64)
        
        def g(w):
            bias = 0.
            vari = 0.
            for j in range(self.m):
                bias += (self.source_samples[j]['eta_j'] - w[reduced_w_idx[j]:reduced_w_idx[j+1]][:,None]*self.source_samples[j]['eta_1']).sum(axis=0)/(n*(1. - self.gamma))
                vari += ((((w[reduced_w_idx[j]:reduced_w_idx[j+1]][:,None]*self.source_samples[j]['eta_1'])**2)/self.source_samples[j]['grp_szs_grad'][:,None]).sum(axis=0) -
                         ((w[reduced_w_idx[j]:reduced_w_idx[j+1]][:,None]*self.source_samples[j]['eta_1']).sum(axis=0)) ** 2 / self.source_sizes[j]) / (n * (1. - self.gamma)) ** 2
            bias = np.abs(bias) + (self.delta_J * self.source_sizes.reshape((-1, 1))).sum(axis=0) / n
            bias = (bias**2).sum()
            vari = vari.sum()
            return bias# + vari

        def grad_g(w):
            bias = 0.
            for j in range(self.m):
                bias += (self.source_samples[j]['eta_j'] - w[reduced_w_idx[j]:reduced_w_idx[j + 1]].reshape((-1, 1)) * self.source_samples[j]['eta_1']).sum(axis=0) / (n*(1. - self.gamma))
            sum_etas = bias
            bias = np.abs(bias) + (self.delta_J * self.source_sizes.reshape((-1, 1))).sum(axis=0) / n
            bias *= n
            grad = np.zeros(w.shape + (2,), dtype=np.float64)
            for j in range(self.m):
                grad[reduced_w_idx[j]:reduced_w_idx[j+1]] = 2*self.source_samples[j]['eta_1']*(np.power(-1., sum_etas > 0) * bias / (1. - self.gamma))/ n ** 2
                grad[reduced_w_idx[j]:reduced_w_idx[j + 1]] += 2 * self.source_samples[j]['eta_1']*\
                                                               (self.source_samples[j]['eta_1'] *w[reduced_w_idx[j]:reduced_w_idx[j+1]][:,None] / (self.source_samples[j]['grp_szs_grad'][:,None]*(1. - self.gamma) ** 2) -
                                                                (self.source_samples[j]['eta_1']*w[reduced_w_idx[j]:reduced_w_idx[j+1]][:,None]).sum(axis=0) / (self.source_sizes[j] * (1. - self.gamma) ** 2))\
                                                            / n ** 2
            grad = grad.sum(axis=1)
            return grad

        bounds = np.ones((self.l_bounds_grad.size + self.u_bounds_grad.size,), dtype=np.float64)
        bounds[0::2] = self.l_bounds_grad
        bounds[1::2] = self.u_bounds_grad
        bounds = tuple(map(tuple, bounds.reshape((-1,2))))
        res = minimize(g, w0, jac=grad_g, bounds=bounds)
        all_w = np.zeros(self.source_sizes.sum(), dtype=np.float64)
        w_idx = np.hstack((0., self.source_sizes)).cumsum().astype(np.int64)
        for j in range(self.m):
            aux = np.repeat(res.x[reduced_w_idx[j]:reduced_w_idx[j+1]], self.source_samples[j]['grp_szs_grad'].astype(np.int32))
            inv = np.empty(self.source_samples[j]['idx_s_grad'].shape[0], dtype=np.int64)
            inv[self.source_samples[j]['idx_s_grad']] = np.arange(self.source_samples[j]['idx_s_grad'].shape[0])
            all_w[w_idx[j]:w_idx[j+1]] = aux[inv]

        return all_w



    def estimate_weights_lstdq(self, target_size):
        reduced_w_idx = np.hstack((0., self.reduced_source_sizes_q)).cumsum().astype(np.int64)
        n = self.source_sizes.sum() + target_size
        w0 = np.ones(self.reduced_source_sizes_q.sum(), dtype=np.float64)
        init_bias_A = (self.delta_A_q * self.source_sizes[:,None,None]).sum(axis=0) / n
        init_bias_b = (self.delta_b_q * self.source_sizes[:, None]).sum(axis=0) / n
        bias_A = np.zeros((self.n_features_q, self.n_features_q), dtype=np.float64)
        sum_etas_A = np.zeros((self.n_features_q, self.n_features_q), dtype=np.float64)
        bias_b = np.zeros((self.n_features_q, self.n_features_q), dtype=np.float64)
        sum_etas_b = np.zeros((self.n_features_q, self.n_features_q), dtype=np.float64)

        def g(w):
            nonlocal bias_A, bias_b
            bias_A = np.zeros((self.n_features_q, self.n_features_q), dtype=np.float64)
            bias_b = np.zeros(self.n_features_q, dtype=np.float64)
            vari_A = np.zeros((self.n_features_q, self.n_features_q), dtype=np.float64)
            vari_b = np.zeros(self.n_features_q, dtype=np.float64)
            for j in range(self.m):
                bias_A += ((1. - w[reduced_w_idx[j]:reduced_w_idx[j+1]])[:,None,None]*self.source_samples[j]['var_phi_q']).sum(axis=0)/n
                vari_A += (((w[reduced_w_idx[j]:reduced_w_idx[j + 1]]**2/self.source_samples[j]['grp_szs_q'])[:,None,None]*self.source_samples[j]['var_phi_q_sq']).sum(axis=0) -
                          (w[reduced_w_idx[j]:reduced_w_idx[j + 1]][:,None,None]*self.source_samples[j]['var_phi_q']).sum(axis=0) ** 2 / self.source_sizes[j])\
                          / n ** 2
                bias_b += ((1. - w[reduced_w_idx[j]:reduced_w_idx[j + 1]])[:, None] * self.source_samples[j]['rho_q']).sum(axis=0) / n
                vari_b += (((w[reduced_w_idx[j]:reduced_w_idx[j + 1]] ** 2 / self.source_samples[j]['grp_szs_q'])[:, None] * self.source_samples[j]['rho_q'] ** 2).sum(axis=0) -
                           (w[reduced_w_idx[j]:reduced_w_idx[j + 1]][:, None] * self.source_samples[j]['rho_q']).sum(axis=0) ** 2 / self.source_sizes[j]) \
                          / n ** 2
            nonlocal sum_etas_A
            sum_etas_A = bias_A
            bias_A = np.abs(bias_A) + init_bias_A
            bias_A_sq = (bias_A**2).sum()
            vari_A = vari_A.sum()
            nonlocal sum_etas_b
            sum_etas_b = bias_b
            bias_b = np.abs(bias_b) + init_bias_b
            bias_b_sq = (bias_b**2).sum()
            vari_b = vari_b.sum()
            return bias_A_sq + vari_A + bias_b_sq + vari_b

        def grad_g(w):
            grad_A = np.zeros(w.shape + (self.n_features_q,self.n_features_q), dtype=np.float64)
            grad_b = np.zeros(w.shape + (self.n_features_q,), dtype=np.float64)
            for j in range(self.m):
                grad_A[reduced_w_idx[j]:reduced_w_idx[j + 1]] = (2 * self.source_samples[j]['var_phi_q'] / n ** 2) *\
                                                                (np.power(-1., sum_etas_A > 0) * bias_A +
                                                                 self.source_samples[j]['var_phi_q'] * (w[reduced_w_idx[j]:reduced_w_idx[j + 1]] / self.source_samples[j]['grp_szs_q'])[:, None, None] -
                                                                 (self.source_samples[j]['var_phi_q'] * w[reduced_w_idx[j]:reduced_w_idx[j + 1]][:, None, None]).sum(axis=0) / self.source_sizes[j])
                #grad_A[reduced_w_idx[j]:reduced_w_idx[j + 1]] = (2 * self.source_samples[j]['var_phi_q']/n**2) * np.power(-1., sum_etas_A > 0) * bias_A
                #grad_A[reduced_w_idx[j]:reduced_w_idx[j + 1]] += (2 * self.source_samples[j]['var_phi_q']/n**2)*self.source_samples[j]['var_phi_q'] * (w[reduced_w_idx[j]:reduced_w_idx[j + 1]]/self.source_samples[j]['grp_szs_q'])[:,None,None]
                #grad_A[reduced_w_idx[j]:reduced_w_idx[j + 1]] -= (2 * self.source_samples[j]['var_phi_q']/n**2)*(self.source_samples[j]['var_phi_q'] * w[reduced_w_idx[j]:reduced_w_idx[j + 1]][:,None,None]).sum(axis=0) / self.source_sizes[j]
                grad_b[reduced_w_idx[j]:reduced_w_idx[j + 1]] = 2 * self.source_samples[j]['rho_q'] *\
                                              (np.power(-1., sum_etas_b > 0) * bias_b +
                                               self.source_samples[j]['rho_q'] * (w[reduced_w_idx[j]:reduced_w_idx[j + 1]]/self.source_samples[j]['grp_szs_q'])[:,None] -
                                               (self.source_samples[j]['rho_q'] * w[reduced_w_idx[j]:reduced_w_idx[j + 1]][:,None]).sum(axis=0) / self.source_sizes[j]) /\
                                              n ** 2
            grad = grad_A.sum(axis=(1,2)) + grad_b.sum(axis=1)
            return grad

        bounds = np.ones((self.l_bounds_lstdq.size + self.u_bounds_lstdq.size,), dtype=np.float64)
        bounds[0::2] = self.l_bounds_lstdq
        bounds[1::2] = self.u_bounds_lstdq
        bounds = tuple(map(tuple, bounds.reshape((-1, 2))))
        res = minimize(g, w0, jac=grad_g, bounds=bounds)
        all_w = np.zeros(self.source_sizes.sum(), dtype=np.float64)
        w_idx = np.hstack((0., self.source_sizes)).cumsum().astype(np.int64)
        for j in range(self.m):
            aux = np.repeat(res.x, self.source_samples[j]['grp_szs_q'].astype(np.int32))
            inv = np.empty(self.source_samples[j]['idx_s_q'].shape[0], dtype=np.int64)
            inv[self.source_samples[j]['idx_s_q']] = np.arange(self.source_samples[j]['idx_s_q'].shape[0])
            all_w[w_idx[j]:w_idx[j + 1]] = aux[inv]

        return all_w



    def estimate_weights_lstdv(self, target_size):
        reduced_w_idx = np.hstack((0., self.reduced_source_sizes_v)).cumsum().astype(np.int64)
        n = self.source_sizes.sum() + target_size
        w0 = np.ones(self.reduced_source_sizes_v.sum(), dtype=np.float64)
        init_bias_A = (self.delta_A_v * self.source_sizes[:, None, None]).sum(axis=0) / n
        init_bias_b = (self.delta_b_v * self.source_sizes[:, None]).sum(axis=0) / n
        bias_A = 0.
        sum_etas_A = 0.
        bias_b = 0.
        sum_etas_b = 0.

        def g(w):
            nonlocal bias_A, bias_b
            bias_A = 0.
            bias_b = 0.
            vari_A = 0.
            vari_b = 0.
            for j in range(self.m):
                bias_A += ((1. - w[reduced_w_idx[j]:reduced_w_idx[j + 1]])[:, None, None] * self.source_samples[j]['var_phi_v']).sum(axis=0) / n
                vari_A += (((w[reduced_w_idx[j]:reduced_w_idx[j + 1]] ** 2 / self.source_samples[j]['grp_szs_v'])[:, None, None] * self.source_samples[j]['var_phi_v_sq']).sum(axis=0) -
                           (w[reduced_w_idx[j]:reduced_w_idx[j + 1]][:, None, None] * self.source_samples[j]['var_phi_v']).sum(axis=0) ** 2 / self.source_sizes[j]) \
                          / n ** 2
                bias_b += ((1. - w[reduced_w_idx[j]:reduced_w_idx[j + 1]])[:, None] * self.source_samples[j]['rho_v']).sum(axis=0) / n
                vari_b += (((w[reduced_w_idx[j]:reduced_w_idx[j + 1]] ** 2 / self.source_samples[j]['grp_szs_v'])[:, None] * self.source_samples[j]['rho_v'] ** 2).sum(axis=0) -
                           (w[reduced_w_idx[j]:reduced_w_idx[j + 1]][:, None] * self.source_samples[j]['rho_v']).sum(axis=0) ** 2 / self.source_sizes[j]) \
                          / n ** 2
            nonlocal sum_etas_A
            sum_etas_A = bias_A
            bias_A = np.abs(bias_A) + init_bias_A
            bias_A_sq = (bias_A ** 2).sum()
            vari_A = vari_A.sum()
            nonlocal sum_etas_b
            sum_etas_b = bias_b
            bias_b = np.abs(bias_b) + init_bias_b
            bias_b_sq = (bias_b ** 2).sum()
            vari_b = vari_b.sum()
            return bias_A_sq + bias_b_sq + vari_A + vari_b

        def grad_g(w):
            grad_A = np.zeros(w.shape + (self.n_features_v, self.n_features_v), dtype=np.float64)
            grad_b = np.zeros(w.shape + (self.n_features_v,), dtype=np.float64)
            for j in range(self.m):
                grad_A[reduced_w_idx[j]:reduced_w_idx[j + 1]] = (2 * self.source_samples[j]['var_phi_v'] / n ** 2) * \
                                                                (np.power(-1., sum_etas_A > 0) * bias_A +
                                                                 self.source_samples[j]['var_phi_v'] * (w[reduced_w_idx[j]:reduced_w_idx[j + 1]] / self.source_samples[j]['grp_szs_v'])[:, None, None] -
                                                                (self.source_samples[j]['var_phi_v'] * w[reduced_w_idx[j]:reduced_w_idx[j + 1]][:, None, None]).sum(axis=0) / self.source_sizes[j])
                #grad_A[reduced_w_idx[j]:reduced_w_idx[j + 1]] = (2 * self.source_samples[j]['var_phi_v']/n**2) * np.power(-1., sum_etas_A > 0) * bias_A
                #grad_A[reduced_w_idx[j]:reduced_w_idx[j + 1]] += (2 * self.source_samples[j]['var_phi_v']/n**2)*self.source_samples[j]['var_phi_v'] * (w[reduced_w_idx[j]:reduced_w_idx[j + 1]]/self.source_samples[j]['grp_szs_v'])[:,None,None]
                #grad_A[reduced_w_idx[j]:reduced_w_idx[j + 1]] -= (2 * self.source_samples[j]['var_phi_v']/n**2)*(self.source_samples[j]['var_phi_v'] * w[reduced_w_idx[j]:reduced_w_idx[j + 1]][:,None,None]).sum(axis=0) / self.source_sizes[j]
                grad_b[reduced_w_idx[j]:reduced_w_idx[j + 1]] = 2 * self.source_samples[j]['rho_v'] * \
                                                                (np.power(-1., sum_etas_b > 0) * bias_b +
                                                                 self.source_samples[j]['rho_v'] * (w[reduced_w_idx[j]:reduced_w_idx[j + 1]] / self.source_samples[j]['grp_szs_v'])[:, None] -
                                                                 (self.source_samples[j]['rho_v'] * w[reduced_w_idx[j]:reduced_w_idx[j + 1]][:, None]).sum(axis=0) / self.source_sizes[j]) / \
                                                                n ** 2
            grad = grad_A.sum(axis=(1, 2)) + grad_b.sum(axis=1)
            return grad

        bounds = np.ones((self.l_bounds_lstdv.size + self.u_bounds_lstdv.size,), dtype=np.float64)
        bounds[0::2] = self.l_bounds_lstdv
        bounds[1::2] = self.u_bounds_lstdv
        bounds = tuple(map(tuple, bounds.reshape((-1, 2))))
        res = minimize(g, w0, jac=grad_g, bounds=bounds)
        all_w = np.zeros(self.source_sizes.sum(), dtype=np.float64)
        w_idx = np.hstack((0., self.source_sizes)).cumsum().astype(np.int64)
        for j in range(self.m):
            aux = np.repeat(res.x, self.source_samples[j]['grp_szs_v'].astype(np.int32))
            inv = np.empty(self.source_samples[j]['idx_s_v'].shape[0], dtype=np.int64)
            inv[self.source_samples[j]['idx_s_v']] = np.arange(self.source_samples[j]['idx_s_v'].shape[0])
            all_w[w_idx[j]:w_idx[j + 1]] = aux[inv]

        return all_w