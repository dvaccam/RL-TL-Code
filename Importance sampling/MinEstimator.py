import numpy as np
import math
from scipy.optimize import minimize, fixed_point, root
from scipy.special import erf
from scipy.stats import ncx2, norm, moment
import gym
import subprocess



class MinWeightsEstimator():
    def __init__(self, gamma):
        self.gamma = gamma

    def add_sources(self, source_samples, source_tasks, source_policies):
        min_power = 0.
        max_power = 0.5
        min_source = gym.make('MountainCarContinuous-v0', min_position=-10., max_position=10., min_action=-1.,
                              max_action=1.,
                              power=min_power, seed=None, model='S', discrete=True,
                              n_position_bins=source_tasks[0].env.position_bins.shape[0],
                              n_velocity_bins=source_tasks[0].env.velocity_bins.shape[0],
                              n_action_bins=source_tasks[0].env.action_bins.shape[0],
                              position_noise=0.025, velocity_noise=0.025)
        max_source = gym.make('MountainCarContinuous-v0', min_position=-10., max_position=10., min_action=-1.,
                              max_action=1.,
                              power=max_power, seed=None, model='S', discrete=True,
                              n_position_bins=source_tasks[0].env.position_bins.shape[0],
                              n_velocity_bins=source_tasks[0].env.velocity_bins.shape[0],
                              n_action_bins=source_tasks[0].env.action_bins.shape[0],
                              position_noise=0.025, velocity_noise=0.025)

        self.source_samples = source_samples
        self.source_tasks = source_tasks
        self.source_policies = source_policies
        self.m = len(source_tasks)

        self.L_P_eps_s_a_s_prime = np.zeros((self.m,) + source_tasks[0].env.transition_matrix.shape, dtype=np.float64)
        self.L_P_eps_s_prime = np.zeros((self.m, source_tasks[0].env.V.shape[0]), dtype=np.float64)
        self.delta_P_eps_theta_s_s_prime = np.zeros(
            (self.m, source_tasks[0].env.V.shape[0], source_tasks[0].env.V.shape[0]), dtype=np.float64)
        self.delta_P_eps_theta_s = np.zeros((self.m, source_tasks[0].env.V.shape[0]), dtype=np.float64)
        self.delta_zeta = np.zeros((self.m,) + source_tasks[0].env.Q.shape, dtype=np.float64)
        self.delta_delta = np.zeros((self.m, source_tasks[0].env.V.shape[0]), dtype=np.float64)
        self.delta_J = np.zeros((self.m, 2), dtype=np.float64)
        self.ns = np.zeros(self.m, dtype=np.int64)

        for j in range(self.m):
            for s in range(source_tasks[0].env.transition_matrix.shape[0]):
                mu_peak_pos_e = np.zeros(source_tasks[j].env.position_reps.shape[0], dtype=np.float64)
                mu_peak_vel_e = np.zeros(source_tasks[j].env.velocity_reps.shape[0], dtype=np.float64)
                for pos_prime in range(source_tasks[j].env.position_reps.shape[0]):
                    if pos_prime != 0 and pos_prime != source_tasks[j].env.position_reps.shape[0] - 1:
                        p1 = source_tasks[j].env.position_bins[pos_prime]
                        p2 = source_tasks[j].env.position_bins[pos_prime + 1]

                        def f(x):
                            return (np.log(p2 - x) - np.log(p1 - x) + (p1 ** 2 - p2 ** 2) / (
                            2. * source_tasks[j].env.position_noise ** 2)) / (
                                   (p1 - p2) / source_tasks[j].env.position_noise ** 2) - x

                        mu_peak_pos_e[pos_prime] = root(f, p1 - 0.1).x
                for vel_prime in range(source_tasks[j].env.velocity_reps.shape[0]):
                    if vel_prime != 0 and vel_prime != source_tasks[j].env.velocity_reps.shape[0] - 1:
                        v1 = source_tasks[j].env.velocity_bins[vel_prime]
                        v2 = source_tasks[j].env.velocity_bins[vel_prime + 1]

                        def f(x):
                            return (np.log(v2 - x) - np.log(v1 - x) + (v1 ** 2 - v2 ** 2) / (
                            2. * source_tasks[j].env.velocity_noise ** 2)) / (
                                   (v1 - v2) / source_tasks[j].env.velocity_noise ** 2) - x

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
                            eps_peak = \
                                (source_tasks[j].env.position_bins[1] - source_tasks[j].env.state_reps[s].sum() +
                                 source_tasks[j].env.rescale(0.0025) * math.cos(
                                     3 * source_tasks[j].env.inverse_transform(source_tasks[j].env.state_reps[s][0]))) / \
                                source_tasks[j].env.action_reps[a]
                            if eps_peak < min_power or eps_peak > max_power:
                                M_e_pos[pos_prime] = max(
                                    np.exp(-(source_tasks[j].env.position_bins[1] - mu_min_source[0]) ** 2 /
                                           (2. * source_tasks[j].env.position_noise ** 2)),
                                    np.exp(-(source_tasks[j].env.position_bins[1] - mu_max_source[0]) ** 2 /
                                           (2. * source_tasks[j].env.position_noise ** 2)))
                            else:
                                M_e_pos[pos_prime] = 1.
                            eps_peak = \
                                (source_tasks[j].env.position_bins[0] - source_tasks[j].env.state_reps[s].sum() +
                                 source_tasks[j].env.rescale(0.0025) * math.cos(
                                     3 * source_tasks[j].env.inverse_transform(source_tasks[j].env.state_reps[s][0]))) / \
                                source_tasks[j].env.action_reps[a]
                            if eps_peak < min_power or eps_peak > max_power:
                                M_P_pos[pos_prime] = max(
                                    (1. + erf((source_tasks[j].env.position_bins[1] - mu_min_source[0]) /
                                              (np.sqrt(2.) * source_tasks[j].env.position_noise))) / 2.,
                                    (1. + erf((source_tasks[j].env.position_bins[1] - mu_max_source[0]) /
                                              (np.sqrt(2.) * source_tasks[j].env.position_noise))) / 2.)
                            else:
                                M_P_pos[pos_prime] = (1. + erf(
                                    (source_tasks[j].env.position_bins[1] - source_tasks[j].env.position_bins[0]) /
                                    (np.sqrt(2.) * source_tasks[j].env.position_noise))) / 2.
                        elif pos_prime == source_tasks[j].env.position_reps.shape[0] - 1:
                            eps_peak = \
                                (source_tasks[j].env.position_bins[-2] - source_tasks[j].env.state_reps[s].sum() +
                                 source_tasks[j].env.rescale(0.0025) * math.cos(
                                     3 * source_tasks[j].env.inverse_transform(source_tasks[j].env.state_reps[s][0]))) / \
                                source_tasks[j].env.action_reps[a]
                            if eps_peak < min_power or eps_peak > max_power:
                                M_e_pos[pos_prime] = max(
                                    np.exp(-(source_tasks[j].env.position_bins[-2] - mu_min_source[0]) ** 2 /
                                           (2. * source_tasks[j].env.position_noise ** 2)),
                                    np.exp(-(source_tasks[j].env.position_bins[-2] - mu_max_source[0]) ** 2 /
                                           (2. * source_tasks[j].env.position_noise ** 2)))
                            else:
                                M_e_pos[pos_prime] = 1.
                            eps_peak = \
                                (source_tasks[j].env.position_bins[-1] - source_tasks[j].env.state_reps[s].sum() +
                                 source_tasks[j].env.rescale(0.0025) * math.cos(
                                     3 * source_tasks[j].env.inverse_transform(source_tasks[j].env.state_reps[s][0]))) / \
                                source_tasks[j].env.action_reps[a]
                            if eps_peak < min_power or eps_peak > max_power:
                                M_P_pos[pos_prime] = max(
                                    (1. - erf((source_tasks[j].env.position_bins[-2] - mu_min_source[0]) /
                                              (np.sqrt(2.) * source_tasks[j].env.position_noise))) / 2.,
                                    (1. - erf((source_tasks[j].env.position_bins[-2] - mu_max_source[0]) /
                                              (np.sqrt(2.) * source_tasks[j].env.position_noise))) / 2.)
                            else:
                                M_P_pos[pos_prime] = (1. - erf(
                                    (source_tasks[j].env.position_bins[-2] - source_tasks[j].env.position_bins[-1]) /
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
                            eps_peak1 = \
                                (mu_peak1 - source_tasks[j].env.state_reps[s].sum() +
                                 source_tasks[j].env.rescale(0.0025) * math.cos(
                                     3 * source_tasks[j].env.inverse_transform(source_tasks[j].env.state_reps[s][0]))) / \
                                source_tasks[j].env.action_reps[a]
                            eps_peak2 = \
                                (mu_peak2 - source_tasks[j].env.state_reps[s].sum() +
                                 source_tasks[j].env.rescale(0.0025) * math.cos(
                                     3 * source_tasks[j].env.inverse_transform(source_tasks[j].env.state_reps[s][0]))) / \
                                source_tasks[j].env.action_reps[a]
                            if (eps_peak1 < min_power or eps_peak1 > max_power) and (
                                    eps_peak2 < min_power or eps_peak2 > max_power):
                                M_e_pos[pos_prime] = max(np.abs(np.exp(-(p1 - mu_min_source[0]) ** 2 /
                                                                       (2 * source_tasks[j].env.position_noise ** 2)) -
                                                                np.exp(-(p2 - mu_min_source[0]) ** 2 /
                                                                       (2 * source_tasks[j].env.position_noise ** 2))),
                                                         np.abs(np.exp(-(p1 - mu_max_source[0]) ** 2 /
                                                                       (2 * source_tasks[j].env.position_noise ** 2))
                                                                - np.exp(-(p2 - mu_max_source[0]) ** 2 /
                                                                         (
                                                                         2 * source_tasks[j].env.position_noise ** 2))))
                            else:
                                M_e_pos[pos_prime] = \
                                    np.abs(
                                        np.exp(-(p1 - mu_peak) ** 2 / (2 * source_tasks[j].env.position_noise ** 2)) -
                                        np.exp(-(p2 - mu_peak) ** 2 / (2 * source_tasks[j].env.position_noise ** 2)))
                            eps_peak = \
                                (pm - source_tasks[j].env.state_reps[s].sum() +
                                 source_tasks[j].env.rescale(0.0025) * math.cos(
                                     3 * source_tasks[j].env.inverse_transform(source_tasks[j].env.state_reps[s][0]))) / \
                                source_tasks[j].env.action_reps[a]
                            if eps_peak < min_power or eps_peak > max_power:
                                M_P_pos[pos_prime] = \
                                    max((erf(
                                        (p2 - mu_min_source[0]) / (np.sqrt(2.) * source_tasks[j].env.position_noise)) -
                                         erf((p1 - mu_min_source[0]) / (
                                         np.sqrt(2.) * source_tasks[j].env.position_noise))) / 2.,
                                        (erf((p2 - mu_max_source[0]) / (
                                        np.sqrt(2.) * source_tasks[j].env.position_noise)) -
                                         erf((p1 - mu_max_source[0]) / (
                                         np.sqrt(2.) * source_tasks[j].env.position_noise))) / 2.)
                            else:
                                M_P_pos[pos_prime] = (erf(
                                    (p2 - pm) / (np.sqrt(2.) * source_tasks[j].env.position_noise)) -
                                                      erf((p1 - pm) / (
                                                      np.sqrt(2.) * source_tasks[j].env.position_noise))) / 2.
                    for vel_prime in range(source_tasks[j].env.velocity_reps.shape[0]):
                        real_vel_min_source = \
                            source_tasks[j].env.state_reps[s][1] + source_tasks[j].env.action_reps[a] * 0. - \
                            source_tasks[j].env.rescale(0.0025) * math.cos(
                                3 * source_tasks[j].env.inverse_transform(source_tasks[j].env.state_reps[s][0]))
                        real_vel_max_source = \
                            source_tasks[j].env.state_reps[s][1] + source_tasks[j].env.action_reps[a] * 0.5 - \
                            source_tasks[j].env.rescale(0.0025) * math.cos(
                                3 * source_tasks[j].env.inverse_transform(source_tasks[j].env.state_reps[s][0]))
                        real_vel_min_source, real_vel_max_source = np.clip([real_vel_min_source, real_vel_max_source],
                                                                           source_tasks[j].env.min_speed,
                                                                           source_tasks[j].env.max_speed)
                        if vel_prime == 0:
                            eps_peak = \
                                (source_tasks[j].env.velocity_bins[1] - source_tasks[j].env.state_reps[s][1] +
                                 source_tasks[j].env.rescale(0.0025) * math.cos(
                                     3 * source_tasks[j].env.inverse_transform(source_tasks[j].env.state_reps[s][0]))) / \
                                source_tasks[j].env.action_reps[a]
                            if (eps_peak < min_power or eps_peak > max_power) and \
                                    (not (real_vel_min_source < 0 and mu_min_source[0] <=
                                        source_tasks[j].env.position_bins[1]) and
                                         not (real_vel_max_source < 0 and mu_max_source[0] <=
                                             source_tasks[j].env.position_bins[1])):
                                M_e_vel[vel_prime] = max(
                                    np.exp(-(source_tasks[j].env.velocity_bins[1] - mu_min_source[1]) ** 2 /
                                           (2. * source_tasks[j].env.velocity_noise ** 2)),
                                    np.exp(-(source_tasks[j].env.velocity_bins[1] - mu_max_source[1]) ** 2 /
                                           (2. * source_tasks[j].env.velocity_noise ** 2)))
                            else:
                                M_e_vel[vel_prime] = 1.
                            eps_peak = \
                                (source_tasks[j].env.velocity_bins[0] - source_tasks[j].env.state_reps[s][1] +
                                 source_tasks[j].env.rescale(0.0025) * math.cos(
                                     3 * source_tasks[j].env.inverse_transform(source_tasks[j].env.state_reps[s][0]))) / \
                                source_tasks[j].env.action_reps[a]
                            if (eps_peak < min_power or eps_peak > max_power) and \
                                    (not (real_vel_min_source < 0 and mu_min_source[0] <=
                                        source_tasks[j].env.position_bins[1]) and
                                         not (real_vel_max_source < 0 and mu_max_source[0] <=
                                             source_tasks[j].env.position_bins[1])):
                                M_P_vel[vel_prime] = max(
                                    (1. + erf((source_tasks[j].env.velocity_bins[1] - mu_min_source[1]) /
                                              (np.sqrt(2.) * source_tasks[j].env.velocity_noise))) / 2.,
                                    (1. + erf((source_tasks[j].env.velocity_bins[1] - mu_max_source[1]) /
                                              (np.sqrt(2.) * source_tasks[j].env.velocity_noise))) / 2.)
                            else:
                                M_P_vel[vel_prime] = (1. + erf(
                                    (source_tasks[j].env.velocity_bins[1] - source_tasks[j].env.velocity_bins[0]) /
                                    (np.sqrt(2.) * source_tasks[j].env.velocity_noise))) / 2.
                        elif vel_prime == source_tasks[j].env.velocity_reps.shape[0] - 1:
                            eps_peak = \
                                (source_tasks[j].env.velocity_bins[-2] - source_tasks[j].env.state_reps[s][1] +
                                 source_tasks[j].env.rescale(0.0025) * math.cos(
                                     3 * source_tasks[j].env.inverse_transform(source_tasks[j].env.state_reps[s][0]))) / \
                                source_tasks[j].env.action_reps[a]
                            if (eps_peak < min_power or eps_peak > max_power) and \
                                    (not (real_vel_min_source < 0 and mu_min_source[0] <=
                                        source_tasks[j].env.position_bins[1]) and
                                         not (real_vel_max_source < 0 and mu_max_source[0] <=
                                             source_tasks[j].env.position_bins[1])):
                                M_e_vel[vel_prime] = max(
                                    np.exp(-(source_tasks[j].env.velocity_bins[-2] - mu_min_source[1]) ** 2 /
                                           (2. * source_tasks[j].env.velocity_noise ** 2)),
                                    np.exp(-(source_tasks[j].env.velocity_bins[-2] - mu_max_source[1]) ** 2 /
                                           (2. * source_tasks[j].env.velocity_noise ** 2)))
                            else:
                                M_e_vel[vel_prime] = 1.
                            eps_peak = \
                                (source_tasks[j].env.velocity_bins[-1] - source_tasks[j].env.state_reps[s][1] +
                                 source_tasks[j].env.rescale(0.0025) * math.cos(
                                     3 * source_tasks[j].env.inverse_transform(source_tasks[j].env.state_reps[s][0]))) / \
                                source_tasks[j].env.action_reps[a]
                            if (eps_peak < min_power or eps_peak > max_power) and \
                                    (not (real_vel_min_source < 0 and mu_min_source[0] <=
                                        source_tasks[j].env.position_bins[1]) and
                                         not (real_vel_max_source < 0 and mu_max_source[0] <=
                                             source_tasks[j].env.position_bins[1])):
                                M_P_vel[vel_prime] = max(
                                    (1. - erf((source_tasks[j].env.velocity_bins[-2] - mu_min_source[1]) /
                                              (np.sqrt(2.) * source_tasks[j].env.velocity_noise))) / 2.,
                                    (1. - erf((source_tasks[j].env.velocity_bins[-2] - mu_max_source[1]) /
                                              (np.sqrt(2.) * source_tasks[j].env.velocity_noise))) / 2.)
                            else:
                                M_P_vel[vel_prime] = (1. - erf(
                                    (source_tasks[j].env.velocity_bins[-2] - source_tasks[j].env.velocity_bins[-1]) /
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
                            eps_peak1 = \
                                (mu_peak1 - source_tasks[j].env.state_reps[s][1] +
                                 source_tasks[j].env.rescale(0.0025) * math.cos(
                                     3 * source_tasks[j].env.inverse_transform(source_tasks[j].env.state_reps[s][0]))) / \
                                source_tasks[j].env.action_reps[a]
                            eps_peak2 = \
                                (mu_peak2 - source_tasks[j].env.state_reps[s][1] +
                                 source_tasks[j].env.rescale(0.0025) * math.cos(
                                     3 * source_tasks[j].env.inverse_transform(source_tasks[j].env.state_reps[s][0]))) / \
                                source_tasks[j].env.action_reps[a]
                            if (eps_peak1 < min_power or eps_peak1 > max_power) and (
                                    eps_peak2 < min_power or eps_peak2 > max_power) and \
                                    (not (real_vel_min_source < 0 and mu_min_source[0] ==
                                        source_tasks[j].env.position_bins[1]) and
                                         not (real_vel_max_source < 0 and mu_max_source[0] ==
                                             source_tasks[j].env.position_bins[1])):
                                M_e_vel[vel_prime] = \
                                    max(np.abs(np.exp(
                                        -(v1 - mu_min_source[1]) ** 2 / (2 * source_tasks[j].env.velocity_noise ** 2)) -
                                               np.exp(-(v2 - mu_min_source[1]) ** 2 / (
                                               2 * source_tasks[j].env.velocity_noise ** 2))),
                                        np.abs(np.exp(-(v1 - mu_max_source[1]) ** 2 / (
                                        2 * source_tasks[j].env.velocity_noise ** 2)) -
                                               np.exp(-(v2 - mu_max_source[1]) ** 2 / (
                                               2 * source_tasks[j].env.velocity_noise ** 2))))
                            else:
                                M_e_vel[vel_prime] = \
                                    np.abs(
                                        np.exp(-(v1 - mu_peak) ** 2 / (2 * source_tasks[j].env.velocity_noise ** 2)) -
                                        np.exp(-(v2 - mu_peak) ** 2 / (2 * source_tasks[j].env.velocity_noise ** 2)))
                            eps_peak = \
                                (vm - source_tasks[j].env.state_reps[s][1] +
                                 source_tasks[j].env.rescale(0.0025) * math.cos(
                                     3 * source_tasks[j].env.inverse_transform(source_tasks[j].env.state_reps[s][0]))) / \
                                source_tasks[j].env.action_reps[a]
                            if (eps_peak < min_power or eps_peak > max_power) and \
                                    (not (real_vel_min_source < 0 and mu_min_source[0] ==
                                        source_tasks[j].env.position_bins[1]) and
                                         not (real_vel_max_source < 0 and mu_max_source[0] ==
                                             source_tasks[j].env.position_bins[1])):
                                M_P_vel[vel_prime] = \
                                    max((erf(
                                        (v2 - mu_min_source[1]) / (np.sqrt(2.) * source_tasks[j].env.velocity_noise)) -
                                         erf((v1 - mu_min_source[1]) / (
                                         np.sqrt(2.) * source_tasks[j].env.velocity_noise))) / 2.,
                                        (erf((v2 - mu_max_source[1]) / (
                                        np.sqrt(2.) * source_tasks[j].env.velocity_noise)) -
                                         erf((v1 - mu_max_source[1]) / (
                                         np.sqrt(2.) * source_tasks[j].env.velocity_noise))) / 2.)
                            else:
                                M_P_vel[vel_prime] = \
                                    (erf((v2 - vm) / (np.sqrt(2.) * source_tasks[j].env.velocity_noise)) -
                                     erf((v1 - vm) / (np.sqrt(2.) * source_tasks[j].env.velocity_noise))) / 2.
                    self.L_P_eps_s_a_s_prime[j, s, a] = \
                        ((np.abs(source_tasks[j].env.action_reps[a]) / (
                        np.sqrt(2. * np.pi) * source_tasks[j].env.velocity_noise)) * M_e_vel.reshape((-1, 1)).dot(
                            M_P_pos.reshape((1, -1))) +
                         (np.abs(source_tasks[j].env.action_reps[a]) / (
                         np.sqrt(2. * np.pi) * source_tasks[j].env.position_noise)) * M_P_vel.reshape((-1, 1)).dot(
                             M_e_pos.reshape((1, -1)))).flatten()

            self.L_P_eps_s_prime[j] = self.L_P_eps_s_a_s_prime[j].max(axis=(0, 1))

            self.ns[j] = self.source_samples[j]['fs'].shape[0]

            self.source_samples[j]['eta_j'] = \
                source_policies[j].log_gradient_matrix[source_samples[j]['fsi'], source_samples[j]['ai']] * \
                (source_tasks[j].env.Q[source_samples[j]['fsi'], source_samples[j]['ai']] -
                 source_tasks[j].env.V[source_samples[j]['fsi']]).reshape((-1, 1))

    def clean_sources(self):
        pass

    def estimate_weights(self, target_samples, target_policy, target_power, target_size, target_Q, target_V, target_grad):
        var_eta_1 = np.var((target_Q[:target_size]-target_V[:target_size]).reshape((-1,1))*target_policy.log_gradient_matrix[target_samples['fsi'], target_samples['ai']], axis=0)
        w_idx = np.hstack((0., self.ns)).cumsum().astype(np.int64)
        n = self.ns.sum() + target_size
        w0 = np.ones(n - target_size, dtype=np.float64)
        l_bounds = np.zeros_like(w0)
        u_bounds = np.zeros_like(w0)
        for i in range(self.m):
            self.delta_P_eps_theta_s_s_prime[i] = \
                (self.source_tasks[i].env.transition_matrix * np.abs(self.source_policies[i].choice_matrix -
                                                                     target_policy.choice_matrix).reshape(
                    (target_policy.choice_matrix.shape) + (1,)) +
                 target_policy.choice_matrix.reshape(target_policy.choice_matrix.shape + (1,)) *
                 self.L_P_eps_s_a_s_prime[i] * np.abs(self.source_tasks[i].env.power - target_power)).sum(axis=1)
            self.delta_P_eps_theta_s[i] = self.delta_P_eps_theta_s_s_prime[i].max(axis=0)
            self.delta_delta[i] = (1. - self.gamma) * self.source_tasks[i].env.P_pi_inv.T.dot(
                0. + self.gamma * np.minimum(self.delta_P_eps_theta_s[i], np.ones_like(self.delta_P_eps_theta_s[i])))

            self.delta_zeta[i] = self.source_tasks[i].env.delta_distr.reshape((-1, 1)) * np.abs(
                self.source_policies[i].choice_matrix -
                target_policy.choice_matrix) + \
                                 target_policy.choice_matrix * np.minimum(self.delta_delta[i],
                                                                          np.ones_like(self.delta_delta[i])).reshape(
                                     (-1, 1))
            self.delta_J[i] = target_grad -\
                              ((self.source_tasks[i].env.zeta_distr*self.source_tasks[i].env.Q).reshape(self.source_tasks[i].env.Q.shape + (1,))*
                              self.source_policies[i].log_gradient_matrix).sum(axis=(0,1))/(1. - self.gamma)
            self.source_samples[i]['eta_1'] = target_policy.log_gradient_matrix[
                                                  self.source_samples[i]['fsi'], self.source_samples[i]['ai']] * \
                                              (target_Q[target_size+w_idx[i]:target_size+w_idx[i + 1]] - target_V[
                                                                                 target_size+w_idx[i]:target_size+w_idx[i + 1]]).reshape(
                                                  (-1, 1))

            l_bounds[w_idx[i]:w_idx[i + 1]] = np.maximum(np.zeros(self.ns[i], dtype=np.float64),
                                                         w0 - np.minimum(np.ones_like(self.delta_zeta[i]),
                                                                         self.delta_zeta[i])[
                                                             self.source_samples[i]['fsi'], self.source_samples[i][
                                                                 'ai']] /
                                                         self.source_tasks[i].env.zeta_distr[
                                                             self.source_samples[i]['fsi'], self.source_samples[i][
                                                                 'ai']])
            u_bounds[w_idx[i]:w_idx[i + 1]] = w0 + np.minimum(np.ones_like(self.delta_zeta[i]),
                                                              self.delta_zeta[i])[
                                                       self.source_samples[i]['fsi'], self.source_samples[i]['ai']] / \
                                                   self.source_tasks[i].env.zeta_distr[
                                                       self.source_samples[i]['fsi'], self.source_samples[i]['ai']]

        def g(w):
            bias = 0.
            vari = 0.
            for j in range(self.m):
                bias += self.ns[j]*(target_grad -
                                    (w[w_idx[j]:w_idx[j + 1]].reshape((-1, 1)) *self.source_samples[j]['eta_1']).mean(axis=0) /
                                    (1. - self.gamma)) / n
                vari += self.ns[j]*np.var(w[w_idx[j]:w_idx[j + 1]].reshape((-1, 1)) * self.source_samples[j]['eta_1'], axis=0) / (n * (1. - self.gamma)) ** 2
            bias = (bias ** 2).sum()
            vari += (1./n**2 - 1./target_size**2)*target_size*var_eta_1/((1. - self.gamma)**2)
            vari = vari.sum()
            return bias + vari

        def grad_g(w):
            bias = 0.
            for j in range(self.m):
                bias += self.ns[j] * (target_grad -
                                      (w[w_idx[j]:w_idx[j + 1]].reshape((-1, 1)) * self.source_samples[j]['eta_1']).mean(axis=0) /
                                      (1. - self.gamma)) / n
            grad = np.zeros(w.shape + (2,), dtype=np.float64)
            for j in range(self.m):
                    grad[w_idx[j]:w_idx[j + 1]] = 2 * self.source_samples[j]['eta_1'] * (-bias / (1. - self.gamma) + self.source_samples[j]['eta_1'] * w[w_idx[j]:w_idx[j + 1]].reshape((-1, 1)) / (1. - self.gamma) ** 2 - (self.source_samples[j]['eta_1'] * w[w_idx[j]:w_idx[j + 1]].reshape((-1, 1))).sum() / (self.ns[j] * (1. - self.gamma) ** 2)) / n ** 2
            grad = grad.sum(axis=1)
            return grad

        def hess_g(w):
            hess = np.zeros([w.shape[0]] * 2 + [2], dtype=np.float64)
            for j1 in range(self.m):
                for j2 in range(self.m):
                    hess[w_idx[j1]:w_idx[j1 + 1], w_idx[j2]:w_idx[j2 + 1]] = (
                    -self.source_samples[j1]['eta_1'] / (1. - self.gamma) ** 2).dot(
                        (1. + (j1 == j2) / self.ns[j1]) * self.source_samples[j2]['eta_1'].T)
            hess = hess * 2 / n ** 2
            return hess

        bounds = np.ones((l_bounds.size + u_bounds.size,), dtype=np.float64)
        bounds[0::2] = l_bounds
        bounds[1::2] = u_bounds
        bounds = tuple(map(tuple, bounds.reshape((-1, 2))))
        res = minimize(g, w0, jac=grad_g, bounds=bounds)
        w = res.x

        bias = 0.
        vari = 0.
        var_ncx = ((self.ns.sum()/n)**2)*var_eta_1/(target_size*(1. - self.gamma)**2)
        var_gauss = 0.
        eta_1 = (target_Q[:target_size]-target_V[:target_size]).reshape((-1,1))*target_policy.log_gradient_matrix[target_samples['fsi'], target_samples['ai']]
        for j in range(self.m):
            bias += self.ns[j] * (target_grad -
                                  (w[w_idx[j]:w_idx[j + 1]].reshape((-1, 1)) * self.source_samples[j]['eta_1']).mean(axis=0) /
                                  (1. - self.gamma)) / n
            vari += self.ns[j] * np.var(w[w_idx[j]:w_idx[j + 1]].reshape((-1, 1)) * self.source_samples[j]['eta_1'],axis=0) / (n * (1. - self.gamma)) ** 2
            var_ncx += self.ns[j] * np.var(w[w_idx[j]:w_idx[j + 1]].reshape((-1, 1)) * self.source_samples[j]['eta_1'],
                                           axis=0) /\
                       (n * (1. - self.gamma)) ** 2
            var_gauss += (self.ns[j]/(n*(1.-self.gamma))**2)**2*\
                         (moment(w[w_idx[j]:w_idx[j + 1]].reshape((-1, 1)) * self.source_samples[j]['eta_1'],moment=4,axis=0) -
                          ((self.ns[j]-3)/(self.ns[j]-1))*np.var(w[w_idx[j]:w_idx[j + 1]].reshape((-1, 1)) * self.source_samples[j]['eta_1'],axis=0)**2)/self.ns[j] +\
                         ((1./n**2 - 1./target_size**2)*target_size/((1. - self.gamma)**2))**2*\
                         (moment(eta_1, moment=4, axis=0) - ((target_size - 3) / (target_size - 1)) * np.var(eta_1, axis=0) ** 2) / target_size
        bias = np.abs(bias)
        vari += (1. / n ** 2 - 1. / target_size ** 2) * target_size * var_eta_1 / ((1. - self.gamma) ** 2)
        vari = vari.sum()
        var_gauss = var_gauss.sum()
        lb, ub = 0., 0.
        c, d = self.build_CI(bias[0]/np.sqrt(var_ncx[0]))
        lb += (c**2)*var_ncx[0]
        ub += (d**2)*var_ncx[0]
        c, d = self.build_CI(bias[1]/np.sqrt(var_ncx[1]))
        lb += (c**2)*var_ncx[1]
        ub += (d**2)*var_ncx[1]
        lb += vari - norm.ppf(0.1/2.)*np.sqrt(var_gauss)
        ub += vari + norm.ppf(0.1/2.)*np.sqrt(var_gauss)
        #print(res.fun, lb, ub)

        return res.x, lb <= 0.


#577
    def build_CI(self, y):
        r_script = 'C:/Users/danie/Documents/Daniel/Dropbox/PoliMi/Semestre 4/Thesis/RL-TL-Code/Importance sampling/CI.r'
        r_command = 'C:/Program Files/R/R-3.3.1/bin/Rscript'
        args = [str(y), str(1), str(0.1)]
        out = str(subprocess.check_output([r_command, r_script] + args, universal_newlines=True))
        out = np.array(list(map(float, str.split(out, ' '))), dtype=np.float64)
        return out