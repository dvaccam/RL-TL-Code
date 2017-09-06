# -*- coding: utf-8 -*-
"""
@author: Olivier Sigaud

A merge between two sources:

* Adaptation of the MountainCar Environment from the "FAReinforcement" library
of Jose Antonio Martin H. (version 1.0), adapted by  'Tom Schaul, tom@idsia.ch'
and then modified by Arnaud de Broissia

* the OpenAI/gym MountainCar environment
itself from 
https://webdocs.cs.ualberta.ca/~sutton/MountainCar/MountainCar1.cp
"""

from __future__ import division
import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from scipy.stats import norm, beta
from scipy.special import erf
import time

class Continuous_MountainCarEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }



    def __init__(self, min_position=-1.2, max_position=0.6, min_action=-1.0, max_action=1.0, power=0.0015, position_noise=0.01,
                 velocity_noise=0.01, seed=None, model='G', discrete=False, n_position_bins=None, n_velocity_bins=None,
                 n_action_bins=None):
        self.min_action = min_action
        self.max_action = max_action
        self.min_position = min_position
        self.max_position = max_position
        self.min_initial_state = self.transform(-0.525)
        self.max_initial_state = self.transform(-0.475)
        self.max_speed = self.rescale(0.07)
        self.min_speed = -self.max_speed
        self.goal_position = self.transform(0.45) # was 0.5 in gym, 0.45 in Arnaud de Broissia's version
        self.power = power
        self.position_noise = (self.max_position - self.min_position)*position_noise#self.rescale(1.8*0.05)
        self.velocity_noise = (self.max_speed - self.min_speed)*velocity_noise#self.rescale(0.14*0.05)
        self.model = model

        self.discrete = discrete
        self.n_position_bins = n_position_bins
        self.n_velocity_bins = n_velocity_bins
        self.n_action_bins = n_action_bins
        if self.discrete:
            assert self.model == 'S' and self.n_position_bins is not None and self.n_velocity_bins is not None and self.n_action_bins is not None
            self.position_bins = np.linspace(self.min_position, self.max_position, self.n_position_bins)
            self.velocity_bins = np.linspace(self.min_speed, self.max_speed, self.n_velocity_bins)
            self.action_bins = np.linspace(self.min_action, self.max_action, self.n_action_bins)
            self.position_reps = (self.position_bins[:-1] + self.position_bins[1:]) / 2.
            self.velocity_reps = (self.velocity_bins[:-1] + self.velocity_bins[1:]) / 2.
            self.action_reps = (self.action_bins[:-1] + self.action_bins[1:]) / 2.
            self.build_model_matrices()

        self.low_state = np.array([self.min_position, self.min_speed])
        self.high_state = np.array([self.max_position, self.max_speed])

        self.viewer = None

        self.action_space = spaces.Box(self.min_action, self.max_action, shape = (1,))
        self.observation_space = spaces.Box(self.low_state, self.high_state)

        self._seed(seed)
        self.reset()



    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]



    def _reset(self):
        self.state = np.array([self.np_random.uniform(low=self.min_initial_state, high=self.max_initial_state), 0])
        if self.discrete:
            self.state = self.discretize_state(self.state)
        return np.array(self.state)



    def _step(self, action):
        if self.discrete:
            state_idx = self.state_to_idx[self.state[0]][self.state[1]]
            action_idx = self.action_to_idx[action[0]]
            next_state = np.random.choice(np.arange(self.state_reps.shape[0]),
                                          p=self.transition_matrix[state_idx, action_idx, :])
            next_state = self.state_reps[next_state]
        else:
            next_state = self.clean_step(self.state, action)
            next_state = self.add_noise(next_state)

        done = bool(next_state[0] >= self.goal_position)

        reward = self.reward_model(self.state, action, next_state)

        self.state = next_state.copy()

        return self.state, reward, done, {}

#    def get_state(self):
#        return self.state



    def _height(self, xs):
        return np.sin(3 * xs)*.45+.55



    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        screen_width = 600
        screen_height = 400

        min_position_view = self.inverse_transform(self.min_position)
        max_position_view = self.inverse_transform(self.max_position)
        goal_position_view = self.inverse_transform(self.goal_position)
        world_width = max_position_view - min_position_view
        scale = screen_width / world_width
        carwidth = 40
        carheight = 20

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            xs = np.linspace(min_position_view, max_position_view, 100)
            ys = self._height(xs)
            xys = list(zip((xs - min_position_view) * scale, ys * scale))

            self.track = rendering.make_polyline(xys)
            self.track.set_linewidth(4)
            self.viewer.add_geom(self.track)

            clearance = 10

            l, r, t, b = -carwidth / 2, carwidth / 2, carheight, 0
            car = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            car.add_attr(rendering.Transform(translation=(0, clearance)))
            self.cartrans = rendering.Transform()
            car.add_attr(self.cartrans)
            self.viewer.add_geom(car)
            frontwheel = rendering.make_circle(carheight / 2.5)
            frontwheel.set_color(.5, .5, .5)
            frontwheel.add_attr(rendering.Transform(translation=(carwidth / 4, clearance)))
            frontwheel.add_attr(self.cartrans)
            self.viewer.add_geom(frontwheel)
            backwheel = rendering.make_circle(carheight / 2.5)
            backwheel.add_attr(rendering.Transform(translation=(-carwidth / 4, clearance)))
            backwheel.add_attr(self.cartrans)
            backwheel.set_color(.5, .5, .5)
            self.viewer.add_geom(backwheel)
            flagx = (goal_position_view - min_position_view) * scale
            flagy1 = self._height(goal_position_view) * scale
            flagy2 = flagy1 + 50
            flagpole = rendering.Line((flagx, flagy1), (flagx, flagy2))
            self.viewer.add_geom(flagpole)
            flag = rendering.FilledPolygon([(flagx, flagy2), (flagx, flagy2 - 10), (flagx + 25, flagy2 - 5)])
            flag.set_color(.8, .8, 0)
            self.viewer.add_geom(flag)

        pos = self.inverse_transform(self.state[0])
        self.cartrans.set_translation((pos - min_position_view) * scale, self._height(pos) * scale)
        self.cartrans.set_rotation(math.cos(3 * pos))

        return self.viewer.render(return_rgb_array = mode=='rgb_array')



    def clean_step(self, initial_state, action):
        if initial_state.size == 2 and action.size == 1:
            position, velocity = initial_state
            force = -1 + 2.0 * (min(max(action[0], self.min_action), self.max_action) - self.min_action)/(self.max_action - self.min_action)
            for i in range(1):
                if self.model == 'B':
                    eps_pos = self.rescale(1e-2)
                    eps_vel = self.rescale(1e-3)
                    max_mu_pos = (self.min_position + self.max_position +
                                  np.sqrt((self.min_position + self.max_position)**2 - 4*(self.min_position*self.max_position + self.position_noise**2)))/2 - eps_pos
                    min_mu_pos = (self.min_position + self.max_position -
                                  np.sqrt((self.min_position + self.max_position) ** 2 - 4 * (self.min_position * self.max_position + self.position_noise**2))) / 2 + eps_pos
                    max_mu_vel = (self.min_speed + self.max_speed +
                                  np.sqrt((self.min_speed + self.max_speed)**2 - 4*(self.min_speed*self.max_speed + self.velocity_noise**2)))/2 - eps_vel
                    min_mu_vel = (self.min_speed + self.max_speed -
                                  np.sqrt((self.min_speed + self.max_speed)**2 - 4*(self.min_speed*self.max_speed + self.velocity_noise**2)))/2 + eps_vel
                    velocity += force * self.power - self.rescale(0.0025) * math.cos(3 * self.inverse_transform(position))
                    if (velocity >= max_mu_vel): velocity = max_mu_vel
                    if (velocity <= min_mu_vel): velocity = min_mu_vel
                    position += velocity
                    if (position >= max_mu_pos): position = max_mu_pos
                    if (position <= min_mu_pos): position = min_mu_pos
                    if (position == self.min_position and velocity < 0): velocity = 0
                elif self.model == 'G':
                    velocity += force * self.power - self.rescale(0.0025) * math.cos(3 * self.inverse_transform(position))
                    if (velocity > self.max_speed): velocity = self.max_speed
                    if (velocity < self.min_speed): velocity = self.min_speed
                    position += velocity
                    if (position > self.max_position): position = self.max_position
                    if (position < self.min_position): position = self.min_position
                    if (position == self.min_position and velocity < 0): velocity = 0
                elif self.model == 'S':
                    velocity += force * self.power - self.rescale(0.0025) * math.cos(3 * self.inverse_transform(position))
                    if (velocity > self.max_speed): velocity = self.max_speed
                    if (velocity < self.min_speed): velocity = self.min_speed
                    position += velocity
                    if (position > self.max_position): position = self.max_position
                    if (position < self.min_position): position = self.min_position
                    if (position <= self.position_bins[0] and velocity < 0): velocity = 0
                    position, velocity = self.discretize_state(np.array([position, velocity]))

            return np.array([position, velocity])
        elif initial_state.ndim == 2 and action.ndim == 2:
            positions = initial_state[:,0].copy()
            velocities = initial_state[:,1].copy()
            forces = -1 + 2.0 * (np.clip(action, self.min_action, self.max_action) - self.min_action) / (self.max_action - self.min_action)
            for i in range(1):
                if self.model == 'B':
                    eps_pos = self.rescale(1e-2)
                    eps_vel = self.rescale(1e-3)
                    max_mu_pos = (self.min_position + self.max_position +
                                  np.sqrt((self.min_position + self.max_position) ** 2 - 4 * (
                                  self.min_position * self.max_position + self.position_noise ** 2))) / 2 - eps_pos
                    min_mu_pos = (self.min_position + self.max_position -
                                  np.sqrt((self.min_position + self.max_position) ** 2 - 4 * (
                                  self.min_position * self.max_position + self.position_noise ** 2))) / 2 + eps_pos
                    max_mu_vel = (self.min_speed + self.max_speed +
                                  np.sqrt((self.min_speed + self.max_speed) ** 2 - 4 * (
                                  self.min_speed * self.max_speed + self.velocity_noise ** 2))) / 2 - eps_vel
                    min_mu_vel = (self.min_speed + self.max_speed -
                                  np.sqrt((self.min_speed + self.max_speed) ** 2 - 4 * (
                                  self.min_speed * self.max_speed + self.velocity_noise ** 2))) / 2 + eps_vel
                    velocity += force * self.power - self.rescale(0.0025) * math.cos(
                        3 * self.inverse_transform(position))
                    if (velocity >= max_mu_vel): velocity = max_mu_vel
                    if (velocity <= min_mu_vel): velocity = min_mu_vel
                    position += velocity
                    if (position >= max_mu_pos): position = max_mu_pos
                    if (position <= min_mu_pos): position = min_mu_pos
                elif self.model == 'G':
                    velocities += forces.flatten() * self.power - self.rescale(0.0025) * np.cos(3 * self.inverse_transform(positions))
                    velocities = np.clip(velocities, self.min_speed, self.max_speed)
                    positions += velocities
                    positions = np.clip(positions, self.min_position, self.max_position)
                    velocities[np.logical_and(positions == self.min_position, velocities < 0)] = 0
                elif self.model == 'S':
                    velocities += forces.flatten() * self.power - self.rescale(0.0025) * np.cos(3 * self.inverse_transform(positions))
                    velocities = np.clip(velocities, self.min_speed, self.max_speed)
                    positions += velocities
                    positions = np.clip(positions, self.min_position, self.max_position)
                    velocities[np.logical_and(positions <= self.position_bins[0], velocities < 0)] = 0
                    positions, velocities = self.discretize_state(np.vstack((positions, velocities)).T).T
            return np.vstack((positions, velocities)).T



    def set_policy(self, policy, gamma):
        self.policy = policy
        self.gamma = gamma
        self.delta_distr = self.calculate_delta_distr()
        self.dseta_distr = self.calculate_dseta_distr()
        self.Q = self.calculate_Q()



    def sample_step(self):
        state_idx = np.random.choice(self.state_reps.shape[0], p=self.delta_distr)
        while self.state_reps[state_idx][0] >= self.goal_position:
            state_idx = np.random.choice(self.state_reps.shape[0], p=self.delta_distr)
        first_state = self.state_reps[state_idx]
        action = np.array([self.policy.produce_action(first_state)])
        next_state_idx = np.random.choice(self.state_reps.shape[0], p=self.transition_matrix[state_idx, self.action_to_idx[action[0]]])
        next_state = self.state_reps[next_state_idx]
        reward = self.reward_model(first_state, action, next_state)
        return first_state, action, next_state, reward



    def transform(self, x):
        return self.min_position + (self.max_position - self.min_position)*(x + 1.2)/1.8



    def inverse_transform(self, x):
        return -1.2 + 1.8*(x - self.min_position)/(self.max_position - self.min_position)



    def rescale(self, x):
        return (self.max_position- self.min_position)*x/1.8



    def restore(self, x):
        return  1.8 * x / (self.max_position - self.min_position)



    def transition_model_pdf(self, next_state, first_state, action, as_array=False):
        if next_state.size == 2 and first_state.size == 2 and action.size == 1:
            means = self.clean_step(first_state, action)

            if self.model == 'G':
                ex = (next_state[0] - means[0]) / self.position_noise
                pos_pdf = np.exp(-ex*ex/2.0)/(np.sqrt(2*np.pi)*self.position_noise)
                ex = (next_state[1] - means[1]) / self.velocity_noise
                vel_pdf = np.exp(-ex * ex / 2.0) / (np.sqrt(2 * np.pi) * self.velocity_noise)
                #pos_pdf = norm.pdf(next_state[0], loc=means[0], scale=self.position_noise)
                #vel_pdf = norm.pdf(next_state[1], loc=means[1], scale=self.velocity_noise)
            elif self.model == 'B':
                mu_pos_std = (means[0] - self.min_position) / (self.max_position - self.min_position)
                sigma2_pos_std = self.position_noise**2 / ((self.max_position - self.min_position) ** 2)
                a_pos = (mu_pos_std ** 2 - mu_pos_std ** 3 - mu_pos_std * sigma2_pos_std) / sigma2_pos_std
                b_pos = (1 - mu_pos_std) * a_pos / mu_pos_std
                pos_pdf = beta.pdf(next_state[0], a=a_pos, b=b_pos, loc=self.min_position, scale=self.max_position - self.min_position)

                mu_vel_std = (means[1] - self.min_speed) / (self.max_speed - self.min_speed)
                sigma2_vel_std = self.velocity_noise**2 / ((self.max_speed - self.min_speed) ** 2)
                a_vel = (mu_vel_std ** 2 - mu_vel_std ** 3 - mu_vel_std * sigma2_vel_std) / sigma2_vel_std
                b_vel = (1 - mu_vel_std) * a_vel / mu_vel_std
                vel_pdf = beta.pdf(next_state[1], a=a_vel, b=b_vel, loc=self.min_speed, scale=self.max_speed - self.min_speed)

            if next_state[1] == 0:
                vel_pdf += (1+erf((self.min_position - means[0])/(self.position_noise*np.sqrt(2.))))/2.

            if as_array:
                return np.array([pos_pdf,vel_pdf])
            else:
                return pos_pdf*vel_pdf
        elif first_state.ndim == 2 and next_state.ndim == 2 and action.ndim == 2:
            means = self.clean_step(first_state, action)

            if self.model == 'G':
                ex = (next_state[:,0] - means[:,0]) / self.position_noise
                pos_pdf = np.exp(-ex * ex / 2.0) / (np.sqrt(2 * np.pi) * self.position_noise)
                ex = (next_state[:,1] - means[:,1]) / self.velocity_noise
                vel_pdf = np.exp(-ex * ex / 2.0) / (np.sqrt(2 * np.pi) * self.velocity_noise)
                # pos_pdf = norm.pdf(next_state[0], loc=means[0], scale=self.position_noise)
                # vel_pdf = norm.pdf(next_state[1], loc=means[1], scale=self.velocity_noise)
            elif self.model == 'B':
                mu_pos_std = (means[0] - self.min_position) / (self.max_position - self.min_position)
                sigma2_pos_std = self.position_noise ** 2 / ((self.max_position - self.min_position) ** 2)
                a_pos = (mu_pos_std ** 2 - mu_pos_std ** 3 - mu_pos_std * sigma2_pos_std) / sigma2_pos_std
                b_pos = (1 - mu_pos_std) * a_pos / mu_pos_std
                pos_pdf = beta.pdf(next_state[0], a=a_pos, b=b_pos, loc=self.min_position,
                                   scale=self.max_position - self.min_position)

                mu_vel_std = (means[1] - self.min_speed) / (self.max_speed - self.min_speed)
                sigma2_vel_std = self.velocity_noise ** 2 / ((self.max_speed - self.min_speed) ** 2)
                a_vel = (mu_vel_std ** 2 - mu_vel_std ** 3 - mu_vel_std * sigma2_vel_std) / sigma2_vel_std
                b_vel = (1 - mu_vel_std) * a_vel / mu_vel_std
                vel_pdf = beta.pdf(next_state[1], a=a_vel, b=b_vel, loc=self.min_speed,
                                   scale=self.max_speed - self.min_speed)

            zero_vels_mask = next_state[:,1] == 0
            vel_pdf[zero_vels_mask] += (1+erf((self.min_position - means[:,0][zero_vels_mask])/(self.position_noise*np.sqrt(2.))))/2.

            if as_array:
                return np.array([pos_pdf, vel_pdf])
            else:
                return pos_pdf * vel_pdf



    def reward_model(self, first_state, action, next_state):
        if first_state.size == 2 and action.size == 1 and next_state.size == 2:
            reward = -math.pow(2, np.abs(min(max(action[0], self.min_action), self.max_action))) * 0.1
            if next_state[0] >= self.goal_position:
                reward += 100.
            if first_state[0] >= self.goal_position:
                reward = 0.
                print("Warning: asking for reward from a terminal state")
            return reward
        elif first_state.ndim == 2 and action.ndim == 2 and next_state.ndim == 2:
            rewards = -np.power(2, np.abs(np.clip(action.flatten(), self.min_action, self.max_action)))*0.1
            mask_1 = next_state[:,0] >= self.goal_position
            rewards[mask_1] += 100.
            mask_2 = first_state[:,0] >= self.goal_position
            rewards[mask_2] = 0.
            return rewards



    def add_noise(self, next_state):
        if self.model == 'G':
            position, velocity = next_state
            aux = position
            aux1 = velocity
            #print("Det:", velocity, position + velocity)
            noise = np.random.randn()
            position += self.position_noise*noise

            noise1 = np.random.randn()
            velocity += self.velocity_noise*noise1
            #print("Noise:",velocity, position)
            #print("Diff:", np.abs(aux-position), np.abs(aux1-velocity))
            #print(0.005*noise1, 0.005*noise)
            #print(norm.pdf(noise1)/self.velocity_noise, norm.pdf(position, loc=aux, scale=self.position_noise))
        elif self.model == 'B':
            eps = 0
            mu_position = next_state[0]
            mu_velocity = next_state[1]
            aux = mu_position
            aux1 = mu_velocity

            mu_position = (mu_position - self.min_position) / (self.max_position - self.min_position)
            sigma2_position = self.position_noise ** 2
            sigma2_position = sigma2_position / ((self.max_position - self.min_position) ** 2)
            a_position = (mu_position ** 2 - mu_position ** 3 - mu_position * sigma2_position) / sigma2_position
            b_position = (1 - mu_position) * a_position / mu_position
            noise = np.random.beta(a=a_position, b=b_position)
            position = (self.max_position - self.min_position) * noise + self.min_position
            while position >= self.max_position or position <= self.min_position:
                noise = np.random.beta(a=a_position, b=b_position)
                position = (self.max_position - self.min_position) * noise + self.min_position

            mu_velocity = (mu_velocity - self.min_speed) / (self.max_speed - self.min_speed)
            sigma2_velocity = self.velocity_noise ** 2
            sigma2_velocity = sigma2_velocity / ((self.max_speed - self.min_speed) ** 2)
            a_velocity = (mu_velocity ** 2 - mu_velocity ** 3 - mu_velocity * sigma2_velocity) / sigma2_velocity
            b_velocity = (1 - mu_velocity) * a_velocity / mu_velocity
            noise1 = np.random.beta(a=a_velocity, b=b_velocity)
            velocity = (self.max_speed - self.min_speed) * noise1 + self.min_speed
            while np.abs(velocity) >= self.max_speed:
                noise1 = np.random.beta(a=a_velocity, b=b_velocity)
                velocity = (self.max_speed - self.min_speed) * noise1 + self.min_speed
            #print(beta.pdf(noise, a=a_position, b=b_position)/(self.max_position - self.min_position),
            #      beta.pdf(noise1, a=a_velocity, b=b_velocity)/(self.max_speed - self.min_speed))
        elif self.model == 'S':
            qwe = 3

        if (velocity >= self.max_speed): velocity = self.max_speed
        if (velocity <= self.min_speed): velocity = self.min_speed
        if (position >= self.max_position): position = self.max_position
        if (position <= self.min_position): position = self.min_position
        if (position == self.min_position and velocity < 0): velocity = 0

        return np.array([position, velocity])



    def discretize_state(self, state):
        if state.size == 2:
            pos_idx = np.digitize(state[0], self.position_bins)
            vel_idx = np.digitize(state[1], self.velocity_bins)
            if pos_idx == self.position_bins.shape[0]:
                pos_idx -= 1
            if vel_idx == self.velocity_bins.shape[0]:
                vel_idx -= 1
            return np.array([self.position_reps[pos_idx - 1], self.velocity_reps[vel_idx - 1]])
        if state.ndim == 2:
            pos_idx = np.digitize(state[:,0], self.position_bins)
            vel_idx = np.digitize(state[:,1], self.velocity_bins)
            pos_idx[pos_idx == self.position_bins.shape[0]] -= 1
            vel_idx[vel_idx == self.velocity_bins.shape[0]] -= 1
            return np.vstack((self.position_reps[pos_idx - 1], self.velocity_reps[vel_idx - 1])).T



    def build_model_matrices(self):
        self.state_reps = np.dstack(np.meshgrid(self.position_reps, self.velocity_reps)).reshape(-1, 2)
        self.transition_matrix = np.zeros((self.state_reps.shape[0], self.action_reps.shape[0], self.state_reps.shape[0]), dtype=np.float64)
        self.R = np.zeros((self.transition_matrix.shape[0], self.transition_matrix.shape[1]), dtype=np.float64)
        self.state_to_idx = {self.position_reps[i]:
                                 {self.velocity_reps[j]:i + j*self.position_reps.shape[0]
                                  for j in range(self.velocity_reps.shape[0])}
                             for i in range(self.position_reps.shape[0])}
        self.action_to_idx = {self.action_reps[i]:i for i in range(self.action_reps.shape[0])}
        idx_grid = np.dstack(np.meshgrid(np.arange(self.state_reps.shape[0]), np.arange(self.action_reps.shape[0]), indexing='ij')).reshape(-1, 2)
        mu = self.clean_step(self.state_reps[idx_grid[:, 0]],
                             self.action_reps[idx_grid[:, 1]].reshape((-1, 1))).reshape((self.state_reps.shape[0], self.action_reps.shape[0], 2))
        pos_mask_1 = np.arange(self.state_reps.shape[0]) % self.position_reps.shape[0] == 0
        self.transition_matrix[:,:,pos_mask_1] = np.dstack([(1 + erf((self.position_bins[1] - mu[:,:,0])/(self.position_noise*np.sqrt(2.))))/2.]*pos_mask_1.sum())
        pos_mask_2 = np.arange(self.state_reps.shape[0]) % self.position_reps.shape[0] == self.position_reps.shape[0] - 1
        self.transition_matrix[:,:,pos_mask_2] = np.dstack([(1 - erf((self.position_bins[-2] - mu[:, :, 0]) / (self.position_noise * np.sqrt(2.)))) / 2.] * pos_mask_2.sum())
        pos_mask_3 = np.logical_not(np.logical_or(pos_mask_1, pos_mask_2))
        mu_pos_rep = np.dstack([mu[:,:,0]]*pos_mask_3.sum())
        pos_bins_rep_right = np.tile(self.position_bins[2:-1], self.velocity_reps.shape[0])
        pos_bins_rep_left = np.tile(self.position_bins[1:-2], self.velocity_reps.shape[0])
        self.transition_matrix[:, :, pos_mask_3] = (erf((pos_bins_rep_right - mu_pos_rep) / (self.position_noise * np.sqrt(2.))) - erf((pos_bins_rep_left - mu_pos_rep) / (self.position_noise * np.sqrt(2.))))/ 2.
        del pos_mask_1, pos_mask_2, pos_mask_3, mu_pos_rep, pos_bins_rep_right, pos_bins_rep_left

        vel_mask_1 = (np.arange(self.state_reps.shape[0]) / self.position_reps.shape[0]).astype(np.int64) == 0
        self.transition_matrix[:, :, vel_mask_1] *= np.dstack([(1 + erf((self.velocity_bins[1] - mu[:, :, 1]) / (self.velocity_noise * np.sqrt(2.)))) / 2.] * vel_mask_1.sum())
        vel_mask_2 = (np.arange(self.state_reps.shape[0]) / self.position_reps.shape[0]).astype(np.int64) == self.position_reps.shape[0] - 1
        self.transition_matrix[:, :, vel_mask_2] *= np.dstack([(1 - erf((self.velocity_bins[-2] - mu[:, :, 1]) / (self.velocity_noise * np.sqrt(2.)))) / 2.] * vel_mask_2.sum())
        vel_mask_3 = np.logical_not(np.logical_or(vel_mask_1, vel_mask_2))
        mu_vel_rep = np.dstack([mu[:, :, 1]] * vel_mask_3.sum())
        vel_bins_rep_right = np.repeat(self.velocity_bins[2:-1], self.position_reps.shape[0])
        vel_bins_rep_left = np.repeat(self.velocity_bins[1:-2], self.position_reps.shape[0])
        self.transition_matrix[:, :, vel_mask_3] *= (erf((vel_bins_rep_right - mu_vel_rep) / (self.velocity_noise * np.sqrt(2.))) - erf((vel_bins_rep_left - mu_vel_rep) / (self.velocity_noise * np.sqrt(2.)))) / 2.
        del vel_mask_1, vel_mask_2, vel_mask_3, mu_vel_rep, vel_bins_rep_right, vel_bins_rep_left, mu

        goal_pos_mask = self.state_reps[:,0] >= self.goal_position
        aux = np.eye(self.state_reps.shape[0], dtype=np.float64)[goal_pos_mask,:]
        self.transition_matrix[goal_pos_mask,:,:] = np.transpose(np.dstack([aux]*self.action_reps.shape[0]), axes=(0,2,1))
        del goal_pos_mask, aux

        idx_cube = np.dstack(np.meshgrid(np.arange(idx_grid.shape[0]), np.arange(self.state_reps.shape[0]), indexing='ij')).reshape(-1, 2)
        idx_cube = np.hstack((idx_grid[idx_cube[:,0]], idx_cube[:,1].reshape(-1,1)))
        r = self.reward_model(self.state_reps[idx_cube[:,0]], self.action_reps[idx_cube[:,1]].reshape((-1,1)), self.state_reps[idx_cube[:,2]]).reshape(self.transition_matrix.shape)
        self.R = (r * self.transition_matrix).sum(axis=2)
        del r, idx_grid, idx_cube

        self.initial_state_distr = np.zeros(self.state_reps.shape[0], dtype=np.float64)
        initial_bins_idx = np.argwhere(np.logical_and(self.min_initial_state <= self.position_bins, self.position_bins  <= self.max_initial_state)).ravel()
        init_rep = self.discretize_state(np.array([self.min_initial_state,0]))
        aux = self.initial_state_distr[self.state_reps[:, 1] == init_rep[1]]
        if initial_bins_idx.size >= 1:
            mass = np.concatenate(([self.min_initial_state], self.position_bins[initial_bins_idx], [self.max_initial_state]))
            mass = np.diff(mass) / (self.max_initial_state - self.min_initial_state)
            initial_state_idx = np.concatenate((initial_bins_idx - 1, [initial_bins_idx[-1]]))
            aux[initial_state_idx] = mass
        else:
            aux[np.where(self.position_reps == init_rep[0])[0]] = 1.
        self.initial_state_distr[self.state_reps[:, 1] == init_rep[1]] = aux



    # Note: empirical discounted distribution of states and the delta (with terminal states going to themselves and zero-reward)
    # differ only in the terminal states, exactly because the ciclic transition is absent in the samples. This does not affect the
    # final result as such states have zero reward
    def calculate_delta_distr(self):
        P_pi_T = np.transpose(self.transition_matrix, axes=(2, 0, 1)).copy()
        P_pi_T = (P_pi_T * self.policy.choice_matrix).sum(axis=2)
        delta_distr = np.eye(self.transition_matrix.shape[0]) - self.gamma * P_pi_T
        delta_distr = np.linalg.inv(delta_distr)
        delta_distr = (1 - self.gamma if self.gamma != 1 else 1) * delta_distr.dot(self.initial_state_distr)
        return delta_distr



    def calculate_dseta_distr(self):
        return (self.policy.choice_matrix.T * self.delta_distr).T



    def calculate_Q(self):
        P_pi = np.transpose(self.transition_matrix, axes=(2, 0, 1)).copy()
        P_pi = (P_pi * self.policy.choice_matrix).sum(axis=2).T
        Q = np.eye(self.state_reps.shape[0]) - self.gamma*P_pi
        Q = np.linalg.inv(Q).dot(self.R)
        return Q