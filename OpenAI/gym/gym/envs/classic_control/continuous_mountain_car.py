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
import multiprocessing as mp
import multiprocessing.sharedctypes as sct
from multiprocessing.dummy import Pool

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
        self.bottom_position = self.transform(-0.5)
        self.power = power
        self.position_noise = (self.max_position - self.min_position)*position_noise
        self.velocity_noise = (self.max_speed - self.min_speed)*velocity_noise
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
            if np.all(self.position_reps < self.goal_position):
                self.goal_position = self.position_reps[-1]
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
            if self.model == 'G':
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
                if (position <= self.position_bins[1] and velocity < 0): velocity = 0
                position, velocity = self.discretize_state(np.array([position, velocity]))
            return np.array([position, velocity])
        elif initial_state.ndim == 2 and action.ndim == 2:
            positions = initial_state[:,0].copy()
            velocities = initial_state[:,1].copy()
            forces = -1 + 2.0 * (np.clip(action, self.min_action, self.max_action) - self.min_action) / (self.max_action - self.min_action)
            if self.model == 'G':
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
                velocities[np.logical_and(positions <= self.position_bins[1], velocities < 0)] = 0
                positions, velocities = self.discretize_state(np.vstack((positions, velocities)).T).T
        return np.vstack((positions, velocities)).T



    def add_noise(self, next_state):
        if self.model == 'G':
            position, velocity = next_state
            noise = np.random.randn()
            position += self.position_noise*noise
            noise1 = np.random.randn()
            velocity += self.velocity_noise*noise1

        if (velocity >= self.max_speed): velocity = self.max_speed
        if (velocity <= self.min_speed): velocity = self.min_speed
        if (position >= self.max_position): position = self.max_position
        if (position <= self.min_position): position = self.min_position
        if (position == self.min_position and velocity < 0): velocity = 0

        return np.array([position, velocity])



    def set_policy(self, policy, gamma):
        self.policy = policy
        P_pi = np.transpose(self.transition_matrix, axes=(2, 0, 1)).copy()
        P_pi = (P_pi * policy.choice_matrix).sum(axis=2).T
        P_pi_inv = np.linalg.inv(np.eye(self.transition_matrix.shape[0]) - gamma * P_pi)
        self.delta_distr = (1 - gamma) * P_pi_inv.T.dot(self.initial_state_distr)
        self.dseta_distr = (policy.choice_matrix.T * self.delta_distr).T
        R_pi = (self.R* policy.choice_matrix).sum(axis=1)
        self.V = P_pi_inv.dot(R_pi)
        self.J = self.V.dot(self.initial_state_distr)
        self.Q = self.R + gamma*(self.transition_matrix*self.V).sum(axis=2)



    def sample_step(self, n_samples=1):
        idx = np.random.choice(self.dseta_distr.size, p=self.dseta_distr.flatten(), size=n_samples)
        state_idx = (idx / self.action_reps.shape[0]).astype(np.int64)
        action_idx = (idx % self.action_reps.shape[0]).astype(np.int64)
        first_states = self.state_reps[state_idx]
        actions = self.action_reps[action_idx].reshape((-1,1))
        cum_p = np.hstack((np.zeros((n_samples, 1), dtype=np.float64), self.transition_matrix[state_idx,action_idx,:].cumsum(axis=1)))
        samps = np.random.random_sample(n_samples).reshape((-1,1))
        next_state_idx = (cum_p <= samps).sum(axis=1) - 1
        #assert np.all(next_state_idx >= 0)
        cum_p = np.hstack((np.zeros((n_samples, 1), dtype=np.float64), self.policy.choice_matrix[next_state_idx, :].cumsum(axis=1)))
        samps = np.random.random_sample(n_samples).reshape((-1, 1))
        next_action_idx = (cum_p <= samps).sum(axis=1) - 1
        #assert np.all(next_action_idx >= 0)
        next_states = self.state_reps[next_state_idx]
        next_actions = self.action_reps[next_action_idx].reshape((-1,1))
        rewards = self.reward_model(first_states, actions, next_states)
        return {'fs':first_states, 'a':actions, 'ns':next_states, 'na':next_actions, 'r':rewards,
                'fsi':state_idx, 'ai':action_idx, 'nsi':next_state_idx, 'nai':next_action_idx}



    def transition_model_pdf(self, next_state, first_state, action, as_array=False):
        if next_state.size == 2 and first_state.size == 2 and action.size == 1:
            means = self.clean_step(first_state, action)
            if self.model == 'G':
                ex = (next_state[0] - means[0]) / self.position_noise
                pos_pdf = np.exp(-ex*ex/2.0)/(np.sqrt(2*np.pi)*self.position_noise)
                ex = (next_state[1] - means[1]) / self.velocity_noise
                vel_pdf = np.exp(-ex * ex / 2.0) / (np.sqrt(2 * np.pi) * self.velocity_noise)
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
            zero_vels_mask = next_state[:,1] == 0
            vel_pdf[zero_vels_mask] += (1+erf((self.min_position - means[:,0][zero_vels_mask])/(self.position_noise*np.sqrt(2.))))/2.
            if as_array:
                return np.array([pos_pdf, vel_pdf])
            else:
                return pos_pdf * vel_pdf



    def reward_model(self, first_state, action, next_state):
        if first_state.size == 2 and action.size == 1 and next_state.size == 2:
            first_state = first_state.ravel()
            action = action.ravel()
            next_state = next_state.ravel()
            reward = -math.pow(2, np.abs(min(max(action[0], self.min_action), self.max_action))) * 0.1
            if next_state[0] >= self.goal_position:
                reward += 100.
            if first_state[0] >= self.goal_position:
                reward = 0.
                #print("Warning: asking for reward from a terminal state")
            return reward
        elif first_state.ndim == 2 and action.ndim == 2 and next_state.ndim == 2:
            rewards = -np.power(2, np.abs(np.clip(action.flatten(), self.min_action, self.max_action)))*0.1
            mask_1 = next_state[:,0] >= self.goal_position
            rewards[mask_1] += 100.
            mask_2 = first_state[:,0] >= self.goal_position
            rewards[mask_2] = 0.
            return rewards



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



    def transform(self, x):
        return self.min_position + (self.max_position - self.min_position)*(x + 1.2)/1.8



    def inverse_transform(self, x):
        return -1.2 + 1.8*(x - self.min_position)/(self.max_position - self.min_position)



    def rescale(self, x):
        return (self.max_position- self.min_position)*x/1.8



    def restore(self, x):
        return  1.8 * x / (self.max_position - self.min_position)