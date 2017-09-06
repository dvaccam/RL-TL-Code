"""
https://webdocs.cs.ualberta.ca/~sutton/MountainCar/MountainCar1.cp
"""

import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np


class MountainCarEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, min_position=-1.2, max_position=0.6, max_speed=0.07, slope=0.45,
                 aceleration=0.001, initial_position='random', goal_position=0.5, discretize=False,
                 seed=None):
        self.min_position = min_position
        self.max_position = max_position

        self.max_speed = max_speed
        self.slope = slope
        self.aceleration = aceleration

        if type(discretize) != bool:
            self.bins = (np.linspace(start=self.min_position, stop=self.max_position, num=discretize[0]),
                         np.linspace(start=-self.max_speed, stop=self.max_speed, num=discretize[1]))
            self.goal_position = self.bins[0][np.array(abs(self.bins[0] - goal_position)).argmin()]
        else:
            self.bins = None
            self.goal_position = goal_position

        self.initial_position = initial_position

        self.low = np.array([self.min_position, -self.max_speed])
        self.high = np.array([self.max_position, self.max_speed])

        self.viewer = None

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(self.low, self.high)

        self._seed(seed)
        self.reset()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        position, velocity = self.state
        velocity += (action - 1) * self.aceleration + math.cos(3 * position) * (-0.0025)
        if self.bins is None:
            velocity = np.clip(velocity, -self.max_speed, self.max_speed)
            position += velocity
            position = np.clip(position, self.min_position, self.max_position)
        else:
            velocity = self.bins[1][np.array(abs(self.bins[1]-velocity)).argmin()]
            position += velocity
            position = self.bins[0][np.array(abs(self.bins[0] - position)).argmin()]

        if (position == self.min_position and velocity < 0): velocity = 0

        done = bool(position >= self.goal_position)
        reward = -1.0

        self.state = (position, velocity)
        return np.array(self.state), reward, done, {}

    def _reset(self):
        try:
            self.state = (float(self.initial_position),0)
        except ValueError:
            if self.initial_position == 'random':
                self.state = np.array([self.np_random.uniform(low=-0.6, high=-0.4), 0])
        if self.bins is not None:
            self.state = (self.bins[0][np.array(abs(self.bins[0] - self.state[0])).argmin()],
                          self.bins[1][np.array(abs(self.bins[1] - self.state[1])).argmin()])
        return np.array(self.state)

    def _height(self, xs):
        return np.sin(3 * xs) * .45 + .55

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        screen_width = 600
        screen_height = 400

        world_width = self.max_position - self.min_position
        scale = screen_width / world_width
        carwidth = 40
        carheight = 20

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            xs = np.linspace(self.min_position, self.max_position, 100)
            ys = self._height(xs)
            xys = list(zip((xs - self.min_position) * scale, ys * scale))

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
            flagx = (self.goal_position - self.min_position) * scale
            flagy1 = self._height(self.goal_position) * scale
            flagy2 = flagy1 + 50
            flagpole = rendering.Line((flagx, flagy1), (flagx, flagy2))
            self.viewer.add_geom(flagpole)
            flag = rendering.FilledPolygon([(flagx, flagy2), (flagx, flagy2 - 10), (flagx + 25, flagy2 - 5)])
            flag.set_color(.8, .8, 0)
            self.viewer.add_geom(flag)

        pos = self.state[0]
        self.cartrans.set_translation((pos - self.min_position) * scale, self._height(pos) * scale)
        self.cartrans.set_rotation(math.cos(3 * pos))

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')
