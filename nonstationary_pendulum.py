import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path


class NonstationaryPendulumEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, drift_speed, drift_type, schedule, dt=0.05, m=1, l=1, b=0.5, g=10):
        self.max_speed = 8
        self.max_torque = 2.
        self.dt = dt
        self.g = g
        self.m = m
        self.l = l
        self.b =b
        self.drifted = None
        self.viewer = None
        self.steps_executed = 0

        high = np.array([1., 1., self.max_speed], dtype=np.float32)
        self.action_space = spaces.Box(
            low=-self.max_torque,
            high=self.max_torque, shape=(1,),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-high,
            high=high,
            dtype=np.float32
        )

        self.seed()

        if drift_type == "constant":
            self.drift_type = self.__class__.constant()
        if drift_type == "decaying":
            self.drift_type = self.__class__.asymptotic()
        if drift_type == "oscillating":
            self.drift_type = self.__class__.oscillating()

        if schedule == "unsupervised":
            self.drift_check = self._unsupervised_check
        if schedule == "supervised":
            self.drift_check = self._pseudo_supervised_check


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, u):
        self.steps_executed += 1
        th, thdot = self.state  # th := theta

        g = self.g
        m = self.m
        l = self.l
        b = self.b
        dt = self.dt


        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        self.last_u = u  # for rendering
        costs = angle_normalize(th) ** 2 + .1 * thdot ** 2 + .001 * (u ** 2)

        alpha = -b*thdot - 3*g/(2*l)*np.sin(th + np.pi) + 3./(m*l ** 2)*u
        newthdot = thdot + alpha * dt
        newth = th + newthdot * dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)

        self.state = np.array([newth, newthdot])
        return self._get_obs(), -costs, False, {}

    def reset(self):
        self.steps_executed=0
        high = np.array([np.pi, 1])
        self.state = self.np_random.uniform(low=-high, high=high)
        self.last_u = None
        return self._get_obs()

    def _get_obs(self):
        theta, thetadot = self.state
        return np.array([np.cos(theta), np.sin(theta), thetadot])

    def update(self):
        self.b = next(self.schedule)

    def asymptotic(init=1.5):
        max_b = 2
        value = init
        assert(value<max_b)
        while True:
            value = 0.1*max_b+0.9*value
            yield value

    def oscillating(values=[0, 1.5]):
        while True:
            for value in values:
                yield value

    def constant(value=0.5):
        while True:
            yield value

    def _unsupervised_check(self):
        return self.steps_executed*self.drift_speed >= 1

    def _pseudo_supervised_check(self):
        drift_trigger = self.steps_executed*self.drift_speed >= 1
        self.drifted = drift_trigger
        return drift_trigger



    def render(self, mode='human'):
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-2.2, 2.2, -2.2, 2.2)
            rod = rendering.make_capsule(1, .2)
            rod.set_color(.8, .3, .3)
            self.pole_transform = rendering.Transform()
            rod.add_attr(self.pole_transform)
            self.viewer.add_geom(rod)
            axle = rendering.make_circle(.05)
            axle.set_color(0, 0, 0)
            self.viewer.add_geom(axle)
            fname = path.join(path.dirname(__file__), "assets/clockwise.png")
            self.img = rendering.Image(fname, 1., 1.)
            self.imgtrans = rendering.Transform()
            self.img.add_attr(self.imgtrans)

        self.viewer.add_onetime(self.img)
        self.pole_transform.set_rotation(self.state[0] + np.pi / 2)
        if self.last_u:
            self.imgtrans.scale = (-self.last_u / 2, np.abs(self.last_u) / 2)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)
