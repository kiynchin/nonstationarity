import numpy as np
import random

class Controller:
    def __init__(self, u_scale, policy):
        self.u_scale = u_scale
        self.policy = policy(self.u_scale)
        next(self.policy)

    def __call__(self, state):
        self.policy.send(state)
        return next(self.policy)


    def random_policy(u_scale):
        while True:
            state = yield u_scale*(random.random()-0.5)

class PendulumDynamics:
    D = 2
    fmap = ["theta", "theta_dot"]
    assert len(fmap)==2
    def __init__(self, dt, m=5, g=9.81, l=2, b=0.5):
        self.m = m
        self.g = g
        self.l = l
        self.b = b
        self.dt = dt

    def __call__(self, x, u):
        m = self.m
        g = self.g
        l = self.l
        b = self.b
        alpha = u/(m*l**2) - (g/l)*np.sin(x[0])-b*x[1]
        theta_new = x[0]+x[1]*self.dt
        theta_dot_new = x[1] + alpha*self.dt
        return np.array((theta_new, theta_dot_new))

    def reset():
        # high = np.array([np.pi, 1])
        # state = self.np_random.uniform(low=-high, high=high)
        return ((np.pi/2,0)) 

    def get_policy():
        return Controller(u_scale = 1, policy=Controller.random_policy)

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

    def update(self, schedule):
        self.b = next(schedule)
