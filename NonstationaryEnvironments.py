import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path
import random

class DriftScheduler:
    def __init__(self, dyn, state, drift_speed, schedule):
        self.steps_executed = 0
        self.drift_speed = drift_speed
        self.state = state
        self.dyn = dyn
        if schedule == "constant":
            self.schedule = dyn.__class__.constant()
        if schedule == "decaying":
            self.schedule = dyn.__class__.asymptotic()
        if schedule == "oscillating":
            self.schedule = dyn.__class__.oscillating()

    def step(self, u):
        self.steps_executed = self.steps_executed+1
        next_state = self.dyn(self.state, u)
        self.state = next_state
        if self.steps_executed*self.drift_speed >= 1:
            self.update()
            self.steps_executed = 0
        return self._get_obs()

    def _get_obs(self):
        return self.state


    def __call__(self, x, u):
        return self.dyn(x,u)

    def update(self):
        self.dyn.update(self.schedule)

def test():
    print(__name__)


