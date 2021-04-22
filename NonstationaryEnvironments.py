import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path
import random

class DriftScheduler:
    def __init__(self, dyn, schedule):
        self.dyn = dyn
        if schedule == "constant":
            self.schedule = dyn.__class__.constant()
        if schedule == "decaying":
            self.schedule = dyn.__class__.asymptotic()
        if schedule == "oscillating":
            self.schedule = dyn.__class__.oscillating()

    def __call__(self, x, u):
        return self.dyn(x,u)

    def update(self):
        self.dyn.update(self.schedule)

def test():
    print(__name__)


