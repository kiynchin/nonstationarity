import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path
import random
from multiprocessing import Process, Queue

def input_loop(channel):
    while True:
        to_drift = bool(input("Enter text to trigger drift:"))
        if not channel.empty():
            channel.get_nowait()
        channel.put(to_drift)


class DriftScheduler:
    def __init__(self, dyn, state, drift_speed, drift_type, schedule):
        self.steps_executed = 0
        self.drift_speed = drift_speed
        self.state = state
        self.dyn = dyn

        self.user_channel = Queue()
        user_process = Process(target=input_loop, args=(self.user_channel,))

        if drift_type == "constant":
            self.drift_type = dyn.__class__.constant()
        if drift_type == "decaying":
            self.drift_type = dyn.__class__.asymptotic()
        if drift_type == "oscillating":
            self.drift_type = dyn.__class__.oscillating()

        if schedule == "intrinsic":
            self.drift_check = self._unsupervised_check
        if schedule == "extrinsic":
            self.drift_check = self._supervised_check

    def step(self, u):
        self.steps_executed = self.steps_executed+1
        next_state = self.dyn(self.state, u)
        self.state = next_state
        if self.drift_check():
            self.update()
            self.steps_executed = 0
        return self._get_obs()

    def _get_obs(self):
        return self.state

    def _unsupervised_check(self):
        return self.steps_executed*self.drift_speed >= 1

    def _supervised_check(self):
        drift_trigger = self.user_channel.get()
        return drift_trigger

        



    def __call__(self, x, u):
        return self.dyn(x,u)

    def update(self):
        self.dyn.update(self.drift_type)

def test():
    print(__name__)


