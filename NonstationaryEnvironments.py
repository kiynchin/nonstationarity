import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path
import random
from multiprocessing import Process, Queue
import time


def input_loop(channel):
    channel.put(False)
    while True:
        time.sleep(0.01)
        try:
            to_drift = bool(input("Enter text to trigger drift:"))
        except EOFError:
            to_drift = False
        if not channel.empty():
            channel.get_nowait()
        else:
            channel.put(to_drift)


class DriftScheduler:
    def __init__(self, dyn, state, drift_speed, drift_type, schedule):
        self.steps_executed = 0
        self.drift_speed = drift_speed
        self.state = state
        self.dyn = dyn
        self.drifted = None

        # self.user_channel = Queue()
        # user_process = Process(target=input_loop, args=(self.user_channel,))


        if drift_type == "constant":
            self.drift_type = dyn.__class__.constant()
        if drift_type == "decaying":
            self.drift_type = dyn.__class__.asymptotic()
        if drift_type == "oscillating":
            self.drift_type = dyn.__class__.oscillating()

        if schedule == "unsupervised":
            self.drift_check = self._unsupervised_check
        if schedule == "supervised":
            self.drift_check = self._pseudo_supervised_check
        # user_process.start()

    def step(self, u):
        self.steps_executed = self.steps_executed+1
        next_state = self.dyn(self.state, u)
        self.state = next_state
        if self.drift_check():
            self.update()
            self.steps_executed = 0
        return self._get_obs()

    def _get_obs(self):
        return self.state, self.drifted

    def _unsupervised_check(self):
        return self.steps_executed*self.drift_speed >= 1

    def _pseudo_supervised_check(self):
        drift_trigger = self.steps_executed*self.drift_speed >= 1
        self.drifted = drift_trigger
        return drift_trigger

    def _supervised_check(self):
        """
        Designed for user input, Do not use without tuning timing to system
        """
        drift_trigger = self.user_channel.get()
        self.drifted = drift_trigger
        return drift_trigger


    def update(self):
        self.dyn.update(self.drift_type)



