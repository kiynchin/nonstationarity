import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.neural_network import MLPRegressor
import sys
from skopt import gp_minimize
from abc import ABC, abstractmethod
from collections import deque
import sklearn.metrics as metrics
import pdb
import time
import argparse
from NonstationaryEnvironments import test, DriftScheduler




def plot_comparison(field, fmap,  axs):
    ax0 = axs[0]
    ax0.plot(np.linspace(0, T, N), traj[:,fmap[field]], c='k')
    ax0.scatter(np.linspace(0, T, N), predicted_traj[:,fmap[field]], c='b')
    for i in range(1, num_dynamics_epochs):
        ax0.axvline(T*i/num_dynamics_epochs)

    ax0.set_xlabel("Time (s)")
    ax0.set_ylabel(field)
    ax0.legend(["Ground Truth", "Model Prediction"])
    ax0.set_title(f"Tracking performance of {field}")

    ax1 = axs[1]
    ax1.plot(np.linspace(0, T, N), np.abs(error_traj[:,fmap[field]]), c='r')
    for i in range(1, num_dynamics_epochs):
        ax1.axvline(T*i/num_dynamics_epochs)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Prediction Squared Error")
    ax1.set_title(f"Error in {field} vs. Dynamics Changes")
    ax1.legend(["Error", "Dynamics Shift"])


class ModelLearner(ABC):
    @abstractmethod
    def observe(self, X, y):
        pass

    @abstractmethod
    def predict(self,X):
        pass

class PendulumDynamics:
    def __init__(self, m, g, l, b, dt):
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



class NeuralModelLearner(ModelLearner):
    def __init__(self, rate):
        self.network = model_learner = MLPRegressor(random_state=1, learning_rate='constant', solver='sgd', learning_rate_init=rate, max_iter=500)

    def observe(self, X, y):
        self.network.partial_fit(X,y)

    def predict(self, X):
        xnew_predicted = self.network.predict(X)
        xnew_predicted = np.array((xnew_predicted[0, 0], xnew_predicted[0, 1]))
        return xnew_predicted


class AnalyticModelLearner(ModelLearner):
    def __init__(self, dt, memory_size):
        self.memory = deque(maxlen=memory_size)
        self.dt =dt
        self.dyn = PendulumDynamics(m=1, g=9.81, l=1, b=0, dt=dt)

    def observe(self, X, y):
        self.memory.append((X,y))

    def predict(self, X):
        self.dyn = self.optimize_model()
        x = np.array((X[0,0], X[0,1]))
        u = X[0,2]
        xnew_predicted = self.dyn(x,u)
        # xnew_predicted = np.array((xnew_predicted[0, 0], xnew_predicted[0, 1]))
        return xnew_predicted


    def optimize_model(self):
        memory = self.memory
        dt = self.dt
        def prediction_loss(dyn_params):
            m,g,l,b = dyn_params
            dyn = PendulumDynamics(m,g,l,b,dt)
            y_pred = np.empty((len(memory),2))
            y_true = np.empty((len(memory),2))
            for i, datum in enumerate(memory):
                true = datum[1][0]
                x = np.array(datum[0][0,:2])
                u = datum[0][0,2]
                pred = dyn(x,u)

                y_true[i] = true
                y_pred[i] = pred

            return metrics.mean_squared_error(y_true, y_pred)

        bounds = [(0.1, 10.0), (0.0, 20.0), (0.1, 10.0), (0.0, 2.0)]
        res = gp_minimize(prediction_loss, dimensions=bounds, n_calls=10, n_restarts_optimizer=1)
        m, g, l, b = res.x
        return PendulumDynamics(m,g,l,b,dt)



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



def loss(xnew_actual, xnew_predicted):
    # error = (xnew_predicted[0]-xnew_actual[0])/xnew_predicted[0],
                 # (xnew_predicted[1]-xnew_actual[1])/xnew_predicted[1]
    error = np.array(((xnew_predicted[0]-xnew_actual[0])**2,(xnew_predicted[1]-xnew_actual[1])**2))
    return error


def setup_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("rate", help="learning rate for neural learner", type=float)
    parser.add_argument("drift_type", choices=["constant", "decaying", "oscillating"], help="type of dynamics drift")
    parser.add_argument("torque_range", help="total range of pendulum torques, 0-centered", type=float)
    parser.add_argument("-num_dynamics_epochs", help="number of distinct dynamics", type=int, default=50)
    parser.add_argument("-num_control_epochs", help="number of distinct dynamics", type=int, default=500)
    parser.add_argument("-learner", help="which type of learner", choices=["neural", "analytic"], default="neural")
    parser.add_argument("--save", help="whether to display save prompt at end", action="store_true")
    parser.add_argument("-T", help="duration of experiment", type=float, default=15)
    parser.add_argument("-dt", help="time step of simulation", type=float, default=0.01)
    return parser




if __name__ == "__main__":
    args = setup_parser().parse_args()
    rate = args.rate
    num_dynamics_epochs = args.num_dynamics_epochs
    drift_type = args.drift_type
    torque_range = args.torque_range
    T = args.T
    dt = args.dt
    num_control_epochs = args.num_control_epochs
    N = int(T/dt)
    control_epoch_length = int(N/num_control_epochs)
    dynamics_epoch_length = int(N/num_dynamics_epochs)

    dyn = DriftScheduler(PendulumDynamics(m=5, g=9.81, l=2, b=0.5, dt=dt), schedule=drift_type)
    policy = Controller(u_scale = torque_range, policy=Controller.random_policy)

    t0 = time.time()
    if args.learner=="analytic":
        model_learner = AnalyticModelLearner(dt=dt, memory_size=10)
    if args.learner=="neural":
        model_learner = NeuralModelLearner(rate=rate)

    x0 = np.array((np.pi/2, 0))
    u0 = policy(x0)


    traj = np.empty((N,2))
    traj[0] = x0
    x1 = dyn(x0, u0)
    traj[1] = x1
    predicted_traj = np.empty((N,2))
    predicted_traj[0] = None
    predicted_traj[1] = None
    error_traj = np.empty((N,2))
    error_traj[0], error_traj[1] = (None, None)

    model_learner.observe(X=np.array([x0[0], x0[1], u0]).reshape(1, -1),
                              y=np.array([x1[0], x1[1]]).reshape(1, -1))


    x = x1
    u = u0
    for i in range(2, N):
        if i % control_epoch_length == 0:
            u = policy(x)

        if i % dynamics_epoch_length == 0:
            dyn.update()

        xnew_actual = dyn(x, u)
        traj[i] = xnew_actual
        xnew_predicted = model_learner.predict(np.array([x[0], x[1], u]).reshape(1, -1))
        predicted_traj[i] = xnew_predicted
        error = loss(xnew_actual, xnew_predicted)
        error_traj[i] = error
        model_learner.observe(X=np.array([x[0], x[1], u]).reshape(1, -1),
                                   y=np.array([xnew_actual[0], xnew_actual[1]]).reshape(1, -1))
        x = xnew_actual

    t1 = time.time()
    print(f"Elapsed time (s): {t1-t0}")
    fig, axs = plt.subplots(2,2)
    fmap ={"theta":0, "theta_dot":1}
    plot_comparison('theta', fmap, [axs[0,0], axs[0,1]])
    plot_comparison('theta_dot', fmap, [axs[1,0], axs[1,1]])
    fig.suptitle(f"Learner: {args.learner}, Dynamics Drift: {drift_type}, Learning rate {rate}, Dynamics Epochs: {num_dynamics_epochs}, Max torque: {torque_range}")
    fig.set_size_inches(12, 12)
    plt.show()
    if args.save:
        save = input(f"Save? (Y/n):")

        if save != "n":
            leader = input("Enter leading filename (<Enter> for default):")
            fig.savefig(f"dd{drift_type}lr{rate}de{num_dynamics_epochs}umax{torque_range}.png")
