import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.neural_network import MLPRegressor

PendulumPhase = np.dtype([('theta', 'f8'), ('theta_dot', 'f8')])


class Dynamics:
    def __init__(self, m, g, l, b, dt):
        self.m = m
        self.g = g
        self.l = l
        self.b = b
        self.dt = dt
        self.schedule = Dynamics.oscillating([0.25, 0.75])
        self.schedule = Dynamics.asymptotic(0)

    def __call__(self, x:PendulumPhase, u):
        m = self.m
        g = self.g
        l = self.l
        b = self.b
        alpha = u/(m*l**2) - (g/l)*np.sin(x['theta'])-b*x['theta_dot']
        theta_new = x['theta']+x['theta_dot']*self.dt
        theta_dot_new = x['theta_dot'] + alpha*self.dt
        return np.array((theta_new, theta_dot_new), dtype=PendulumPhase)

    def asymptotic(init):
        max_b = 2
        value = init
        assert(value<max_b)
        while True:
            value = 0.1*max_b+0.9*value
            yield value

    def oscillating(values):
        while True:
            for value in values:
                yield value

    def update(self):
        self.b = next(self.schedule)



def plot_comparison(field: str, axs):
    ax0 = axs[0]
    ax0.plot(np.linspace(0, T, N), traj[field], c='k')
    ax0.scatter(np.linspace(0, T, N), predicted_traj[field], c='b')
    for i in range(1, num_dynamics_epochs):
        ax0.axvline(T*i/num_dynamics_epochs)

    ax0.set_xlabel("Time (s)")
    ax0.set_ylabel(field)
    ax0.legend(["Ground Truth", "Model Prediction"])
    ax0.set_title(f"Tracking performance of {field}")

    ax1 = axs[1]
    ax1.plot(np.linspace(0, T, N), np.abs(error_traj[field]), c='r')
    for i in range(1, num_dynamics_epochs):
        ax1.axvline(T*i/num_dynamics_epochs)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Prediction % Error")
    ax1.set_title(f"Error in {field} vs. Dynamics Changes")
    ax1.legend(["Error", "Dynamics Shift"])
    ax1.set_ylim((0,100))



class ModelLearner:
    def __init__(self, x0, x1):
        pass


if __name__ == "__main__":
    dt = 0.01
    dyn = Dynamics(m=5, g=9.81, l=2, b=0.5, dt=dt)
    T = 15
    rate = 0.001


    N = int(T/dt)

    u_scale = 0
    u_traj = [u_scale*random.random()-0.5 for i in range(N)]
    u = u_traj[0]
    theta_0 = np.pi/2
    theta_dot_0 = 0
    x0 = np.array((theta_0, theta_dot_0), dtype=PendulumPhase)
    traj = np.empty((N,), dtype=PendulumPhase)
    traj[0] = x0
    x1 = dyn(x0, u_traj[0])
    traj[1] = x1
    model_learner = MLPRegressor(random_state=1, learning_rate='constant', solver='sgd', learning_rate_init=rate, max_iter=500)
    predicted_traj = np.empty((N,), dtype=PendulumPhase)
    predicted_traj[0] = None
    predicted_traj[1] = None
    model_learner.partial_fit(X=np.array([x0['theta'], x0['theta_dot'], u]).reshape(1, -1),
                              y=np.array([x1['theta'], x1['theta_dot']]).reshape(1, -1))

    error_traj = np.empty((N,), dtype=PendulumPhase)
    error_traj[0], error_traj[1] = (None, None)
    num_dynamics_epochs = 10
    num_control_epochs = num_dynamics_epochs*10

    control_epoch_length = int(N/num_control_epochs)
    dynamics_epoch_length = int(N/num_dynamics_epochs)
    x = x1

    for i in range(2, N):
        if i % control_epoch_length == 0:
            u = u_traj[i]

        if i % dynamics_epoch_length == 0:
            dyn.update()

        xnew_actual = dyn(x, u)
        traj[i] = xnew_actual
        xnew_predicted = model_learner.predict(np.array([x['theta'], x['theta_dot'], u]).reshape(1, -1))
        xnew_predicted = np.array((xnew_predicted[0, 0], xnew_predicted[0, 1]), dtype=PendulumPhase)
        predicted_traj[i] = xnew_predicted
        error = ((xnew_predicted['theta']-xnew_actual['theta'])/xnew_predicted['theta'],
                 (xnew_predicted['theta_dot']-xnew_actual['theta_dot'])/xnew_predicted['theta_dot'])
        error_traj[i] = np.array(error, dtype=PendulumPhase)
        model_learner.partial_fit(X=np.array([x['theta'], x['theta_dot'], u]).reshape(1, -1),
                                   y=np.array([xnew_actual['theta'], xnew_actual['theta_dot']]).reshape(1, -1))
        x = xnew_actual

    fig, axs = plt.subplots(2,2)
    plot_comparison('theta', [axs[0,0], axs[0,1]])
    plot_comparison('theta_dot', [axs[1,0], axs[1,1]])
    plt.show()


