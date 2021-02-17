import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.neural_network import MLPRegressor

State = np.dtype([('theta', 'f8'), ('theta_dot', 'f8')])


class Dynamics:
    def __init__(self, m, g, l, b, dt):
        self.m = m
        self.g = g
        self.l = l
        self.b = b
        self.dt = dt

    def __call__(self, x:State, u):
        m = self.m
        g = self.g
        l = self.l
        b = self.b
        alpha = u/(m*l**2) - (g/l)*np.sin(x['theta'])-b*x['theta_dot']
        theta_new = x['theta']+x['theta_dot']*self.dt
        theta_dot_new = x['theta_dot'] + alpha*self.dt
        return np.array((theta_new, theta_dot_new), dtype=State)


def plot_comparison(field: str):
    plt.plot(np.linspace(0, T, N+1), traj[field], c='k')
    plt.scatter(np.linspace(2*dt, T+dt, N), neural_traj[field], c='b')
    plt.xlabel("Time (s)")
    plt.ylabel(field)
    plt.legend(["ground_truth", "neural_predictor"])
    plt.title(f"Tracking performance of {field}")
    plt.show()


if __name__ == "__main__":
    dt = 0.01
    dyn = Dynamics(m=5, g=9.81, l=2, b=0.05, dt=dt)
    T = 10
    N = int(T/dt)

    x = np.array((np.pi/2, 0), dtype=State)
    traj = np.empty((N+1,), dtype=State)
    traj[0] = x

    neural_traj = np.empty((N,), dtype=State)
    neural_network = MLPRegressor(random_state=1, max_iter=500)

    u = 0
    u0 = 2
    for i in range(N):
        if i%int(N/N) == 0:
            u = u0*(random.random()-0.5)
        x_new = dyn(x, u)
        traj[i+1] = x_new

        neural_network.partial_fit(np.array([x['theta'], x['theta_dot'], u]).reshape(1, -1),
                                   np.array([x_new['theta'], x_new['theta_dot']]).reshape(1, -1))
        x_neural = neural_network.predict(np.array([x_new['theta'], x_new['theta_dot'], u]).reshape(1, -1))
        x_neural = np.array((x_neural[0, 0], x_neural[0, 1]), dtype=State)
        neural_traj[i] = x_neural
        x = x_new
    
    plot_comparison('theta')
    plot_comparison('theta_dot')
        


