import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.neural_network import MLPRegressor

PendulumPhase = np.dtype([('theta', 'f8'), ('theta_dot', 'f8')])


class PendulumDynamics:
    def __init__(self, m, g, length, b):
        self.m = m
        self.g = g
        self.length = length
        self.b = b

    def __call__(self, x: PendulumPhase, u):
        m = self.m
        g = self.g
        length = self.length
        b = self.b
        alpha = u/(m*length**2) - (g/length)*np.sin(x['theta'])-b*x['theta_dot']
        theta_new = x['theta']+x['theta_dot']*self.dt
        theta_dot_new = x['theta_dot'] + alpha*self.dt
        return np.array((theta_new, theta_dot_new), dtype=PendulumPhase)

    def update(self):
        self.b = random.random()



class Env:
    def __init__(self, dyn, T=15, dt=0.01, control_epochs=1, dynamics_epochs=10):
        self.dynamics = dyn
        self.dt = dt
        self.N = int(T/dt)
        self.num_control_epochs = control_epochs
        self.num_dynamics_epochs = dynamics_epochs
        self.control_epoch_length = int(self.N/self.num_control_epochs)
        self.dynamics_epoch_length = int(self.N/self.num_dynamics_epochs)

    def run_experiment(self, x):
        for i in range(2, self.N):
            if i % self.control_epoch_length == 0:
                u = u_traj[i]

            if i % self.dynamics_epoch_length == 0:
                self.dyn.update()
                
            xnew_actual = dyn(x, u, env.dt)
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
    
    def plot_comparison(self, field: str):
        plt.figure()
        plt.plot(np.linspace(0, self.T, self.N), self.traj[field], c='k')
        plt.scatter(np.linspace(0, self.T, self.N), self.predicted_traj[field], c='b')

        plt.xlabel("Time (s)")
        plt.ylabel(field)
        plt.legend(["Ground Truth", "Model Prediction"])
        plt.title(f"Tracking performance of {field}")
        plt.show()

        plt.figure()
        plt.plot(np.linspace(0, self.T, self.N), np.abs(error_traj[field]), c='r')
        for i in range(1, self.num_dynamics_epochs):
            plt.axvline(self.T*i/self.num_dynamics_epochs)
        plt.xlabel("Time (s)")
        plt.ylabel("Prediction % Error")
        plt.title(f"Error in {field} vs. Dynamics Changes")
        plt.legend(["Error", "Dynamics Shift"])
        plt.show()



class ModelLearner:
    def __init__(self, x0, x1):
        pass


if __name__ == "__main__":
    dyn = PendulumDynamics(m=5, g=9.81, length=2, b=0.5)
    env = Env(dyn, T=15, dt=0.01)

    u_scale = 5
    u_traj = [u_scale*random.random()-0.5 for i in range(env.N)]
    u = u_traj[0]
    theta_0 = np.pi/2
    theta_dot_0 = 0

    x0 = np.array((theta_0, theta_dot_0), dtype=PendulumPhase)
    traj = np.empty((N,), dtype=PendulumPhase)
    traj[0] = x0
    x1 = dyn(x0, u_traj[0])
    traj[1] = x1
    
    model_learner = MLPRegressor(random_state=1, max_iter=500)
    predicted_traj = np.empty((N,), dtype=PendulumPhase)
    predicted_traj[0] = None
    predicted_traj[1] = None
    model_learner.partial_fit(X=np.array([x0['theta'], x0['theta_dot'], u]).reshape(1, -1),
                              y=np.array([x1['theta'], x1['theta_dot']]).reshape(1, -1))

    error_traj = np.empty((N,), dtype=PendulumPhase)
    error_traj[0], error_traj[1] = (None, None)

    env.run_experiment(x1)
   
    env.plot_comparison('theta')
    env.plot_comparison('theta_dot')
        

