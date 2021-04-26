from sklearn.neural_network import MLPRegressor
from abc import ABC, abstractmethod
from skopt import gp_minimize
import sklearn.metrics as metrics
import random
import numpy as np

class ModelLearner(ABC):
    @abstractmethod
    def observe(self, X, y, drifted):
        pass

    @abstractmethod
    def predict(self,X):
        pass


class NeuralModelLearner(ModelLearner):
    def __init__(self, rate):
        self.learning_rate=rate
        self.network = model_learner = MLPRegressor(random_state=1, learning_rate='constant', solver='sgd', learning_rate_init=rate, max_iter=500)
        self.to_adapt = False

    def observe(self, X, y, drifted):
        if drifted != False:
            self.to_adapt = True
            print("Time to adapt neural")
            # self.network = model_learner = MLPRegressor(random_state=1, learning_rate='constant', solver='sgd', learning_rate_init=self.learning_rate, max_iter=500)
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
        self.to_adapt = False

    def observe(self, X, y, drifted):
        if drifted != False:
            self.to_adapt = True
            print("Time to adapt analytic")
        self.memory.append((X,y))

    def predict(self, X):
        if self.to_adapt:
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


