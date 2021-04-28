from sklearn.neural_network import MLPRegressor
from abc import ABC, abstractmethod
from skopt import gp_minimize
import sklearn.metrics as metrics
import random
import numpy as np
from sklearn.exceptions import NotFittedError


class ModelLearner(ABC):
    @abstractmethod
    def observe(self, X, y, drifted):
        pass

    @abstractmethod
    def predict(self,X):
        pass

class PartialModelLearner(ModelLearner):
    def __init__(self, error_thresh, model_type, **kwargs):
        self.error_thresh = error_thresh
        self.create_model = self.register(model_type, **kwargs)
        self.model_ensemble = [self.create_model()]
        self.model_errors = [0]
        self.active_idx = 0
        self.min_error = 0

    def register(self, model_type, **kwargs):
        def create_model():
            print("TIME FOR A NEW MODEL")
            return model_type(**kwargs)
        return create_model



    def update_model_errors(self, X, y):
        errors = self.model_errors
        ensemble = self.model_ensemble
        sensitivity = 0.5

        for i, (model, curr_err) in enumerate(zip(ensemble, errors)):
            try:
                y_pred = model.predict(X)
                error = np.mean([(y[0][i] - y_pred[i]) ** 2 for i in range(len(y[0]))])
                print((error, i))
                errors[i] = errors[i] + sensitivity*(error - errors[i])
            except NotFittedError:
                print("Training new model")


    def observe(self, X, y, drifted):
        adapted = False
        errors = self.model_errors
        ensemble = self.model_ensemble
        active = self.active_idx

        self.update_model_errors(X, y)
        self.min_error = np.min(errors)


        adapted = self.min_error > self.error_thresh
        if adapted:
           ensemble.append(self.create_model())
           errors.append(0)
           self.min_error = 0

        active = np.argmin(errors)
        ensemble[active].observe(X, y, drifted)
        print(active)
        return adapted

    def predict(self, X):
        return self.model_ensemble[self.active_idx].predict(X)




class NeuralModelLearner(ModelLearner):
    def __init__(self, rate):
        self.learning_rate=rate
        self.network = MLPRegressor(random_state=1, learning_rate='constant', solver='sgd', learning_rate_init=rate, max_iter=500)
        self.to_adapt = False

    def observe(self, X, y, drifted):
        if drifted != False:
            self.to_adapt = True
        self.network.partial_fit(X,y)

    def predict(self, X):
        xnew_predicted = self.network.predict(X)[0]
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








