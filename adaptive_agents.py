from sklearn.neural_network import MLPRegressor
from abc import ABC, abstractmethod
from skopt import gp_minimize
from collections import deque
import sklearn.metrics as metrics
import random
import numpy as np
from sklearn.exceptions import NotFittedError
import copy
from pendulum import PendulumDynamics



class ModelLearner(ABC):
    @abstractmethod
    def observe(self, X, y, drifted):
        pass

    @abstractmethod
    def predict(self,X):
        pass

class PartialModelLearner(ModelLearner):
    def __init__(self, error_thresh, adaptation_strategy, memory_size, model_type, **kwargs):
        self.adaptation_strategy = adaptation_strategy
        self.fork = False
        self.memory = deque(maxlen=memory_size)
        self.error_thresh = error_thresh
        self.create_model = self.register(model_type, **kwargs)
        self.model_ensemble = [self.create_model()]
        self.model_errors = [0]
        self.active_idx = 0
        self.min_error = 0
        self.adaptation_lockout = 0

    def register(self, model_type, **kwargs):
        def create_model():
            return model_type(**kwargs)

        def fork_model():
            try:
                model = copy.deepcopy(self.model_ensemble[self.active_idx])
            except:
                model = model_type(**kwargs)
            return model


        if self.fork:
            return fork_model
        return create_model


    def update_model_errors(self):
        errors = self.model_errors
        ensemble = self.model_ensemble
        memory = self.memory
        D = len(np.squeeze(self.memory[0][1]))
        sensitivity = 0.1

        try:
            for k, (model, curr_err) in enumerate(zip(ensemble, errors)):
                y_pred = np.empty((len(memory),D))
                y_true = np.empty((len(memory),D))
                for i, (X,y) in enumerate(self.memory):
                    y_pred[i] = model.predict(X)
                    y_true[i] = y
                    # error = np.mean([np.abs(y[0][k] - y_pred[k]) for k in range(len(y[0]))])
                    # errors[k] = errors[k] + sensitivity*(error - errors[k])
                errors[k] = metrics.mean_squared_error(y_true, y_pred)
        except NotFittedError:
            self.adaptation_lockout = 0 # proxy for confidence normalization of the predictor
            print("Training new model")

    def status(self):
        data = {"Num models": len(self.model_ensemble)}
        return data


    def observe(self, X, y, drifted):
        self.memory.append((X,y))
        adapting = False
        errors = self.model_errors
        ensemble = self.model_ensemble
        # active = self.active_idx

        self.update_model_errors()
        self.min_error = np.min(errors)


        no_good_model = self.min_error > self.error_thresh
        strategies = {
            "detection": no_good_model and not self.adaptation_lockout,
            "supervision": drifted,
            "blind": False
                     }


        # print(errors)
        need_new_model = strategies[self.adaptation_strategy]

        # if (no_good_model or active_model_is_not_best) and not self.adaptation_lockout:
        #     print((no_good_model, active_model_is_not_best))


        if need_new_model:
           self.model_ensemble.append(self.create_model())
           errors.append(0)
           self.min_error = 0
           self.adaptation_lockout = max(self.adaptation_lockout-1, 0)

        best_model = np.argmin(errors)
        have_better_model = (best_model != self.active_idx)
        adapting =  have_better_model


        if adapting:
            print(f"Switching models to {best_model} from {self.active_idx}")
            self.active_idx = best_model
        ensemble[self.active_idx].observe(X, y, drifted)
        return adapting

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

    def trig_to_theta(self,X):
        cos = X[0][0]
        sin = X[0][1]
        return np.array((np.arctan2(sin,cos), *X[0][2:])).reshape(1,-1)

    def theta_to_trig(self, X):
        theta = X[0]
        return np.array((np.cos(theta), np.sin(theta), *X[1:]))


    def observe(self, X, y, drifted):
        if drifted != False:
            self.to_adapt = True
            print("Time to adapt analytic")
        self.memory.append((self.trig_to_theta(X), self.trig_to_theta(y)))

    def predict(self, X):
        if self.to_adapt:
            self.dyn = self.optimize_model()
        x = np.array((X[0,0], X[0,1]))
        u = X[0,2]
        xnew_predicted = self.dyn(x,u)
        # xnew_predicted = np.array((xnew_predicted[0, 0], xnew_predicted[0, 1]))
        return self.theta_to_trig(xnew_predicted)

    def status(self):
        data = {
            "m":self.dyn.m,
            "g":self.dyn.g,
            "l":self.dyn.l,
            "b": self.dyn.b}
        return data



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








