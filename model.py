import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt


class Model:
    def __init__(self, C: int = 100, method: str = 'L-BFGS-B', options: dict = None):
        """
        :param C:  Weight of the logistic loss function
        """
        self.optimal_b = None
        self.optimal_w = None
        self.optimal_p = None
        self.optimal_n = None
        self.min_loss = None
        self.options = options
        self.C = C
        self.loss_history = []
        self.method = method

    def _logistic_loss(self, params, X, y):
        n_samples, n_features = X.shape
        n = params[:n_features]
        p = params[n_features:2 * n_features]
        b = params[-1]

        loss = self.C * np.mean(np.log(1 + np.exp(-y * (np.dot(X, n - p) + b)))) + np.sum(n + p)
        self.loss_history.append(loss)
        return loss

    def fit(self, X, y, verbose=False):
        self.loss_history.clear()
        n_samples, n_features = X.shape

        # model parameters
        bounds = [(0, None)] * n_features + [(0, None)] * n_features + [(None, None)]
        n = np.random.normal(0, 1, n_features)
        p = np.random.normal(0, 1, n_features)
        initial_b = np.random.normal(0, 1, 1)
        initial_params = np.append(np.append(n, p), initial_b)
        if self.options is None:
            result = minimize(self._logistic_loss, initial_params, args=(X, y), method=self.method, bounds=bounds)
        else:
            result = minimize(self._logistic_loss, initial_params, args=(X, y), method=self.method, bounds=bounds, options=self.options)

        optimal_params = result.x
        self.optimal_n = optimal_params[:n_features]
        self.optimal_p = optimal_params[n_features:2 * n_features]
        self.optimal_w = self.optimal_n - self.optimal_p
        self.optimal_b = optimal_params[-1]
        self.min_loss = result.fun

        if verbose:
            print(f"Optimal n:\n {self.optimal_n}")
            print("-" * 75)
            print(f"Optimal p:\n {self.optimal_p}")
            print("-" * 75)
            print(f"Optimal w:\n {self.optimal_w}")
            print("-" * 75)
            print(f"Optimal bias:\n {self.optimal_b}")
            print("-" * 75)
            print(f"Minimum logistic loss:\n {result.fun}")

    def predict_proba(self, X):
        linear_model = np.dot(X, self.optimal_w) + self.optimal_b
        proba = 1 / (1 + np.exp(-linear_model))
        return proba

    def predict(self, X):
        proba = self.predict_proba(X)
        return np.where(proba >= 0.5, 1, -1)

    def get_optimal_parameters(self):
        params = {
            'Optimal n': self.optimal_n,
            'Optimal p': self.optimal_p,
            'Optimal w': self.optimal_w,
            'Optimal b': self.optimal_b,
            'Minimum loss': self.min_loss
        }
        return params

    def get_loss_history(self):
        return self.loss_history

    def plot_loss_history(self, w=10, h=6):
        plt.figure(figsize=(w, h))
        plt.plot(self.loss_history, label='Logistic Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Logistic Loss')
        plt.title(f'Learning Curve, method: {self.method}')
        plt.legend()
        plt.grid(True)
        plt.show()
