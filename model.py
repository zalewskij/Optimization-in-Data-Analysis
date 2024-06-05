import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt


class Model:
    def __init__(self, C: int = 100):
        """
        :param C:  Weight of the logistic loss function
        """
        self.optimal_b = None
        self.optimal_w = None
        self.optimal_p = None
        self.optimal_n = None
        self.C = C
        self.loss_history = []

    def _logistic_loss(self, params, X, y):
        n_samples, n_features = X.shape
        n = params[:n_features]
        p = params[n_features:2 * n_features]
        b = params[-1]

        loss = self.C * np.mean(np.log(1 + np.exp(-y * (np.dot(X, n-p) + b)))) + np.sum(n + p)
        self.loss_history.append(loss)
        return loss

    def fit(self, X, y, verbose = True):
        self.loss_history.clear()
        n_samples, n_features = X.shape

        # model parameters
        bounds = [(0, None)] * n_features + [(0, None)] * n_features + [(None, None)]
        n = np.random.normal(0, 1, n_features)
        p = np.random.normal(0, 1, n_features)
        initial_b = np.random.normal(0, 1, 1)
        initial_params = np.append(np.append(n, p), initial_b)

        result = minimize(self._logistic_loss, initial_params, args=(X, y), method='L-BFGS-B', bounds=bounds)

        if verbose:
            optimal_params = result.x
            self.optimal_n = optimal_params[:n_features]
            self.optimal_p = optimal_params[n_features:2 * n_features]
            self.optimal_w = self.optimal_n - self.optimal_p
            self.optimal_b = optimal_params[-1]

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

    def plot_loss_history(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.loss_history, label='Logistic Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Logistic Loss')
        plt.title('Learning Curve')
        plt.legend()
        plt.grid(True)
        plt.show()