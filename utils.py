import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from model import Model
from matplotlib import pyplot as plt


def plot_loss_histories(loss_histories_, idx=0):
    fig_, axs_ = plt.subplots(1, len(loss_histories_), figsize=(15, 5))
    for i_, (dataset, history_list) in enumerate(loss_histories_.items()):
        history = history_list[idx]
        axs_[i_].plot(history, label='Logistic Loss')
        axs_[i_].set_title(f"Loss History, dataset: {dataset}")
        axs_[i_].set_xlabel("Iteration")
        axs_[i_].set_ylabel("Loss")
        axs_[i_].legend()
    plt.show()


def plot_loss_histories_one_plot_log(loss_histories_, idx=0):
    plt.figure(figsize=(5, 4))
    for method, history_list in loss_histories_.items():
        history = history_list[idx]
        plt.plot(np.log(history), label=f'Method: {method}')
    plt.title("Loss Histories")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


def plot_loss_histories_one_plot(loss_histories_, idx=0):
    plt.figure(figsize=(5, 4))
    for method, history_list in loss_histories_.items():
        history = history_list[idx]
        plt.plot(history, label=f'Method: {method}')
    plt.title("Loss Histories")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


def run_experiment(X_, y_, iter_=10, methods_=['L-BFGS-B', 'COBYLA', 'SLSQP'], tol = None, max_iter=1000):
    accuracies_ = {m: [] for m in methods_}
    loss_histories_ = {m: [] for m in methods_}
    optimal_parameters_ = {m: [] for m in methods_}
    X_train_, X_test_, y_train_, y_test_ = train_test_split(X_, y_, test_size=0.2)
    for i in range(iter_):
        for m in methods_:
            model_ = Model(method=m, tol=tol, max_iter=max_iter)
            model_.fit(X_train_, y_train_)
            loss_histories_[m].append(model_.get_loss_history())
            y_pred_ = model_.predict(X_test_)
            accuracies_[m].append(accuracy_score(y_test_, y_pred_))
            optimal_parameters_[m] = model_.get_optimal_parameters()
    return accuracies_, loss_histories_, optimal_parameters_

def run_experiment_and_collect_weights(X_, y_, tol = None, max_iter=1000, C = [500,400,300,200,100,50,25,10,5,2,1,0.5,0.25,0]):
    accuracies_ = {m: [] for m in C}
    loss_histories_ = {m: [] for m in C}
    optimal_parameters_ = {m: [] for m in C}
    X_train_, X_test_, y_train_, y_test_ = train_test_split(X_, y_, test_size=0.2, random_state=0)

    for c in C:
        model_ = Model(method='L-BFGS-B', tol=tol, max_iter=max_iter, C=c)
        model_.fit(X_train_, y_train_)
        loss_histories_[c].append(model_.get_loss_history())
        y_pred_ = model_.predict(X_test_)
        accuracies_[c].append(accuracy_score(y_test_, y_pred_))
        optimal_parameters_[c].append(model_.get_optimal_parameters())

    return accuracies_, loss_histories_, optimal_parameters_





