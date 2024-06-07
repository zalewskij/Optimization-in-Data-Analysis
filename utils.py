import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo

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


def plot_loss_histories_one_plot_log(loss_histories_, dataset, idx=0):
    plt.figure(figsize=(5, 4))
    for method, history_list in loss_histories_.items():
        history = history_list[idx]
        plt.plot(np.log(history), label=f'Method: {method}')
    plt.title(f"Log loss histories - {dataset}")
    plt.xlabel("Iteration")
    plt.ylabel("Log loss")
    plt.legend()
    plt.show()


def plot_loss_histories_one_plot(loss_histories_, idx=0):
    plt.figure(figsize=(5, 4))
    for method, history_list in loss_histories_.items():
        history = history_list[idx]
        plt.plot(history, label=f'Method: {method}')
    plt.title("Loss histories")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


def run_experiment(X_, y_, iter_=10, methods_=['L-BFGS-B', 'SLSQP'], options_={'L-BFGS-B': None, 'SLSQP': None}):
    accuracies_ = {m: [] for m in methods_}
    loss_histories_ = {m: [] for m in methods_}
    optimal_parameters_ = {m: [] for m in methods_}
    X_train_, X_test_, y_train_, y_test_ = train_test_split(X_, y_, test_size=0.2)
    for i in range(iter_):
        for m in methods_:
            model_ = Model(method=m, options=options_.get(m, None))
            model_.fit(X_train_, y_train_)
            loss_histories_[m].append(model_.get_loss_history())
            y_pred_ = model_.predict(X_test_)
            accuracies_[m].append(accuracy_score(y_test_, y_pred_))
            optimal_parameters_[m] = model_.get_optimal_parameters()
    return accuracies_, loss_histories_, optimal_parameters_


def plot_accuracies(all_accuracies):
    fig, axs = plt.subplots(1, 3, figsize=(20, 6))

    for i, (dataset_name, accuracies) in enumerate(all_accuracies.items()):
        dataset_names = []
        accuracies_data = []

        for method, acc_list in accuracies.items():
            dataset_names.append(method)
            accuracies_data.append(acc_list)

        axs[i].boxplot(accuracies_data, labels=dataset_names)
        axs[i].set_ylim([0.85, 1.0])
        axs[i].set_title(dataset_name)
        axs[i].set_ylabel('Accuracy')
        axs[i].grid(True)

    plt.tight_layout()
    plt.show()


def get_dataset(id, l1, l2):
    data = fetch_ucirepo(id=id)
    X = data.data.features.to_numpy()
    y = data.data.targets.to_numpy().ravel()

    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    label_mapping = {l1: -1, l2: 1}
    y = np.vectorize(label_mapping.get)(y)
    return X, y


def test_and_plot_parameters(X, y, params_, iter_=10):
    fig, axes = plt.subplots(3, len(params_), figsize=(10, 9), sharey='row')
    fig.tight_layout(pad=5.0, h_pad=3.0, w_pad=3.0)

    for i, (method, param) in enumerate(params_.items()):
        for j, (param_name, values) in enumerate(param.items()):
            avg_min_losses, values = test_parameter(X, y, params_[method], method, param_name, iter_=iter_)
            plot_parameter_results(avg_min_losses, values, method, param_name, axes[j, i])
    plt.show()


def test_parameter(X_, y_, params_, method_, param_, iter_=10):
    values = []
    avg_losses = []
    for value in params_[param_]:
        options = {method_: {param_: value}}
        min_losses = []
        for _ in range(iter_):
            _, _, optimal_params_ = run_experiment(X_, y_, iter_=1, methods_=[method_], options_=options)
            min_losses.append(optimal_params_[method_]['Minimum loss'])

        avg_losses.append(np.mean(min_losses))
        values.append(value)
    return avg_losses, values


def plot_parameter_results(avg_losses_, values_, method_, param_, ax):
    ax.plot(values_, avg_losses_, marker='o')
    ax.set_title(f'{method_} - {param_}')
    ax.set_xlabel(param_)
    ax.set_ylabel('Average minimum loss')
    ax.grid(True)


def run_experiment_and_collect_weights(X_, y_, tol = None, max_iter=1000, C = [500,400,300,200,100,50,25,10,5,2,1,0.5,0.25,0]):
    accuracies_ = {m: [] for m in C}
    loss_histories_ = {m: [] for m in C}
    optimal_parameters_ = {m: [] for m in C}
    X_train_, X_test_, y_train_, y_test_ = train_test_split(X_, y_, test_size=0.2, random_state=0)

    for c in C:
        model_ = Model(method='L-BFGS-B', C=c)
        model_.fit(X_train_, y_train_)
        loss_histories_[c].append(model_.get_loss_history())
        y_pred_ = model_.predict(X_test_)
        accuracies_[c].append(accuracy_score(y_test_, y_pred_))
        optimal_parameters_[c].append(model_.get_optimal_parameters())

    return accuracies_, loss_histories_, optimal_parameters_





