import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

def f(x):
    return .5 * x + np.sqrt(np.max(x, 0)) - np.cos(x) + 2

def f_hat(x, w):
    d = len(w) - 1
    return np.sum(w * np.power(x, np.expand_dims(np.arange(d, -1, -1), 1)).T, 1)

N = 1000
n = int(.02 * N)
R = 1000
d_arr = [1, 2, 3, 5]
colors = np.array(['tab:green', 'tab:purple', 'tab:cyan', 'tab:orange'])
sigma_epsilon = 1
x_max = 3
x_range = np.linspace(-x_max, x_max, 1000)
x_test = 3.2
x = x_max * (2 * np.random.rand(N) - 1)
epsilon = sigma_epsilon * np.random.randn(N)

y = f(x) + epsilon
y_test = f(x_test) + sigma_epsilon * np.random.randn()

def plot_sample():
    plt.figure(figsize=(12, 6))
    plt.scatter(x, y)
    plt.plot(x_range, f(x_range), 'r', linewidth=3.0)
    plt.scatter(x_test, y_test, c='r')
    plt.xlabel('x', size=12)
    plt.ylabel('y', size=12)
    plt.xticks(np.arange(-x_max, x_max + 1))
    plt.show()


def plot_experiments():

    cnt = 1
    fig, axs = plt.subplots(2, 3, sharey=True, figsize=(15, 9))
    for i in range(2):
        for j in range(3):
            idx = np.random.permutation(N)[:n]
            x_train, y_train = x[idx], y[idx]

            w = []
            for d in d_arr:
                w.append(np.polyfit(x_train, y_train, d))

            axs[i, j].scatter(x_train, y_train)
            axs[i, j].plot(x_range, f(x_range), 'r', linewidth=3.0)
            for k in range(len(w)):
                axs[i, j].plot(x_range, f_hat(x_range, w[k]), colors[k], linewidth=3.0)

            axs[i, j].scatter(x_test, y_test, c='r')
            for k in range(len(w)):
                axs[i, j].scatter(x_test, f_hat(x_test, w[k]), c=colors[k])

            axs[i, j].set_xlabel('x', size=12)
            axs[i, j].set_ylabel('y', size=12)
            axs[i, j].legend([r'$f$', r'$\hat{f}$ (d = 1)', r'$\hat{f}$ (d = 2)', 
                              r'$\hat{f}$ (d = 3)', r'$\hat{f}$ (d = 5)'], fontsize=12)
            axs[i, j].title.set_text('experiment {}'.format(cnt))
            cnt += 1
    plt.tight_layout()

    
def plot_test_hists(n_hists):
    y_hat_test = np.zeros((n_hists, R))

    for r in range(R):
        idx = np.random.permutation(N)[:n]
        x_train, y_train = x[idx], y[idx]

        for k in range(n_hists):
            d = d_arr[k]
            w = np.polyfit(x_train, y_train, d)
            y_hat_test[k, r] = f_hat(x_test, w)

    y_hat_test_mean = np.mean(y_hat_test, 1)
    y_hat_test_std = np.std(y_hat_test, 1)

    fig, axs = plt.subplots(n_hists, 1, sharex=True, sharey=True, figsize=(12, 3*n_hists))
    for k in range(n_hists):
        axs[k].hist(y_hat_test[k], density=True, color=colors[k], alpha=0.6, range=[2, 12])             
        xlim = axs[k].get_xlim()
        axs[k].plot([f(x_test), f(x_test)], [0, 1], 'r', linewidth=3.0)
        axs[k].plot([y_hat_test_mean[k], y_hat_test_mean[k]], [0, 1], c='k', linewidth=3.0)
        axs[k].title.set_text('d = {}'.format(d_arr[k]))
        axs[k].legend([r'$f(x_{test})$', r'$\mathbb{E}[\hat{f}(x_{test})]$', r'$\hat{f}(x_{test})$'], fontsize=12)

    for k in range(n_hists):
        x_range = np.linspace(xlim[0], xlim[1], 1000)
        axs[k].plot(x_range, stats.norm.pdf(x_range, y_hat_test_mean[k], y_hat_test_std[k]), color=colors[k], ls='--')

    plt.suptitle(r'Histogram of $\hat{f}(x_{test})$', size=12)
    

def plot_bias_variance_tradeoff(R=R, n_test=1000, d_arr=np.arange(5)):

    x_test = x_max + np.random.rand(n_test) - .5
    epsilon = sigma_epsilon * np.random.randn(n_test)
    y_test = f(x_test) + epsilon

    train_squared_error = np.zeros((len(d_arr), R))
    y_hat_test = np.zeros((len(d_arr), R, n_test))
    for r in range(R):
        idx = np.random.permutation(N)[:n]
        x_train, y_train = x[idx], y[idx]
        for k in range(len(d_arr)):
            d = d_arr[k]
            w = np.polyfit(x_train, y_train, d)
            train_squared_error[k, r] = np.mean((y_train - f_hat(x_train, w)) ** 2)
            y_hat_test[k, r, :] = f_hat(x_test, w)

    test_squared_error = np.mean((y_hat_test - y_test) ** 2, 1)
    bias_squared = (np.mean(y_hat_test, 1) - f(x_test)) ** 2
    var_y_hat_test = np.var(y_hat_test, 1)

    plt.figure(figsize=(12, 8))
    plt.plot(d_arr, np.mean(test_squared_error, 1), 'g', linewidth=3.0, label='test error')
    plt.plot(d_arr, np.mean(train_squared_error, 1), 'k', linewidth=3.0, label='training error')
    plt.plot(d_arr, np.mean(bias_squared, 1), 'y--', label=r'bias squared: $(\mathbb{E}[\hat{f}(x)] - f(x))^2$')
    plt.plot(d_arr, np.mean(var_y_hat_test, 1), 'b--', label=r'$var(\hat{f}(x))$')
    plt.plot(d_arr, (sigma_epsilon ** 2) * np.ones_like(d_arr), 'r--', label= r'irreducible error: $\sigma_\epsilon^2$')
    # plt.plot(d_arr, np.mean(bias_squared + var_y_hat_test + sigma_epsilon ** 2, 1), 'm--')
    plt.xticks(d_arr)
    plt.xlabel('d', size=12)
    plt.legend(loc='upper center', fontsize=12)
    plt.ylim([0, 12])
    plt.show()
    