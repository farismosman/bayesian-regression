import pystan
import pickle
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from os.path import exists


model = """

data {
    int<lower=0> N;
    vector[N] x;
    vector[N] y;
}

parameters {
    real alpha;
    real beta;
    real<lower=0> sigma;
}

model {
    y ~ normal(alpha + beta * x, sigma);
}

"""


def generate_data(alpha, beta, sigma):
    x = 10 * np.random.rand(100)
    y = alpha + beta * x
    y = np.random.normal(y, scale=sigma)
    _data = {'N': len(x), 'x': x, 'y': y}

    return _data


def sampling(filename, data, iter=1000, chains=4, warmup=500, thin=1, seed=101, verbose=True):
    if exists(filename):
        sm = pickle.load(open(filename, 'rb'))
        return sm.sampling(data, iter=1000, chains=4, warmup=500, thin=1, seed=101, verbose=True)

    sm = pystan.StanModel(model_code=model)
    with open(filename, 'wb') as f:
        pickle.dump(sm, f)

    return sm.sampling(data, iter=1000, chains=4, warmup=500, thin=1, seed=101, verbose=True)


def summary(fit):
    _summary = fit.summary()
    df = pd.DataFrame(_summary['summary'], columns=_summary['summary_colnames'], index=_summary['summary_rownames'])
    return df


def plot_data(x, y):
    plt.scatter(x, y)
    plt.xlabel('$X$')
    plt.ylabel('$Y$')
    plt.title('Scatter Plot of Data')
    plt.show()


def traces(fit):
    return {
        'alpha': fit['alpha'],
        'beta': fit['beta'],
        'sigma': fit['sigma'],
        'lp': fit['lp__']
    }


def plot_regression_lines(x, y, xmin, xmax, alpha, beta, alpha_mean, beta_mean):
    xaxis = np.linspace(xmin, xmax, 100)

    np.random.shuffle(alpha) 
    np.random.shuffle(beta)

    for i in range(1000):
        plt.plot(xaxis, alpha[i] + beta[i] * xaxis, color='lightsteelblue', alpha=0.005)

    plt.plot(xaxis, alpha_mean + beta_mean * xaxis)
    plt.scatter(x, y)

    plt.xlabel('$X$')
    plt.ylabel('$Y$')
    plt.title('Fitted Regression Line')
    plt.xlim(xmin, xmax)
    plt.show()


def plot_trace(param, param_name='parameter'):
    """Plot the trace and posterior of a parameter."""
    
    # Summary statistics
    mean = np.mean(param)
    median = np.median(param)
    cred_min, cred_max = np.percentile(param, 2.5), np.percentile(param, 97.5)
    
    # Plotting
    plt.subplot(2,1,1)
    plt.plot(param)
    plt.xlabel('samples')
    plt.ylabel(param_name)
    plt.axhline(mean, color='r', lw=2, linestyle='--')
    plt.axhline(median, color='c', lw=2, linestyle='--')
    plt.axhline(cred_min, linestyle=':', color='k', alpha=0.2)
    plt.axhline(cred_max, linestyle=':', color='k', alpha=0.2)
    plt.title('Trace and Posterior Distribution for {}'.format(param_name))

    plt.subplot(2,1,2)
    plt.hist(param, 30, density=True); sns.kdeplot(param, shade=True)
    plt.xlabel(param_name)
    plt.ylabel('density')
    plt.axvline(mean, color='r', lw=2, linestyle='--',label='mean')
    plt.axvline(median, color='c', lw=2, linestyle='--',label='median')
    plt.axvline(cred_min, linestyle=':', color='k', alpha=0.2, label=r'95% CI')
    plt.axvline(cred_max, linestyle=':', color='k', alpha=0.2)
    
    plt.gcf().tight_layout()
    plt.legend()
    plt.show()


if __name__ == "__main__":
    data = generate_data(alpha=4.0, beta=0.5, sigma=1.0)
    x, y = data['x'], data['y']

    plot_data(x, y)

    fit = sampling(filename='lr.pkl', data=data)
    summary_df = summary(fit)
    alpha_mean, beta_mean = summary_df['mean']['alpha'], summary_df['mean']['beta']


    extracted_traces = traces(fit)
    _alpha = extracted_traces['alpha']
    _beta = extracted_traces['beta']
    _sigma = extracted_traces['sigma']
    _lp = extracted_traces['lp']

    plot_regression_lines(x=x, y=y, xmin=-0.5, xmax=10.5, alpha=_alpha, beta=_beta, alpha_mean=alpha_mean, beta_mean=beta_mean)

    plot_trace(_alpha, r'$\alpha$')
    plot_trace(_beta, r'$\beta$')
    plot_trace(_sigma, r'$\sigma$')
    plot_trace(_lp, r'lp__') 