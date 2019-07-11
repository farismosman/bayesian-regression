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
    int noOfTrainingObservation;
    int noOfTestObservation;
    int noOfFeatures;
    int y[noOfTrainingObservation];
    matrix[noOfTrainingObservation, noOfFeatures] x;
    matrix[noOfTestObservation, noOfFeatures] xTest;
}
parameters {
    real alpha;
    vector[noOfFeatures] beta;
}                                                                                                    
transformed parameters {                                                                             
    vector[noOfTrainingObservation] linpred;
    linpred = alpha + x * beta;                                                                            
}
model {
    alpha ~ cauchy(0,10);

    for(i in 1 : noOfFeatures)
        beta[i] ~ student_t(1, 0, 0.03);

    y ~ bernoulli_logit(linpred);
}
generated quantities {
  vector[noOfTestObservation] yTest;
  yTest = alpha + xTest * beta;
}
"""


def load_data(filename):
    return pd.read_csv(filename)


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


def traces(fit):
    return {
        'alpha': fit['alpha'],
        'beta': fit['beta'],
        'lp': fit['lp__']
    }

def predict(x, alpha, beta):
    y = alpha + np.dot(beta, x)
    probability = 1.0/(1.0 + np.exp(-y))
    if probability > 0.5:
        return [1, probability]
    return [0, 1 - probability]



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
    training_data = load_data('data/train.csv')

    training_id = training_data.pop('id')
    training_target = training_data.pop('target').astype(int)

    testing_data = load_data('data/test.csv')

    testing_id = testing_data.pop('id')

    data = {
        'noOfTrainingObservation': 250,
        'noOfTestObservation': 19750,
        'noOfFeatures': 300,
        'y': training_target,                                                                                     
        'x': training_data,
        'xTest': testing_data
    }


    fit = sampling(filename='logr.pkl', data=data)
    summary_df = summary(fit)
    
    beta_means = []
    alpha_mean = summary_df['mean']['alpha']
    for i in range(1, data['noOfFeatures'] + 1):
        beta_means.append(summary_df['mean']['beta[%s]'%(str(i))])

    y_prediction = testing_data.apply(lambda x: predict(x, alpha=alpha_mean, beta=beta_means), axis=1)

    extracted_traces = traces(fit)
    _alpha = extracted_traces['alpha']
    _beta = extracted_traces['beta']
    _lp = extracted_traces['lp']

    plot_trace(_alpha, r'$\alpha$')
    plot_trace(_lp, r'lp__')

    for i in range(data['noOfFeatures']):
        plot_trace(_beta[:, i], r'$\beta$')
