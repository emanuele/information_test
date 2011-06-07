"""Hierarchical model connecting mutual information (I), Bayes error
(epsilon_B), the generalization error (epsilon) of a classifier and
the observed number of errors (e) on a testset of size m.

See: Olivetti et al., Testing for Information with Brain Decoding,
Pattern Recognition for NeuroImaging (PRNI), 2011.

Copyright Emanuele Olivetti, 2011.

This program is distributed under the GNU General Public Licence v3.0.
"""

import numpy as np
import pymc
import information_bayes_error_bounds as ibeb
import pprint
import matplotlib.pyplot as plt

def information_model(e, m, epsilon=1.0e-5):
    I = pymc.Uniform(name='mutual information', lower=epsilon, upper=1.0-epsilon)
    belb = ibeb.bayes_error_lower_bound(Y_entropy=1.0, num_classes=2)
    @pymc.deterministic(name='Bayes error lower bound')
    def bayes_error_lb(I=I):
        return belb(I)
    @pymc.deterministic(name='Bayes error upper bound')
    def bayes_error_ub(I=I):
        return ibeb.bayes_error_upper_bound(I)
    epsilon_B = pymc.Uniform(name='Bayes error', lower=bayes_error_lb, upper=bayes_error_ub)
    epsilon = pymc.Uniform(name='generalization error', lower=epsilon_B, upper=0.5)
    error = pymc.Binomial(name='observed number of errors', n=m, p=epsilon, observed=True, value=e)
    return locals()
    

if __name__=='__main__':

    m = 40
    e = 8
    model = information_model(e=e, m=m)
    M = pymc.MCMC(model)
    M.isample(iter=100000, burn=10000, thin=10)

    pymc.Matplot.plot(M)
    pprint.pprint(M.stats())
    
    posterior_mutual_information = M.trace('mutual information')[:]
    posterior_Bayes_error = M.trace('Bayes error')[:]    
    posterior_Bayes_error_lb = M.trace('Bayes error lower bound')[:]
    posterior_Bayes_error_ub = M.trace('Bayes error upper bound')[:]
    posterior_generalization_error = M.trace('generalization error')[:]    

    plt.figure()
    plt.hist(posterior_mutual_information, bins=50, normed=True)
    plt.title('$p(I(X;Y)|e=%s,m=%s,H_1)$' % (e,m))
    plt.xlim([0.0,1.0])
    plt.savefig('posterior_I_e_%s_m_%s.pdf' % (e, m))

    plt.figure()
    plt.hist(posterior_Bayes_error, bins=50, normed=True)
    plt.title('$p(\epsilon_B|e=%s,m=%s,H_1)$' % (e,m))
    plt.xlim([0.0,0.5])
    plt.savefig('posterior_epsilonB_e_%s_m_%s.pdf' % (e, m))
    
    plt.figure()
    plt.hist(posterior_generalization_error, bins=50, normed=True)
    plt.title('$p(\epsilon|e=%s,m=%s,H_1)$' % (e,m))
    plt.xlim([0.0,0.5])
    plt.savefig('posterior_epsilon_e_%s_m_%s.pdf' % (e, m))
    
