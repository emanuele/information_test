"""A simple implementation of the Binomial test.

Copyright Emanuele Olivetti, 2011.

This program is distributed under the GNU General Public Licence v3.0.
"""

import numpy as np
import scipy.misc as sm
import matplotlib.pyplot as plt

class Binomial(object):
    """Binomial distribution.

    PMF, CDF and test accept both scalars and vectors as input.

    This implementation is necessary since scipy.stats.binom as of
    v0.7.0 does not work.
    """
    def __init__(self, n, p):
        self.n = n
        self.p = p

    def __call__(self, k):
        """Returns PMF value in k.
        """
        return self.pmf(k)

    def pmf(self, k):
        """Binomial PMF evaluate in k.
        """
        return sm.comb(self.n, k) * self.p**k * (1.0 - self.p)**(self.n-k)

    def cdf(self, k):
        """Binomial CDF in k.
        """
        k = np.atleast_1d(np.round(k))
        return np.array([self.pmf(np.arange(ki+1)).sum() for ki in k])

    def test(self, k):
        """Binomial test, i.e., P(k<=e) under Binomial distribution.
        """
        return self.cdf(k)

if __name__=='__main__':

    testset_size = 50
    epsilon = 0.5
    num_errors = 18
    alpha = 0.05
    
    # not working in scipy 0.7.0:
    # import scipy.stats as ss
    # binomial = ss.binom(n=testset_size, pr=epsilon)
    # binomial.pdf(num_errors)
    # This motivates the implementation of Binomial in this file.
    
    print "Binomial Test:"
    binomial = Binomial(n=testset_size, p=epsilon)
    p_value = binomial.test(num_errors)
    print "P(k <= e=%s | testset_size=%s, epsilon=%s) = %s" % (num_errors, testset_size, epsilon, p_value)
    print "Confidence level is alpha =", alpha
    print "The Binomial test",
    if p_value <= alpha :
        print "REJECTS the null hypothesis of no information."
    else:
        print "DOES NOT REJECT the hypothesis of no information."

    k = np.arange(testset_size+1)

    plt.figure()
    plt.plot(k, binomial.pmf(k), 'o')
    plt.xlabel('number of errors (e)')
    plt.ylabel('Binomial(%s, %s) PMF' % (testset_size, epsilon))
    plt.figure()
    plt.plot(k, binomial.cdf(k), 'o')
    plt.xlabel('number of errors (e)')
    plt.ylabel('Binomial(%s, %s) CDF' % (testset_size, epsilon))
    plt.figure()
    plt.plot(k,binomial.test(k), 'o')
    plt.title("Binomial test, testset_size=%i epsilon=%s" % (testset_size, epsilon))
    plt.xlabel('number of errors (e)')
    plt.ylabel('p(k,<=e) under Binomial(%s, %s)' % (testset_size, epsilon))
