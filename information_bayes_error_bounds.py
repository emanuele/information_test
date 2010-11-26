"""Functions and plot about the Fano's inequality:
http://en.wikipedia.org/wiki/Fano%27s_inequality
http://www.scholarpedia.org/article/Fano_inequality
"""

import numpy as np

def conditional_entropy_upper_bound(bayes_error, num_classes=2):
    """Fano's inequality: upper bound of the conditional entropy of a
    class label Y given input X once the Bayes error is known.

    Fano's inequality: H(Y|X) <= H(E) + P(e) log(num_classes-1)
    P(e) = bayes_error
    H(E) = -(P(e) log(P(e)) + (1-P(e))log(1-P(e)))

    The return value is the bound, i.e., the case where:
    H(Y|X) = H(E) + P(e) log(num_classes-1)
    """
    bound = - (bayes_error * np.log2(bayes_error) +  \
               (1.0 - bayes_error) * np.log2(1.0 - bayes_error)) +  \
               bayes_error * np.log2(num_classes -1)
    return np.nan_to_num(bound)

def conditional_entropy_lower_bound(bayes_error):
    """Lower bound of conditional entropy, from Hellman-Raviv 1970:
@article{hellman1970probability,
    abstract = {Abstract-Relationships between the probability of error, the equivocation, and the Chemoff bound are examined for the two-hypothesis decision problem. The effect of rejections on these bounds is derived. Finally, the results are extended to the case of any finite number of hypotheses. I.},
    author = {Hellman, Martin E. and Raviv, Josef},
    citeulike-article-id = {8071356},
    citeulike-linkout-0 = {http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.131.2865},
    journal = {IEEE Transactions on Information Theory},
    pages = {368--372},
    posted-at = {2010-10-22 21:07:06},
    priority = {2},
    title = {Probability of error, equivocation and the chernoff bound},
    url = {http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.131.2865},
    volume = {16},
    year = {1970}
}
    """
    return 2.0 * bayes_error

def mutual_information_lower_bound(bayes_error, num_classes=2, Y_entropy=1.0):
    """Lower bound of Mutual Information given the Bayes error. It
    depends on the Fano's inequality upper bound on conditional
    entropy.
    """
    return Y_entropy - conditional_entropy_upper_bound(bayes_error, num_classes=num_classes)

def mutual_information_upper_bound(bayes_error, Y_entropy=1.0):
    """Upper bound of Mutual Information given the Bayes error. It
    depends on the Hellman-Raviv lower bound on conditional entropy.
    """
    return Y_entropy - conditional_entropy_lower_bound(bayes_error)


class bayes_error_lower_bound(object):
    """Compute the lower bound of the bayes error given I(X;Y) by
    means of the Fano's inequelity.

    Since it is not straightforward to invert the inequality we rely
    on interpolation after setting up a cache of bayes errors and
    corresponding mutual informations.

    Note that this class behaves like a function (see __call__())
    after instantiation in order to have a uniform behavior with the
    other functions in this module.

    Technical note: the bayes error cached values are generated in
    descending order so that the corresponding mutual information is
    not decreasing, which is a requirement of np.interp().
    """
    
    def __init__(self, Y_entropy=1.0 , num_classes=2, cache_size=1000):
        self.Y_entropy = Y_entropy
        self.num_classes = num_classes
        self.bayes_error_lb_cache = np.linspace(0.5, 0.0, cache_size)
        self.mutual_information_lb_cache = mutual_information_lower_bound(self.bayes_error_lb_cache)

    def __call__(self, mutual_information):
        """Computation of the Bayes error given mutual information by
        means of interpolation from a cache of precomputed pairs of
        values.
        """
        mutual_information = np.nan_to_num(mutual_information) # casting necessary for pymc
        return np.interp(mutual_information,self.mutual_information_lb_cache,self.bayes_error_lb_cache)


def bayes_error_upper_bound(mutual_information):
    """Bayes error upper given I(X;Y) according to the Hellman-Raviv
    bound.
    """
    return 0.5 * (1.0 - mutual_information)


if __name__=='__main__':

    import matplotlib.pyplot as plt

    plt.figure()

    bayes_error = np.linspace(0 , 0.5 , 200)

    conditional_entropy_ub = conditional_entropy_upper_bound(bayes_error)
    conditional_entropy_lb = conditional_entropy_lower_bound(bayes_error)

    plt.plot(bayes_error, conditional_entropy_ub, 'k-', label="Fano's inequality bound", linewidth=3)
    plt.plot(bayes_error, conditional_entropy_lb, 'k--', label="Hellman-Raviv bound", linewidth=3)
    plt.legend()
    plt.ylabel('conditional entropy : H(Y|X)')
    plt.xlabel('Bayes error')

    mutual_information_lb = mutual_information_lower_bound(bayes_error)
    mutual_information_ub = mutual_information_upper_bound(bayes_error)
    plt.figure()
    plt.plot(bayes_error, mutual_information_ub, 'k--', label="Hellman-Raviv bound", linewidth=3)
    plt.plot(bayes_error, mutual_information_lb, 'k-', label="Fano's inequality bound", linewidth=3)
    plt.legend()
    plt.ylabel('Mutual Information : I(Y;X)')
    plt.xlabel('Bayes error')

    mi = np.linspace(0,1,200)
    
    belb=bayes_error_lower_bound()
    be_lb = belb(mi)
    be_ub = bayes_error_upper_bound(mi)

    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.1, 0.85, 0.85])
    plt.plot(mi, be_ub, 'k--', label="Hellman-Raviv bound", linewidth=3)
    plt.plot(mi, be_lb, 'k-', label="Fano's inequality bound", linewidth=3)
    plt.legend()
    plt.xlabel('Mutual Information $I(Y;X)$')
    plt.ylabel('Bayes error $\epsilon_B$')
    
