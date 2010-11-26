import numpy as np
import information_bayes_error_bounds as ibeb
import binomial_test

def data_likelihood(m, lower=0.0, upper=1.0, N=100000, counts_min=5):
    """Compute p(e|0>I(X;Y)>=1,m) via Monte Carlo with N iterations.

    upper, lower = bounds of mutual information.
    """
    be_lb = ibeb.bayes_error_lower_bound(Y_entropy=1.0, num_classes=2, cache_size=100000)
    be_ub = ibeb.bayes_error_upper_bound

    # Monte Carlo:
    MI = np.random.uniform(low=lower, high=upper, size=N)
    epsilon_B_lower = be_lb(MI)
    epsilon_B_upper = be_ub(MI)
    epsilon_B = np.random.uniform(low=epsilon_B_lower, high=epsilon_B_upper)
    epsilon = np.random.uniform(low=epsilon_B, high=0.5)
    e = np.random.binomial(m,epsilon)

    # Statistics:
    e_counts, bins = np.histogram(e,bins=range(m+2))
    e_frequency = e_counts / np.double(N)
    e_frequency[e_counts<counts_min] = np.NaN # mark less reliable values with NaN
    return e_frequency

def Bayes_factor(m, N=100000):
    """Compute Bayes factor for all observed errors from 0 to m
    on a test set of size m.
    """
    # H1:
    e_frequency_H1 = data_likelihood(m=m, N=N)    
    # H0:
    b = binomial_test.Binomial(n=m, p=0.5)
    e_pmf_H0 = b.pmf(np.arange(m+1))
    Bf = e_frequency_H1 / e_pmf_H0
    return Bf

def Bayes_factor_lower_bound(p_value=0.05):
    """Lower bound of the Bayes factor given p-value.

    See: Sellke, Bayarri and Berger, J.Am.Stat. (2001)
    """
    assert(p_value < 1.0/np.e)
    return - 1.0 / (np.e * p_value * np.log(p_value))


if __name__=='__main__':

    np.random.seed(0)

    m = 50 # size of the test set

    N = 1000000 # number of iterations of Monte Carlo

    p_value = 0.05
    B = Bayes_factor_lower_bound(p_value=p_value)

    print "Computation of the Bayes factor (Bf) for the information test given the observed number of errors (e)."

    Bf = Bayes_factor(m, N)
    e_limit = m
    tmp = np.where(np.isnan(Bf))[0]
    if tmp.size > 0:
        print "Results are unreliable when e =", tmp
        e_limit = tmp.min()

    print "Bayes factor given e:"
    print 'e \t Bf'
    for e in range(e_limit):
        print e, '\t', Bf[e]
        
    # plot:
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.1, 0.85, 0.85])
    plt.semilogy(range(e_limit), Bf[:e_limit], 'ko')
    left = 0.0
    height = np.nan_to_num(Bf).max()
    width = np.where(Bf>=B)[0].max()
    bottom = B
    plt.bar(left=left,height=height, width=width, bottom=bottom, fill=True, color='#EAEAEA')
    Bf_min = Bf[-np.isnan(Bf)].min()
    plt.plot([width, width], [B, Bf_min], 'k-')
    left1 = np.where(Bf<=1.0/B)[0].min()
    height1 = 1.0/B - Bf_min
    width1 = e_limit - left1
    bottom1 = Bf_min
    plt.bar(left=left1,height=height1, width=width1, bottom=bottom1, fill=True, color='whitesmoke')
    plt.plot([0.0, left1], [1.0/B, 1.0/B], 'k-')
    plt.text(width*0.6, bottom+10**(np.log10(height)*0.75),"$H_1$",{'fontsize':24})
    plt.text(left1+width1*0.6,bottom1+10**(np.log10(height1)*1.05),"$H_0$",{'fontsize':24})
    plt.axis('tight')
    plt.xlim([0,m])
    plt.xlabel('number of observed errors $e$')
    plt.ylabel('Bayes factor = $p(e|H_1)/p(e|H_0)$')
    
    # plt.show() # usually not necessary.
