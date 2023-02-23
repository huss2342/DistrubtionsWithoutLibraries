#********************************************HELPER*************************************************#
#import scipy.stats as stats

#********************************************1-DNBINOM*************************************************#

def dnbinom(k,r,p):
    """Returns the probability of k failures before r successes in a sequence of
    Bernoulli trials with probability of success p."""
    if p < 0 or p > 1:
        raise ValueError("p must be between 0 and 1.")
    if r <= 0 or not isinstance(r, int):
        raise ValueError("r must be a positive integer.")
    if k < 0 or not isinstance(k, int):
        raise ValueError("k must be a non-negative integer.")

    # Calculate the binomial coefficient
    def binom(n, k):
        """Returns the binomial coefficient 'n choose k'."""
        if 0 <= k <= n:
            ntok = 1
            ktok = 1
            for t in range(1, min(k, n - k) + 1):
                ntok *= n
                ktok *= t
                n -= 1
            return ntok // ktok
        else:
            return 0

    # Calculate the probability mass function of the negative binomial distribution
    return binom(r + k - 1, k) * p ** r * (1 - p) ** k

#********************************************2-QNBINOM*************************************************#
def pnbinom(x, size, prob):
    """
    Calculates the cumulative distribution function of the negative binomial distribution
    for a given number of failures x, a given number of successes size, and a given
    probability of success prob, using the parameterization used in R.
    """

    # Compute the binomial coefficient using integer arithmetic
    def binomial(n, k):
        if 0 <= k <= n:
            ntok = 1
            ktok = 1
            for t in range(1, min(k, n - k) + 1):
                ntok *= n
                ktok *= t
                n -= 1
            return ntok // ktok
        else:
            return 0

    # Calculate the cumulative distribution function using the negative binomial parameterization used in R
    cdf = 0
    for i in range(x + 1):
        cdf += binomial(i + size - 1, size - 1) * (prob ** size) * ((1 - prob) ** i)

    return cdf
#AFTER ADDING ALL THESE METHODS... ADD EXP AND VAR CLACULATOR
#********************************************3-DBINOM*************************************************#
#********************************************4-PBINOM*************************************************#
#********************************************5-DHYPER*************************************************#
#********************************************6-PHYPER*************************************************#
def phyper(x,M1,M2,n1):
    N = M1 + M2  # population size
    n = M1      # number of successes in population
    M = n1      # sample size
    k = x      # number of successes in sample

    # Calculate binomial coefficients
    def choose(a, b):
        if b > a:
            return 0
        if b == 0 or b == a:
            return 1
        return choose(a-1, b-1) + choose(a-1, b)

    # Calculate the CDF using the formula
    cdf = 0
    for i in range(k+1):
        cdf += (choose(M, i) * choose(N-M, n-i)) / choose(N, n)
    return cdf



#********************************************7-PPOIS*************************************************#
#********************************************8-PNORM*************************************************#
#********************************************9-QNORM**************************************************#
#********************************************10-QQNORM*************************************************#
#********************************************11-QQLINE*************************************************#

#********************************************PRINTING*************************************************#
while(True):
    #goal is the python line... x,M1,M2,sampleSize

    #selecting a choice
    print("1: dnbinom    2: pnbinom")
    print("3:            4:")
    print("4:            6:phyper")
    choice = int(input())

    #method based on choice
    if(choice==1):
        print("tot-suc, suc, prob")
        a = int(input())
        b = int(input())
        c = float(input())
        out = dnbinom(a,b,c)
    elif(choice==2):
        print("ex: x>n, n>suc");
        print("n-suc, suc, prob")
        a = int(input())
        b = int(input())
        c = float(input())
        out = pnbinom(a,b,c)
    elif (choice == 6):
        print("x, M1(not tag),")
        print("M2(tag), sample")
        x = int(input())
        M1 = int(input())
        M2 = int(input())
        n = int(input())
        out = phyper(x,M1,M2,n)

    print("OUT:", out, "\n")