# ********************************************HELPER*************************************************#
# import scipy.stats as stats
# from scipy.stats import hypergeom
#import scipy.stats as stats
from scipy.stats import poisson
def factorial(n):
    """
    Calculates the factorial of a non-negative integer n.

    :param n: The integer to calculate the factorial of.
    :return: The factorial of n.
    """
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result
def choose(n, k):
    """
    Calculates the binomial coefficient (n choose k).

    :param n: The total number of items.
    :param k: The number of items to choose.
    :return: The binomial coefficient (n choose k).
    """
    if k > n:
        return 0
    if k == 0:
        return 1
    if k > n / 2:
        k = n - k
    result = 1
    for i in range(k):
        result *= (n - i)
        result //= (i + 1)
    return result
def exp(x):
    """
    Calculates the exponential of x.
    """
    term = 1.0
    sum = 0.0
    for i in range(100):
        sum += term
        term *= x / (i + 1)
    return sum

# ********************************************1-DNBINOM*************************************************#
def dnbinom(k, r, p):
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

# ********************************************2-QNBINOM*************************************************#
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

# ********************************************3-DBINOM*************************************************#
def dbinom(x, size, prob):
    """
    Calculates the probability mass function (PMF) of the binomial distribution.

    :param x: The number of successes.
    :param size: The number of trials.
    :param prob: The probability of success.
    :return: The probability mass function of the binomial distribution at x.
    """
    coef = factorial(size) / (factorial(x) * factorial(size - x))  # Calculate the binomial coefficient
    return coef * prob ** x * (1 - prob) ** (size - x)  # Calculate the PMF

# ********************************************4-PBINOM*************************************************#
def pbinom(q, size, prob, lower_tail=True):
    """
    Calculates the cumulative distribution function (CDF) of the binomial distribution.

    :param q: The upper limit of the summation for the CDF.
    :param size: The number of trials.
    :param prob: The probability of success.
    :param lower_tail: If True, calculates the lower tail (default). If False, calculates the upper tail.
    :return: The cumulative distribution function of the binomial distribution at q.
    """
    if lower_tail:
        return sum([dbinom(i, size, prob) for i in range(q + 1)])
    else:
        return sum([dbinom(i, size, prob) for i in range(q, size + 1)])

# ********************************************5-DHYPER*************************************************#
def dhyper(x, m, n, k):
    """
    Calculates the probability mass function (PMF) of the hypergeometric distribution.

    :param x: The number of successes in the sample.
    :param m: The number of successes in the population.
    :param n: The number of failures in the population.
    :param k: The sample size.
    :return: The probability mass function of the hypergeometric distribution at x.
    """
    N = m + n  # Total population size
    numerator = choose(m, x) * choose(n, k - x)  # Calculate the numerator of the PMF
    denominator = choose(N, k)  # Calculate the denominator of the PMF
    return numerator / denominator  # Calculate the PMF

# ********************************************6-PHYPER*************************************************#
def phyper(x, M1, M2, n1):
    N = M1 + M2  # population size
    n = M1  # number of successes in population
    M = n1  # sample size
    k = x  # number of successes in sample

    # Calculate the CDF using the formula
    cdf = 0
    for i in range(k + 1):
        cdf += (choose(M, i) * choose(N - M, n - i)) / choose(N, n)
    return cdf

# ********************************************7-PPOIS*************************************************#
def ppois(q, mu, lower_tail=True):
    """
    Calculates the cumulative distribution function (CDF) of the Poisson distribution.

    :param q: the quantile(s)
    :param mu: the mean parameter of the Poisson distribution
    :param lower_tail: if True, returns P(X <= q), otherwise returns P(X > q)

    :return: the CDF of the Poisson distribution evaluated at q
    """
    p = 0.0
    for k in range(int(q) + 1):
        p += ((mu ** k) / factorial(k)) * exp(-mu)
    if lower_tail:
        return p
    else:
        return 1 - p

#================================================================================================================#

# ********************************************PNORM*************************************************#
# ********************************************-QNORM-**************************************************#
# ********************************************-QQNORM-*************************************************#
# ********************************************-QQLINE-*************************************************#

# ********************************************-PRINTING DISTS-*********************************************#
def dists():
    # selecting a choice of distribution
    print("1:dnbinom  2:pnbinom")
    print("3:dbinom   4:pbinom")
    print("5:dhyper   6:phyper")
    print("7:ppois           ")
    choice = int(input())

    # method based on choice
    if (choice == 1):  # 1:dnbinom
        print("tot-suc, suc, prob")
        out = dnbinom(int(input()), int(input()), float(input()))

    elif (choice == 2):  # 2:pnbinom
        print("ex: x>n, n>suc");
        print("n-suc, suc, prob")
        out = pnbinom(int(input()), int(input()), float(input()))

    elif (choice == 3):  # 3:dbinom
        print("x,size,prob");
        out = dbinom(int(input()), int(input()), float(input()))

    elif (choice == 4):  # 4:pbinom
        print("x,size,prob");
        out = pbinom(int(input()), int(input()), float(input()))

    elif (choice == 5):  # 5:dhyper
        print("x, M1(not tag),")
        print("M2(tag), sample")
        out = dhyper(int(input()), int(input()), int(input()), int(input()))

    elif (choice == 6):  # 6:phyper
        print("x, M1(not tag),")
        print("M2(tag), sample")
        out = phyper(int(input()), int(input()), int(input()), int(input()))

    elif (choice == 7):  # 7:ppois
        print("q, lambda")
        out = ppois(int(input()), int(input()))

    print("OUT:", out, "\n")
    return
# ********************************************-PRINTING NORMS-*********************************************#
def norms():
    return

#*******************************************-PRINTING expVar()-*********************************************#
def expVar():
    return

#********************************************-SELECTING-*************************************************#
while (True):
    #selecting the mode
    print("1: distibutions")
    print("2: norms")
    print("3: exp(), var()")

    selection = int(input())

    if(selection == 1):
        dists() #sends you to printing dists

    elif(selection == 2):
        norms() #sends you to printing norms
    elif (selection == 3):
        expVar()#sends you to printing exp()var()

    delay = input()