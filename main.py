# ********************************************HELPER*************************************************#
#from scipy.special import erfinv
import math
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
def sqrt(x):
    # Babylonian method to approximate the square root of a number
    if x == 0:
        return 0
    a = x
    b = 1.0
    while abs(a - b) > 0.0001:
        a, b = (a + b) / 2, x / a
    return a
def erf(x):
    # Approximation of the error function using a truncated Taylor series
    p = 0.3275911
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    t = 1.0 / (1.0 + p * x)
    poly = a1 * t + a2 * t ** 2 + a3 * t ** 3 + a4 * t ** 4 + a5 * t ** 5
    return 1.0 - poly * x ** 2
def sign(x):
    if x < 0:
        return -1
    elif x > 0:
        return 1
    else:
        return 0
def polevl(x, coefs, N):
    ans = 0
    power = len(coefs) - 1
    for coef in coefs:
        ans += coef * x**power
        power -= 1
    return ans
#chad from stackoverflow
def p1evl(x, coefs, N):
    return polevl(x, [1] + coefs, N)
def erfinv(z):
    if z < -1 or z > 1:
        raise ValueError("`z` must be between -1 and 1 inclusive")

    if z == 0:
        return 0
    if z == 1:
        return math.inf
    if z == -1:
        return -math.inf

    # From scipy special/cephes/ndrti.c
    def ndtri(y):
        # approximation for 0 <= abs(z - 0.5) <= 3/8
        P0 = [
            -5.99633501014107895267E1,
            9.80010754185999661536E1,
            -5.66762857469070293439E1,
            1.39312609387279679503E1,
            -1.23916583867381258016E0,
        ]

        Q0 = [
            1.95448858338141759834E0,
            4.67627912898881538453E0,
            8.63602421390890590575E1,
            -2.25462687854119370527E2,
            2.00260212380060660359E2,
            -8.20372256168333339912E1,
            1.59056225126211695515E1,
            -1.18331621121330003142E0,
        ]

        # Approximation for interval z = sqrt(-2 log y ) between 2 and 8
        # i.e., y between exp(-2) = .135 and exp(-32) = 1.27e-14.
        P1 = [
            4.05544892305962419923E0,
            3.15251094599893866154E1,
            5.71628192246421288162E1,
            4.40805073893200834700E1,
            1.46849561928858024014E1,
            2.18663306850790267539E0,
            -1.40256079171354495875E-1,
            -3.50424626827848203418E-2,
            -8.57456785154685413611E-4,
        ]

        Q1 = [
            1.57799883256466749731E1,
            4.53907635128879210584E1,
            4.13172038254672030440E1,
            1.50425385692907503408E1,
            2.50464946208309415979E0,
            -1.42182922854787788574E-1,
            -3.80806407691578277194E-2,
            -9.33259480895457427372E-4,
        ]

        # Approximation for interval z = sqrt(-2 log y ) between 8 and 64
        # i.e., y between exp(-32) = 1.27e-14 and exp(-2048) = 3.67e-890.
        P2 = [
            3.23774891776946035970E0,
            6.91522889068984211695E0,
            3.93881025292474443415E0,
            1.33303460815807542389E0,
            2.01485389549179081538E-1,
            1.23716634817820021358E-2,
            3.01581553508235416007E-4,
            2.65806974686737550832E-6,
            6.23974539184983293730E-9,
        ]

        Q2 = [
            6.02427039364742014255E0,
            3.67983563856160859403E0,
            1.37702099489081330271E0,
            2.16236993594496635890E-1,
            1.34204006088543189037E-2,
            3.28014464682127739104E-4,
            2.89247864745380683936E-6,
            6.79019408009981274425E-9,
        ]

        s2pi = 2.50662827463100050242
        code = 1

        if y > (1.0 - 0.13533528323661269189):      # 0.135... = exp(-2)
            y = 1.0 - y
            code = 0

        if y > 0.13533528323661269189:
            y = y - 0.5
            y2 = y * y
            x = y + y * (y2 * polevl(y2, P0, 4) / p1evl(y2, Q0, 8))
            x = x * s2pi
            return x

        x = math.sqrt(-2.0 * math.log(y))
        x0 = x - math.log(x) / x

        z = 1.0 / x
        if x < 8.0:                 # y > exp(-32) = 1.2664165549e-14
            x1 = z * polevl(z, P1, 8) / p1evl(z, Q1, 8)
        else:
            x1 = z * polevl(z, P2, 8) / p1evl(z, Q2, 8)

        x = x0 - x1
        if code != 0:
            x = -x

        return x

    result = ndtri((z + 1) / 2.0) / math.sqrt(2)

    return result
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
def pnorm(x, mu=0, sigma=1):
    """
    Calculates the cumulative distribution function (CDF) of the standard normal distribution.

    :param x: The point at which to evaluate the CDF.
    :param mu: The mean of the normal distribution. Default is 0.
    :param sigma: The standard deviation of the normal distribution. Default is 1.
    :return: The probability that a random variable from a standard normal distribution is less than or equal to x.
    """
    z = (x - mu) / sigma
    t = 1 / (1 + 0.2316419 * abs(z))
    d = 0.3989423 * exp(-z * z / 2)
    prob = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))))
    if z > 0:
        prob = 1 - prob
    return prob

# ********************************************-QNORM-**************************************************#
def qnorm(q, mean=0, std=1):
    """Returns the x-value that corresponds to the given quantile q in the normal distribution with mean and standard deviation."""
    if q < 0 or q > 1:
        raise ValueError("Quantile q must be between 0 and 1.")

    # Approximation for x-value using inverse CDF of standard normal distribution
    # Uses the identity x = mean + std * qnorm(p), where p is the probability corresponding to q
    p = (1 + erf((q - 0.5) / sqrt(2))) / 2
    return mean + std * sqrt(2) * erfinv(2 * q - 1)

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

    print("OUT:", out)
    return
# ********************************************-PRINTING NORMS-*********************************************#
def norms():
    # selecting a choice of distribution
    print("1:Pnorm   2:Qnorm")
    print("3:QQnorm  4:QQline")
    choice = int(input())

    if(choice == 1):
        print("q, mean, std")
        out = pnorm(int(input()), int(input()), int(input()))
    elif (choice == 2):
        print("percentile=p")
        print("p, mean, std")
        out = qnorm(float(input()), int(input()), int(input()))
    elif (choice == 3):
        out=1
    elif (choice == 4):
        out = 1

    print("OUT:", out)
    return

#*******************************************-PRINTING expVar()-*********************************************#
def expVar():
    return

#********************************************-SELECTING-*************************************************#
#print(qnorm(0.9,19000,2100))
def run_interactive_menu():
    while True:
        print("1: distributions")
        print("2: norms")
        print("3: exp(), var()")

        selection = int(input())

        if selection == 1:
            dists()
        elif selection == 2:
            norms()
        elif selection == 3:
            expVar()

        delay = input()
        print("\n" * 3)

if __name__ == "__main__":
    run_interactive_menu()