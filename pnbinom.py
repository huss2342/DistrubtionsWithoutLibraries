def pnbinom(x, r, p):
    """
    Calculates the probability mass function of the negative binomial distribution
    for a given number of failures x, a given number of successes r, and a given
    probability of success p.
    """
    if x < 0:
        return 0
    else:
        # Calculate the binomial coefficient using the factorial function
        def factorial(n):
            if n == 0:
                return 1
            else:
                return n * factorial(n - 1)

        binom_coef = factorial(x + r - 1) / (factorial(x) * factorial(r - 1))

        # Calculate the probability mass function using the binomial coefficient
        pmf = binom_coef * pow(p, r) * pow(1 - p, x)

        return pmf


print("tot-suc, suc, prob" )

a = int(input())
b = int(input())
c = float(input())
out = pnbinom(a,b,c)
print("OUT : \n", out)