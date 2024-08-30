# unit_tests/test_statistical_functions.py

import unittest
import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import factorial, choose, dnbinom, dbinom, dhyper, ppois, pnorm, qnorm


class TestStatisticalFunctions(unittest.TestCase):

    def test_factorial(self):
        self.assertEqual(factorial(0), 1)
        self.assertEqual(factorial(5), 120)
        self.assertEqual(factorial(10), 3628800)

    def test_choose(self):
        self.assertEqual(choose(5, 2), 10)
        self.assertEqual(choose(10, 3), 120)
        self.assertEqual(choose(20, 5), 15504)

    def test_dnbinom(self):
        self.assertAlmostEqual(dnbinom(3, 5, 0.5), 0.1318359375, places=2)
        self.assertAlmostEqual(dnbinom(2, 3, 0.7), 0.1852200, places=2)

    def test_dbinom(self):
        self.assertAlmostEqual(dbinom(2, 5, 0.5), 0.3125, places=7)
        self.assertAlmostEqual(dbinom(3, 10, 0.3), 0.2668279, places=7)

    def test_dhyper(self):
        self.assertAlmostEqual(dhyper(1, 6, 15, 3), 0.474, places=2)
        self.assertAlmostEqual(dhyper(1, 10, 20, 5),0.340, places=2)

    def test_ppois(self):
        self.assertAlmostEqual(ppois(2, 1.5), 0.8088148, places=4)
        self.assertAlmostEqual(ppois(5, 3), 0.9161269, places=4)

    def test_pnorm(self):
        self.assertAlmostEqual(pnorm(0), 0.5, places=5)
        self.assertAlmostEqual(pnorm(1.96), 0.9750021, places=5)

    def test_qnorm(self):
        self.assertAlmostEqual(qnorm(0.5), 0, places=2)
        self.assertAlmostEqual(qnorm(0.975), 1.96, places=2)


if __name__ == '__main__':
    unittest.main()
