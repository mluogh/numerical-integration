import unittest
import numpy as np
import sys
import monte
import math

class TestSampling(unittest.TestCase):

    def test_values_possible(self):
        np.random.seed(0)
        # test that it generates values that have non-zero density
        n = 10000
        vals = monte.metropolis_hastings(
                lambda x: 5 * math.exp(-x) if x >= 0 else 0,
                lambda: np.random.rand(),
                lambda x: np.random.normal(loc=x),
                lambda x, x_p: math.exp(-((x_p - x) ** 2 / 2)),
                n)

        vals = list(vals)
        self.assertTrue(min(vals) >= 0)

    def test_exp_plausible(self):
        np.random.seed(0)
        # generate exp of param 1
        # test that about (1-e^-1) of the values are below 1
        n = 100000
        vals = monte.metropolis_hastings(
                lambda x: 234.6432 * math.exp(-x) if x >= 0 else 0,
                lambda: np.random.rand(),
                lambda x: np.random.normal(loc=x),
                lambda x, x_p: math.exp(-((x_p - x) ** 2 / 2)),
                n,
                burn_in=10000,
                skip=3)

        vals = np.array(list(vals))
        
        below = vals[np.where(vals < 1)]

        p = len(below) / n

        actual = 1 - math.exp(-1)
        self.assertTrue(abs(p - actual) < 0.01)

class TestIntegration(unittest.TestCase):
    def test_same(self):
        np.random.seed(0)
        n = 100000
        
        val = monte.importance_sample(
                lambda x: 83 * math.exp(-x),
                lambda x: math.exp(-x) if x >=0 else 0,
                lambda: np.random.rand(1, 1),
                [(0, 1)],
                n=n)

        actual = 83 * (math.e - 1) / math.e
        self.assertTrue(abs((val - actual) / actual) < 0.01)

    def test_different(self):
        np.random.seed(0)
        n = 100000
        
        val = monte.importance_sample(
                lambda x: 83 * math.exp(-x),
                lambda x: (1/math.sqrt(2 * math.pi)) * math.exp(-((x) ** 2 / 2)),
                lambda: np.random.rand(1),
                [(0, 1)],
                n=n,
                burn_in=10000)

        actual = 83 * (math.e - 1) / math.e
        self.assertTrue(abs((val - actual) / actual) < 0.01)

if __name__ == '__main__':
    sys.unittesting = True

    unittest.main()
