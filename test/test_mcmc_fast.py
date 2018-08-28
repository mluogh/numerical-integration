import unittest
import numpy as np
import sys
import math
import sampling
import integration

class TestSampling(unittest.TestCase):

    def test_values_possible(self):
        np.random.seed(0)
        # test that it generates values that have non-zero density
        n = 10000
        init_x = np.random.rand(1)

        fn = lambda x: 5 * math.exp(-x) if x >= 0 else 0
        proposal = lambda x: np.random.normal(loc=x)
        acceptance = lambda x, x_p: min(1.0, (fn(x_p) / fn(x)) * (density(x, x_p) / density(x_p, x)))
        density = lambda x, x_p: (1 / math.pow(2 * math.pi, len(x)/2)) * math.exp(-((x_p - x).T @ (x_p - x) / 2))

        vals = sampling._metropolis_hastings(
                init_x,
                proposal,
                acceptance,
                n,
                1000,
                1)

        vals = list(vals)
        self.assertTrue(min(vals) >= 0)

    def test_exp_plausible(self):
        np.random.seed(0)
        # generate exp of param 1
        # test that about (1-e^-1) of the values are below 1
        n = 100000
        init_x = np.random.rand(1)

        fn = lambda x: 5 * math.exp(-x) if x[0] >= 0 else 0
        proposal = lambda x: np.random.normal(loc=x)
        acceptance = lambda x, x_p: min(1.0, (fn(x_p) / fn(x)) * (density(x, x_p) / density(x_p, x)))
        density = lambda x, x_p: (1 / math.pow(2 * math.pi, len(x)/2)) * math.exp(-((x_p - x).T @ (x_p - x) / 2))

        vals = sampling._metropolis_hastings(
                init_x,
                proposal,
                acceptance,
                n,
                1000,
                1)

        vals = np.array(list(vals))
        
        below = vals[np.where(vals < 1)]

        p = len(below) / n

        actual = 1 - math.exp(-1)
        self.assertTrue(abs(p - actual) < 0.01)

class TestIntegration(unittest.TestCase):
    def test_proportional_same(self):
        # function to integrat is literally the same
        # so it should be really good
        np.random.seed(0)
        n = 150000
        
        val = integration.importance_sample(
                lambda x: 83 * math.exp(-x[0]),
                lambda x: math.exp(-x[0]) if x[0] >=0 else 0,
                lambda: np.random.rand(1),
                [(0, 1)],
                n=n,
                burn_in=20000)

        actual = 83 * (math.e - 1) / math.e
        print(val)
        self.assertTrue(abs((val - actual) / actual) < 0.01)

    def test_proportional_different(self):
        # function to integrate is not same
        # should be a little worse
        np.random.seed(0)
        n = 100000
        
        val = integration.importance_sample(
                lambda x: 83 * math.exp(-x[0]),
                lambda x: (1/math.sqrt(2 * math.pi)) * math.exp(-((x[0]) ** 2 / 2)),
                lambda: np.random.rand(1),
                [(0, 1)],
                n=n,
                burn_in=10000)

        actual = 83 * (math.e - 1) / math.e
        self.assertTrue(abs((val - actual) / actual) < 0.01)

    def test_multivariate(self):
        np.random.seed(0)
        n = 1000000
        d = 2

        val = integration.importance_sample(
                lambda x: x[0] * x[1] + x[0] + x[1],
                lambda x: (1 / math.pow(2 * math.pi, d/2)) * math.exp(-(x.T @ x) / 2),
                lambda: np.random.rand(2),
                [(0, 1), (0, 1)],
                n=n,
                burn_in=10000)

        actual = 1.25

        # more lenient since cant 1000000 takes forever and reseeding problem with numpy
        self.assertTrue(abs((val - actual) / actual) < 0.05)

if __name__ == '__main__':
    sys.unittesting = True

    unittest.main()
