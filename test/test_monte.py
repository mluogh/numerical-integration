import unittest
import numpy as np
import sys
import monte

np.random.seed(0)

class TestSampling(unittest.TestCase):

    def test_simple_fn(self):
        n = monte._hypercube_sample(lambda x: x, [0], [1], 10000)
        n /= 10000

        # sampling identity from 0 to 1 should average out to about 0.5
        self.assertTrue(n > 0.49 and n < 0.51)

    def test_simple_fn_improper(self):
        # should also work if low > high
        n = monte._hypercube_sample(lambda x: x, [1], [0], 10000)
        n /= 10000

        self.assertTrue(n > 0.49 and n < 0.51)

class TestIntegration(unittest.TestCase):
    
    def test_single_dimension(self):
        # x^2 from -1 to 1 which is 2/3
        n = monte.integrate_hypercube(lambda x: x[0] * x[0], [(-1, 1)], 1000000)

        self.assertTrue(n > 0.66 and n < 0.67)

    def test_single_dimension_reversed(self):
        # x^2 from -1 to 1 which is 2/3
        n = monte.integrate_hypercube(lambda x: x[0] * x[0], [(1, -1)], 1000000)

        self.assertTrue(n < -0.66 and n > -0.67)

    def test_multiple_dimensions(self):
        # x + y^2 + x + y
        n = monte.integrate_hypercube(lambda x: x[0] * x[1] * x[1] + x[0] + x[1], [(-2, 3), (2, 7)], n=1000000)

        self.assertTrue(abs(n - 2425/6) / (2456/6) < 0.01)


if __name__ == '__main__':
    sys.unittesting = True

    unittest.main()
