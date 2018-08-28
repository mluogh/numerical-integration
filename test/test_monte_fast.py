import unittest
import numpy as np
import sys
import sampling
import integration

class TestSampling(unittest.TestCase):

    def test_simple_fn(self):
        np.random.seed(0)
        n = 100000
        val = sampling._sample_hypercube(lambda x: x[0], [0.0], [1.0], n)
        val /= n

        # sampling identity from 0 to 1 should average out to about 0.5
        self.assertTrue(abs(val - 0.5) < 0.01)

    def test_simple_fn_improper(self):
        np.random.seed(0)
        # should also work if low > high
        n = 100000
        val = sampling._sample_hypercube(lambda x: x[0], [1.0], [0.0], n)
        val /= n

        self.assertTrue(abs(val - 0.5) < 0.01)

class TestIntegration(unittest.TestCase):
    
    def test_single_dimension(self):
        np.random.seed(0)
        # x^2 from -1 to 1 which is 2/3
        actual = 2/3
        val = integration.integrate(lambda x: x[0] * x[0], [(-1, 1)], n=1000000)

        self.assertTrue(abs(val - actual) < 0.01)

    def test_single_dimension_reversed(self):
        np.random.seed(0)
        # x^2 from -1 to 1 which is -2/3
        actual = -2/3
        val = integration.integrate(lambda x: x[0] * x[0], [(1, -1)], n=1000000)

        self.assertTrue(abs(val - actual) < 0.01)

    def test_multiple_dimensions(self):
        np.random.seed(0)
        n = 1000000
        # x + y^2 + x + y
        actual = 2425 / 6
        val = integration.integrate(lambda x: x[0] * x[1] * x[1] + x[0] + x[1], [(-2, 3), (2, 7)], n=n)

        self.assertTrue(abs((val - actual) / actual) < 0.01)

    def test_function_as_limit(self):
        np.random.seed(0)
        # integrate x + y from x=0 to 1 and y = x to 1-x
        actual = -1/6
        val = integration.integrate(
                lambda x: x[0] + x[1],
                [(0, 1), (lambda x: x[0], lambda x: 1 - x[0])],
                cube = [(0, 1), (0, 1)],
                n = 2000000)

        self.assertTrue(abs((val - actual) / actual) < 0.01)

    def test_function_as_limit_reversed(self):
        np.random.seed(0)
        # integrate x + y from x= 1 to 0 and y = x to 1-x
        # cube ordering shouldnt matter for this!
        actual = 1/6
        val = integration.integrate(
                lambda x: x[0] + x[1],
                [(1, 0), (lambda x: x[0], lambda x: 1 - x[0])],
                cube = [(1, 0), (1, 0)],
                n = 2000000)

        self.assertTrue(abs((val - actual) / actual) < 0.01)

    def test_function_as_limit_double_reversed(self):
        np.random.seed(0)
        # integrate x + y from x=1 to 0 and y = 1-x to x
        # cube ordering shouldnt matter for this!
        actual = -1/6
        val = integration.integrate(
                lambda x: x[0] + x[1],
                [(1, 0), (lambda x: 1 - x[0], lambda x: x[0])],
                cube = [(0, 1), (1, 0)],
                n = 2000000)

        self.assertTrue(abs((val - actual) / actual) < 0.01)
        
if __name__ == '__main__':
    sys.unittesting = True

    unittest.main()
