import numpy as np
import math
from pathos import multiprocessing
import sys
import os
import numbers
import utils
import sampling

def _hypercube_process_wrapper(fn, lows, highs, n=1000):
    ''' Sample uniformly across a hypercube

    Arguments:
    fn - actual function f: R^n -> R, should take np.array as sole argument
    lows, highs 
    n - number of samples to take '''

    # want deterministic seeding for unittesting
    if 'UNITTESTING' not in os.environ:
        np.random.seed()

    return sampling._sample_hypercube(fn, lows, highs, n) 

def integrate(fn, limits, cube=None, n=1000):
    ''' Integrate a given function in a bounded interval

    Arguments:
    fn - actual function f: R^n -> R, should take np.array as sole argument
    limits - list of tuples representing intervals [a(x), b(x)] for each dimension
    cube - if limits are not simple real numbers, provide a hypercube of form [(a, b), (c, d), ... ] s.t. it contains the domain of integration 
    n - number of samples to take '''
    is_hypercube = cube == None

    num_cores = multiprocessing.cpu_count()
    # change n to something that is easily split up by processes
    # makes process code a little cleaner
    samples_per_core = n // num_cores
    rectified_n = samples_per_core * num_cores

    c = 1 / rectified_n

    lows = []
    highs = []

    # if is hypercube, this means limits are some hypercube, so just set
    if is_hypercube:
        cube = limits

    lows, highs, volume = utils.get_cube_info(cube)
    c *= volume

    if not is_hypercube:
        limits = utils.build_limit_fns(limits)
        # exotic domains
        # get unsigned volume (mc part signs it according to integration rules and limits)
        c = abs(c)
        fn = utils.fn_limit_wrapper(fn, limits)

    # launch some processes to sample

    p = multiprocessing.Pool(num_cores)

    results = [p.apply_async(_hypercube_process_wrapper, args=(fn, lows, highs, samples_per_core)) for i in range(num_cores)] 
    p.close()
    p.join()

    return sum([r.get() for r in results]) * c

# def metropolis_hastings(fn, init_fn, proposal_fn, proposal_density, n, burn_in=1000, skip=1):

    # def acceptance(x, x_p):
        # return min(1, (fn(x_p) / fn(x)) * (proposal_density(x, x_p)/ proposal_density(x_p, x)))

    # # pick initial state - user must care to pick prior s.t. always state with non-zero density
    # x = init_fn()

    # for i in range(n * skip + burn_in):
        # x_p = proposal_fn(x)

        # a = acceptance(x, x_p)

        # if np.random.rand() < a:
            # x = x_p

        # if i >= burn_in and (i - burn_in) % skip == 0:
            # yield x

def _mcmc_process_wrapper(score_fn, *args, **kwargs):
    s = sampling._metropolis_hastings(*args, **kwargs)
    return sum(np.apply_along_axis(score_fn, 1, s))

def importance_sample(integrate_fn, dist_fn, init_fn, limits, n=10000, proposal_fn=None, proposal_density=None, burn_in=1000, skip=1):

    num_cores = multiprocessing.cpu_count()
    # change n to something that is easily split up by processes
    # makes process code a little cleaner
    samples_per_core = n // num_cores
    rectified_n = samples_per_core * num_cores

    d = len(limits)

    # default proposal and densities are unit gaussian
    if proposal_fn == None:
        # multivar normal runs like 10x slower than normal for some reason, so if d == 1, hardcode univariate distribution
        cov = np.diag(np.ones(d))
        if d > 1:
            proposal_fn = lambda x: np.random.multivariate_normal(x, cov)
        else:
            proposal_fn = lambda x: np.random.normal(x)

        proposal_density = lambda x, x_p: (1 / math.pow(2 * math.pi, d/2)) * math.exp(-((x_p - x).T @ (x_p - x) / 2))


    limits = utils.build_limit_fns(limits)
    integrate_fn = utils.fn_limit_wrapper(integrate_fn, limits)

    def acceptance_fn(x, x_p):
        return min(1, (dist_fn(x_p) / dist_fn(x)) * (proposal_density(x, x_p)/ proposal_density(x_p, x)))

    def score_fn(x):
        return integrate_fn(x) / dist_fn(x) 

    initial_x = init_fn()

    # launch some processes to sample
    p = multiprocessing.Pool(num_cores)

    results = [p.apply_async(_mcmc_process_wrapper, args=(score_fn, initial_x, proposal_fn, acceptance_fn, samples_per_core, burn_in, skip)) for i in range(num_cores)] 
    p.close()
    p.join()

    return sum([r.get() for r in results]) / rectified_n
