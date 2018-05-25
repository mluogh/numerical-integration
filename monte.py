import numpy as np
import math
from pathos import multiprocessing
import sys
import os
import numbers

def _hypercube_sample(fn, lows, highs, n=1000):
    ''' Sample uniformly across a hypercube

    Arguments:
    fn - actual function f: R^n -> R, should take np.array as sole argument
    lows, highs 
    n - number of samples to take '''

    # want deterministic seeding for unittesting
    if 'UNITTESTING' not in os.environ:
        np.random.seed()
        
    CHUNK_SIZE = 1000

    count = 0

    # max out the n count so it doesnt allocate a ton of memory and die
    num_chunks = n // CHUNK_SIZE

    for i in range(num_chunks):
        points = np.random.uniform(lows, highs, size=(CHUNK_SIZE, len(highs)))
        count += sum(map(fn, points))

    points = np.random.uniform(lows, highs, size=(n % CHUNK_SIZE, len(highs)))
    count += sum(map(fn, points))

    return count

def _fn_limit_wrapper(fn, limits):
    def altered_fn(x):
        sign = 1
        for i in range(len(limits)):
            a, b = limits[i][0](x), limits[i][1](x)

            # basically even number of integrals with a > b => positive
            # odd => negative
            # this is from the definition of an integral
            # going backwards is basically -dx
            if a > b:
                sign *= -1

            if x[i] < min(a, b) or x[i] > max(a, b):
                return 0
        
        return fn(x) * sign

    return altered_fn

def _build_limit_fns(limits):
    new_limits = []
    for l in limits:
        a = l[0] if callable(l[0]) else lambda x, n=l[0]: n
        b = l[1] if callable(l[1]) else lambda x, n=l[1]: n
        new_limits.append((a, b))
    return new_limits

def get_cube_info(cube):
    lows, highs, volume = [], [], 1
    for i in cube:
        # store low, high for later random generation
        assert isinstance(i[0], numbers.Real) and isinstance(i[1], numbers.Real), 'if box is not provided, then limits must be real numbers (no functions)'

        lows.append(i[0])
        highs.append(i[1])

        volume *= i[1] - i[0]
    
    return lows, highs, volume

def integrate_hypercube(fn, limits, cube=None, n=1000):
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

    lows, highs, volume = get_cube_info(cube)
    c *= volume

    if not is_hypercube:
        limits = _build_limit_fns(limits)
        # exotic domains
        # get unsigned volume (mc part signs it according to integration rules and limits)
        c = abs(c)
        fn = _fn_limit_wrapper(fn, limits)

    # launch some processes to sample

    p = multiprocessing.Pool(num_cores)

    results = [p.apply_async(_hypercube_sample, args=(fn, lows, highs, samples_per_core)) for i in range(num_cores)] 
    p.close()
    p.join()

    return sum([r.get() for r in results]) * c

def metropolis_hastings(fn, init_fn, proposal_fn, proposal_density, n, burn_in=1000, skip=1):

    def acceptance(x, x_p):
        return min(1, (fn(x_p) / fn(x)) * (proposal_density(x, x_p)/ proposal_density(x_p, x)))

    # pick initial state - user must care to pick prior s.t. always state with non-zero density
    x = init_fn()

    for i in range(n * skip + burn_in):
        x_p = proposal_fn(x)

        a = acceptance(x, x_p)

        if np.random.rand() < a:
            x = x_p

        if i >= burn_in and (i - burn_in) % skip == 0:
            yield x

def importance_sample(integral_fn, proportional_fn, init_fn, limits, box=None, n=10000, proposal_fn=None, proposal_density=None, burn_in=1000, skip=1):

    # default proposal and densities are unit gaussian
    if proposal_fn == None:
        proposal_fn = lambda x: np.random.normal(loc=x)
        proposal_density = lambda x, x_p: math.exp(-((x_p - x) ** 2 / 2))

    limits = _build_limit_fns(limits)
    integral_fn = _fn_limit_wrapper(integral_fn, limits)

    simulated = metropolis_hastings(proportional_fn, init_fn, proposal_fn, proposal_density, n, burn_in=burn_in, skip=skip)
    
    samples = (integral_fn(s)/proportional_fn(s) for s in simulated)

    return sum(samples) / n
