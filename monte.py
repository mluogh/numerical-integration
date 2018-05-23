import numpy as np
import math
import multiprocessing

def sample_in_process(fn, lows, highs, n):
    CHUNK_SIZE = 1000

    np.random.seed()

    count = 0

    # max out the n count so it doesnt allocate a ton of memory and die
    num_chunks = n // CHUNK_SIZE

    for i in range(num_chunks):
        points = np.random.uniform(lows, highs, size=(CHUNK_SIZE, len(highs)))
        count += sum(map(fn, points))

    points = np.random.uniform(lows, highs, size=(n % CHUNK_SIZE, len(highs)))
    count += sum(map(fn, points))

    return count

def integrate(fn, intervals, n=1000):
    ''' Integrate a given function in a bounded interval

    Arguments:
    fn - actual function f: R^n -> R, should take np.array as sole argument
    intervals - list of tuples representing intervals [a, b] for each dimension
    n - number of samples to take '''

    num_cores = multiprocessing.cpu_count()
    # change n to something that is easily split up by processes
    # makes process code a little cleaner

    samples_per_core = n // num_cores
    rectified_n = samples_per_core * num_cores

    c = 1 / rectified_n

    lows = []
    highs = []

    for i in intervals:
        # store low, high for later random generation
        lows.append(i[0])
        highs.append(i[1])

        c *= i[1] - i[0]

    # launch some processes to sample

    p = multiprocessing.Pool(num_cores)

    results = [p.apply_async(sample_in_process, args=(fn, lows, highs, samples_per_core)) for i in range(num_cores)] 
    p.close()
    p.join()

    return sum([r.get() for r in results]) * c
