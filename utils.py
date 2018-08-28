import numpy as np
import math
import sys
import os
import numbers

def fn_limit_wrapper(fn, limits):
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

def build_limit_fns(limits):
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

