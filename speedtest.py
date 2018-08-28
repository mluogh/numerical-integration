import time
import example
import monte
n = 10

speeds = []

for i in range(n):
    t = time.time()
    monte._hypercube_sample(lambda x: x[0] * 2, [1, 2, 3], [4, 5, 6], 10000000)
    speeds.append(time.time() - t)

mean = sum(speeds) / n
var = sum(map(lambda x: (x - mean) ** 2, speeds)) / (n - 1)

print(mean, var)
