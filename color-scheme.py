import numpy as np
import numpy.linalg as lin
import sys
import itertools as it
import random

import npnn

data = []

for line in sys.stdin:
    line.strip()
    data.append(np.array([[int(x, 16)/(1<<16)] for x in [line[:4], line[4:8], line[8:12]]]))

network = [
    np.random.random((3, 4)),
    np.random.random((4, 4)),
    np.random.random((4, 2)),
    np.random.random((2, 4)),
    np.random.random((4, 4)),
    np.random.random((4, 3)),
]

observations = [
    [d.T, d.T] for d in data
]

net1 = npnn.train(network, observations, np.tanh, npnn.dtanh, limit=200000, learning_rate=0.5)
final = npnn.train(net1, observations, np.tanh, npnn.dtanh, limit=200000, learning_rate=0.1)

for i in range(10):
    d = npnn.choice(data)
    outs, acts = npnn.feed_forward(final, d.T, np.tanh)
    print(d.T, outs, acts[3], file=sys.stderr)

print("6661726266656c64")
print("000001f4000001f4")

half_net = final[3:]

def clip(x):
    if x < 0: return 0
    if x > 1: return 1
    return x

for i in range(500):
    for j in range(500):
        out, acts = npnn.feed_forward(half_net, [i/500*.8, j/500*.8], np.tanh)
        print("".join(["{:04x}".format(int(clip(o)*(1<<16))) for o in out]) + "ffff")
