import random
import torch
import numpy as np
from tsaug import Crop, Drift, Reverse, AddNoise, Resize

max_magnitude = 103

class TimeSeriesAutoAugment(object):
    def __init__(self):
        self.policies = []
        for _ in range(10):
            policy = []
            for _ in range(6):
                policy.append(random.choice(list(operations.keys())))
                policy.append(random.random())
                policy.append(random.random())
            self.policies.append(policy)
                
        #save in a txt file
        with open('policies.txt', 'w') as f:
            for policy in self.policies:
                f.write(str(policy) + '\n')
                
    def __call__(self, series):
        n = series.size(1)
        series_XY = [np.linspace(1, n, n), series.squeeze(0).numpy()]
        series_XY = apply_policy(series_XY, self.policies[random.randrange(len(self.policies))])
        series =  torch.tensor(series_XY[1], dtype=torch.float32).unsqueeze(0)
        return series

operations = {
    'Crop' : lambda series, magnitude: crop(series, magnitude),
    'Drift' : lambda series, magnitude: drift(series, magnitude),
    'AddNoise' : lambda series, magnitude: add_noise(series, magnitude),
    'Reverse' : lambda series, magnitude: reverse(series, magnitude)
}

def apply_policy(series_XY, policy):
    if random.random() < policy[1]:
        series_XY = operations[policy[0]](series_XY, policy[2])
    if random.random() < policy[4]:
        series_XY = operations[policy[3]](series_XY, policy[5])
    return series_XY

def crop(series_XY, magnitude):
    max_size = len(series_XY[0])
    min_size = 0
    magnitude = round(min_size + magnitude * (max_size - min_size))
    X = series_XY[0]
    Y = series_XY[1]
    croped = Crop(size=magnitude).augment(X, Y)
    return Resize(size=len(X)).augment(croped[0], croped[1])

def drift(series_XY, magnitude):
    max_max_drift = 1
    min_max_drift = 0
    magnitude = min_max_drift + magnitude * (max_max_drift - min_max_drift)
    X = series_XY[0]
    Y = series_XY[1]
    return Drift(max_drift=magnitude).augment(X, Y)

def add_noise(series_XY, magnitude):
    max_scale = 10
    min_scale = 0 
    magnitude = min_scale + magnitude * (max_scale - min_scale)
    X = series_XY[0]
    Y = series_XY[1]
    return AddNoise(scale=magnitude).augment(X, Y)

def reverse(series_XY, magnitude):
    X = series_XY[0]
    Y = series_XY[1]
    return Reverse().augment(X, Y)