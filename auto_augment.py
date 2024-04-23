import random
import torch
import numpy as np
from tsaug import Crop, Drift, Reverse, AddNoise, Resize

max_magnitude = 103

class TimeSeriesAutoAugment(object):
    def __init__(self):
        self.policies = [
            ['Crop', 0.23, 0.9, 'AddNoise', 0.29, 0.1],
            ['Drift', 0.4, 0.1, 'Reverse', 0.8, 5],
            ['Crop', 0.8, 0.8, 'AddNoise', 0.99, 0.04],
            ['Crop', 0.6, 0.99, 'Reverse', 0.4, 9],
            ['Crop', 0.79, 0.7, 'Drift', 0.72, 0.3],
            ['Drift', 0.5, 0.5, 'AddNoise', 0.21, 0.2],
            ['Drift', 0.84, 0.9, 'Reverse', 0.94, 6],
            ['Drift', 0.31, 0.4, 'Crop', 0.6, 0.3],
            ['AddNoise', 0.26, 0.1, 'Drift', 0.64, 0.2],
            ['AddNoise', 0.8, 0.3, 'Crop', 0.3, 0.7],
            ['Reverse', 0.62, 0.3, 'Drift', 0.58, 0.4],
            ['Reverse', 0.5, 7, 'Crop', 0.89, 0.96],
            ['Reverse', 0.47, 1, 'AddNoise', 0.7, 0.4],
        ]
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
    X = series_XY[0]
    Y = series_XY[1]
    croped = Crop(size=round(magnitude * len(X))).augment(X, Y)
    return Resize(size=len(X)).augment(croped[0], croped[1])

def drift(series_XY, magnitude):
    X = series_XY[0]
    Y = series_XY[1]
    return Drift(max_drift=magnitude).augment(X, Y)

def add_noise(series_XY, magnitude):
    X = series_XY[0]
    Y = series_XY[1]
    return AddNoise(scale=magnitude).augment(X, Y)

def reverse(series_XY, magnitude):
    X = series_XY[0]
    Y = series_XY[1]
    return Reverse().augment(X, Y)