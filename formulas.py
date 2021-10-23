import math
import numpy as np
from typing import Callable
from random import random
from scipy.misc import derivative

DELTA_NOISE =  np.random.randn(100)
alpha = random()
beta = random()

"""noise generator from 2nd task"""
def gen_noisy_data(k: int) -> float:
    delta = DELTA_NOISE[k]
    return alpha * k / 100 + beta + delta

"""Linear approximant of the function from the first task"""
def linear_approximation(x: float, a: float, b: float) -> float:
    return x*a + b

"""Rational approximant of the function from the first task"""
def rational_approximation(x: float, a: float, b: float) -> float:
    return a / (1 + b * x)

NOISY_DATA = [] 
X_ES = []

for k in range(0,100):
    NOISY_DATA.append(gen_noisy_data(k))
    X_ES.append(k/100)

"""Least squares method that for function func calculates lest square sum"""
def least_squares_custom(func: Callable, a: float, b: float) -> float:
    sum = 0
    for k in range(0,100):
        y = NOISY_DATA[k]
        x = X_ES[k]
        sum += (func(x, a, b) - y) **2
    return sum

def least_squares_lin(point: (float, float)) -> float:
    return least_squares_custom(linear_approximation, point[0], point[1])

def least_squares_rat(point: (float, float)) -> float:
    return least_squares_custom(rational_approximation, point[0], point[1])

def least_squares_lin_gradient_norm(point: (float, float)) -> float:
    return math.sqrt(least_squares_lin_der_a(point) ** 2 + least_squares_lin_der_b(point) ** 2)

def lin_to_approximate(point: (float, float)) -> [float]:
    return [linear_approximation(X_ES[k], point[0], point[1]) - NOISY_DATA[k] for k in range(0, len(X_ES))]

def rat_to_approximate(point: (float, float)) -> [float]:
    return [rational_approximation(X_ES[k], point[0], point[1]) - NOISY_DATA[k] for k in range(0, len(X_ES))]

def least_squares_gradient(func: Callable, point: (float, float)) -> (float, float):
    der_a = derivative(lambda a: least_squares_custom(func, a, point[1]), point[0])
    der_b = derivative(lambda b: least_squares_custom(func, point[0], b), point[1])
    return (der_a, der_b)