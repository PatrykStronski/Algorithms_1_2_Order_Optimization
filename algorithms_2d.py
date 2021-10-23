from typing import Callable
from formulas import least_squares_custom, least_squares_gradient
from random import random
from scipy.optimize import minimize, least_squares, newton

def calculate_learning_rate(func: Callable, point: (float, float), grad: (float, float), point_pre: (float, float)) -> (float, float):
    grad_pre = least_squares_gradient(func, point)
    numerator_a = abs((point[0] - point_pre[0]) * (grad[0] - grad_pre[0]))
    numerator_b = abs((point[1] - point_pre[1]) * (grad[1] - grad_pre[1]))
    denominator = (grad[0] - grad_pre[0]) ** 2 + (grad[1] - grad_pre[1]) ** 2
    return numerator_a / denominator, numerator_b / denominator

def gradient_descent(func: Callable, start: (float, float), end: (float, float), precision: float) -> (float, (float, float), int, int):
    curr_a = start[0]
    curr_b = start[1]

    res_pre = float('inf')
    learning_rate_a = precision
    learning_rate_b = precision
    res_curr = least_squares_custom(func, curr_a, curr_b)

    iter_nmb = 0
    # Performing Gradient Descent
    while(res_curr <= res_pre):
        iter_nmb += 1
        point_pre = (curr_a, curr_b)
        res_pre = res_curr
        grad = least_squares_gradient(func, (curr_a, curr_b))
        curr_a = curr_a - learning_rate_a * grad[0] 
        curr_b = curr_b - learning_rate_b * grad[1] 
        res_curr = least_squares_custom(func, curr_a, curr_b)
        learning_rate_a, learning_rate_b = calculate_learning_rate(func, (curr_a, curr_b), grad, point_pre)

    return (res_pre, point_pre, iter_nmb * 2, iter_nmb)

def conjugate_gradient_descent(func: Callable, start: (float, float), end: (float, float), precision: float) -> (float, (float, float), int, int):
    x0 = [start[0], start[1]]
    ans = minimize(func, x0, method='CG', tol=precision)
    val = func(ans.x)
    return (val, ans.x, ans.nfev ,ans.nit)

def newton_algorithm(func: Callable, start: (float, float), end: (float, float), precision: float) -> (float, (float, float), int, int):
    x0 = [start[0], start[1]]
    ans = minimize(func, x0, method='BFGS', tol=precision)
    val = func(ans.x)
    return (val, ans.x, ans.nfev ,ans.nit)

def levenberg_marquadt(func: Callable, start: (float, float), end: (float, float), precision: float) -> (float, (float, float), int, int):
    x0 = [start[0], start[1]]
    ans = least_squares(func, x0, method='lm', ftol=precision, xtol=precision)
    val = sum(func(ans.x))
    return (val, ans.x, ans.nfev, ans.njev)