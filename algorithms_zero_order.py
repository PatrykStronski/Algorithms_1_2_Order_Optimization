from typing import Callable
from formulas import least_squares_custom
from random import random
import math

GOLDEN_RATIO = (3 - math.sqrt(5)) * 0.5

def exhaustive_search(approx: Callable, start: (float, float), end: (float, float), precision: float) -> (float, (float, float), int, int):
    division_number_a = int((end[0] - start[0])/precision)
    division_number_b = int((end[1] - start[1])/precision)
    min = start
    a = start[0]
    b = start[1]
    min_val = float('inf')
    func_calc = 0
    iter_calc = 0
    for ind_a in range(0, division_number_a):
        a += ind_a * precision
        b = start[1]
        for ind_b in range(0, division_number_b):
            iter_calc += 1
            b += ind_b * precision
            val = least_squares_custom(approx, a, b)
            func_calc += 1
            if min_val > val:
                min_val = val
                min = (a, b)
    return (min_val, min, func_calc, iter_calc)

def gauss_search(approx: Callable, start: (float, float), end: (float, float), precision: float) -> (float, (float, float), int, int):
    a = start[0]
    b = start[1]
    prev = (float('inf'), float('inf'))
    min = (float('inf'), float('inf'))
    min_val = float('inf')
    prev_val = 0.0
    func_calc = 0
    iter_calc = 0
    terminate = False

    while abs(prev_val - min_val) > precision:
        a1 = start[0] + GOLDEN_RATIO * (end[0] - start[0])
        a2 = end[0] - GOLDEN_RATIO * (end[0] - start[0])
        a_start = start[0]
        a_end = end[0]
        val1 = least_squares_custom(approx, a1, b)
        val2 = least_squares_custom(approx, a2, b)
        func_calc += 2
        while abs(a_start - a_end) > precision:
            iter_calc += 1
            if val1 <= val2:
                a_end = a2
                a2 = a1
                val2 = val1
                a1 = a_start + GOLDEN_RATIO * (a_end - a_start)
                val1 = least_squares_custom(approx, a1, b)
            else:
                a_start = a1
                a1 = a2
                val1 = val2
                a2 = a_end - GOLDEN_RATIO * (a_end - a_start)
                val2 = least_squares_custom(approx, a2, b)
            func_calc += 1

        if val1 < val2:
            prev = (min[0], min[1])
            min = (a1, b)
            prev_val = min_val
            min_val = val1
            a = a1
        else:
            prev = (min[0], min[1])
            min = (a2, b)
            prev_val = min_val
            min_val = val2
            a = a2

        b1 = start[1] + GOLDEN_RATIO * (end[1] - start[1])
        b2 = end[1] - GOLDEN_RATIO * (end[1] - start[1])
        b_start = start[1]
        b_end = end[1]
        val1 = least_squares_custom(approx, a, b1)
        val2 = least_squares_custom(approx, a, b2)
        func_calc += 2
        while abs(b_start - b_end) > precision:
            iter_calc += 1
            if val1 <= val2:
                b_end = b2
                b2 = b1
                val2 = val1
                b1 = b_start + GOLDEN_RATIO * (b_end - b_start)
                val1 = least_squares_custom(approx, a, b1)
            else:
                b_start = b1
                b1 = b2
                val1 = val2
                b2 = b_end - GOLDEN_RATIO * (b_end - b_start)
                val2 = least_squares_custom(approx, a, b2)
            func_calc += 1

        if val1 < val2:
            prev = (min[0], min[1])
            min = (a, b1)
            prev_val = min_val
            min_val = val1
            b = b1
        else:
            prev = (min[0], min[1])
            min = (a, b2)
            prev_val = min_val
            min_val = val2
            b = b2

        if abs(prev[0]-min[0]) > precision and abs(prev[1]-min[1]) > precision:
            return (min_val, min, func_calc, iter_calc)
        
    return (min_val, min, func_calc, iter_calc)

def sort_points(p1: (float, (float, float)), p2:(float, (float, float)), p3:(float, (float, float))) -> ((float, (float, float)), (float, (float, float)), (float, (float, float)), int):
    p_h = ()
    p_g = ()
    p_l = ()
    l = 0
    if p1[0] > p2[0]:
        if p1[0] > p3[0]:
            p_h = p1
            if p2[0] > p3[0]:
                p_g = p2
                p_l = p3
                l = 3
            else:
                p_g = p3
                p_l = p2
                l= 2
        else:
            p_h = p3
            p_g = p1
            p_l = p2
            l = 2
    else:
        if p2[0] > p3[0]:
            p_h = p2
            if p1[0] > p3[0]:
                p_g = p1
                p_l = p3
                l = 3
            else:
                p_l = p1
                p_g = p3
                l = 1
        else:
            p_h = p3
            p_g = p2
            p_l = p1
            l = 1
    return (p_h, p_g, p_l, l)

def points_in_precision(x1: (float, float), x2: (float, float), x3: (float, float), precision: float) -> bool:
    return abs(x1[0] - x2[0]) > precision and abs(x1[1] - x2[1]) > precision and abs(x2[0] - x3[0]) > precision and abs(x2[1] - x3[1]) > precision and abs(x1[0] - x3[0]) > precision and abs(x1[1] - x3[1]) > precision

def reflect_point(xh: (float, float), xc: (float, float), alpha: float) -> (float, float):
    return (1 + alpha) * xc[0] - alpha * xh[0], (1 + alpha) * xc[1] - alpha * xh[1]

def calc_e(xc: (float, float), xr: (float, float), gamma: float) -> (float, float):
    return (1 - gamma) * xc[0] + gamma * xr[0], (1 - gamma) * xc[1] + gamma * xr[1]

def nelder_mead(approx: Callable, start: (float, float), end: (float, float), precision: float) -> (float, (float, float), int, int):
    alpha = 1.0
    beta = 0.5
    gamma = 2

    x1 = start
    x2 = end
    x3 = (end[0]/2, end[1]/2)

    f1 = least_squares_custom(approx, x1[0], x1[1])
    f2 = least_squares_custom(approx, x2[0], x2[1])
    f3 = least_squares_custom(approx, x3[0], x3[1])
    
    func_calc = 3
    iter_calc = 0

    while points_in_precision(x1, x2, x3, precision):
        iter_calc += 1

        x1_pre, x2_pre, x3_pre = x1, x2, x3
        (f1, x1), (f2, x2), (f3, x3), l = sort_points((f1, x1), (f2, x2), (f3, x3))
        
        xc = ((x2[0] + x3[0])/2, (x2[1] + x3[1])/2)

        xr = reflect_point(x1, xc, alpha)
        fr = least_squares_custom(approx, xr[0], xr[1])

        func_calc += 1

        if fr < f3:
            xe = calc_e(xc, xr, gamma)
            fe = least_squares_custom(approx, xe[0], xe[1])
            func_calc += 1

            if fe < fr:
                f1, x1 = (fe, xe)
            elif fr < fe:
                f1, x1 = fr, xr
        elif f3 < fr < f2:
            f1, x1 = fr, xr
        elif f2 < fr < f1 or f1 < fr:
            if f2 < fr < f1:
                tmp = (fr, xr)
                fr, xr = f1, x1
                f1, x1 = tmp

            xs = (beta * x1[0] + (1 - beta) * xc[0], beta * x1[1] + (1 - beta) * xc[1])
            fs = least_squares_custom(approx, xs[0], xs[1])

            if fs < f1: 
                f1, x1 = (fs, xs)
            else: 
                if l == 3:
                    x1 = (x3[0] + (x1_pre[0] - x3[0])/2, x3[1] + (x1_pre[1] - x3[1])/2)
                    x2 = (x3[0] + (x2_pre[0] - x3[0])/2, x3[1] + (x2_pre[1] - x3[1])/2)
                    f1 = least_squares_custom(approx, x1[0], x1[1])
                    f2 = least_squares_custom(approx, x2[0], x2[1])
                elif l == 2:
                    x1 = (x3[0] + (x1_pre[0] - x3[0])/2, x3[1] + (x1_pre[1] - x3[1])/2)
                    x3 = (x3[0] + (x3_pre[0] - x3[0])/2, x3[1] + (x3_pre[1] - x3[1])/2)
                    f1 = least_squares_custom(approx, x1[0], x1[1])
                    f3 = least_squares_custom(approx, x3[0], x3[1])
                elif l == 1:
                    x2 = (x3[0] + (x2_pre[0] - x3[0])/2, x3[1] + (x2_pre[1] - x3[1])/2)
                    x3 = (x3[0] + (x3_pre[0] - x3[0])/2, x3[1] + (x3_pre[1] - x3[1])/2)
                    f2 = least_squares_custom(approx, x2[0], x2[1])
                    f3 = least_squares_custom(approx, x3[0], x3[1])
                func_calc += 2

    (f1, x1), (f2, x2), (f3, x3), l = sort_points((f1, x1), (f2, x2), (f3, x3))
    return (f3, x3, func_calc, iter_calc)
