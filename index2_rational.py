import matplotlib.pyplot as plt

from algorithms_zero_order import exhaustive_search, gauss_search, nelder_mead
from algorithms_2d import gradient_descent, conjugate_gradient_descent, newton_algorithm, levenberg_marquadt
from formulas import least_squares_rat, rational_approximation, rational_approximation, NOISY_DATA, X_ES, rat_to_approximate

PRECISION = 0.001
START = (0.0, 0.0)
END = (1.0, 1.0)

plt.scatter(X_ES, NOISY_DATA, label='Y noisy data')

print('Rational approximant')

answ = exhaustive_search(rational_approximation, START, END, PRECISION)
print(f"Exhaustive Search: result={answ[0]} a={answ[1][0]}; b={answ[1][1]};function_calculation={answ[2]}; iterations={answ[3]}")

plt.plot(X_ES, [rational_approximation(x, answ[1][0], answ[1][1]) for x in X_ES], label='exhaustive search')

answ = gauss_search(rational_approximation, START, END, PRECISION)
print(f"Gauss Search: result={answ[0]} a={answ[1][0]}; b={answ[1][1]};function_calculation={answ[2]}; iterations={answ[3]}")

plt.plot(X_ES, [rational_approximation(x, answ[1][0], answ[1][1]) for x in X_ES], label='Gauss coordinate descent')

answ = nelder_mead(rational_approximation, START, END, PRECISION)
print(f"Nelder-Mead Search: result={answ[0]} a={answ[1][0]}; b={answ[1][1]};function_calculation={answ[2]}; iterations={answ[3]}")

plt.plot(X_ES, [rational_approximation(x, answ[1][0], answ[1][1]) for x in X_ES], label='Nelder-Mead method')

answ = gradient_descent(rational_approximation, START, END, PRECISION)
print(f"Gradient Descent: result={answ[0]} a={answ[1][0]}; b={answ[1][1]};function_calculation={answ[2]}; iterations={answ[3]}")
plt.plot(X_ES, [rational_approximation(x, answ[1][0], answ[1][1]) for x in X_ES], label='Gradient Descent', linewidth=4)

answ = conjugate_gradient_descent(least_squares_rat, START, END, PRECISION)
print(f"Conjugate Gradient Descent: result={answ[0]} a={answ[1][0]}; b={answ[1][1]};function_calculation={answ[2]}; iterations={answ[3]}")
plt.plot(X_ES, [rational_approximation(x, answ[1][0], answ[1][1]) for x in X_ES], label='Conjugate Gradient Descent', linewidth=3)

answ = newton_algorithm(least_squares_rat, START, END, PRECISION)
print(f"Newton Algorithm: result={answ[0]} a={answ[1][0]}; b={answ[1][1]};function_calculation={answ[2]}; iterations={answ[3]}")
plt.plot(X_ES, [rational_approximation(x, answ[1][0], answ[1][1]) for x in X_ES], label='Newton Algorithm', linewidth=2)

answ = levenberg_marquadt(rat_to_approximate, START, END, PRECISION)
print(f"Levenberg-Marquadt: result={answ[0]} a={answ[1][0]}; b={answ[1][1]};function_calculation={answ[2]}; iterations={answ[3]}")
plt.plot(X_ES, [rational_approximation(x, answ[1][0], answ[1][1]) for x in X_ES], label='Levenberg-Marquadt', linewidth=1)

plt.xlabel('X-value')
plt.ylabel('Y-value')
plt.legend()
plt.grid(True)
plt.title(f"Data and its rational approximation with precision {PRECISION}")
plt.show()