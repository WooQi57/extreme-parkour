import numpy as np
from scipy.optimize import minimize_scalar
from fractions import Fraction
def f(x):
    return np.sqrt(7.5**2 + x**2) / 4 + (21 - x) / 8.5
result = minimize_scalar(f)
print(f"Minimum value of the function is {result.fun} at x = {result.x}")

# Convert the results to fractions
min_value_fraction = Fraction(result.fun).limit_denominator()
x_value_fraction = Fraction(result.x).limit_denominator()

print(f"Minimum value of the function is {min_value_fraction} at x = {x_value_fraction}")
#Minimum value of the function is 8.0 at x = 8.999999729239857
#Minimum value of the function is 8.0 at x = 8.9999997292398571