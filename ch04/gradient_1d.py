import numpy as np
import matplotlib.pyplot as plt

def numerical_diff(f, x):
    h = 1e-4
    return (f(x+h) - f(x-h)) / (2*h)

def function_1(x):
    return 0.01*x**2 + 0.1*x

def tangent_line(f, x):
    k = numerical_diff(f, x)
    print(k)
    b = f(x) - k*x
    return lambda t: k*t + b

x = np.arange(0.0, 20.0, 0.1)
y1 = function_1(x)
tf = tangent_line(function_1, 5)
y2 = tf(x)

plt.plot(x, y1)
plt.plot(x, y2)

plt.xlabel("x")
plt.ylabel("f(x)")
plt.show()