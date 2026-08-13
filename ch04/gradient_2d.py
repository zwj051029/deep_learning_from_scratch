import numpy as np
import matplotlib.pyplot as plt

def _numerical_gradient_no_batch(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    for index in range(x.size):
        temp = x[index]

        x[index] = temp + h
        fxh1 = f(x)
        x[index] = temp - h
        fxh2 = f(x)

        grad[index] = (fxh1 - fxh2) / (2 * h)
        x[index] = temp

    return grad

def numerical_gradient(f, X):
    if X.ndim == 1:
        return _numerical_gradient_no_batch(f, X)
    else:
        grad = np.zeros_like(X)

        for index, x in enumerate(X):
            grad[index] = _numerical_gradient_no_batch(f, x)

        return grad

def function_2(x):
    if x.ndim == 1:
        return np.sum(x**2)
    else:
        return np.sum(x**2, axis=1)

def tangent_line(f, x):
    k = numerical_gradient(f, x)
    print(k)
    b = f(x) - k*x
    return lambda t: k*t + b

def main():
    x0 = np.arange(-2, 2.5, 0.25)
    x1 = np.arange(-2, 2.5, 0.25)

    X, Y = np.meshgrid(x0, x1)
    X = X.flatten()
    Y = Y.flatten()

    points = np.array([X, Y]).T
    grad = numerical_gradient(function_2, points)

    plt.figure()
    plt.quiver(
        X,
        Y,
        -grad[:, 0],
        -grad[:, 1],
        angles="xy",
        color="#A01212",
    )

    plt.xlim([-2, 2])
    plt.ylim([-2, 2])
    plt.xlabel("x0")
    plt.ylabel("x1")
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()