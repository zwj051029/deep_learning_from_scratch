import numpy as np
import matplotlib.pyplot as plt

def step_function(x):
    return np.array(x > 0, dtype = np.int32)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x = np.arange(-5.0, 5.0, 0.1)
y1 = step_function(x)
y2 = sigmoid(x)

plt.plot(x, y1, label = "step_function")
plt.plot(x, y2, label = "sigmoid", linestyle = "--")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()