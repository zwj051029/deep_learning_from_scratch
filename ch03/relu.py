import numpy as np
import matplotlib.pyplot as plt

def relu(x):
    return np.maximum(x, 0)

def main():
    x = np.arange(-5.0, 5.0, 0.1)
    y = relu(x)

    plt.plot(x, y)
    plt.show()

if __name__ == "__main__":
    main()