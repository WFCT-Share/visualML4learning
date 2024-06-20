from visualML4learning import *
import pandas as pd
import numpy as np

if __name__ == '__main__':
    np.random.seed(0)

    a = 2.0
    b = 1.0

    x = np.linspace(0, 10, 256)

    noise = np.random.normal(0, 1, x.shape)
    y = a * x + b + noise
    model = Least_Squares_Method()
    model.fit(x, y)
    model.visualize(x, y)
