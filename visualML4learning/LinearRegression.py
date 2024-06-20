import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import dump, load
import math
import os
import datetime
from easylogger4dev_alpha import *
from abc import ABC, abstractmethod
from visualML4learning.VisualizerBase import *

"""设置字符集防止乱码问题"""
plt.rcParams['font.sans-serif'] = [u'simHei']
plt.rcParams['axes.unicode_minus'] = False

"""使用动态规划分步回归实现回归过程可视化  """


class LinearBase(VisualizerBase):
    """线性对象的基类"""
    @abstractmethod
    def predict(self, x):
        pass

    @abstractmethod
    def fit(self, X, Y):
        pass

    @abstractmethod
    def predict_alter(self, X):
        """预测器的迭代器"""
        pass


class Normal_equations:
    pass

class Least_Squares_Method(LinearBase):
    def __init__(self, model=None):
        super().__init__()
        self.beta0 = None
        self.beta1 = None
        self.model = model

    def fit(self, X, Y):
        self.beta1 = np.sum((X - np.mean(X)) * (Y - np.mean(Y))) / np.sum((X - np.mean(X)) ** 2)
        self.beta0 = np.mean(Y) - self.beta1 * np.mean(X)
        epsilon_list = 0.5 * (self.beta0 + self.beta1 * X - Y) ** 2
        epsilon = np.sum(epsilon_list)

        def f(x):
            return self.beta0 + self.beta1 * x

        return f, epsilon
    pass

    def predict(self, x):
        if self.beta0 is None or self.beta1 is None:
            raise ValueError("未拟合模型")
        else:
            return self.beta0 + self.beta1 * x

    def visualize(self, X, Y):
        frames = []
        for i in range(1, X.shape[0] + 1):
            X_slice = X[:i]
            Y_slice = Y[:i]
            min_x = np.min(X_slice)
            max_x = np.max(X_slice)
            X_points = np.linspace(min_x, max_x, num=50)
            plot = super().Scatter2D_trendline(X_slice, Y_slice, X_points, self.predict(X_points), "最小二乘法")
            frames = super().set_frame(plot, "测试", frames)
            plt.close(plot[1])
        super().generate_gif(frames, title="测试")

    def predict_alter(self, X):
        pass

class Linear_Regression(LinearBase):
    pass

class Univariate_Linear_Regression():
    pass


