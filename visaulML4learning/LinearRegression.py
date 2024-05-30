import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import dump, load
import math
import os
import datetime
from easylogger4dev_alpha import *
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt

"""设置字符集防止乱码问题"""
plt.rcParams['font.sans-serif'] = [u'simHei']
plt.rcParams['axes.unicode_minus'] = False

"""使用动态规划分步回归实现回归过程可视化  """


class LinearBase(ABC):
    """线性对象的基类"""
    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict_alter(self, X):
        """预测器的迭代器"""
        pass


class Normal_equations:
    pass

class Least_Squares_Method():
    pass

class Univariate_Linear_Regression():
    pass
