"""构建一种可视化展示特征工程的预训练算法"""
import pandas as pd
import matplotlib.pyplot as plt
from visualML4learning.VisualizerBase import *


class FeatureEngineering(VisualizerBase):
    """定义特征工程对象"""

    def __init__(self, data):
        super().__init__()
        self.data = data

        self.x = None
        self.y = None
        self.z = None
        self.name = ""

    def head(self, n):
        """查看前n行的数据，方法同pandas.DataFrame"""
        return self.data.head(n)

    def tail(self, n):
        """查看后n行的数据，方法同pandas.DataFrame"""
        return self.data.tail(n)

    def info(self):
        return self.data.info()

    def describe(self):
        return self.data.describe()

    def __deepcopy__(self):
        pass

    def columns_counts(self, columns):
        return self.data[columns].value_counts()

    def column_hist(self, columns):
        for column in self.columns:
            super().hist(self.data[column], self.name + "—" + self.x.columns[0], self.x.columns[0])

    def visualize(self):
        if self.x is None:
            pass
        else:
            if self.z is None:
                return super().Scatter2D(self.x, self.y, self.name, self.x.columns[0], self.y.columns[0])
            else:
                return super().Scatter3D(self.x, self.y, self.z, self.name, self.x.columns[0], self.y.columns[0],
                                         self.z.columns[0])

    def auto_outlier_processing(self):
        pass

    def auto_missing_data_processing(self):
        pass

    def set_x(self, column):
        self.x = self.data[column]
        pass

    def set_y(self, column):
        self.y = self.data[column]

    def set_z(self, column):
        self.z = self.data[column]

    def remove_column(self, columns):
        self.data = self.data.drop([columns], axis=1)

        pass

    def count_null(self):
        self.data.isnull().any()
        return self.data.isnull().sum()

    def fill_mean(self, columns):
        """均值填充法"""
        for column in columns:
            mean_value = self.data[column].mean()
            self.data[column] = self.data[column].fillna(mean_value)

    def fill_majority(self, columns):
        """众数填充法"""
        for column in columns:
            majority_value = self.data[column].value_counts().idmax()
            self.data[column] = self.data[column].fillna(majority_value)

    def min_max_scaling(self, columns, visual=False):
        """进行最小-最大归一化"""
        for column in columns:
            min_vals = self.data[column].min(axis=0)
            max_vals = self.data[column].max(axis=0)
            norm_features = (self.data[column] - min_vals) / (max_vals - min_vals)

    def one_hot(self):
        pass
