import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import dump, load
import math
import os
import datetime
from abc import ABC, abstractmethod
from visualML4learning.VisualizerBase import *
class KNNBase(VisualizerBase):
    """这是KNN的抽象基类，不可实例化，为各类KNN对象创造固定的规范"""

    @abstractmethod
    def predict(self, data):
        pass


    def save(self, path=None):
        """使用joblib包保存数据，并返回保存的路径，可以指定路径保存，不指定路径时，默认保存在./data/KNN/中并以当前时间命名"""
        if path is None:
            path = './data/KNN/' + auto_file_name('joblib')
        self.model_path = path
        if not os.path.exists('./data/KNN/'):
            os.makedirs(path)
        stored_data = np.column_stack((self.X, self.y))
        dump(stored_data, path)
        return path


    def load(self, path):
        """使用joblib包读取数据，可以指定路径进行读取，不指定路径时，默认读取模型地址，当模型地址不存在时抛出报错"""
        if path is None:
            path = self.model_path
            if path is None:
                raise KNNError("未保存模型故无法读取，请先保存模型或指定读取路径")
        if not os.path.exists(path):
            raise FileNotFoundError(f"文件 '{path}' 不存在")
        stored_data = load(path)
        self.X = stored_data[:, :-1]  # 选择除了最后一列之外的所有列
        self.y = stored_data[:, -1]
        return f"{path}已成功读取"


class KNNClassify(KNNBase):
    """创建一个KNN分类器，计算距离的方式为闵可夫斯基距离(Minkowski Distance)"""


    def __init__(self, X: np.array = None, y: np.array = None, neighbors: int = 5, p: int = 2, *, KNN=None,
                 model_path=None):
        """构造一个KNN分类器，输入数据是非必须的，通过输入特征矩阵和结果向量，设定相邻点数（k值）构建，也可通过输入其他KNN对象构建

           Args:
                X(np.array):特征向量
                y(np.array):分类结果向量
                neighbors(int):相邻元素数
                p(int): 特征矩阵的维度数，作为明可夫斯基距离的参数p

           """
        super().__init__()
        if KNN is not None:
            self.X = KNN.X
            self.y = KNN.y
            self.neighbors = KNN.neighbors
            self.p = KNN.p
        elif model_path is not None:
            self.load(model_path)
        elif X is not None and y is not None:
            check_X_y(X, y)

            self.X = X
            self.y = y
        self.neighbors = neighbors
        self.p = p


    def fit(self, X, y, evaluate=False, division=0.7):
        """非必须，虽然函数名称为fit，但是实际计算过程中仅仅只是存储数据以方便计算并不包含拟合这一过程，
        当用户需要改变数据时，或者一开始并未对数据进行输入时使用。当评估模式开启时会划分数据的比例进行评估，开启评估时返回模型预测准确率。
            :Args:
                X(np.array):特征向量
                y(np.array):分类结果向量
                evaluate(bool):
                division(float):划分训练数据和测试数据的比例关系：比例 = 训练数据 : 总数据
        """
        if not evaluate:
            self.X = X
            self.y = y
        else:
            index = int(X.shape[0] * division)
            self.X = X[:index]
            self.y = y[:index]
            result = self.predict(X[index:])
            num_rights = 0
            for i in range(result.shape[0]):
                if result[i] == y[index:][i]:
                    num_rights += 1
            return num_rights / result.shape[0]


    def predict(self, data: np.array) -> np:
        """使用KNN模型进行预测"""
        knn_check(self)
        if data.ndim == 2:
            result = np.zeros((data.shape[0]))
            for j in range(data.shape[0]):
                check_data(data[j], self.X)
                result[j] = self.row_predict(data[j], self.p)
            return result
        else:
            check_data(data, self.X)
            return self.row_predict(data, self.p)

    def row_predict(self, data, p):
        """仅适用于一维向量进行预测，返回单个预测结果"""
        distance_matrix = np.zeros((self.X.shape[0], 2))
        for i in range(self.X.shape[0]):
            distance_matrix[i, 0] = np.sum(np.abs(self.X[i, :] - data) ** p) ** (1 / p)
            distance_matrix[i, 1] = self.y[i]
        sorted_indices = np.argsort(distance_matrix[:, 0])
        nearest_labels = [self.y[i] for i in sorted_indices[:self.neighbors]]
        return max(set(nearest_labels), key=nearest_labels.count)




class KNNRegressor(KNNBase):


    def __init__(self, X: np.array = None, y: np.array = None, neighbors: int = 5, p: int = 2, *, KNN=None,
                 model_path=None, weighed=False, weights=None):
        """构造一个KNN回归器，输入数据是非必须的，通过输入特征矩阵和结果向量，设定相邻点数（k值）构建，也可通过输入其他KNN对象构建

           Args:
                X(np.array):特征向量
                y(np.array):回归结果向量
                neighbors(int):相邻元素数
                p(int): 特征矩阵的维度数，作为明可夫斯基距离的参数p
                weighed(bool):是否开启权重模式，为False时默认以等权回归
                weights(np.array):权重向量

           """
        super().__init__()
        if KNN is not None:
            self.X = KNN.X
            self.y = KNN.y
            self.neighbors = KNN.neighbors
            self.p = KNN.p
        elif model_path is not None:
            self.load(model_path)
        elif X is not None and y is not None:
            check_X_y(X, y)

            self.X = X
            self.y = y
        self.neighbors = neighbors
        self.p = p
        if weighed == True and weights is not None:
            check_weights(weights, self.X)
        self.weighed = weighed
        self.weights = weights

    def set_weighs(self, weights):
        """非必须，设置权重将使得weighed变成True"""
        check_weights(weights, self.X)
        self.weighed = True
        self.weights = weights


    def fit(self, X, y, weighed=False, weights=None, evaluate=False, division=0.7):
        """非必须，虽然函数名称为fit，但是实际计算过程中仅仅只是存储数据以方便计算并不包含拟合这一过程，
        当用户需要改变数据时，或者一开始并未对数据进行输入时使用。当评估模式开启时会划分数据的比例进行评估，开启评估时返回模型预测准确率。
            :Args:
                X(np.array):特征向量
                y(np.array):分类结果向量
                evaluate(bool): 是否划分训练数据和测试数据进行评估
                division(float):划分训练数据和测试数据的比例关系：比例 = 训练数据 : 总数据
        """
        if not evaluate:
            self.X = X
            self.y = y
        elif weighed and weights is not None:
            check_weights(weights, self.X)
            self.weighed = weighed
            self.weights = weights
        else:
            index = int(X.shape[0] * division)
            self.X = X[:index]
            self.y = y[:index]
            result = self.predict(X[index:])
            M = 1 / result.shape[0] * sum((result - y[index:]) ** 2)
            return M


    def predict(self, data):
        """使用KNN模型进行预测"""
        knn_check(self)
        if data.ndim == 2:
            result = np.zeros((data.shape[0]))
            for j in range(data.shape[0]):
                check_data(data[j], self.X)
                result[j] = self.row_predict(data[j], self.p)
            return result
        else:
            check_data(data, self.X)
            return self.row_predict(data, self.p)

    def row_predict(self, data, p):
        """仅适用于一维向量进行预测，返回单个预测结果"""
        distance_matrix = np.zeros((self.X.shape[0], 2))
        for i in range(self.X.shape[0]):
            distance_matrix[i, 0] = np.sum(np.abs(self.X[i, :] - data) ** p) ** (1 / p)
            distance_matrix[i, 1] = self.y[i]
        sorted_indices = np.argsort(distance_matrix[:, 0])
        if not self.weighed:
            # 简单平均回归
            return np.mean([self.y[i] for i in sorted_indices[:self.neighbors]])
        elif self.weights is None:
            self.weights = np.zeros(self.X.shape[0])
            for i in range(sorted_indices.shape[0]):
                self.weights[i] = 1 / (sorted_indices[i, 0] + 1)
            return np.average([self.y[i] for i in sorted_indices[:self.neighbors]],
                              weights=self.weights[:self.neighbors])
        else:
            return np.average([self.y[i] for i in sorted_indices[:self.neighbors]],
                              weights=self.weights[:self.neighbors])


class KNNAuto:
    """对本包不熟悉，或者不想亲历亲为调节相应参数的用户，可以通过这一套预设方式获得理想的效果"""
    pass


def auto_file_name(extension):
    """根据文件后缀自动根据当前时间为文件命名"""
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime('%Y-%m-%d_%H-%M-%S')
    return formatted_time + '.' + extension


class KNNTester():
    pass


def check_X_y(X, y):
    """检查用户输入的特征矩阵和相应的结果向量是否符合规范"""
    if not isinstance(X, np.ndarray):
        raise ValueError("输入的特征矩阵必须是一个 NumPy 数组")
    if X.ndim != 2:
        raise ValueError("输入的特征矩阵的数组维度错误，必须是是二维数组")
    if X.shape[0] != y.shape[0]:
        raise ValueError("输入的特征矩阵行数与分类结果向量长度不等")


def check_data(data, X):
    """检查用户输入的特征矩阵或特征向量是否符合模型规范"""
    if data.ndim > X.ndim or data.ndim == 0:
        raise ValueError("输入的数组维度数不符合要求，预测的特征向量应当为一维或二维")
    if data.shape[0] != X.shape[1]:
        raise ValueError("第一维度的特征数不相等")

    pass


def check_weights(weights, X):
    if X.shape[0] != weights.shape[0]:
        raise ValueError("输入的特征矩阵行数与权重向量长度不等")
    pass


def knn_check(KNN: KNNBase):
    """检查模型的数据是否被正确键入"""
    if KNN.X is None:
        raise KNNError("特征矩阵X错误，未进行数据拟合")
    if KNN.y is None:
        raise KNNError("回归结果向量y错误，未进行数据拟合")

def knn_visual_check(KNN):
    pass



def check_path():
    pass



class KNNError(Exception):
    """自定义KNN异常类，用于特定错误情况的处理"""

    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

