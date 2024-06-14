import pandas as pd
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from mpl_toolkits.mplot3d import Axes3D

plt.rcParams['font.sans-serif'] = [u'simHei']
plt.rcParams['font.size'] = 16
plt.rcParams['axes.unicode_minus'] = False

class VisualizerBase(ABC):
    def __init__(self):
        pass

    def Scatter2D(self, x, y, title, xlabel=None, ylabel=None):
        plt.figure(figsize=(8, 4))
        plt.plot(x, y)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        return plt.show()

    def Scatter3D(self, x, y, z, title, xlabel=None, ylabel=None):
        # 生成随机数据
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')  # 注意这里的 'projection='3d''
        ax.scatter(x, y, z)

        # 设置坐标轴标签
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_zlabel('Z Coordinate')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        # 显示图形
        return plt.show()


    def PCA(self):
        """线性降维"""
        pass

    def t_SNE(self):
        """非线性降维"""
        pass

    def set_frame(self, pic):
        pass

    def generate_gif(self, frames, save_path, frame_time, loop_times):
        pass