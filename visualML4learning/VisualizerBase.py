import pandas as pd
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
import io
import imageio
import os
import shutil
plt.rcParams['font.sans-serif'] = [u'simHei']
plt.rcParams['font.size'] = 16
plt.rcParams['axes.unicode_minus'] = False


class VisualizerBase(ABC):
    def __init__(self):
        pass

    def Scatter2D(self, x, y, title=None, xlabel=None, ylabel=None):
        fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
        ax.scatter(x, y, marker='o')
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        plot = [ax, fig]
        return plot

    def Scatter2D_trendline(self,  x, y, x_points, y_predict, title=None, xlabel=None, ylabel=None, trendline_label=None):
        plot = self.Scatter2D(x, y, title, xlabel, ylabel)
        plot[0].plot(x_points, y_predict, 'r-', label=trendline_label)
        return plot

    def Scatter3D(self, x, y, z, title, xlabel=None, ylabel=None, zlabel=None):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x, y, z)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_zlabel(zlabel)
        plt.title(title)
        return plt.show()

    def hist(self, x, title, ylabel=None):
        plt.figure(figsize=(16, 8))
        plt.hist(x, bins=20)
        plt.title(title)
        plt.ylabel(ylabel)
        return plt.show()

    def PCA(self):
        """线性降维"""
        pass

    def t_SNE(self):
        """非线性降维"""
        pass

    def set_frame(self, plot, title, frames, path="./frames/"):
        path = path + title
        if not os.path.exists(path):
            os.makedirs(path)
        whole_path = path + "/" + str(len(frames)) + ".png"
        frames.append(whole_path)
        plot[1].savefig(whole_path)
        return frames

    def generate_gif(self, frames,  title, frame_time=1/12, loop=0, path="./gifs/",):
        if not os.path.exists(path):
            os.makedirs(path)
        images = []
        for frame in frames:
            images.append(imageio.imread(frame))
        whole_path = f"{path + title}.gif"
        imageio.mimsave(whole_path, images, duration=frame_time, loop=loop)
        for frame in frames:
            if os.path.exists(frame):
                os.remove(frame)

    def store_plt_config(self):
        points_label = {
            'Xlabel': None,
            'Ylabel': None,
            'Zlabel': None
        }
        pic_info = {'Xlabel': 1}
        pass
