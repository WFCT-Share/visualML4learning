from visualML4learning.VisualizerBase import *



class DecisionTreeBase(VisualizerBase):
    def __init__(self):
        super().__init__()



class DecisionTreeClassifier(DecisionTreeBase):
    def __init__(self):
        super().__init__()
        pass

    import numpy as np

    # 计算基尼指数
    def gini_index(self, groups, classes):
        # 计算所有样本的数量
        n_instances = float(sum([len(group) for group in groups]))
        # 初始化基尼指数
        gini = 0.0
        # 对每个组进行基尼指数计算
        for group in groups:
            size = float(len(group))
            # 避免空组
            if size == 0:
                continue
            score = 0.0
            # 计算属于每个类的比例
            for class_val in classes:
                p = [row[-1] for row in group].count(class_val) / size
                score += p * p

            gini += (1.0 - score) * (size / n_instances)
        return gini

    # 分割数据集
    def test_split(self, index, value, dataset):
        left, right = list(), list()
        for row in dataset:
            if row[index] < value:
                left.append(row)
            else:
                right.append(row)
        return left, right

    # 选择最优分割点
    def get_split(self, dataset):
        class_values = list(set(row[-1] for row in dataset))
        b_index, b_value, b_score, b_groups = 999, 999, 999, None
        for index in range(len(dataset[0]) - 1):
            for row in dataset:
                groups = self.test_split(index, row[index], dataset)
                gini = self.gini_index(groups, class_values)
                if gini < b_score:
                    b_index, b_value, b_score, b_groups = index, row[index], gini, groups
        return {'index': b_index, 'value': b_value, 'groups': b_groups}

    # 构建树
    def build_tree(self, train, max_depth, min_size):
        root = self.get_split(train)
        self.split(root, max_depth, min_size, 1)
        return root

    # 创建子树或终止分支
    def split(self, node, max_depth, min_size, depth):
        left, right = node['groups']
        del (node['groups'])
        # 检查分支
        if not left or not right:
            node['left'] = node['right'] = self.to_terminal(left + right)
            return
        # 检查最大深度
        if depth >= max_depth:
            node['left'], node['right'] = self.to_terminal(left), self.to_terminal(right)
            return
        # 处理左分支
        if len(left) <= min_size:
            node['left'] = self.to_terminal(left)
        else:
            node['left'] = self.get_split(left)
            self.split(node['left'], max_depth, min_size, depth + 1)
        # 处理右分支
        if len(right) <= min_size:
            node['right'] = self.to_terminal(right)
        else:
            node['right'] = self.get_split(right)
            self.split(node['right'], max_depth, min_size, depth + 1)

    # 终止分支
    def to_terminal(self, group):
        outcomes = [row[-1] for row in group]
        return max(set(outcomes), key=outcomes.count)


class DecisionTreeRegressor(DecisionTreeBase):
    def __init__(self):
        super().__init__()
        pass