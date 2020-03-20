"""
@Author: Dustin Xu
@Date: 2020/2/9 12:24 PM
@Description: data generator (single or batch)
"""
import os
from .dataset import Dataset
from .data import Data

class DataGenerator:
    """
    输入：不同数据集
    输出：数据迭代器
    """
    def __init__(self, base_path, dataset):
        # 数据集选择，
        self.PATH = base_path
        self.DATASET = dataset
        pass

    def data_generator(self):

        pass
