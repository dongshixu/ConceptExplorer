""" Data interface for trainning and suit for tornado
"""
import os
import json
import numpy as np
from scipy import sparse

from data_structures.attribute import Attribute
from dictionary.tornado_dictionary import TornadoDic
from rewrite.dataset import Dataset


class Data(object):
    def __init__(self, dataset, sub_dataset, split=True):
        """	Initialize Data
            @params:
                dataset: dataset name, belong to Dataset.DATASET
                sub_dataset: sub_dataset name, belong to Dataset
                split: bool, set dataset as split or not, default(True)
        """
        self.dataset = dataset
        self.sub_dataset = sub_dataset
        self.split = split
        self.path = Dataset.get_path(dataset, sub_dataset)
        if not os.path.exists(self.path):
            ValueError("The path '{}' is not exist!\nPlease ensure dataset exist!".format(self.path))

        self.label_attr = [0, 1]
        self.cur_data_slit = 0
        if self.dataset == Dataset.MOVIE_DATASET_NAME:
            self.dim = Dataset.MOVIE_DIM
        elif self.dataset == Dataset.PRSA_DATASET_NAME:
            self.dim = Dataset.PRSA_DIM
        self.configuration = self.get_configuration()

    def data(self, batch_size):
        """ generator for data.

            @returns:
            x: numpy, features of data
            y: int, label of data
            attributes: Attributes, not None means transfer another file.
        """
        data_slit = self.__get_file_split()

        cur_data_slit = 0
        while cur_data_slit < data_slit:
            data = np.load(self.path + 'data_{}.npy'.format(cur_data_slit))
            label = np.load(self.path + 'label_{}.npy'.format(cur_data_slit))
            num = len(data)
            index = 0
            while index < num:
                next_index = min(num, index + batch_size)
                if self.dataset == Dataset.MOVIE_DATASET_NAME:
                    yield data[index: next_index], label[index: next_index, np.newaxis], None
                elif self.dataset == Dataset.PRSA_DATASET_NAME:
                    yield data[index: next_index], label[index: next_index, np.newaxis] > 100, None
                index = index + batch_size

            # reset attributes again
            labels, attributes = self.get_attributes()
            # means we translate another file
            yield None, None, attributes

            cur_data_slit += 1
            self.cur_data_slit = cur_data_slit

    def get_attributes(self):
        """ return attributes by tornado Attributes type

            @return:
            attributes: Attributes
        """
        attributes = []
        dim_index = 0
        dataset_attributes = Dataset.get_attributes(self.dataset)

        for attr in dataset_attributes:
            for index in range(attr[1]):
                attribute = Attribute()
                attribute.set_name('{}_{}'.format(attr[0], index))
                attribute.set_type(TornadoDic.NUMERIC_ATTRIBUTE)
                attribute.set_possible_values([])
                if self.split:
                    range_dim = self.configuration['split']['range'][self.cur_data_slit][dim_index]
                else:
                    range_dim = self.configuration['all']['range'][dim_index]
                attribute.set_bounds_values(range_dim[0], range_dim[1])
                if attr[0] !='cast':
                    attributes.append(attribute)
                dim_index += 1
        return [0, 1], attributes

    def get_configuration(self):
        """ Get configuration of current dataset. contain record num, attributes of all dim.

            @return:
            result: dict, {'split': [[]], 'all': []}
        """
        if os.path.exists(self.path + 'configuration.json'):
            with open(self.path + 'configuration.json', 'r') as fp:
                result = json.load(fp)
                return result

        data_slit = self.__get_file_split()

        cur_data_slit = 0
        result = {
            'split': {
                'range': [],
                'num': []
            },
            'all': {
                'range': [],
                'num': 0
            }
        }
        while cur_data_slit < data_slit:
            data = np.load(self.path + 'data_{}.npy'.format(cur_data_slit))
            result['split']['range'].append([(float(data[:, i].min()), float(data[:, i].max())) for i in range(self.dim)])
            result['split']['num'].append(data.shape[0])
            result['all']['num'] += data.shape[0]
            cur_data_slit += 1

        for i in range(self.dim):
            result['all']['range'].append(
                (min([v[i][0] for v in result['split']['range']]),
                max([v[i][1] for v in result['split']['range']]))
            )

        with open(self.path + 'configuration.json', 'w') as fp:
            json.dump(result, fp)
        return result

    def __get_file_split(self):
        all_files = os.listdir(self.path)
        all_files.sort()
        # return int(sum([1 if '.npy' in f else 0 for f in all_files]) / 2)
        return int(sum([1 if 'data' in f else 0 for f in all_files]))

    def __len__(self):
        if self.split:
            return self.configuration['split']['num'][self.cur_data_slit]
        else:
            return self.configuration['all']['num']
