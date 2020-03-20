"""
@Author: Dustin Xu
@Date: 2020/2/23 13:02 PM
@Description: Make the time interval of the movie data consistent
"""

from data_stream_xu.data import Data
from data_stream_xu.dataset import Dataset
from data_structures.attribute_scheme import AttributeScheme
from result_save_xu.__init__ import *

import numpy as np
import json


def save_file(file, path, type_name=None):
    if type_name is not None:
        filename = path + '{}.json'.format(type_name)
    else:
        filename = path + '.json'
    with open(filename, 'w') as file_obj:
        json.dump(file, file_obj)
        print(filename + '==>' + 'saved ok!')


if __name__ == '__main__':
    for data_set in Dataset.DATASET:
        data_num_statistic = {}
        folder_create = Folder('E:/zju/result/{}'.format(data_set))
        for sub_data_set in Dataset.get_sub_dataset(data_set):
            _instance_count = 0
            save_data = []
            save_label = []
            time_start = 77855
            time_interval = 1
            data_num_statistic[sub_data_set] = []
            day_count = 0
            sub_folder_path = folder_create.sub_folder(sub_data_set)
            if sub_data_set != 'netflix':
                data = Data(data_set, sub_data_set)
                labels, attributes = data.get_attributes()
                for x, y, attribute in data.data(batch_size=1):
                    if attribute is not None:
                        attributes_scheme = AttributeScheme.get_scheme(attributes)
                        __numeric_attribute_scheme = attributes_scheme['nominal']
                        __numeric_attribute_scheme = attributes_scheme['numeric']
                        continue

                    instance = x.tolist()[0] + [int(y.tolist()[0][0])]
                    specific_time = instance[-2]
                    _instance_count += 1
                    if 77855 <= specific_time <= 78616:
                        save_data.append(instance[:-1])
                        save_label.append(instance[-1])
                        difference_time_value = specific_time - time_start
                        if difference_time_value >= time_interval:
                            batch_interval = int(difference_time_value / time_interval)
                            for i in range(batch_interval):
                                time_start += time_interval
                                data_num_statistic[sub_data_set].append(day_count)
                                day_count = 0
                            day_count += 1
                        else:
                            day_count += 1
                    if specific_time > 78616:
                        data_num_statistic[sub_data_set].append(day_count-1)
                        break
            if sub_data_set == 'rotten_tomatoes':
                num = 400
            elif sub_data_set == 'twitter':
                num = 2000
            else:
                num = 6000

            save_data = np.array(save_data)[num:, :-1]
            save_label = np.array(save_label)[num:]
            print(sub_data_set, np.array(save_data).shape)

            save_data_path = sub_folder_path + '/data_{}.npy'.format(0)
            save_label_path = sub_folder_path + '/label_{}.npy'.format(0)
            np.save(save_data_path, save_data)
            np.save(save_label_path, save_label)

        save_statistic_path = folder_create.get_path() + '/'
        save_file(data_num_statistic, save_statistic_path, type_name='data_num')