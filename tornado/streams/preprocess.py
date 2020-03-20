""" Data pre-process model for datasets.
"""
import os
import csv
import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt

from streams.dataset import Dataset
from streams.aqi import AQI


class DataPreProcess(object):
    def __init__(self, dataset, sub_dataset, split=True):
        """	Initialize Data
            @params:
                dataset: dataset name, belong to Dataset.DATASET
                sub_dataset: sub_dataset name, belong to Dataset
                split: bool, set dataset as split or not, default(True)
        """
        self.dataset = dataset
        self.sub_dataset = sub_dataset
        self.path = Dataset.get_path(dataset, sub_dataset)
        if dataset == Dataset.PRSA_DATASET_NAME:
            self.path = '{}/{}/csv'.format(Dataset.PATH, dataset)

        if not os.path.exists(self.path):
            ValueError("The path '{}' is not exist!\nPlease ensure dataset exist!".format(self.path))
        # if dataset == Dataset.MOVIE_DATASET_NAME:
        #     self.path = '{}/{}/'.format(Dataset.PATH, dataset)

    def transfer_npz_2_npy(self):
        """ Transfer npz file into numpy data
        """
        if os.path.exists(self.path + 'data.npy') or os.path.exists(self.path + 'data_0.npy'):
            return

        print('Begin transfer npz to npy in {}'.format(self.sub_dataset))
        file_list = os.listdir(self.path)
        file_list.sort()
        feature = []
        label = []
        row_size = 0
        tmp_row_size = 0
        for file_name in file_list:
            if 'feature' in file_name:
                feature_tmp = sparse.load_npz(self.path + file_name).toarray()
                feature.append(feature_tmp)
                row_size += feature_tmp.shape[0]
            else:
                label_tmp = np.squeeze(sparse.load_npz(self.path + file_name).toarray())
                tmp_row_size += label_tmp.shape[0]
                label.append(label_tmp)
        self.__sort_data_by_timeline(feature, label, row_size)

    def transfer_csv_2_npy(self, reset=True):
        """ Transfer csv file into numpy data
        """
        save_path = Dataset.get_path(self.dataset, self.sub_dataset)
        if os.path.exists(save_path + 'data_0.npy') and not reset:
            return

        keys = ['year', 'month', 'day', 'hour', 'PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3',
            'TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM']
        data_keys = ['year', 'month', 'day', 'hour', \
            'SO2', 'NO2', 'CO', 'O3', 'O3_8hours', 'PM2.5', 'PM2.5_day', 'PM10', 'PM10_day',\
            'TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM']
        wd = {
            'E': -4,
            'S': -3,
            'W': -2,
            'N': -1
        }

        csv_path = '{}/{}_20130301-20170228.csv'.format(self.path, self.sub_dataset)
        csv_file = open(csv_path, 'r')
        dict_reader = csv.DictReader(csv_file)

        row_history = []
        dataset = []
        labels = []
        for row in dict_reader:
            # transfer one row data into dict
            r = {}
            for key in keys:
            # set default NA as prevalue
                if 'NA' not in row[key]:
                    r[key] = float(row[key])
                else:
                    if len(row_history) == 0:
                        r[key] = 0
                    else:
                        r[key] = row_history[-1][key]

            if row['wd'] == 'NA':
                if len(row_history) == 0:
                    r['wd'] = ''
                else:
                    r['wd'] = row_history[-1]['wd']
            else:
                r['wd'] = row['wd']

            row_history.append(r.copy())
            if len(row_history) > 24:
                row_history.pop(0)

            # get data and set it
            label = self.__set_prsr_label(row_history, r)
            data = np.zeros((4 + 9 + 5 + 4), dtype=np.float32)

            for i, key in enumerate(data_keys):
                data[i] = r[key]
            for v in r['wd']:
                data[wd[v]] += 1 / len(r['wd'])
            dataset.append(data)
            labels.append(label)

        # save data
        np.save(save_path + 'data_0.npy', np.array(dataset[:-24]))
        np.save(save_path + 'label_0.npy', np.array(labels[24:]))

    def plot_label(self):
        save_path = Dataset.get_path(self.dataset, self.sub_dataset)
        labels = np.load(save_path + 'label_0.npy').tolist()
        values_cnt = [0 for i in range(500)]
        for v in labels:
            values_cnt[min(int(v), 499)] += 1
        for i in range(500):
            values_cnt[i] /= len(labels)
        values_sum = [sum(values_cnt[:i]) for i in range(500)]
        x = [i for i in range(len(labels))]
        plt.figure()
        plt.subplot(131)
        plt.plot(x, labels)
        plt.subplot(132)
        plt.plot([i for i in range(500)], values_cnt)
        plt.subplot(133)
        plt.plot([i for i in range(500)], values_sum)
        plt.title(self.sub_dataset)
        plt.show()

    def __set_prsr_label(self, rows, row):
        """ Calculate AQI(Air quality index) by
            http://www.mee.gov.cn/ywgz/fgbz/bz/bzwb/jcffbz/201203/W020120410332725219541.pdf
            @param:
            rows: 24-rows of past 24-hours observation value.
            r: current hour observation value.
            @return:
            row: not return, but changed by ref.
            AQI: on time AQI.
        """
        row['PM2.5_day'] = sum([r['PM2.5'] for r in rows]) / len(rows)
        row['PM10_day'] = sum([r['PM10'] for r in rows]) / len(rows)
        row['PM2.5'] = rows[-1]['PM2.5']
        row['PM10'] = rows[-1]['PM10']
        row['SO2'] = rows[-1]['SO2']
        row['NO2'] = rows[-1]['NO2']
        row['CO'] = rows[-1]['CO']
        row['O3'] = rows[-1]['O3']
        row['O3_8hours'] = sum([r['O3'] for r in rows[-8:]]) / min(len(rows), 8)
        return max([AQI.IAQI_P(p, row[p]) for p in AQI.POLLUTANT])

    def __sort_data_by_timeline(self, feature, label, row_size):
        """ Sort split dataset by review timeline
        @params:
            feature: list of all split dataset [numpy, ..]
            label: list of all split label [numpy, ..]
            row_size: row number of all dataset
        """
        time_line = np.zeros((row_size, 3), dtype=np.int32)
        before_index = 0
        for index, f in enumerate(feature):
            shape = f.shape
            time_line[before_index: before_index + shape[0], 0] = index
            time_line[before_index: before_index + shape[0], 1] = np.arange(0, shape[0])
            time_line[before_index: before_index + shape[0], 2] = f[:, -1].copy()
            before_index += shape[0]
        time_line = time_line[time_line[:, 2].argsort()]

        shape = feature[0].shape
        result_feature = np.zeros((row_size, shape[1]), dtype=np.float32)
        shape = label[0].shape
        result_label = np.zeros((row_size), dtype=np.int8)
        for i, v in enumerate(time_line):
            result_feature[i] = feature[v[0]][v[1]]
            result_label[i] = label[v[0]][v[1]]

        cur_index = 1000000
        pre_index = 0
        index = 0
        while pre_index < row_size:
            np.save(self.path + 'data_{}.npy'.format(index), result_feature[pre_index: cur_index])
            np.save(self.path + 'label_{}.npy'.format(index), result_label[pre_index: cur_index])
            pre_index = cur_index
            index += 1
            cur_index = min(row_size, cur_index + 1000000)

        print('data shape: {}'.format(result_feature.shape))
        print('Finish sort data by timeline! Save data into..{}'.format(self.path))

    def statistic_label(self):
        all_files = os.listdir(self.path)
        movie_len = int(sum([1 if '.npy' in f else 0 for f in all_files]) / 2)
        count = 0
        x_ = -1
        y = []
        x = []
        for i in range(movie_len):
            label = np.load('{}label_{}.npy'.format(self.path, i))
            m = np.shape(label)[0]
            for value in label:
                if value == 0:
                    count -= 1
                else:
                    count += 1
                y.append(count)
                x_ += 1
                x.append(x_)
        plt.figure()
        plt.plot(x, y)
        plt.title(self.sub_dataset)
        plt.show()