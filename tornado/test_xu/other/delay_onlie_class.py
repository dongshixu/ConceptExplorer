"""
@Author: Dustin Xu
@Date: 2020/2/12 16:01 PM
@Description: Get ready for the code to come online
"""
from data_structures.attribute_scheme import AttributeScheme
from classifier_xu.__init__ import *
from drift_detector_xu.__init__ import *
from evaluate_xu.__init__ import *
from data_stream_xu.data import Data
from data_stream_xu.dataset import Dataset
from filters.attribute_handlers import *
from filters.project_creator import Project

import time
import datetime
import numpy as np
import json


class SaveDifferentData:
    def __init__(self, leaner, drift_detector, evaluator):
        if leaner is not None:
            self.leaner = leaner
        if drift_detector is not None:
            self.drift_detector = drift_detector
        if evaluator is not None:
            self.evaluator = evaluator

        self.__window_size = 500
        self.__step = 1000
        self.__classifier_limit = 10
        self.__numeric_attribute_scheme = 0
        self.__numeric_attribute_scheme = 0
        self.__instance_count = 0
        self.__data_length = 0

        self.detection = True
        self.warning_status = False
        self.drift_status = False
        self.__last_warning_status = False
        self.__last_drift_status = False
        self.current_warning_status = {}
        self.current_drift_status = {}

        self.date_time_flag = True
        self.prsa_flag = False

        self.__start_point = 0
        self.__batch_size = 0
        self.__count = 0
        self.__bingo = 0
        self.unix_time = 0
        self.__last_unix_time = 0

        self.warning = []
        self.drift = []
        self.batch_count = []
        self.batch_start_time = []
        self.weights = []
        self.accuracy = []
        self.right_count = []
        self.warning_level = []
        self.warning_level_set = []
        self.hit = []
        self.predict_corr = []

        self.attributes_set = {}
        self.last_batch_attributions = None
        self.spilt_inform = None

    def run(self, data_set, sub_data_set):

        if data_set == 'prsa_data':
            self.__batch_size = 24 * 3600  # 3600 represent 1 hour
        elif data_set == 'movie_data':
            self.__batch_size = 24 * 7 * 3600  # 3600 represent 1 hour

        self.__data_length = Dataset.get_length(data_set, sub_data_set)
        self.attributes_set = SaveDifferentData.construct_correlation(data_set)
        self.spilt_inform = Dataset.get_spilt_inform(data_set)

        data = Data(data_set, sub_data_set)
        labels, attributes = data.get_attributes()
        attributes_scheme = AttributeScheme.get_scheme(attributes)
        self.__numeric_attribute_scheme = attributes_scheme['nominal']
        self.__numeric_attribute_scheme = attributes_scheme['numeric']

        # Initializing a learner
        learner = Logistic(labels, attributes_scheme['numeric'])
        learner = OnlineAccuracyUpdatedEnsemble(labels, attributes_scheme['numeric'], learner,
                                                windowSize=self.__window_size, classifierLimit=self.__classifier_limit)

        # Initializing a drift detector
        drift_detector = DDM()

        # Creating a save content
        project = Project('./projects/distributed/{}'.format(data_set), sub_data_set)

        # Initializing a evaluator
        evaluator = EvaluateWithWindowSize(learner, drift_detector, project, self.__window_size)

        # train & test
        for x, y, attribute in data.data(batch_size=1):
            if attribute is not None:
                attributes_scheme = AttributeScheme.get_scheme(attributes)
                self.__numeric_attribute_scheme = attributes_scheme['nominal']
                self.__numeric_attribute_scheme = attributes_scheme['numeric']
                continue

            instance = x.tolist()[0] + [int(y.tolist()[0][0])]

            # 每条数据的unix时间戳
            # prsa data
            if data_set == 'prsa_data':
                self.date_time_flag = True
                date_time = list(map(int, instance[:4]))
                d = datetime.date(date_time[0], date_time[1], date_time[2])
                t = datetime.time(date_time[3])
                datetime_str = str(d) + ' ' + str(t)
                self.unix_time = int(time.mktime(time.strptime(datetime_str, '%Y-%m-%d %H:%M:%S')))
                if self.unix_time >= 1363881600:
                    self.prsa_flag = True
            elif data_set == 'movie_data':
                # movie data
                if instance[-2] > 62091:
                    self.date_time_flag = True
                    self.prsa_flag = True
                    date_time = self._get_date(instance[-2])
                    datetime_str = str(date_time) + ' ' + '00:00:00'
                    self.unix_time = int(time.mktime(time.strptime(datetime_str, '%Y-%m-%d %H:%M:%S')))

            instance[0:len(instance) - 1] = Normalizer.normalize(instance[0:len(instance) - 1],
                                                                 self.__numeric_attribute_scheme)
            self.__instance_count += 1

            if self.__instance_count > self.__window_size and self.date_time_flag and self.prsa_flag:
                if self.unix_time == 1363881600:
                    self.__start_point = self.unix_time
                    self.batch_start_time.append(self.__start_point)
                if self.unix_time - self.__start_point >= self.__batch_size:
                    self.__start_point = self.unix_time
                    # for every batch
                    # 权重
                    self.weights.append(learner.currentClassifierWeights.tolist())
                    # 属性记录
                    self.calculate_correlation()
                    # 准确度（可能会变, 目前是后端计算drift的准确率）
                    self.accuracy.append(drift_detector.accuracy)
                    # batch开始时间
                    self.batch_start_time.append(self.__start_point)
                    # batch 数量
                    self.batch_count.append(self.__count)
                    # 目前是 batch 中预测正确的个数， 后面改为concept drift 预测正确个数（后面为drift预测正确的个数）
                    self.right_count.append(self.__bingo)
                    # warning level
                    self.calculate_warning_level()

                    # print(batch_start_time, batch_count)
                    self.predict_corr = [instance]
                    self.__count = 1
                    self.__bingo = 0
                else:
                    self.__count += 1
                    self.predict_corr.append(instance)
                self.warning_level_set.append(drift_detector.risk)

                predicted_value = learner.do_testing(instance)

                prediction_status = evaluator.calculate_accuracy(predicted_value, instance[-1],
                                                                 output_size=self.__step, output_flag=False)
                if prediction_status:
                    self.__bingo += 1

                if self.detection is True:
                    self.warning_status, self.drift_status = drift_detector.detect(prediction_status)
                    if self.warning_status is not self.__last_warning_status:
                        if self.warning_status:
                            self.current_warning_status['start'] = self.unix_time
                            self.current_warning_status['max_accuracy'] = [drift_detector.o_s_d_min]
                            self.current_warning_status['max_accuracy_time'] = [self.unix_time]
                        else:
                            self.current_warning_status['end'] = self.__last_unix_time
                            self.warning.append(self.current_warning_status)
                            self.current_warning_status = {}
                    else:
                        if self.warning_status:
                            self.current_warning_status['max_accuracy'].append(drift_detector.o_s_d_min)
                            self.current_warning_status['max_accuracy_time'].append(self.unix_time)
                    if self.drift_status is not self.__last_drift_status:
                        if self.drift_status:
                            self.current_drift_status['start'] = self.unix_time
                            self.current_drift_status['max_accuracy'] = [drift_detector.o_s_d_min]
                            self.current_drift_status['max_accuracy_time'] = [self.unix_time]
                        else:
                            self.current_drift_status['end'] = self.__last_unix_time
                            self.drift.append(self.current_drift_status)
                            self.current_drift_status = {}
                    else:
                        if self.drift_status:
                            self.current_drift_status['max_accuracy'].append(drift_detector.o_s_d_min)
                            self.current_drift_status['max_accuracy_time'].append(self.unix_time)

                    self.__last_warning_status = self.warning_status
                    self.__last_drift_status = self.drift_status
                    self.__last_unix_time = self.unix_time
                else:
                    self.warning_status = False
                    self.drift_status = False

                if self.__instance_count == self.__data_length:  # 最后一个batch可能只有少部分数据，要考虑
                    # 权重
                    self.weights.append(learner.currentClassifierWeights.tolist())
                    # 属性记录
                    self.calculate_correlation()
                    # 准确度（可能会变, 目前是后端计算drift的准确率）
                    self.accuracy.append(drift_detector.accuracy)
                    # batch 数量
                    self.batch_count.append(self.__count)
                    # 目前是 batch 中预测正确的个数， 后面改为concept drift 预测正确个数
                    self.right_count.append(self.__bingo)
                    # warning level
                    self.calculate_warning_level()

                # training
                learner.do_training(instance, self.drift_status)
            else:
                # training
                learner.do_training(instance, self.drift_status)

        self.save_file()

    def calculate_warning_level(self):
        _avg = round(sum(self.warning_level_set) / len(self.warning_level_set), 4)
        _max = round(max(self.warning_level_set), 4)
        if len(self.warning_level) == 0:
            pass
        else:
            if _max >= 3:
                self.hit.append(1)
            else:
                self.hit.append(0)
        self.warning_level.append({
            'avg': _avg,
            'max': _max
        })
        self.warning_level_set = []

    def calculate_correlation(self):
        self.predict_corr = np.array(self.predict_corr)
        label = self.predict_corr[:, -1]
        correlation_coefficient = []
        batch2batch = []
        count = 0
        if self.last_batch_attributions is None:
            self.last_batch_attributions = self.normalization_attribution(self.predict_corr, self.spilt_inform)
        else:
            _matrix = self.normalization_attribution(self.predict_corr, self.spilt_inform)
            for j in range(len(self.last_batch_attributions)):
                batch_c = round(self.cosin_relation(self.last_batch_attributions[j], _matrix[j]), 4)
                batch2batch.append(batch_c)
            self.last_batch_attributions = _matrix
        for i in range(np.shape(self.predict_corr)[1] - 1):
            c = round(self.cosin_relation(self.predict_corr[:, i], label), 4)
            correlation_coefficient.append(c)
        for arr in self.attributes_set:
            if arr['num'] == 1:
                if self.last_batch_attributions is None:
                    arr['correlation'].append(0)
                else:
                    arr['correlation'].append(batch2batch[count])
                arr['predict'].append(correlation_coefficient[count])
                count += arr['num']
            else:
                if self.last_batch_attributions is None:
                    arr['correlation'].append([arr['num'] * [0]][0])
                else:
                    arr['correlation'].append(batch2batch[count:count + arr['num']])
                arr['predict'].append(correlation_coefficient[count:count + arr['num']])
                count += arr['num']

    def save_file(self):
        pass

    @staticmethod
    def cosin_relation(a, b):
        denominator = (np.linalg.norm(a) * np.linalg.norm(b))
        if denominator == 0:
            return 0
        else:
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    @staticmethod
    def normalization_attribution(matrix, spilt_info):
        normalized_array = []
        for i in range(np.shape(matrix)[1] - 1):
            column = matrix[:, i].tolist()
            column_length = len(column)
            if len(spilt_info[i]) == 11:
                _temp = [[0] * 10][0]
                for unit in column:
                    for k in range(len(spilt_info[i]) - 1):
                        if spilt_info[i][k] <= unit <= spilt_info[i][k + 1]:
                            _temp[k] += 1
                            break
                normalized_array.append([value / column_length for value in _temp])
            else:
                temp = []
                for value in spilt_info[i]:
                    temp.append(column.count(value) / column_length)
                normalized_array.append(temp)
        return normalized_array

    @staticmethod
    def _get_date(days):
        aa = time.strptime('1800-01-01', "%Y-%m-%d")
        datetime.date(aa[0], aa[1], aa[2])
        return datetime.date(aa[0], aa[1], aa[2]) + datetime.timedelta(days=days)

    @staticmethod
    def construct_correlation(data_source):
        dataset_attributes = Dataset.get_attributes(data_source)
        attribute_object = []
        a = dict()
        for arr in dataset_attributes:
            a['name'] = arr[0]
            a['predict'] = []
            a['correlation'] = []
            a['num'] = arr[1]
            attribute_object.append(a)
            a = dict()
        return attribute_object