"""
@Author: Dustin Xu
@Date: 2020/2/20 11:16 AM
@Description: you know
"""
from data_structures.attribute_scheme import AttributeScheme
from classifier_xu.__init__ import *
from drift_detector_xu.__init__ import *
from evaluate_xu.__init__ import *
from result_save_xu.__init__ import *
from data_stream_xu.data import Data
from data_stream_xu.dataset import Dataset
from filters.attribute_handlers import *
from filters.project_creator import Project
from data_structures.attribute import Attribute

import time
import datetime
import numpy as np
import json
import threading

# 1.data generator

# 2.test

# 3.evaluate

# 4.train

class DistributedOnlineLearning:
    def __init__(self, con):
        self.con = con
        self.attributes = []

        self.construct_attribute()
        self.nb = NaiveBayes([0, 1], self.attributes)
        self.last_wl_status = {}
        self.nb_set = {}
        self.instance_set = {}
        self.nb_classifier_accuracy = {}

    def construct_attribute(self):
        for attr in [('warning_level', 2, [0, 1, 2])]:
            for index in range(attr[1]):
                attribute = Attribute()
                attribute.set_name('{}_{}'.format(attr[0], index))
                attribute.set_type(TornadoDic.NOMINAL_ATTRIBUTE)
                attribute.set_possible_values(attr[2])
                self.attributes.append(attribute)

    @staticmethod
    def save_file(file, path, type_name=None):
        if type_name is not None:
            filename = path + '{}.json'.format(type_name)
        else:
            filename = path + '.json'
        with open(filename, 'w') as file_obj:
            json.dump(file, file_obj)
            print(filename + '==>' + 'saved ok!')

    @staticmethod
    def cosin_relation(a, b):
        denominator = (np.linalg.norm(a) * np.linalg.norm(b))
        if denominator == 0:
            return 0
        else:
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    @staticmethod
    def get_date(days):
        aa = time.strptime('1800-01-01', "%Y-%m-%d")
        datetime.date(aa[0], aa[1], aa[2])
        return datetime.date(aa[0], aa[1], aa[2]) + datetime.timedelta(days=days)

    @staticmethod
    def calculate_warning_level(warning_list, warning_output, hit_list, w_l_m, repeated_flag=False):
        if repeated_flag:
            warning_output.append(warning_output[-1])
            if len(hit_list) == 0:
                hit_list.append(0)
            else:
                hit_list.append(hit_list[-1])
        else:
            _avg = round(sum(warning_list) / len(warning_list), 4)
            _max = round(max(warning_list), 4)
            if _max > w_l_m:
                w_l_m = _max
            if _max >= 3:
                hit_list.append(1)
            else:
                hit_list.append(0)
            warning_output.append({
                'avg': _avg,
                'max': _max
            })
            warning_list = []
        return warning_output, warning_list, hit_list, w_l_m

    @staticmethod
    def normalization_attribution(matrix, spilt_info):
        matrix = np.array(matrix)
        normalized_array = []
        for i in range(np.shape(matrix)[1]-1):
            column = matrix[:, i].tolist()
            column_length = len(column)
            if len(spilt_info[i]) == 11:
                _temp = [[0]*10][0]
                for unit in column:
                    for k in range(len(spilt_info[i])-1):
                        if spilt_info[i][k] <= unit <= spilt_info[i][k+1]:
                            _temp[k] += 1
                            break
                normalized_array.append([value/column_length for value in _temp])
            else:
                temp = []
                for value in spilt_info[i]:
                    temp.append(column.count(value)/column_length)
                normalized_array.append(temp)
        return normalized_array

    @staticmethod
    def calculate_correlation(matrix, last_batch, attr_dict, SI, label_correlation, batch_correlation,
                              repeat_flag=False):
        if repeat_flag:
            last_batch_nor = last_batch
            count = 0
            for arr in attr_dict:
                if arr['num'] == 1:
                    if last_batch is None:
                        arr['correlation'].append(0)
                    else:
                        arr['correlation'].append(batch_correlation[count])
                    arr['predict'].append(label_correlation[count])
                    count += arr['num']
                else:
                    if last_batch is None:
                        arr['correlation'].append([arr['num'] * [0]][0])
                    else:
                        arr['correlation'].append(batch_correlation[count:count + arr['num']])
                    arr['predict'].append(label_correlation[count:count + arr['num']])
                    count += arr['num']
        else:
            matrix = np.array(matrix)
            label = matrix[:, -1]
            correlation_coefficient = []
            batch2batch = []
            count = 0
            if last_batch is None:
                last_batch_nor = DistributedOnlineLearning.normalization_attribution(matrix, SI)
            else:
                _matrix = DistributedOnlineLearning.normalization_attribution(matrix, SI)
                for j in range(len(last_batch)):
                    batch_c = round(DistributedOnlineLearning.cosin_relation(last_batch[j], _matrix[j]), 4)
                    batch2batch.append(batch_c)
                batch_correlation = batch2batch
                last_batch_nor = _matrix
            for i in range(np.shape(matrix)[1] - 1):
                c = round(DistributedOnlineLearning.cosin_relation(matrix[:, i], label), 4)
                correlation_coefficient.append(c)
            label_correlation = correlation_coefficient
            for arr in attr_dict:
                if arr['num'] == 1:
                    if last_batch is None:
                        arr['correlation'].append(0)
                    else:
                        arr['correlation'].append(batch2batch[count])
                    arr['predict'].append(correlation_coefficient[count])
                    count += arr['num']
                else:
                    if last_batch is None:
                        arr['correlation'].append([arr['num'] * [0]][0])
                    else:
                        arr['correlation'].append(batch2batch[count:count + arr['num']])
                    arr['predict'].append(correlation_coefficient[count:count + arr['num']])
                    count += arr['num']
        return attr_dict, last_batch_nor, label_correlation, batch_correlation

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

    @staticmethod
    def statistic_spilt_information(inst, container):
        inst = inst[:-1]
        if len(container.keys()) == 0:
            for i, value in enumerate(inst):
                container[i] = {value}
        else:
            for i, value in enumerate(inst):
                container[i].add(value)
        return container

    @staticmethod
    def spilt(array):
        spilt_information = {}
        for i in range(len(array)):
            if len(array[i]) > 10:
                spilt_information[i] = [i*0.1 for i in range(0, 11)]
            else:
                spilt_information[i] = sorted(array[i])
        return spilt_information

    @staticmethod
    def wl_transformer(num):
        if num < 2:
            return 0
        elif 2 <= num < 3:
            return 1
        else:
            return 2

    def sub_thread(self, dataset, d_id, sub_dataset, spilt_inform, folder_create, length):

        global global_count

        self.nb_set[sub_dataset] = NaiveBayes([0, 1], self.attributes)
        self.last_wl_status[sub_dataset] = []
        self.instance_set[sub_dataset] = []
        self.nb_classifier_accuracy[sub_dataset] = dict(all_count=0, right_count=0, accuracy=[])

        # Set variables
        date_time_flag = False
        data_set_id = d_id

        __batch_size = 0

        __instance_count = 0
        __window_size = 500
        __step = 1000
        __start_point = 0
        __count = 0
        # __bingo = 0
        __last_unix_time = 0
        __last_warning_status = False
        __last_drift_status = False
        __warning_level_max = 0
        __data_length = Dataset.get_length(dataset, sub_dataset)

        lc = []
        bc = []

        configure = {}
        data_statistics = dict()
        delay = {}
        online = {}

        weights = []
        accuracy = []
        batch_start_time = []
        batch_count = []
        # right_count = []
        warning = []
        current_warning_status = {}
        warning_level = []
        warning_level_set = []
        drift = []
        current_drift_status = {}
        predict_corr = []
        hit = []
        last_batch_attributions = None

        detection = True
        drift_status = False

        # classifier flag
        prsa_flag = False

        attributes_set = DistributedOnlineLearning.construct_correlation(dataset)

        # Creating a data stream
        data = Data(dataset, sub_dataset)
        labels, attributes = data.get_attributes()
        attributes_scheme = AttributeScheme.get_scheme(attributes)
        __numeric_attribute_scheme = attributes_scheme['numeric']

        # Creating a save content
        project = Project('projects/single/{}'.format(dataset), sub_dataset)
        sub_folder_path = folder_create.sub_folder(sub_dataset)

        # Initializing a learner
        learner = Logistic(labels, attributes_scheme['numeric'])
        learner = OnlineAccuracyUpdatedEnsemble(labels, attributes_scheme['numeric'], learner,
                                                windowSize=__window_size, classifierLimit=10)

        # Initializing a drift detector
        drift_detector = DDM()

        # Initializing a evaluator
        evaluator = EvaluateWithWindowSize(learner, drift_detector, project, __window_size)

        # train & test
        for x, y, attribute in data.data(batch_size=1):
            if attribute is not None:
                attributes_scheme = AttributeScheme.get_scheme(attributes)
                __numeric_attribute_scheme = attributes_scheme['numeric']
                continue

            instance = x.tolist()[0] + [int(y.tolist()[0][0])]

            # 每条数据的unix时间戳
            # prsa data
            if dataset == 'prsa_data':
                date_time_flag = True
                date_time = list(map(int, instance[:4]))
                d = datetime.date(date_time[0], date_time[1], date_time[2])
                tt = datetime.time(date_time[3])
                datetime_str = str(d) + ' ' + str(tt)
                unix_time = int(time.mktime(time.strptime(datetime_str, '%Y-%m-%d %H:%M:%S')))
                if unix_time >= 1363881600:
                    prsa_flag = True
            elif dataset == 'movie_data':
                # movie data
                if instance[-2] > 62091:
                    date_time_flag = True
                    prsa_flag = True
                    date_time = DistributedOnlineLearning.get_date(instance[-2])
                    datetime_str = str(date_time) + ' ' + '00:00:00'
                    unix_time = int(time.mktime(time.strptime(datetime_str, '%Y-%m-%d %H:%M:%S')))

            instance[0:len(instance) - 1] = Normalizer.normalize(instance[0:len(instance) - 1],
                                                                 __numeric_attribute_scheme)
            __instance_count += 1

            if __instance_count > __window_size and date_time_flag and prsa_flag:
                if dataset == 'prsa_data':
                    if unix_time == 1363881600:
                        __start_point = unix_time
                        batch_start_time.append(__start_point)
                elif dataset == 'movie_data':
                    if __instance_count == __window_size + 1:
                        __start_point = unix_time
                        batch_start_time.append(__start_point)

                difference_value = unix_time - __start_point
                if difference_value >= __batch_size:
                    self.con.acquire()  # 获得锁
                    batch_interval = int(difference_value / __batch_size)
                    for cc in range(batch_interval):
                        if cc == 0:
                            r_f = False
                        else:
                            r_f = True
                        __start_point += __batch_size
                        # for every batch
                        # 权重
                        weights.append(learner.currentClassifierWeights.tolist())
                        # 属性记录
                        attributes_set, last_batch_attributions, lc, bc = self.calculate_correlation(
                            predict_corr, last_batch_attributions, attributes_set, spilt_inform, lc, bc,
                            repeat_flag=r_f)
                        # 准确度（可能会变, 目前是后端计算drift的准确率）
                        accuracy.append(drift_detector.accuracy)
                        # batch开始时间
                        batch_start_time.append(__start_point)
                        # batch 数量
                        batch_count.append(__count)
                        # warning level
                        warning_level, warning_level_set, hit, __warning_level_max = self.calculate_warning_level(
                            warning_level_set, warning_level, hit, __warning_level_max, repeated_flag=r_f)
                        # 保存每个数据源的当前状态
                        self.last_wl_status[sub_dataset] = [self.wl_transformer(warning_level[-1]['max']), hit[-1]]
                        __count = 0

                        global_count += 1

                        if global_count == length:
                            global_count = 0
                            # 训练和测试每个模型的贝叶斯
                            if len(warning_level) > 1:
                                d_s_n = self.nb_set.keys()
                                for data_set_name in d_s_n:
                                    self.instance_set[data_set_name] = [self.last_wl_status[value][0]
                                                                        for value in d_s_n if value != data_set_name]\
                                                                       + [self.last_wl_status[data_set_name][-1]]
                                if len(warning_level) > 2:
                                    # testing
                                    for temple_name in d_s_n:
                                        self.nb_set[temple_name].set_ready()
                                        predict = self.nb_set[temple_name].do_testing(self.instance_set[temple_name])
                                        if predict == self.instance_set[temple_name][-1]:
                                            self.nb_classifier_accuracy[temple_name]['right_count'] += 1
                                        self.nb_classifier_accuracy[temple_name]['all_count'] += 1
                                        self.nb_classifier_accuracy[temple_name]['accuracy'].append(
                                            round(self.nb_classifier_accuracy[temple_name]['right_count']
                                                  / self.nb_classifier_accuracy[temple_name]['all_count'], 4))
                                    # training
                                    for temple_name in d_s_n:
                                        self.nb_set[temple_name].do_training(self.instance_set[temple_name], drift_status)
                                else:
                                    for temple_name in d_s_n:
                                        self.nb_set[temple_name].do_training(self.instance_set[temple_name], drift_status)
                            # print(self.nb_classifier_accuracy)
                            self.con.notifyAll()
                        else:
                            self.con.wait()

                    predict_corr = [instance]
                    __count = 1
                else:
                    __count += 1
                    predict_corr.append(instance)

                warning_level_set.append(drift_detector.risk)

                predicted_value = learner.do_testing(instance)

                prediction_status = evaluator.calculate_accuracy(predicted_value, instance[-1],
                                                                 output_size=__step, output_flag=False)
                # if prediction_status:
                #     __bingo += 1

                if detection is True:
                    warning_status, drift_status = drift_detector.detect(prediction_status)
                    if warning_status is not __last_warning_status:
                        if warning_status:
                            current_warning_status['start'] = unix_time
                            current_warning_status['max_accuracy'] = [drift_detector.o_s_d_min]
                            current_warning_status['max_accuracy_time'] = [unix_time]
                            current_warning_status['backend_accuracy'] = [drift_detector.accuracy]
                        else:
                            current_warning_status['end'] = __last_unix_time
                            warning.append(current_warning_status)
                            current_warning_status = {}
                    else:
                        if warning_status:
                            current_warning_status['max_accuracy'].append(drift_detector.o_s_d_min)
                            current_warning_status['max_accuracy_time'].append(unix_time)
                            current_warning_status['backend_accuracy'].append(drift_detector.accuracy)
                    if drift_status is not __last_drift_status:
                        if drift_status:
                            current_drift_status['start'] = unix_time
                            current_drift_status['max_accuracy'] = [drift_detector.o_s_d_min]
                            current_drift_status['max_accuracy_time'] = [unix_time]
                            current_drift_status['backend_accuracy'] = [drift_detector.accuracy]
                        else:
                            current_drift_status['end'] = __last_unix_time
                            drift.append(current_drift_status)
                            current_drift_status = {}
                    else:
                        if drift_status:
                            current_drift_status['max_accuracy'].append(drift_detector.o_s_d_min)
                            current_drift_status['max_accuracy_time'].append(unix_time)
                            current_drift_status['backend_accuracy'].append(drift_detector.accuracy)

                    __last_warning_status = warning_status
                    __last_drift_status = drift_status
                    __last_unix_time = unix_time
                else:
                    # warning_status = False
                    drift_status = False

                if __instance_count == __data_length:  # 最后一个batch可能只有少部分数据，要考虑

                    self.con.acquire()  # 获得锁

                    # 权重
                    weights.append(learner.currentClassifierWeights.tolist())
                    # 属性记录
                    attributes_set, last_batch_attributions, lc, bc = self.calculate_correlation(
                        predict_corr, last_batch_attributions, attributes_set, spilt_inform, lc, bc,
                        repeat_flag=False)
                    # 准确度（可能会变, 目前是后端计算drift的准确率）
                    accuracy.append(drift_detector.accuracy)
                    # batch 数量
                    batch_count.append(__count)
                    # # 目前是 batch 中预测正确的个数， 后面改为concept drift 预测正确个数
                    # right_count.append(__bingo)
                    # warning level
                    warning_level, warning_level_set, hit, __warning_level_max = self.calculate_warning_level(
                        warning_level_set, warning_level, hit, __warning_level_max, repeated_flag=False)

                    global_count += 1

                    if global_count == length:
                        global_count = 0
                        # 训练和测试每个模型的贝叶斯
                        if len(warning_level) > 1:
                            d_s_n = self.nb_set.keys()
                            for data_set_name in d_s_n:
                                self.instance_set[data_set_name] = [self.last_wl_status[value][0]
                                                                    for value in d_s_n if value != data_set_name] \
                                                                   + [self.last_wl_status[data_set_name][-1]]
                            if len(warning_level) > 2:
                                # testing
                                for temple_name in d_s_n:
                                    predict = self.nb_set[temple_name].do_testing(self.instance_set[temple_name])
                                    if predict == self.instance_set[temple_name][-1]:
                                        self.nb_classifier_accuracy[temple_name]['right_count'] += 1
                                    self.nb_classifier_accuracy[temple_name]['all_count'] += 1
                                    self.nb_classifier_accuracy[temple_name]['accuracy'].append(
                                        round(self.nb_classifier_accuracy[temple_name]['right_count']
                                              / self.nb_classifier_accuracy[temple_name]['all_count'], 4))
                                # training
                                for temple_name in d_s_n:
                                    self.nb_set[temple_name].do_training(self.instance_set[temple_name], drift_status)
                            else:
                                for temple_name in d_s_n:
                                    self.nb_set[temple_name].do_training(self.instance_set[temple_name], drift_status)
                        # 保存最后一个数据源的状态
                        configure['timeStart'] = batch_start_time[0]
                        configure['timeEnd'] = batch_start_time[-1]
                        configure['timeUnit'] = __batch_size
                        configure['dataNumMax'] = max(batch_count)
                        configure['warningLevelMax'] = __warning_level_max

                        data_statistics['name'] = sub_dataset
                        delay['time'] = batch_start_time
                        delay['accuracy'] = accuracy
                        delay['bingo'] = []
                        delay['hit'] = hit[1:]
                        delay['warning'] = warning
                        delay['drift'] = drift
                        delay['warningLevel'] = warning_level
                        delay['attributes'] = attributes_set

                        online['weight'] = weights
                        online['time'] = batch_start_time
                        online['dataNum'] = batch_count

                        data_statistics['delay'] = delay
                        data_statistics['online'] = online

                        save_path = sub_folder_path + '/'
                        self.save_file(configure, save_path, type_name='configure')
                        self.save_file(data_statistics, save_path, type_name=None)

                        # 提示所有数据训练完成，可结束主进程
                        print('All data has been trained. Please finish the main process manually!')
                        Zip.plot_multi(self.nb_classifier_accuracy)
                        Zip(folder_create.get_path())
                        self.con.notifyAll()
                    else:
                        configure['timeStart'] = batch_start_time[0]
                        configure['timeEnd'] = batch_start_time[-1]
                        configure['timeUnit'] = __batch_size
                        configure['dataNumMax'] = max(batch_count)

                        data_statistics['name'] = sub_dataset
                        delay['time'] = batch_start_time
                        delay['accuracy'] = accuracy
                        delay['bingo'] = []
                        delay['hit'] = hit[1:]
                        delay['warning'] = warning
                        delay['drift'] = drift
                        delay['warningLevel'] = warning_level
                        delay['attributes'] = attributes_set

                        online['weight'] = weights
                        online['time'] = batch_start_time
                        online['dataNum'] = batch_count

                        data_statistics['delay'] = delay
                        data_statistics['online'] = online

                        save_path = sub_folder_path + '/'
                        self.save_file(configure, save_path, type_name='configure')
                        self.save_file(data_statistics, save_path, type_name=None)

                        self.con.wait()

                # training
                learner.do_training(instance, drift_status)
            else:
                # training
                learner.do_training(instance, drift_status)


if __name__ == "__main__":
    global_count = 0
    global_test_flag = False
    lock_con = threading.Condition()
    dol = DistributedOnlineLearning(lock_con)
    threads = []
    for ds in Dataset.DATASET:
        information = Dataset.get_spilt_inform(ds)
        f_c = Folder('E:/zju/result/{}'.format(ds))
        sub_data_set_list = Dataset.get_sub_dataset(ds)
        for sds_id, sds in enumerate(sub_data_set_list):
            t = threading.Thread(target=dol.sub_thread, args=(ds, sds_id, sds, information, f_c, len(sub_data_set_list)))
            t.start()
            threads.append(t)
        for thread in threads:
            thread.join()