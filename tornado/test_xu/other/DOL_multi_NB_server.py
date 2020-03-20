"""
@Author: Dustin Xu
@Date: 2020/2/24 18:13 AM
@Description: for server
"""
import sys
sys.path.insert(1, '/home/fhz_11821062/czx_21821062/xds/tornado')
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
        self.nb_batch_count = {}
        self.nb_drift_prob = {}
        self.plot_risk_level = {}
        self.data_statistics = {}
        self.configure = {}
        self.sub_file_path = {}
        self.warning_level_max = {}
        self.__ubuntu_time_difference = 8*3600

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
    def save_file_1(file, path, type_name=None):
        key = file.keys()
        for d_name in key:
            if type_name is not None:
                filename = path[d_name] + '{}.json'.format(type_name)
            else:
                filename = path[d_name] + d_name + '.json'
            if type_name is not 'configure':
                print(len(file[d_name]['delay']['time']), len(file[d_name]['delay']['warningLevel']), len(file[d_name]['delay']['nb_prob']))
            with open(filename, 'w') as file_obj:
                json.dump(file[d_name], file_obj)
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
            if _max < w_l_m:
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
                spilt_information[i] = [i * 0.1 for i in range(0, 11)]
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

    def sub_thread(self, dataset, sub_dataset, spilt_inform, folder_create, length):

        global global_count
        global naive_bayes_batch_count

        self.nb_set[sub_dataset] = NaiveBayes([0, 1], self.attributes)
        self.last_wl_status[sub_dataset] = dict(r_l=0, hit=[])
        self.instance_set[sub_dataset] = []
        self.nb_classifier_accuracy[sub_dataset] = dict(all_count=0, right_count=0, accuracy=[])
        self.nb_batch_count[sub_dataset] = 0
        self.nb_drift_prob[sub_dataset] = dict(prob=[], ground_truth=[])
        self.plot_risk_level[sub_dataset] = 0
        self.data_statistics[sub_dataset] = dict(name=sub_dataset,
                                                 delay=dict(time=[], accuracy=[], nb_prob=[], bingo=[], hit=[], warning=[], drift=[], warningLevel=[], attributes=self.construct_correlation(dataset), batch_delay=1),
                                                 online=dict(weight=[], time=[], dataNum=[]))
        self.configure[sub_dataset] = {}
        self.warning_level_max[sub_dataset] = 0

        # Set variables
        date_time_flag = False

        if dataset == 'prsa_data':
            __batch_size = 24 * 3600  # 3600 represent 1 hour
        elif dataset == 'movie_data':
            __batch_size = 24 * 7 * 3600  # 3600 represent 1 hour
        else:
            __batch_size = 0

        __instance_count = 0
        __window_size = 500
        __step = 1000
        __start_point = 0
        __count = 0
        __last_unix_time = 0
        __last_warning_status = False
        __last_drift_status = False
        __data_length = Dataset.get_length(dataset, sub_dataset)

        lc = []
        bc = []

        current_warning_status = {}
        warning_level_set = []
        current_drift_status = {}
        predict_corr = []
        last_batch_attributions = None

        detection = True
        drift_status = False

        # classifier flag
        prsa_flag = False

        # Creating a data stream
        data = Data(dataset, sub_dataset)
        labels, attributes = data.get_attributes()
        attributes_scheme = AttributeScheme.get_scheme(attributes)
        __numeric_attribute_scheme = attributes_scheme['numeric']

        # Creating a save content
        project = Project('projects/single/{}'.format(dataset), sub_dataset)
        self.sub_file_path[sub_dataset] = folder_create.sub_folder(sub_dataset)

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
                unix_time = unix_time - self.__ubuntu_time_difference
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
                    unix_time = unix_time - self.__ubuntu_time_difference

            instance[0:len(instance) - 1] = Normalizer.normalize(instance[0:len(instance) - 1],
                                                                 __numeric_attribute_scheme)
            __instance_count += 1

            if __instance_count > __window_size and date_time_flag and prsa_flag:
                if dataset == 'prsa_data':
                    if unix_time == 1363881600:
                        __start_point = unix_time
                        self.data_statistics[sub_dataset]['delay']['time'].append(__start_point)
                elif dataset == 'movie_data':
                    if __instance_count == __window_size + 1:
                        __start_point = unix_time
                        self.data_statistics[sub_dataset]['delay']['time'].append(__start_point)

                difference_value = unix_time - __start_point
                if difference_value >= __batch_size:
                    batch_interval = int(difference_value / __batch_size)
                    for cc in range(batch_interval):
                        if cc == 0:
                            r_f = False
                        else:
                            r_f = True
                        self.con.acquire()  # 获得锁
                        __start_point += __batch_size
                        # for every batch
                        # 权重
                        self.data_statistics[sub_dataset]['online']['weight'].append(learner.currentClassifierWeights.tolist())
                        # 属性记录
                        self.data_statistics[sub_dataset]['delay']['attributes'], last_batch_attributions, lc, bc = self.calculate_correlation(
                            predict_corr, last_batch_attributions, self.data_statistics[sub_dataset]['delay']['attributes'], spilt_inform, lc, bc,
                            repeat_flag=r_f)
                        # 准确度（可能会变, 目前是后端计算drift的准确率）
                        self.data_statistics[sub_dataset]['delay']['accuracy'].append(drift_detector.accuracy)
                        # batch开始时间
                        self.data_statistics[sub_dataset]['delay']['time'].append(__start_point)
                        # batch 数量
                        self.data_statistics[sub_dataset]['online']['dataNum'].append(__count)
                        # warning level
                        self.data_statistics[sub_dataset]['delay']['warningLevel'], warning_level_set, self.data_statistics[sub_dataset]['delay']['hit'], self.warning_level_max[sub_dataset] = self.calculate_warning_level(
                            warning_level_set, self.data_statistics[sub_dataset]['delay']['warningLevel'], self.data_statistics[sub_dataset]['delay']['hit'], self.warning_level_max[sub_dataset], repeated_flag=r_f)

                        __count = 0

                        self.last_wl_status[sub_dataset]['r_l'] = self.wl_transformer(self.data_statistics[sub_dataset]['delay']['warningLevel'][-1]['max'])
                        if len(self.last_wl_status[sub_dataset]['hit']) >= 2:
                            self.last_wl_status[sub_dataset]['hit'].pop(0)
                            self.last_wl_status[sub_dataset]['hit'].append(self.data_statistics[sub_dataset]['delay']['hit'][-1])
                        else:
                            self.last_wl_status[sub_dataset]['hit'].append(self.data_statistics[sub_dataset]['delay']['hit'][-1])

                        global_count += 1

                        self.plot_risk_level[sub_dataset] = self.data_statistics[sub_dataset]['delay']['warningLevel'][-1]['max']

                        if global_count == length:
                            global_count = 0
                            # 训练和测试每个模型的贝叶斯
                            d_s_n = self.nb_set.keys()
                            # 训练贝叶斯
                            if len(self.data_statistics[sub_dataset]['delay']['warningLevel']) >= 2:
                                for data_set_name in d_s_n:
                                    self.instance_set[data_set_name] = [self.last_wl_status[value]['r_l']
                                                                        for value in d_s_n if value != data_set_name] \
                                                                       + [max(self.last_wl_status[data_set_name]['hit'])]
                                if len(self.data_statistics[sub_dataset]['delay']['warningLevel']) >= 3:
                                    # testing
                                    for temple_name in d_s_n:
                                        self.nb_set[temple_name].set_ready()
                                        predict = self.nb_set[temple_name].do_testing(self.instance_set[temple_name])
                                        self.data_statistics[temple_name]['delay']['nb_prob'].append(self.nb_set[temple_name].drift_prob)
                                        self.nb_drift_prob[temple_name]['prob'].append(
                                            self.nb_set[temple_name].drift_prob)
                                        self.nb_drift_prob[temple_name]['ground_truth'].append(
                                            self.plot_risk_level[temple_name])

                                        if predict == self.instance_set[temple_name][-1]:
                                            self.nb_classifier_accuracy[temple_name]['right_count'] += 1
                                        self.nb_classifier_accuracy[temple_name]['all_count'] += 1
                                        self.nb_classifier_accuracy[temple_name]['accuracy'].append(
                                            round(self.nb_classifier_accuracy[temple_name]['right_count']
                                                  / self.nb_classifier_accuracy[temple_name]['all_count'], 4))
                                    # training
                                    for temple_name in d_s_n:
                                        self.nb_set[temple_name].do_training(self.instance_set[temple_name],
                                                                             drift_status)
                                else:
                                    for temple_name in d_s_n:
                                        self.nb_set[temple_name].do_training(self.instance_set[temple_name],
                                                                             drift_status)
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
                            self.data_statistics[sub_dataset]['delay']['warning'].append(current_warning_status)
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
                            self.data_statistics[sub_dataset]['delay']['drift'].append(current_drift_status)
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
                    self.data_statistics[sub_dataset]['online']['weight'].append(learner.currentClassifierWeights.tolist())
                    # 属性记录
                    self.data_statistics[sub_dataset]['delay']['attributes'], last_batch_attributions, lc, bc = self.calculate_correlation(
                        predict_corr, last_batch_attributions, self.data_statistics[sub_dataset]['delay']['attributes'], spilt_inform, lc, bc,
                        repeat_flag=False)
                    # 准确度（可能会变, 目前是后端计算drift的准确率）
                    self.data_statistics[sub_dataset]['delay']['accuracy'].append(drift_detector.accuracy)
                    # batch 数量
                    self.data_statistics[sub_dataset]['online']['dataNum'].append(__count)
                    # warning level
                    self.data_statistics[sub_dataset]['delay']['warningLevel'], warning_level_set, self.data_statistics[sub_dataset]['delay']['hit'], self.warning_level_max[sub_dataset] = self.calculate_warning_level(
                        warning_level_set, self.data_statistics[sub_dataset]['delay']['warningLevel'], self.data_statistics[sub_dataset]['delay']['hit'], self.warning_level_max[sub_dataset], repeated_flag=False)

                    self.last_wl_status[sub_dataset]['r_l'] = self.wl_transformer(self.data_statistics[sub_dataset]['delay']['warningLevel'][-1]['max'])
                    if len(self.last_wl_status[sub_dataset]['hit']) >= 2:
                        self.last_wl_status[sub_dataset]['hit'].pop(0)
                        self.last_wl_status[sub_dataset]['hit'].append(self.data_statistics[sub_dataset]['delay']['hit'][-1])
                    else:
                        self.last_wl_status[sub_dataset]['hit'].append(self.data_statistics[sub_dataset]['delay']['hit'][-1])

                    global_count += 1

                    # 画 drift probability
                    Zip.plot_multi_1(self.nb_drift_prob[sub_dataset], sub_dataset)

                    if global_count == length:
                        global_count = 0
                        # # 训练和测试每个模型的贝叶斯
                        d_s_n = self.nb_set.keys()
                        for data_set_name in d_s_n:
                            self.instance_set[data_set_name] = [self.last_wl_status[value]['r_l']
                                                                for value in d_s_n if value != data_set_name] \
                                                               + [max(self.last_wl_status[data_set_name]['hit'])]
                        # testing
                        for temple_name in d_s_n:
                            self.nb_set[temple_name].set_ready()
                            predict = self.nb_set[temple_name].do_testing(self.instance_set[temple_name])
                            self.data_statistics[temple_name]['delay']['nb_prob'].append(
                                self.nb_set[temple_name].drift_prob)
                            self.nb_drift_prob[temple_name]['prob'].append(self.nb_set[temple_name].drift_prob)
                            self.nb_drift_prob[temple_name]['ground_truth'].append(
                                self.plot_risk_level[temple_name])

                            if predict == self.instance_set[temple_name][-1]:
                                self.nb_classifier_accuracy[temple_name]['right_count'] += 1
                            self.nb_classifier_accuracy[temple_name]['all_count'] += 1
                            self.nb_classifier_accuracy[temple_name]['accuracy'].append(
                                round(self.nb_classifier_accuracy[temple_name]['right_count']
                                      / self.nb_classifier_accuracy[temple_name]['all_count'], 4))
                        # training
                        for temple_name in d_s_n:
                            self.nb_set[temple_name].do_training(self.instance_set[temple_name], drift_status)

                        # 保存每个数据源的状态
                        # ① 每个数据源概念漂移检测+贝叶斯drift概率  + configure

                        for key_name in self.data_statistics.keys():
                            self.configure[key_name]['timeStart'] = self.data_statistics[key_name]['delay']['time'][0]
                            self.configure[key_name]['timeEnd'] = self.data_statistics[key_name]['delay']['time'][-1]
                            self.configure[key_name]['timeUnit'] = __batch_size
                            self.configure[key_name]['dataNumMax'] = self.data_statistics[key_name]['online']['dataNum']
                            self.configure[key_name]['warningLevelMax'] = self.warning_level_max[key_name]
                            self.configure[key_name]['warningLevel'] = [[0, 2], [2, 3], [3, 100000]]
                            self.data_statistics[key_name]['delay']['hit'] = self.data_statistics[key_name]['delay']['hit'][:-1]

                        self.save_file_1(self.configure, self.sub_file_path, type_name='configure')
                        self.save_file_1(self.data_statistics, self.sub_file_path, type_name=None)

                        # 提示所有数据训练完成，可结束主进程
                        print('All data has been trained. Please finish the main process manually!')
                        self.save_file(self.nb_drift_prob, folder_create.get_path(),
                                       type_name='experiment_with_the_figure')
                        # Zip.plot_multi(self.nb_classifier_accuracy)
                        Zip(folder_create.get_path())
                        self.con.notifyAll()
                    else:
                        self.con.wait()

                # training
                learner.do_training(instance, drift_status)
            else:
                # training
                learner.do_training(instance, drift_status)


if __name__ == "__main__":
    global_count = 0
    naive_bayes_batch_count = 0
    global_test_flag = False
    lock_con = threading.Condition()
    dol = DistributedOnlineLearning(lock_con)
    threads = []
    for ds in Dataset.DATASET:
        information = Dataset.get_spilt_inform(ds)
        f_c = Folder('/home/fhz_11821062/czx_21821062/xds/result/{}'.format(ds))
        sub_data_set_list = Dataset.get_sub_dataset(ds)
        for sds_id, sds in enumerate(sub_data_set_list):
            t = threading.Thread(target=dol.sub_thread, args=(ds, sds, information, f_c, len(sub_data_set_list)))
            t.start()
            threads.append(t)
        for thread in threads:
            thread.join()