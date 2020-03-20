"""
@Author: Dustin Xu
@Date: 2020/2/9 21:15 PM
@Description: the method of evaluate with window size (multi)
"""
import numpy as np
from dictionary.tornado_dictionary import TornadoDic
from plotter.performance_plotter import *

class EvaluateWithWindowSizeMulti:
    def __init__(self, pairs, project, color_set, windowSize=500):
        self.window_size = windowSize
        self.pairs = pairs
        self.project = project

        self.__project_path = project.get_path()
        self.__project_name = project.get_name()
        self.color_set = color_set

        self.count = []
        self.allCount = []
        self.accuracyList = []
        self.accuracyAverageList = []
        self.NameList = []
        self.unique_learners_names = []

        for pair in self.pairs:
            self.accuracyList.append([])
            self.accuracyAverageList.append([])
            self.NameList.append(TornadoDic.get_short_names(pair[0].LEARNER_NAME) + " + " + pair[1].DETECTOR_NAME)
            if self.unique_learners_names.__contains__(pair[0].LEARNER_NAME) is False:
                self.unique_learners_names.append(pair[0].LEARNER_NAME)
            self.count.append(0)
            self.allCount.append(0)
        self.prediction_status = False

    def calculate_accuracy(self, prob, ground_truth, index, output_size=1, output_flag=False):
        """
        :param prob: Predicted value
        :param ground_truth: Ground truth
        :param output_size: Calculate the step size of accuracy
        :param output_flag: Print out ?
        :return: True or False
        """
        predicted_class = prob.index(max(prob))
        if predicted_class == ground_truth:
            self.count[index] += 1
            self.prediction_status = True
        else:
            self.prediction_status = False
        self.allCount[index] += 1
        acc = self.count[index] / self.allCount[index]
        if self.allCount[index] % output_size == 0:
            self.accuracyList[index].append(round(acc, 4))
            average_accuracy = round(sum(self.accuracyList[index])/len(self.accuracyList[index]), 4)
            self.accuracyAverageList[index].append(average_accuracy)
            if output_flag:
                print("Current Accuracy:", acc)
                print("Average Accuracy:", average_accuracy)
        return self.prediction_status

    def store_stats(self):
        # 保存各个模型的准确率
        for i in range(len(self.pairs)):
            np.save(self.__project_path + self.NameList[i] + '_accuracy.npy', np.vstack([self.accuracyList[i], self.accuracyAverageList[i]]))
        # README
        # 保存模型准确率等等等各种前端数据
        # stats_writer = open(self.__project_path + "INFORMATION.txt", "w")
        # stats_writer.

    def plot(self, step=1, dataset=None, data=None):

        z_orders = []
        for i in range(len(self.NameList)):
            z_orders.append(len(self.NameList) - i + 1)

        file_name = self.__project_name + "_multi"

        # === Plotting Accuracy
        Plotter.plot_multiple(self.NameList, len(self.accuracyList[0]), self.accuracyList, 'Accuracy',
                              self.__project_name, self.__project_path, file_name, None, (1, 1.0125), 2,
                              len(self.unique_learners_names), 313, self.color_set, z_orders, step=step,
                              print_legend=True, datasetName=dataset, dataName=data)
