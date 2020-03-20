"""
@Author: Dustin Xu
@Date: 2020/2/9 14:56 PM
@Description: the method of evaluate with window size
"""
import numpy as np
from dictionary.tornado_dictionary import TornadoDic
from plotter.performance_plotter import *

class EvaluateWithWindowSize:
    def __init__(self, learner, detector, project, windowSize=500):
        self.window_size = windowSize
        self.learner = learner
        self.detector = detector

        self.count = 0
        self.allCount = 0
        self.accuracy = 0
        self.accuracyList = []
        self.accuracyAverageList = []

        self.__project_path = project.get_path()
        self.__project_name = project.get_name()
        self.learner_name = TornadoDic.get_short_names(self.learner.LEARNER_NAME)
        self.detector_name = self.detector.DETECTOR_NAME

        self.prediction_status = False

    def calculate_accuracy(self, prob, ground_truth, output_size=1, output_flag=False):
        """
        :param prob: Predicted value
        :param ground_truth: Ground truth
        :param output_size: Calculate the step size of accuracy
        :param output_flag: Print out ?
        :return: True or False
        """
        predicted_class = prob.index(max(prob))
        if predicted_class == ground_truth:
            self.count += 1
            self.prediction_status = True
        else:
            self.prediction_status = False
        self.allCount += 1
        self.accuracy = self.count / self.allCount
        if self.allCount % output_size == 0:
            self.accuracyList.append(self.accuracy)
            average_accuracy = round(sum(self.accuracyList)/len(self.accuracyList), 4)
            self.accuracyAverageList.append(average_accuracy)
            if output_flag:
                print("Current Accuracy:", self.accuracy)
                print("Average Accuracy:", average_accuracy)
        return self.prediction_status

    def store_stats(self):
        # 保存模型的准确率
        np.save(self.__project_path + self.learner_name + '+' + self.detector_name + '_accuracy.npy', np.vstack([self.accuracyList, self.accuracyAverageList]))
        # README
        # 保存模型准确率等等等各种前端数据
        # stats_writer = open(self.__project_path + "INFORMATION.txt", "w")
        # stats_writer.

    def plot(self, step=1, dataset=None, data=None):

        file_name = self.__project_name + "_single"

        Plotter.plot_single(self.learner_name + '+' + self.detector_name, self.accuracyList, "Accuracy",
                            self.__project_name, self.__project_path, file_name, None, 'upper right', 200,
                            datasetName=dataset, dataName=data, step=step)