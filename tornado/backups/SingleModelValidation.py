"""
@Author: Dustin Xu
@Date: 2020/2/7 10:48 AM
@Description: Online Accuracy Updated Ensemble
"""
import copy as cp
import random

import numpy as np
from pympler import asizeof
from matplotlib.pyplot import MultipleLocator

from archiver.archiver import Archiver
from data_structures.attribute_scheme import AttributeScheme
from evaluators.classifier_evaluator import PredictionEvaluator
from evaluators.detector_evaluator import DriftDetectionEvaluator
from plotter.performance_plotter import *
from filters.attribute_handlers import *
from streams.readers.arff_reader import *


class OnlineAccuracyUpdatedEnsemble:
    """This class lets one run a classifier with a drift detector against a data stream,
    and evaluate it prequentially over time. Also, one is able to measure the detection
    false positive as well as false negative rates."""

    def __init__(self, learner, drift_detector, attributes, attributes_scheme,
                 actual_drift_points, drift_acceptance_interval, memory_check_step=-1, test_flag=False):

        self.learner = learner
        self.drift_detector = drift_detector
        self.testFlag = test_flag

        self.__instance_counter = 0

        self.__learner_error_rate_array = []
        self.__learner_memory_usage = []
        self.__learner_runtime = []

        self.__actual_drift_points = actual_drift_points
        self.__drift_acceptance_interval = drift_acceptance_interval

        self.__located_drift_points = []
        self.__drift_points_boolean = []
        self.__drift_detection_memory_usage = []
        self.__drift_detection_runtime = []
        self.drift_status = False
        self.warning_status = False
        self.prediction_status = True

        self.__attributes = attributes
        self.__numeric_attribute_scheme = attributes_scheme['numeric']
        self.__nominal_attribute_scheme = attributes_scheme['nominal']

        self.__memory_check_step = memory_check_step

        # set constant (remove after) maximum number of classifiers
        self.max_classifier = 10

        # size of d
        self.windowSize = 500
        # classifier list
        self.ensemble = dict(classifier=[], mse_it=[])
        # weights of classifiers
        self.weights = []
        # classes
        self.attr_class = [0, 1]
        # class distribute
        self.classDistribute = np.zeros([len(self.attr_class), 1])
        # New candidate classifier
        self.candidate_classifier = cp.deepcopy(learner)
        print(self.candidate_classifier.WEIGHTS)
        self.candidate_classifier.squareErrors = np.zeros(self.windowSize)
        self.candidate_classifier.id = 0
        self.duration = 0
        #
        self.mse_r = 0.0
        #
        self.currentWindow = [self.windowSize*[0]][0]

        #
        self.min_value = 4.9e-324

        # statistic variables
        self.accuracyList = []
        self.accuracyAverageList = []
        self.count = 0
        self.prob = 0
        self.allCount = 0
        self.creat_count = 0
        self.loss = []

        self.dataset_name = None

        # statistic
        self.duration_list = []

    def run(self, data, random_seed=1, detector=False, dataset=None):

        random.seed(random_seed)

        self.dataset_name = dataset

        for x, y, attributes in data.data(batch_size=1):

            if attributes != None:
                self.__attributes = attributes
                attributes_scheme = AttributeScheme.get_scheme(attributes)
                self.__numeric_attribute_scheme = attributes_scheme['numeric']
                self.__nominal_attribute_scheme = attributes_scheme['nominal']
                self.duration_list = []
                continue

            # deal single instance: combine & normalize
            inst = np.hstack([x, y]).tolist()[0]
            inst[0:len(inst) - 1] = Normalizer.normalize(inst[0:len(inst) - 1], self.__numeric_attribute_scheme)
            inst[-1] = int(inst[-1])
            self.__instance_counter += 1

            if self.__instance_counter >= 500:
                # predict & judge the drift status
                self.prob = self.candidate_classifier.getPredict(inst)

                # Accuracy of calculation
                self.calculate_accuracy(self.prob, y, output_size=1, output_flag=False)

                self.candidate_classifier.do_training(inst)
            else:
                self.candidate_classifier.do_training(inst)

    def calculate_accuracy(self, prob, ground_truth, output_size=1, output_flag=False):
        predicted_class = prob.index(max(prob))
        if predicted_class == ground_truth:
            self.count += 1
            self.prediction_status = True
        else:
            self.prediction_status = False
        self.allCount += 1
        acc = self.count / self.allCount
        if self.allCount % output_size == 0:
            self.accuracyList.append(round(acc, 4))
            average_accuracy = round(sum(self.accuracyList)/len(self.accuracyList), 4)
            self.accuracyAverageList.append(average_accuracy)
            if output_flag:
                print("Current Accuracy:", acc)
                print("Average Accuracy:", average_accuracy)

    def recordStatusData(self):
        # self.__drift_points_boolean.append(1)
        # self.__located_drift_points.append(self.__instance_counter)
        #
        # learner_error_rate = PredictionEvaluator.calculate(TornadoDic.ERROR_RATE,
        #                                                    self.learner.get_global_confusion_matrix())
        # self.__learner_error_rate_array.append(round(learner_error_rate, 4))
        # self.__learner_memory_usage.append(asizeof.asizeof(self.learner, limit=20))
        # self.__learner_runtime.append(self.learner.get_running_time())
        #
        # self.__drift_detection_memory_usage.append(asizeof.asizeof(self.drift_detector, limit=20))
        # self.__drift_detection_runtime.append(self.drift_detector.RUNTIME)

        self.drift_detector.reset()
        # print(learner_error_rate, '\n')

    def __plot(self, y):
        x = [i for i in range(len(y))]
        # fig = plt.figure()
        plt.plot(x, y)
        # y_major_locator = MultipleLocator(0.05)
        ax = plt.gca()
        # ax.yaxis.set_major_locator(y_major_locator)
        plt.ylim(-0.04, 1.04)
        plt.title(self.dataset_name)
        plt.show()
