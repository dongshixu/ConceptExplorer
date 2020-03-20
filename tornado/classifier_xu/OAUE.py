"""
@Author: Dustin Xu
@Date: 2020/2/9 13:30 PM
@Description: Online Accuracy Updated Ensemble
"""
import copy as cp
import numpy as np
from classifier_xu.classifier import SuperClassifier
from dictionary.tornado_dictionary import TornadoDic

class OnlineAccuracyUpdatedEnsemble(SuperClassifier):
    """
    OAUE ALGOROTHM
    """
    LEARNER_NAME = TornadoDic.OAUE
    LEARNER_TYPE = TornadoDic.TRAINABLE
    LEARNER_CATEGORY = TornadoDic.NUM_CLASSIFIER

    def __init__(self, labels, attributes, learner, windowSize=500, classifierLimit=10):
        super().__init__(labels, attributes)
        self.learner = learner
        self.__instance_counter = 0

        self.drift_status = False
        self.currentClassifierWeights = 0

        # set constant (remove after) maximum number of classifiers
        self.max_classifier = classifierLimit

        # size of d
        self.windowSize = windowSize
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

    def train(self, instance, drift_status):
        self.drift_status = drift_status
        y = instance[-1]
        if self.__instance_counter < self.windowSize:
            self.classDistribute[int(y)][0] += 1
        else:
            self.classDistribute[self.currentWindow[self.__instance_counter % self.windowSize]] -= 1
            self.classDistribute[int(y)][0] += 1
        self.currentWindow[self.__instance_counter % self.windowSize] = int(y)
        self.__instance_counter += 1
        self.computeMseR()

        if self.__instance_counter % self.windowSize == 0 or self.drift_status:
            # create a new classifier
            self.createNewClassifier(instance)
            self.creat_count += 1
            if self.drift_status:
                # record data and status
                pass
        else:
            # train classifier & update weights of classifier
            self.candidate_classifier.do_training(instance, self.drift_status)

            # self.getLossVotesForInstance(inst)
            for i in range(len(self.ensemble['classifier'])):
                self.weights[i] = self.computeWeight(i, instance)

        # train classifiers
        for classifier in self.ensemble['classifier']:
            classifier.do_training(instance, self.drift_status)

    def predict(self):
        print("error")
        pass

    def test(self, instance):
        return self.getVotesForInstance(instance)

    def computeMseR(self):
        self.mse_r = 0
        for value in self.classDistribute:
            p_c = value[0] / self.windowSize
            self.mse_r += p_c * ((1 - p_c) * (1 - p_c))

    def createNewClassifier(self, inst):
        # print("标签分布：", self.mse_r)
        candidate_classifier_weight = 1.0 / (self.mse_r + self.min_value)
        self.candidate_classifier.squareErrors = np.zeros(self.windowSize)

        for classifier in self.ensemble['classifier']:
            classifier.do_training(inst, self.drift_status)

        for i in range(len(self.ensemble['classifier'])):
            self.weights[i] = self.computeWeight(i, inst)

        self.candidate_classifier.birthday = self.__instance_counter

        if len(self.ensemble['classifier']) < self.max_classifier:
            self.ensemble['classifier'].append(self.candidate_classifier)
            self.weights.append(candidate_classifier_weight)
        else:
            w_index = self.weights.index(min(self.weights))
            worst_weight = self.weights[w_index]
            if candidate_classifier_weight > worst_weight:
                self.weights[w_index] = candidate_classifier_weight
                self.duration = self.__instance_counter - self.ensemble['classifier'][w_index].birthday
                self.duration_list.append(self.duration)
                self.ensemble['classifier'][w_index] = self.candidate_classifier

        self.candidate_classifier = cp.deepcopy(self.learner)
        self.candidate_classifier.reset()
        self.candidate_classifier.id = self.creat_count + 1

    def computeWeight(self, i, inst):

        d = self.windowSize
        t = self.__instance_counter - self.ensemble['classifier'][i].birthday

        votes = self.getVotesForInstance(inst)
        vote_sum = sum(votes)
        try:
            if vote_sum > 0:
                f_it = 1 - (votes[int(inst[-1])] / vote_sum)
                e_it = f_it * f_it
            else:
                e_it = 1.0
        except ValueError:
            e_it = 1.0

        if t > d:
            mse_it = self.ensemble['classifier'][i].mse_it + e_it / d - self.ensemble['classifier'][i].squareErrors[t % d] / d
        else:
            mse_it = self.ensemble['classifier'][i].mse_it * (t-1) / t + e_it / t

        self.ensemble['classifier'][i].squareErrors[t % d] = e_it
        self.ensemble['classifier'][i].mes_it = mse_it

        return 1.0 / (self.mse_r + mse_it + self.min_value)

    def getVotesForInstance(self, inst):
        votes = [0, 0]
        sum_weights = sum(self.weights)
        classifier_weights = np.zeros(len(self.ATTRIBUTES))
        if sum_weights == 0:
            sum_weights = 1.0
        for i, classifier in enumerate(self.ensemble['classifier']):
            prediction = classifier.test(inst)
            prediction = [value/sum(prediction) for value in prediction]
            prediction = [(self.weights[i]*value)/(1.0*len(self.ensemble['classifier'])+1.0) for value in prediction]
            votes = [votes[0] + prediction[0], votes[1] + prediction[1]]
            classifier_weights += np.dot(np.array(list(classifier.WEIGHTS.values())), self.weights[i]/sum_weights)
        self.currentClassifierWeights = classifier_weights
        return votes

