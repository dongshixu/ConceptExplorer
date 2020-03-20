"""

*** The logistic regression ***

"""

import random
from collections import OrderedDict
import numpy as np

from classifier_xu.classifier import SuperClassifier
from data_structures.attribute import Attribute
from dictionary.tornado_dictionary import TornadoDic


class Logistic(SuperClassifier):
    """basic logistic"""

    LEARNER_NAME = TornadoDic.LOGISTIC
    LEARNER_TYPE = TornadoDic.TRAINABLE
    LEARNER_CATEGORY = TornadoDic.NUM_CLASSIFIER

    __BIAS_ATTRIBUTE = Attribute()
    __BIAS_ATTRIBUTE.set_name("bias")
    __BIAS_ATTRIBUTE.set_type(TornadoDic.NUMERIC_ATTRIBUTE)
    __BIAS_ATTRIBUTE.set_possible_values(1)

    def __init__(self, labels, attributes, learning_rate=0.8):
        super().__init__(labels, attributes)

        attributes.append(self.__BIAS_ATTRIBUTE)
        self.seen_label = labels
        self.WEIGHTS = OrderedDict()
        self.__initialize_weights()
        self.LEARNING_RATE = learning_rate
        self.birthday = 0
        self.dataOfDeath = 0
        self.mse_it = 0
        self.squareErrors = 0
        self.id = 0
        random.seed(1)

    def __initialize_weights(self):
        for a in self.ATTRIBUTES:
            self.WEIGHTS[a.NAME] = 0.2 * random.random() - 0.1
            # self.WEIGHTS[a.NAME] = 1.0

    def train(self, instance, drift_status):
        x = instance[0:len(instance) - 1]
        x.append(1)
        y_real = instance[len(instance) - 1]
        prediction = self.predict(x)
        p = np.clip(prediction, 0.00001, 1-0.00001)
        err = y_real - p
        # update the weights and bias
        for i in range(len(instance)):
            self.WEIGHTS[self.ATTRIBUTES[i].NAME] += self.LEARNING_RATE * x[i] * err
        self._IS_READY = True

    def predict(self, x):
        s = 0
        for i in range(0, len(x)):
            s += self.WEIGHTS[self.ATTRIBUTES[i].NAME] * x[i]
        # print("value of s ==>", s)
        if s >= 0:
            return 1.0 / (1 + np.exp(-s))
        else:
            return np.exp(s) / (1 + np.exp(s))
        # p = 1.0 / (1 + np.exp(-s))
        # return p

    def getLoss(self, instance):
        x = instance[0:len(instance) - 1]
        y = instance[len(instance) - 1]
        x.append(1)
        y_predicted = self.predict(x)
        y_predicted = np.clip(y_predicted, 0.00001, 1-0.00001)
        loss = - y * np.log(y_predicted) - (1 - y) * np.log(1 - y_predicted)
        return loss

    def test(self, instance):
        x = instance[0:len(instance) - 1]
        x.append(1)
        y_predicted = self.predict(x)
        return [1-y_predicted, y_predicted]

    def reset(self):
        super()._reset_stats()
        self.WEIGHTS = OrderedDict()
        self.__initialize_weights()