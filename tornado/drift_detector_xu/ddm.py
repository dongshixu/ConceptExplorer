"""
@Author: Dustin Xu
@Date: 2020/2/11 14:00 PM
@Description: DDM with slider window
"""

import math
import sys

from dictionary.tornado_dictionary import TornadoDic
from drift_detection.detector import SuperDetector


class DDM(SuperDetector):
    """The traditional Drift Detection Method (DDM) class."""

    DETECTOR_NAME = TornadoDic.DDM

    def __init__(self, min_instance=100, interval=1):

        super().__init__()

        self.MINIMUM_NUM_INSTANCES = min_instance
        self.NUM_INSTANCES_SEEN = 0
        self.DETECTION_INTERVAL = interval

        self.__P = 1
        self.__S = 0
        self.__P_min = sys.maxsize
        self.__S_min = sys.maxsize
        self.__sliderWindow = []

        self.warning_level = 0
        self.drift_level = 0
        self.current_level = 0
        self.P = 0
        self.S = 0

        self.risk = 0
        self.accuracy = 0
        self.o_s_d_min = 0

    def run(self, pr):

        warning_status, drift_status = False, False

        pr = 1 if pr is False else 0

        # 1. UPDATING STATS
        # self.__P += (pr - self.__P) / self.NUM_INSTANCES_SEEN
        # self.P = self.__P
        # self.__S = math.sqrt(self.__P * (1 - self.__P) / self.NUM_INSTANCES_SEEN)
        # self.S = self.__S

        self.NUM_INSTANCES_SEEN += 1

        # if self.NUM_INSTANCES_SEEN <= self.MINIMUM_NUM_INSTANCES:
        #     self.__sliderWindow.append(pr)
        #     return False, False
        # elif len(self.__sliderWindow) < self.MINIMUM_NUM_INSTANCES + self.DETECTION_INTERVAL - 1:
        #     self.__sliderWindow.pop(0)
        #     return False, False
        # else:
        #     self.__sliderWindow.pop(0)
        #     self.__sliderWindow.append(pr)
        #     self.__P = self.__sliderWindow.count(1) / self.MINIMUM_NUM_INSTANCES
        #     self.P = self.__P
        #     self.accuracy = 1 - self.__P
        #     self.__S = math.sqrt(self.__P * (1 - self.__P) / self.MINIMUM_NUM_INSTANCES)
        #     self.S = self.__S
        #     self.o_s_d_min = 1 - self.__P_min

        if len(self.__sliderWindow) < self.MINIMUM_NUM_INSTANCES + self.DETECTION_INTERVAL-1:
            self.__sliderWindow.append(pr)
            return False, False
        else:
            self.__sliderWindow.append(pr)
            self.__sliderWindow = self.__sliderWindow[self.DETECTION_INTERVAL:]
            self.__P = self.__sliderWindow.count(1) / self.MINIMUM_NUM_INSTANCES
            self.P = self.__P
            self.accuracy = 1 - self.__P
            self.__S = math.sqrt(self.__P * (1 - self.__P) / self.MINIMUM_NUM_INSTANCES)
            self.S = self.__S

        if self.__P + self.__S <= self.__P_min + self.__S_min:
            self.__P_min = self.__P
            self.__S_min = self.__S

        self.o_s_d_min = 1 - self.__P_min

        # 2. UPDATING WARNING AND DRIFT STATUSES
        self.current_level = self.__P + self.__S
        self.warning_level = self.__P_min + 2 * self.__S_min
        self.drift_level = self.__P_min + 3 * self.__S_min

        # print("P, S, P_min, S_min：", self.__P, self.__S, self.__P_min, self.__P_min)
        # print("current_level, warning_level, drift_level：", self.current_level, self.warning_level, self.drift_level)

        if self.current_level > self.warning_level:
            warning_status = True

        if self.current_level > self.drift_level:
            drift_status = True

        if self.__S_min == 0:
            if drift_status:
                self.risk = 3.
            else:
                self.risk = 0.
        else:
            self.risk = (self.current_level - self.__P_min) / self.__S_min

        if drift_status:
            self.__P_min = self.__P
            self.__S_min = self.__S
            # self.__P_min = sys.maxsize
            # self.__S_min = sys.maxsize

        # self.o_s_d_min = 1 - self.__P_min

        return warning_status, drift_status

    def reset(self):
        super().reset()
        self.NUM_INSTANCES_SEEN = 1
        self.__P = 1
        self.__S = 0
        self.__P_min = sys.maxsize
        self.__S_min = sys.maxsize

    def get_settings(self):
        return [str(self.MINIMUM_NUM_INSTANCES), "$n_{min}$:" + str(self.MINIMUM_NUM_INSTANCES)]
