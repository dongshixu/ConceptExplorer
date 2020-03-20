"""
@Author: Dustin Xu
@Date: 2020/2/9 21:00 AM
@Description: multi test with prequential
"""
from data_structures.attribute_scheme import AttributeScheme
from classifier_xu.__init__ import *
from drift_detector_xu.__init__ import *
from evaluate_xu.__init__ import *
from filters.attribute_handlers import *
from data_stream_xu.data import Data
from data_stream_xu.dataset import Dataset
from filters.project_creator import Project
import copy as cp

# 1.data generator

# 2.test

# 3.evaluate

# 3.train

if __name__ == '__main__':
    for dataset in Dataset.DATASET:
        for sub_dataset in Dataset.get_sub_dataset(dataset):
            # Set variables
            __instance_count = 0
            __window_size = 500
            __step = 1000

            detection = True
            warning_status = False
            drift_status = False

            # classifier flag
            flag = False

            # Creating a data stream
            data = Data(dataset, sub_dataset)
            labels, attributes = data.get_attributes()
            attributes_scheme = AttributeScheme.get_scheme(attributes)
            __numeric_attribute_scheme = attributes_scheme['nominal']
            __numeric_attribute_scheme = attributes_scheme['numeric']

            # Initializing a learner
            learner = Logistic(labels, attributes_scheme['numeric'])
            learner_copy = cp.deepcopy(learner)

            # Initializing a Classifier-Detector Pairs
            pairs = [
                [OnlineAccuracyUpdatedEnsemble(labels, attributes_scheme['numeric'], learner_copy,
                                               windowSize=__window_size, classifierLimit=10), DDM()],
                [learner, DDM()]
            ]

            # Creating a save content
            project = Project('projects/multi/{}'.format(dataset), sub_dataset)

            # Creating a color set for plotting results
            colors = ['#FF0000', '#3E8ABF', '#1891FF',  '#4A0083', '#00FFFF']

            # Initializing a evaluator
            evaluator = EvaluateWithWindowSizeMulti(pairs, project, colors, __window_size)

            # train & test
            for x, y, attribute in data.data(batch_size=1):
                if attribute is not None:
                    attributes_scheme = AttributeScheme.get_scheme(attributes)
                    __numeric_attribute_scheme = attributes_scheme['nominal']
                    __numeric_attribute_scheme = attributes_scheme['numeric']
                    continue

                instance = x.tolist()[0] + [int(y.tolist()[0][0])]
                instance[0:len(instance) - 1] = Normalizer.normalize(instance[0:len(instance) - 1], __numeric_attribute_scheme)
                __instance_count += 1

                #
                for i, pair in enumerate(pairs):
                    classifier = pair[0]
                    drift_detector = pair[1]
                    if __instance_count > __window_size:
                        predicted_value = classifier.do_testing(instance)
                        prediction_status = evaluator.calculate_accuracy(predicted_value, instance[-1], i, output_size=__step, output_flag=False)
                        if detection is True:
                            warning_status, drift_status = drift_detector.detect(prediction_status)
                        else:
                            warning_status = False
                            drift_status = False
                        if drift_status:
                            print("Drift Points detected:", __instance_count)
                        classifier.do_training(instance, drift_status)
                    else:
                        classifier.do_training(instance, drift_status)
            evaluator.plot(step=__step, dataset=dataset, data=sub_dataset)
            evaluator.store_stats()
