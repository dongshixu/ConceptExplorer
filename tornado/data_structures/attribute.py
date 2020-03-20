
class Attribute:

    def __init__(self):
        self.NAME = None
        self.TYPE = None
        self.POSSIBLE_VALUES = []
        self.DATASET_NAME = None

        self.MAXIMUM_VALUE = None
        self.MINIMUM_VALUE = None

    def set_name(self, attr_name):
        self.NAME = attr_name

    def set_dataset_name(self, attr_name):
        self.NAME = attr_name

    def set_type(self, attr_type):
        self.TYPE = attr_type

    def set_possible_values(self, attr_possible_values):
        self.POSSIBLE_VALUES = attr_possible_values

    def set_bounds_values(self, attr_min_value, attr_max_value):
        self.MINIMUM_VALUE = attr_min_value
        if attr_min_value == attr_max_value:
            self.MAXIMUM_VALUE = self.MINIMUM_VALUE + 1
        else:
            self.MAXIMUM_VALUE = attr_max_value
