"""
@Author: Dustin Xu
@Date: 2020/2/14 19:31 PM
@Description: Create a folder where the results are saved
"""
import os
import time

class Folder:
    def __init__(self, project_folder):
        self.__project_path = self.__create(project_folder)

    @staticmethod
    def __create(project_folder):
        project_path = project_folder + '_result_' + str(time.strftime("%Y%m%d")) + "_" + str(
            time.strftime("%H%M%S"))

        if not os.path.exists(project_path):
            os.makedirs(project_path)

        print('The project path "' + project_path + '" is created.')
        return project_path

    def sub_folder(self, project_name):
        sub_folder = self.__project_path + '/' + project_name + '/'
        if not os.path.exists(sub_folder):
            os.makedirs(sub_folder)
        return sub_folder

    def get_path(self):
        return self.__project_path

