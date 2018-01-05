"""
Created on Thu Jan  4 16:19:13 2018

@author: Matin Kheirkhahan (matinkheirkhahan@ufl.edu)
"""
import pandas as pd
import numpy as np

class UCI_Handler(object):
    def __readUCIFiles(self, data_folder: str, dataset_type: str):
        """
        Read downloaded and extracted files from UCI repository and returns five pandas.DataFrames.
        - sensor_x: one vector containing x-axis data.
        - sensor_y: one vector containing y-axis data.
        - sensor_z: one vector containing z-axis data.
        (No process is performed on them)
        - subjects: one vector showing which participant performed each row of data (128 sensor readings)
        - activities: one vector showing each row is for what type of activity.
             1: Walking
             2: Walking upstairs
             3: Walking downstairs
             4: Sitting
             5: Standing
             6: Laying

        Files downloaded from UCI repository, "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/"

        :param data_folder: address of extracted folder. For example, "~/Desktop/UCI HAR Dataset/"
        :param dataset_type: either "train" or "test".
        :return: five pandas.DataFrame; sensor_x, sensor_y, sensor_z, subjects, activities. All of them have only one column.
        """
        sensor_x = pd.read_csv(r"{}Inertial Signals/total_acc_x_{}.txt".format(data_folder, dataset_type), header=None)
        sensor_y = pd.read_csv(r"{}Inertial Signals/total_acc_y_{}.txt".format(data_folder, dataset_type), header=None)
        sensor_z = pd.read_csv(r"{}Inertial Signals/total_acc_z_{}.txt".format(data_folder, dataset_type), header=None)
        subjects = pd.read_csv(r"{}subject_{}.txt".format(data_folder, dataset_type), header=None)
        activities = pd.read_csv(r"{}y_{}.txt".format(data_folder, dataset_type), header=None)

        return sensor_x, sensor_y, sensor_z, subjects, activities

    def __convertToAVector(self, rowSample):
        """
        Each row in the original files have 128 sensor readings. Row values are character strings and have several whitespaces.
        This function takes one row of 128 readings, handles missing values and extra whitespaces,
        and returns a column vector of numeric values.

        :param rowSample: a character string which contains 128 sensor readings to be extracted.
        :return: a column vector of 128 numeric values.
        """
        tokens = rowSample.split(" ")
        vec = []
        for t in tokens:
            if len(t) > 0:
                vec.append(float(t))
        return vec

    def saveUCIFilesAsCSV(self, data_folder: str, dataset_type: str):
        """
        Reads raw sensor data (Inertial Signals/total_acc_?_<dataset_type>.txt) and generates csv files in the following format.
                    Subject  |  Activity  |  Axis_X  |  Axis_Y  |  Axis_Z
                    -------------------------------------------------------
                        1    |  Walking   |  1.0214  |  0.0254  |  0.0019

        Subject information is found in "<dataset_type>/subject_<dataset_type>.txt" file.
        Activity information is found in "<dataset_type>/y_<dataset_type>.txt" file.

        The output is save in "<data_folder>/<dataset_type>_data.csv".

        :param data_folder: address of extracted folder. For example, "~/Desktop/UCI HAR Dataset/"
        :param dataset_type: either "train" or "test".
        """
        activity_dict = {1: "WALKING",
                         2: "WALKING_UPSTAIRS",
                         3: "WALKING_DOWNSTAIRS",
                         4: "SITTING",
                         5: "STANDING",
                         6: "LAYING"}
        # Reading raw sensor data and additional information from UCI files.
        sensor_x, sensor_y, sensor_z, subjects, activities = self.__readUCIFiles(data_folder, dataset_type)
        # preparing the output DataFrame
        final_df = None
        # Since rows have 50% overlapping parts, we skip every other row and append the rest.
        for i in range(0, sensor_x.shape[0], 2):
            vec_x = self.__convertToAVector(sensor_x.iloc[i, 0])
            vec_y = self.__convertToAVector(sensor_y.iloc[i, 0])
            vec_z = self.__convertToAVector(sensor_z.iloc[i, 0])
            activity = activity_dict[activities.iloc[i, 0]]
            currDF = pd.DataFrame(data={"subject": subjects.iloc[i, 0],
                                        "activity": activity,
                                        "X": vec_x,
                                        "Y": vec_y,
                                        "Z": vec_z})
            if final_df is None:
                final_df = currDF
            else:
                final_df = final_df.append(currDF, ignore_index=True)

        # Saving the final DataFrame as a csv file.
        final_df.to_csv(r"{}{}_data.csv".format(data_folder, dataset_type), index=False)

