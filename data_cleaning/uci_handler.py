"""
Created on Thu Jan  4 16:19:13 2018

@author: Matin Kheirkhahan (matinkheirkhahan@ufl.edu)
"""
import pandas as pd
import numpy as np
import time

class UCI_Handler(object):
    def __readUCIFiles(self, data_folder: str, dataset_type: str, type_sensor: str):
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
        :param type_sensor: one of the following: 1) "total_acc", 2) "body_acc", 3) "body_gyro".
        :return: five pandas.DataFrame; sensor_x, sensor_y, sensor_z, subjects, activities. All of them have only one column.
        """
        sensor_x = pd.read_csv(r"{}Inertial Signals/{}_x_{}.txt".format(data_folder, type_sensor, dataset_type), header=None)
        sensor_y = pd.read_csv(r"{}Inertial Signals/{}_y_{}.txt".format(data_folder, type_sensor, dataset_type), header=None)
        sensor_z = pd.read_csv(r"{}Inertial Signals/{}_z_{}.txt".format(data_folder, type_sensor, dataset_type), header=None)
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

    def saveUCIFilesAsCSV(self, data_folder: str, dataset_type: str, type_sensor: str):
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
        :param type_sensor: one of the following: 1) "total_acc", 2) "body_acc", 3) "body_gyro".
        """
        activity_dict = {1: "WALKING",
                         2: "WALKING_UPSTAIRS",
                         3: "WALKING_DOWNSTAIRS",
                         4: "SITTING",
                         5: "STANDING",
                         6: "LAYING"}
        # Reading raw sensor data and additional information from UCI files.
        sensor_x, sensor_y, sensor_z, subjects, activities = self.__readUCIFiles(data_folder, dataset_type, type_sensor)
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
        final_df.to_csv(r"{}{}_{}_data.csv".format(data_folder, dataset_type, type_sensor), index=False)

    def downsampleAndSeparateSubjectData(self, data_folder: str, data_filename: str, out_folder: str):
        """
        Reads the csv data file generated by `saveUCIFilesAsCSV`, downsample data to 10 Hz and save the data for each
        subject separately.

        :param data_folder: address of the folder data files are located.
        :param data_filename: name of the csv file `saveUCIFilesAsCSV` generated.
        :param out_folder: address of the folder to save subjects' data. In this folder, files like `subjectXX.csv` will be saved.
        """
        tic = time.time()
        out_folder = out_folder if out_folder.endswith("/") else out_folder + "/"
        all_df = pd.read_csv("{}{}".format(data_folder, data_filename))
        subjectIDs = all_df.subject.unique()
        print("Data for {} subjects loaded.".format(len(subjectIDs)))

        for p, subjectID in enumerate(subjectIDs):
            ppt_df = all_df.loc[all_df.subject == subjectID, :]
            ppt_df.reset_index(inplace=True)
            ppt_10hz_df = None

            # since original files have 50Hz sampling rate, we reduce every 5 samples to 1 to achieve 10Hz data.
            for i in range(0, ppt_df.shape[0], 5):
                s, e = max(0, i - 2), min(i+3, ppt_df.shape[0])
                x, y, z = np.mean(ppt_df.loc[s:e, "X"]), np.mean(ppt_df.loc[s:e, "Y"]), np.mean(ppt_df.loc[s:e, "Z"])
                vm = np.sqrt(x**2 + y**2 + z**2)
                curr_df = pd.DataFrame(data={"subject": [ppt_df.loc[i, "subject"]],
                                             "activity": [ppt_df.loc[i, "activity"]],
                                             "X": [x],
                                             "Y": [y],
                                             "Z": [z],
                                             "VM": [vm]})
                if ppt_10hz_df is None:
                    ppt_10hz_df = curr_df
                else:
                    ppt_10hz_df = ppt_10hz_df.append(curr_df, ignore_index=True)
            ppt_10hz_df.to_csv(r"{}subject{}.csv".format(out_folder, subjectID), index=False)
            print("\r({} of {})".format(p+1, len(subjectIDs)), end="")
        toc = time.time()
        print("\n{} new files are saved in {}. (elapsed time: {:.2f} seconds)".format(len(subjectIDs), out_folder, toc - tic))
