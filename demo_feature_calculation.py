from feature_calculation.feature_calculation import FeatureCalculation
import pandas as pd
import numpy as np
import time
import os

def featCalc_3():

    fc = FeatureCalculation()

    dictionary_folder = r"/Users/matin/Dropbox/Work-Research/Current Directory/Shape Features/Data/UF dataset/final_dictionaries/"
    atom_length = 3
    dictionary_filename = dictionary_folder + str(atom_length) + "/final_dictionary_{}sec.csv".format(atom_length)
    dictionary_df = pd.read_csv(dictionary_filename)
    dataset = "Test_Set"

    in_folder = r"/Users/matin/Dropbox/Work-Research/Current Directory/Shape Features/Data/UF dataset/{}/".format(
        dataset)
    filenames = os.listdir(in_folder)

    for f in range(len(filenames)):
        print("{} out of {} ".format(f, len(filenames)), end="")
        filename = in_folder + filenames[f]
        if filename.endswith(".csv"):
            participant_df = pd.read_csv(filename)
            shape_df = fc.calculate_features_for_one_participant(participant_df, dictionary_df,
                                                                 atom_length*10)

            out_folder = r"/Users/matin/Dropbox/Work-Research/Current Directory/Shape Features/Data/UF dataset/shape_features/{}/{}/".format(
                atom_length, dataset)
            shape_df.to_csv("{}{}".format(out_folder, filenames[f]))


def avgFeatures_15secEpoch():
    atom_length = 6
    epoch_length = 15
    merge_size = int(epoch_length / (atom_length / 2)) - 1
    dataset = "Test_Set"
    in_folder = r"/Users/matin/Dropbox/Work-Research/Current Directory/Shape Features/Data/UF dataset/shape_features/{}/{}/".format(
        atom_length, dataset)
    out_folder = r"/Users/matin/Dropbox/Work-Research/Current Directory/Shape Features/Data/UF dataset/shape_features/"
    filenames = os.listdir(in_folder)

    result = None
    for f in range(len(filenames)):
        tic = time.time()
        filename = filenames[f]
        if filename.endswith(".csv"):
            df = pd.read_csv(in_folder + filename)
            dataMat = df.iloc[:, 5:]
            indices = [i for i in range(0, df.shape[0] - merge_size, merge_size)]

            avgMat = [dataMat[i:i + merge_size].mean(axis=0) for i in indices]
            new_df = pd.DataFrame(avgMat)
            atom_colnames = new_df.columns.values
            new_df.loc[:, "Participant"] = df.Participant.values[0]
            new_df.loc[:, "Activity"] = [df.Activity.values[i] for i in indices]
            new_df.loc[:, "time_start"] = [df.time_start.values[i] for i in indices]
            new_df.loc[:, "time_end"] = [df.time_end.values[i + merge_size - 1] for i in indices]
            colnames = ["Participant", "Activity", "time_start", "time_end"]
            colnames.extend(atom_colnames)
            new_df = new_df.loc[:, colnames]

            if result is None:
                result = new_df
            else:
                result = result.append(new_df, ignore_index=True)
            toc = time.time()
            print("{} of {} ... Done (elapsed time: {:.2f} seconds)".format(f, len(filenames), toc - tic))

    result.to_csv("{}shape_{}_{}sec.csv".format(out_folder, dataset, atom_length), index=False)


if __name__ == "__main__":
    #featCalc_3()
    avgFeatures_15secEpoch()