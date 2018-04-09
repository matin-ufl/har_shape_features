from dictionary_learning.supervised_learner import Utils
import numpy as np
import pandas as pd
import time

class FeatureCalculation(object):

    def __calculate_distances_to_atoms(self, vector, dictionary_df, normalize=False, dist_func=Utils.dtw):
        """

        :param vector: [float] * atom_length. It is the vector of VM and the size is the same as a dictionary atom.
        :param dictionary_df: a DataFrame of size (m*n) where m is the number of atoms and columns are <VM1, ..., VM_n>
        :param normalize: if true, the `vector` will be normalized to take values in [0, 1]
        :return: a vector of [float]*n, where i^{th} point is the distance of the vector to i^{th} atom.
        """
        if normalize:
            minVal, maxVal = min(vector), max(vector)
            normalized_vector = (np.array(vector) - minVal) / (maxVal - minVal) if maxVal > minVal else np.array(
                [0] * len(vector))
            vector = normalized_vector
        else:
            vector = np.array(vector)
        number_of_atoms = dictionary_df.shape[0]
        result = [5] * number_of_atoms
        for atom_number in range(number_of_atoms):
            atom_df = dictionary_df.iloc[atom_number, :]
            result[atom_number] = dist_func(atom_df.values, vector)
        return result

    def calculate_features_for_one_participant(self, participant_df, dictionary_df, atom_length=None):
        """

        :param participant_df: <'Participant', 'Activity', 'Timestamp', 'X', 'Y', 'Z', 'VM'>
        :param dictionary_df: <'atom_number', 'frequency', 'VM1', ..., 'VM_n', 'activity'>
        :return:
        """
        if atom_length is None:
            atom_length = dictionary_df.shape[1] - 3

        VM = participant_df.VM.values
        Activity = participant_df.Activity.values
        timestamps = participant_df.Timestamp.values
        indeces = [i for i in range(0, participant_df.shape[0] - (atom_length), int(atom_length/2))]

        tic = time.time()
        colnames = dictionary_df.columns
        VM_colnames = []
        i = 1
        while "VM{}".format(i) in colnames:
            VM_colnames.append("VM{}".format(i))
            i += 1
        del i

        # Passing VM vector of atom_length size and the dictionary with only its VM values.
        mat = [self.__calculate_distances_to_atoms(VM[i:(i+atom_length)], dictionary_df.loc[:, VM_colnames]) for i in indeces]

        ppt = [participant_df.Participant.values[0]] * len(mat)
        activity = [Activity[i] for i in indeces]
        ts = [[timestamps[i], timestamps[i+atom_length]] for i in indeces]

        # Arranging the output dataframe
        atom_columns = ["atom_{}".format(i) for i in range(dictionary_df.shape[0])]
        new_df = pd.DataFrame(mat, columns=atom_columns)
        new_df.loc[:, "Participant"] = ppt
        new_df.loc[:, "Activity"] = activity
        new_df.loc[:, "time_start"] = [ts[i][0] for i in range(len(ts))]
        new_df.loc[:, "time_end"] = [ts[i][1] for i in range(len(ts))]

        # Reordering the output dataframe
        new_columns = ["Participant", "Activity", "time_start", "time_end"]
        new_columns.extend(atom_columns)
        new_df = new_df.loc[:, new_columns]
        toc = time.time()
        print("({}) Elapsed time: {:.2f} seconds.".format(ppt[0], toc - tic))

        return new_df