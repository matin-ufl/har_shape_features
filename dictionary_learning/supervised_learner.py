"""
Created on Sun Jan  7 16:28:11 2018

@author: Matin Kheirkhahan (matinkheirkhahan@ufl.edu)
"""

import pandas as pd
import numpy as np
import time
from dictionary_learning.utils import Utils
import matplotlib.pyplot as plt
from sklearn.manifold import MDS, TSNE
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

class SupervisedLearner(object):
    """
    This class provides functions to learn dictionary of recurring patterns by examining acceleration patterns in the
    activity of interest.
    """

    @staticmethod
    def hasOverlapWithPreviousOnes(atom_s, atom_e, mp_s, mp_e, previousAtoms: list):
        """
        Checks whether part of the selected acceleration pattern was previously covered in other atoms.
        To make sure unique information is preserved, we check overlap between
             a. current window and its similar window in the future (we refer to it as mp_window)
             b. previously selected atom and its similar window.
        Therefore, we have four comparisons in total.

        :param atom_s: start of the current acceleration pattern.
        :param atom_e: end of the current acceleration pattern.
        :param mp_s: start of the current acceleration pattern's similar window.
        :param mp_e: end of the current acceleration pattern's similar window.
        :param previousAtoms: List of (start, end, mp_start, mp_end) tuples showing previously selected acceleration patterns start and end points and their similar periods in the future.
        :return: True if there is an overlap between:
                  a. current window and a previously selected atom.
                  b. current window and a previously selected atom's similar pattern.
                  c. current window's similar pattern and a previously selected atom.
                  d. current window's similar pattern and a previously selected atom's similar pattern.
        """
        for prev_atom_s, prev_atom_e, prev_mp_s, prev_mp_e in previousAtoms:
            if atom_s < prev_atom_e and atom_e > prev_atom_s:
                return True
            if atom_s < prev_mp_e and atom_e > prev_mp_s:
                return True
            if mp_s < prev_atom_e and mp_e > prev_atom_s:
                return True
            if mp_s < prev_mp_e and mp_e > prev_mp_s:
                return True
        return False

    def learnDraftDictionary_triAxial(self, activity_df, draftDictionary_df, atom_length_sec, prc: int, dist_func=Utils.dtw, samplingRate=10):
        """
        Finds recurring patterns by calculating "matrix profile" and selecting the top `prc`% of them. The selected
        acceleration patterns will be appended to the `draftDictionary_df` DataFrame.

        This function finds recurring patterns using X, Y, and Z vectors.

        :param activity_df: pandas.DataFrame containing the accelerometer data for a specific participant and an activity.
        This DataFrame is populated by reading csv files produced by `UCI_Handler.downsampleAndSeparateSubjectData()`.
                       -> subject: int <-  |  -> activity: str <-  |  X: float  |  Y: float  |  Z: float  |  VM: float
                      -------------------------------------------------------------------------------------------------
                       --->   1      <--   |     -> SITTING <-     |   0.5670   |   0.1234   |   0.4567   |   0.7384
        :param draftDictionary_df: pandas.DataFrame containing previously found atoms from previous participants.
                       atom_number: int  |  X: float  |  Y: float  |  Z: float
                       -----------------------------------------------------------
                                 1       |   0.7301   |   0.0123   |   0.1024

        :param atom_length_sec: length of the atoms (in seconds) to be found.
        :param prc: integer (0 to 100) showing what proportion of the accelerometer patterns should be considered.
        :param dist_func: a distance function which receives two vectors of teh same length.
                          Use distance functions implemented in `Utils` class. (default: dtw(euclidean))
        :param samplingRate: (integer) number of samples per second.
        :return: Appends the newly found atoms to `draftDictionary_df` and returns it.
        """
        tic = time.time()
        print("Finding recurring accelerometer patterns for subject {} and activity {} started.".format(activity_df.subject.values[0], activity_df.activity.values[0]))

        atomLen = atom_length_sec * samplingRate
        index = activity_df.index.values # index values will be used to detect breaks in activity data.
        x = activity_df.X.values
        y = activity_df.Y.values
        z = activity_df.Z.values
        nrow = len(index)
        mp = [1] * nrow # Placeholder for distance values. Minimum distance of current window with its successors.
        mp_idx = [-1] * nrow # Placeholder for the index where minimum difference to the current window was observed.
        idx = [i for i in range(nrow)]

        i = 0
        pbTxtNo = 0 # For progress bar
        while i < nrow - atomLen:
            progress = i * 100 / nrow
            pbTxtNo = int(progress / 10)
            print('\r[{0}] ({1} out of {2}) {3:.2f}%'.format('#' * pbTxtNo, i, nrow, progress), end="")
            curr_s, curr_e = i, i + atomLen
            noBreak = (index[curr_e] - index[curr_s] == atomLen)
            if noBreak:
                # For the current window, we calculate its distance (average of distances for each axis) with its successors.
                currX = x[curr_s:curr_e]
                currY = y[curr_s:curr_e]
                currZ = z[curr_s:curr_e]
                next_s = curr_e
                while next_s < nrow - atomLen:
                    next_e = next_s + atomLen
                    noBreak = (index[next_e] - index[next_s] == atomLen)
                    if noBreak:
                        nextX = x[next_s:next_e]
                        distX = dist_func(currX, nextX)
                        nextY = y[next_s:next_e]
                        distY = dist_func(currY, nextY)
                        nextZ = z[next_s:next_e]
                        distZ = dist_func(currZ, nextZ)
                        curDist = np.mean((distX, distY, distZ))
                        if curDist < mp[curr_s]:
                            mp[curr_s] = curDist
                            mp_idx[curr_s] = next_s
                        next_s += 1
                    else:
                        next_s += atomLen
                i += 1
            else:
                i += atomLen
        print('\r[{0}] ({1} of {1}) 100%'.format('#' * pbTxtNo, nrow))

        # Put the difference vector (MP) and the corresponding indeces (MP_idx) into a DataFrame.
        mp_df = pd.DataFrame(data={'idx': idx, 'index': index, 'MP': mp, 'MP_idx': mp_idx})
        sorted_mp_df = mp_df.sort_values(by="MP")

        number_of_atoms = int(np.ceil((nrow / atomLen) * (prc / 100)))
        atom_start_end = []
        c = 0
        lastJ = 0
        while c < number_of_atoms and lastJ < sorted_mp_df.shape[0]:
            curr_s = sorted_mp_df.idx.values[lastJ]
            curr_e = curr_s + atomLen
            mp_s = sorted_mp_df.MP_idx.values[lastJ]
            mp_e = mp_s + atomLen
            # Selecting atoms which do not have overlapping parts with previously selected atoms.
            # Their similar windows (i.e., mp_windows) are also checked.
            # This is to make sure we are preserving non-redundant information.
            if not self.__hasOverlapWithPreviousOnes(curr_s, curr_e, mp_s, mp_e, atom_start_end):
                atom_start_end.append((curr_s, curr_e, mp_s, mp_e))
                c += 1
            lastJ += 1

        # Appending new atoms to previously learned draft dictionary for this activity
        atom_number = 0 if draftDictionary_df is None else draftDictionary_df.atom_number.max()  # read from the previous DataFrame
        for start, end, mp_s, mp_e in atom_start_end:
            atom_number += 1
            atom_df = pd.DataFrame(data={"atom_number": atom_number, "X": x[start:end], 'Y': y[start:end], 'Z': z[start:end]})
            if draftDictionary_df is None:
                draftDictionary_df = atom_df
            else:
                draftDictionary_df = draftDictionary_df.append(atom_df, ignore_index=True)

        toc = time.time()
        print("{} atoms found and appended. (elapsed time: {:.2f} seconds)".format(number_of_atoms, toc - tic))
        return draftDictionary_df

    def learnDraftDictionary_VM(self, activity_df, draftDictionary_df, atom_length_sec, prc: int, dist_func=Utils.dtw, samplingRate=10):
        """
        Finds recurring patterns by calculating "matrix profile" and selecting the top `prc`% of them. The selected
        acceleration patterns will be appended to the `draftDictionary_df` DataFrame.

        This function finds recurring patterns using Vector Magnitude values (one vector).

        :param activity_df: pandas.DataFrame containing the accelerometer data for a specific participant and an activity.
        This DataFrame is populated by reading csv files produced by `UCI_Handler.downsampleAndSeparateSubjectData()`.
                -> subject: int <-  |  -> activity: str <-  |  X: float  |  Y: float  |  Z: float  |  -> VM: float <-
                -------------------------------------------------------------------------------------------------------
                --->   1      <--   |     -> SITTING <-     |   0.5670   |   0.1234   |   0.4567   |  ->  0.7384   <-
        :param draftDictionary_df: pandas.DataFrame containing previously found atoms from previous participants.
                       atom_number: int  |  VM: float
                       -------------------------------
                                 1       |   0.7301
        :param atom_length_sec: length of the atoms (in seconds) to be found.
        :param prc: integer (0 to 100) showing what proportion of the accelerometer patterns should be considered.
        :param dist_func: a distance function which receives two vectors of the same length. Use distance functions implemented in the `Utils` class. (default: dtw(euclidean))
        :param samplingRate: (integer) number of samples per second.
        :return: Appends the newly found atoms to `draftDictionary_df` and returns the updated `draftDictionary_df`.
        """
        tic = time.time()
        atomLen = samplingRate * atom_length_sec
        subjectID = activity_df.subject.values[0]
        activity = activity_df.activity.values[0]
        print("Finding recurring accelerometer patterns for subject {} and {} started.".format(subjectID, activity))


        vm = activity_df.VM.values # Vector Magnitude values.
        vm_index = activity_df.VM.index # This is used to make sure there is no gap between neighboring data points.

        nrow = len(vm)
        # initializing matrix profile values
        mp = [1] * nrow
        # for debug purposes, we check which index (starting point) was found for the minimum matrix profile value.
        mp_idx = [-1] * nrow
        # index value for the intermediate DataFrame.
        idx = [i for i in range(nrow)]
        print("Matrix Profile calculation started.")
        i = 0
        pbTxtNo = 0
        while i < (nrow - atomLen):
            progress = i / nrow * 100
            pbTxtNo = int(progress / 10)
            print('\r[{0}] ({1} of {2}) {3:.2f}%'.format('#' * pbTxtNo, i, nrow, progress), end="")
            curr_s, curr_e = i, i + atomLen
            noBreak = (vm_index[curr_e] - vm_index[curr_s] == atomLen)
            if noBreak:
                currVM = vm[curr_s:curr_e]
                next_s = curr_e
                while next_s < (nrow - atomLen):
                    next_e = next_s + atomLen
                    noBreak = (vm_index[next_e] - vm_index[next_s] == atomLen)
                    if noBreak:
                        nextVM = vm[next_s:next_e]
                        curDist = dist_func(currVM, nextVM)
                        if curDist < mp[curr_s]:
                            mp[curr_s] = curDist
                            mp_idx[curr_s] = next_s
                        next_s += 1
                    else:
                        next_s += atomLen
                i += 1
            else:
                i += atomLen
        print("\r[{0}] ({1} of {1}) 100%".format('#' * pbTxtNo, nrow))

        # Putting matrix profile values and other information in one DataFrame
        mp_df = pd.DataFrame(data={"idx": idx, "vm_idx": vm_index, "MP": mp, "MP_idx": mp_idx})
        sorted_mp_df = mp_df.sort_values(by="MP")

        # Selecting (prc) top atoms. They should be non-overlapping to maximize the amount of information saved.
        number_of_atoms = int(np.ceil((nrow / atomLen) * (prc / 100)))
        atom_start_end = []
        c = 0
        lastJ = 0
        while c < number_of_atoms and lastJ < sorted_mp_df.shape[0]:
            curr_s = sorted_mp_df.idx.values[lastJ]
            curr_e = curr_s + atomLen
            mp_s = sorted_mp_df.MP_idx.values[lastJ]
            mp_e = mp_s + atomLen
            # Selecting atoms which do not have overlapping parts with previously selected atoms.
            # Their similar windows (i.e., mp_windows) are also checked.
            # This is to make sure we are preserving non-redundant information.
            if not self.__hasOverlapWithPreviousOnes(curr_s, curr_e, mp_s, mp_e, atom_start_end):
                atom_start_end.append((curr_s, curr_e, mp_s, mp_e))
                c += 1
            lastJ += 1

        # Appending new atoms to previously learned draft dictionary for this activity
        atom_number = 0  if draftDictionary_df is None else draftDictionary_df.atom_number.max()# read from the previous DataFrame
        for start, end, mp_s, mp_e in atom_start_end:
            atom_number += 1
            atom_df = pd.DataFrame(data={"atom_number": atom_number, "VM": vm[start:end]})
            if draftDictionary_df is None:
                draftDictionary_df = atom_df
            else:
                draftDictionary_df = draftDictionary_df.append(atom_df, ignore_index=True)

        toc = time.time()
        print("{} atoms found and appended. (elapsed time: {:.2f} seconds)".format(number_of_atoms, toc - tic))
        return draftDictionary_df

    def reshapeToTabular(self, dictionary_df, atom_len=None):
        """
        Reshapes from (atom_number*atom_len, 2) to (atom_number, atom_len) DataFrame.

        :param dictionary_df: Learned dictionary. It is a pandas.DataFrame with the following format
                                   atom_number  |    VM
                                   =======================
                                       1        |  0.0012
                                       1        |  0.0023
        :param atom_len: length of atoms. Number of rows dedicated to an atom in `dictionary_df`.
        :return: tabular dictionary:
                                      atom_number  |   V0   |   V1   |  ...  |  V{atom_len}
                                    =====================================================
                                           1       | 0.0012 | 0.0023 |       |   0.0055
                                           2       | 1.0001 | 1.0005 |       |   0.9800
        """
        atom_numbers = sorted(dictionary_df.atom_number.unique())
        if atom_len is None:
            atom_len = dictionary_df.loc[dictionary_df.atom_number == atom_numbers[0], :].shape[0]

        # --------- Reshaping dictionary into a tabular format -----------
        data_mat = np.zeros((max(atom_numbers), atom_len+1))  # Matrix, each row is an atom.
        colNames = ["atom_number"] # first column holds atom_numbers
        for atom_number in atom_numbers:
            currPart_df = dictionary_df.loc[dictionary_df.atom_number == atom_number, :]
            data_mat[atom_number - 1, 0] = int(atom_number)
            data_mat[atom_number - 1, 1:] = currPart_df.VM.values

        restOfNames = ["V{}".format(int(i)) for i in range(atom_len)]
        colNames.extend(restOfNames)
        # Putting the data (now in tabular format) back in the DataFrame format.
        tab_dictionary_df = pd.DataFrame(data=data_mat, columns=colNames)

        return tab_dictionary_df

    def atomVisualization_2DScatterPoints(self, dictionary_df, dist_func=Utils.dtw, cluster_numbers=None, just_centers=False):
        """
        Calculates dissimilarities between atoms and plots them in a 2D figure.
        Plot is an approximation.
        :param dictionary_df: DataFrame (m x n) containing atoms in tabular format.
                                      atom_number  |   V0   |   V1   |  ...  |  V{atom_len}
                                    =====================================================
                                           1       | 0.0012 | 0.0023 |       |   0.0055
                                           2       | 1.0001 | 1.0005 |       |   0.9800
        :param dist_func: dissimilarity function. (default: Utils.dtw())
        :param cluster_numbers: a vector (m x 1) containing cluster numbers for each atom.
        :param just_centers: if True, only cluster centers will be plotted. Otherwise, all the points will be plotted.
        """
        # ---------- Calculating dissimilarities ----------
        tic = time.time()
        justData = dictionary_df.iloc[:, 1:].values
        dissimilarity = np.zeros((len(justData), len(justData)))
        checked = np.zeros((len(justData), len(justData)))
        total = int(len(justData) ** 2 / 2)
        c = 0
        for i in range(len(justData)):
            for j in range(len(justData)):
                if i != j and checked[i][j] < 1:
                    c += 1
                    progress = c / total * 100
                    pbTxtNo = int(progress / 10)
                    print('\r[{0}] ({1} of {2}) {3:.2f}%'.format('#' * pbTxtNo, c, total, progress), end="")
                    diss = dist_func(justData[i], justData[j])
                    dissimilarity[i][j] = np.round(diss, decimals=4)
                    dissimilarity[j][i] = np.round(diss, decimals=4)
                    checked[i][j] = checked[j][i] = 1
        print("\r[{0}] ({1} of {1}) 100%".format('#' * 10, total))

        # ---------- 2D plot ----------
        seed = 5855
        mds = MDS(n_components=2, dissimilarity="precomputed", random_state=seed)
        pos = mds.fit(dissimilarity).embedding_
        x = [pos[i][0] for i in range(pos.shape[0])]
        y = [pos[i][1] for i in range(pos.shape[0])]

        # Preparing pallet for plotting
        plot_df = pd.DataFrame(data={'X': x, 'Y': y, 'c': cluster_numbers})
        plt.figure(figsize=(6, 6))
        markers = ['o', 'o', '*', 'v', 'D', ',', '^', '<', '>', '8', 's', 'p', 'P']
        colors = [(0, 0, 0), (0, 0, 0), (1, 0, 0), (0, 0.5, 0), (0, 0, 1), (1, 0.4, 0), (0.5, 0, 0.6), (0.1, 0.9, 0.3),
                  (0.5, 0.6, 0), (1, 0, 1), (0.8, 0.1, 0), (0, 1, 1), (1, 0.5, 1)]

        # If cluster numbers are not provided, everything is plotted as black dots.
        if cluster_numbers is None:
            plt.scatter(plot_df.X.values, plot_df.Y.values, c=colors[0], marker=markers[0])
        else:
            # Otherwise, each cluster points gets a unique color
            for c in plot_df.c.unique():
                c_df = plot_df.loc[plot_df.c == c, :]
                c_df.reset_index(inplace=True)
                if just_centers:
                    c_x = c_df.X.mean()
                    c_y = c_df.Y.mean()
                    plt.scatter(c_x, c_y, c=colors[c], marker=markers[0], s=c_df.shape[0]*200, alpha=0.4)
                else:
                    plt.scatter(c_df.X.values, c_df.Y.values, c=colors[c], marker=markers[c])
        plt.show()
        toc = time.time()
        print("Elapsed time: {:.2f} seconds.".format(toc - tic))

    def atomVisualization_dendrogram(self, dictionary_df, dist_func=Utils.dtw):
        """
        Shows dendrogram of atoms based on their dissimilarities.
        :param dictionary_df: DataFrame containing atoms in tabular format.
                                      atom_number  |   V0   |   V1   |  ...  |  V{atom_len}
                                    =====================================================
                                           1       | 0.0012 | 0.0023 |       |   0.0055
                                           2       | 1.0001 | 1.0005 |       |   0.9800
        :param dist_func: dissimilarity function. (default: Utils.dtw())
        """
        justData = dictionary_df.iloc[:, 1:].values
        lnkg = linkage(justData, method='complete', metric=dist_func)
        fig = plt.figure(figsize=(10, 10))
        dn = dendrogram(lnkg)
        plt.show()

    def applyClustering(self, draftDictionary_df, cutoff, dist_func=Utils.dtw):
        """
        Applies hierarchical clustering based on the given dissimilarity function and distance cutoff.

        :param draftDictionary_df: DataFrame containing atoms in tabular format.
                                      atom_number  |   V0   |   V1   |  ...  |  V{atom_len}
                                    =====================================================
                                           1       | 0.0012 | 0.0023 |       |   0.0055
                                           2       | 1.0001 | 1.0005 |       |   0.9800
        :param cutoff: (float) distance cutoff to define clusters.
        :param dist_func: dissimilarity function. (default: Utils.dtw())
        :return: a DataFrame like following:
                                      atom_number  |  cluster_number
                                    ==================================
                                           1       |        3
                                           2       |        1
                                           3       |        1
        """
        justData = draftDictionary_df.iloc[:, 1:].values
        lnkg = linkage(justData, method='complete', metric=dist_func)
        cluster_numbers = fcluster(lnkg, cutoff, criterion='distance')
        result_df = pd.DataFrame(data={"atom_number": draftDictionary_df.atom_number.values,
                                       "cluster_number": cluster_numbers})
        return result_df

    def learnFinalDictionary(self, draftDictionary_df, cutoff, dist_func=Utils.dtw):
        """
        Among points belonging to the same cluster, picks the one which is closest to the center.
        For each cluster, one centroid is selected.

        :param dictionary_w_cluster_df: DataFrame containing atoms in tabular format. By applying clustering, we find which atoms belong to which clusters.
        :param cutoff: (float) distance cutoff to define clusters.
        :param dist_func: dissimilarity function. (default: Utils.dtw())
        :return: a DataFrame similar to draftDictionary_df but with fewer atoms; the final dictionary where the selected atoms have the minimum information overlap.
                                      atom_number  |   V0   |   V1   |  ...  |  V{atom_len}   |  cluster_number
                                    =============================================================================
                                           1       | 0.0012 | 0.0023 |       |   0.0055       |        5
                                           2       | 1.0001 | 1.0005 |       |   0.9800       |        1
        """
        # We apply clustering to find similar atoms.
        clst_df = self.applyClustering(draftDictionary_df, cutoff, dist_func)
        # For each cluster, centroids are found.
        draftDictionary_df["cluster_number"] = clst_df.cluster_number
        # Column names
        colnames = draftDictionary_df.columns[1:-1]
        # Initializing final dictionary
        finalDictioanry_df = None
        # For each cluster, (1) we find the center and (2) select the atom which is closest to the center as the exemplar for that cluster.
        # (3) the selected atom will be added to the final dictionary.
        cluster_numbers = clst_df.cluster_number.unique()
        for an, c in enumerate(cluster_numbers):
            atoms_df = draftDictionary_df.loc[draftDictionary_df["cluster_number"] == c, :]
            atoms_df.reset_index(inplace=True)
            center = np.array([atoms_df.loc[:, colname].mean() for colname in colnames]) # Only V0 ... V{atom_len} are considered to find the centroid. (Discarding 'index', 'atom_number' & 'cluster_number')
            centroid = np.array(atoms_df.loc[0, colnames].values)
            diss = dist_func(centroid, center) # Discarding 'index', 'atom_number' and `cluster_number` for distance calculations.
            for i in range(1, atoms_df.shape[0], 1):
                atom = np.array(atoms_df.loc[i, colnames].values)
                newDiss = dist_func(atom, center) # Discarding 'index', 'atom_number' and `cluster_number` for distance calculations.
                if newDiss < diss:
                    centroid = atom
                    diss = newDiss
            vals = [an]
            vals.extend(centroid)
            cnames = ["atom_number"]
            cnames.extend(colnames)
            centroid = pd.DataFrame(data=[vals], columns=cnames)
            if finalDictioanry_df is None:
                finalDictioanry_df = centroid
            else:
                finalDictioanry_df = finalDictioanry_df.append(centroid, ignore_index=True)
        # In the end, finalDictionary_df has one atom for each cluster; the exemplar.
        return finalDictioanry_df

    def visualizeDictionaryWaveforms(self, dictionary_df, plot_save_address: str, atom_len=None):
        """
        Suppose you have merged all your dictionary atoms in a way that
            Each row represents an atom.
            "atom_number" column shows the atom number (not important).
            "V0...V{atom_len}" are columns which show data points for an atom.
            "activity" shows the atom belonged to what activity.

        Plots all atoms in a tabular form where
            Each row represents an activity type.
            Each cell represents an atom.
        :param dictionary_df: a pandas.DataFrame having the following columns (atom_number, V0, ..., V{atom_len-1}, activity)
        :param atom_len: (int) an integer indicating the number of data points in an atom. If None, it will be automatically found.
        """
        activities = dictionary_df.activity.unique()
        colnames = dictionary_df.columns
        atom_len = 0
        for c in colnames:
            if c.startswith("V"):
                atom_len += 1
        X = [i for i in range(atom_len)]
        colnames = ["V{}".format(i) for i in range(atom_len)]

        justData = dictionary_df.loc[:, colnames]
        ymin, ymax = min(justData.min()), max(justData.max())
        f, ax = plt.subplots(len(activities), 6, figsize=(10, 10))
        for i in range(ax.shape[0]):
            for j in range(ax.shape[1]):
                ax[i, j].set_xticks([])
                ax[i, j].set_ylim((ymin, ymax))
                if j == 0:
                    ax[i, j].set_ylabel(activities[i])
                    ax[i, j].set_yticks([0, 0.5, 1])
                else:
                    ax[i, j].set_yticks([])

        for i, activity in enumerate(activities):
            activity_df = dictionary_df.loc[dictionary_df["activity"] == activity, colnames]
            activity_df.reset_index(inplace=True)
            for j in range(activity_df.shape[0]):
                ax[i, j].plot(X, activity_df.loc[j, colnames].values, 'b-')

        f.savefig(plot_save_address)


class UF_SupervisedLearner(SupervisedLearner):
    def __convertTimeToInt(self, timeStr: str):
        """

        :param timeStr: "HH:MM:SS.msc". E.g., 10:34:30.100
        :return: the milliseconds passed from 00:00:00.000.
        """
        tokens = timeStr.split(":")
        hour, minute = tokens[0], tokens[1]
        second, ms = tokens[2].split(".")
        return int(ms) + (int(second) * 1000) + (int(minute) * 60 * 1000) + (int(hour) * 60 * 60 * 1000)

    def __convertAllTimestampsToInt(self, timestamps:list):
        """
        Dataset from UF ChoresXL dataset has a column Timestamp with the format
        mm/dd/yyyy hh:mm:ss.msc. For each timestamp an integer is generated and a vector of incrementing integers
        is returned.

        :param timestamps: e.g., [3/22/2016 10:48:56.000, 3/22/2016 10:48:56.100, ..., ]
        :return: an integer vector. e.g., [0, 1, ...]
        """
        dates = [t.split(" ")[0] for t in timestamps]
        times = [t.split(" ")[1] for t in timestamps]
        days = np.unique(dates).tolist()
        dayVals = {}
        dayVal = 0
        for day in days:
            dayVals[day] = dayVal
            dayVal += 1
        # Dates are not considered fully for the conversion to avoid creating super large integers.
        # Different dates are separated by a 24-hour difference.
        initialResult = [self.__convertTimeToInt(times[i]) + dayVals[dates[i]] * 24 * 3600 * 1000 for i in range(len(times))]

        # To obtain values like [0, 1, ...] we should subtract the minimum value of the initialResult vector from the rest.
        # Also, since the sampling rate might be < 1000 Hz, the difference between adjacent points might be > 1.
        # Therefore, an adjustment is needed.
        minVal = min(initialResult)
        step = initialResult[1] - initialResult[0]
        adjResult = [int((initialResult[i] - minVal) / step) for i in range(len(initialResult))]
        return adjResult

    def __similarities(self, vector, ids, length, step, dist_func):
        """
        Calculates similarities between each selected part (by a sliding window) and parts coming after it.
        Once the minimum distance between the current part and another part is found, the window slides to
        the next part (by `step` size) and etc.

        :param vector: list(float). A time-series to find the recurring patterns from.
        :param ids: indexes showing the order of each data point. Also used to show time gap between two data points.
        :param length: length of the sliding window.
        :param step: sliding window stride length.
        :param dist_func: function to calculate similarities between the current window and windows after that.
        :return: a vector of float values showing the minimum similarity between one window and another window after that.
        """
        tic = time.time()
        nrow = len(vector)
        mp = [1] * nrow
        mp_idx = [-1] * nrow
        i = 0
        pbTxtNo = 0
        while i < nrow - length:
            progress = i * 100 / nrow
            pbTxtNo = int(progress / 10)
            print("\r[{0}] ({1} out of {2}) {3:.2f}% - {4:.2f} seconds".format('#' * pbTxtNo, i, nrow, progress,
                                                                               time.time() - tic), end="")
            curr_s, curr_e = i, i + length
            noBreak = (ids[curr_e] - ids[curr_s] == length)
            if noBreak:
                next_s = curr_e
                while next_s < nrow - length:
                    next_e = next_s + length
                    noBreak = (ids[next_e] - ids[next_s] == length)
                    if noBreak:
                        dist = dist_func(vector[curr_s:curr_e], vector[next_s:next_e])
                        if dist < mp[curr_s]:
                            mp[curr_s] = dist
                            mp_idx[curr_s] = next_s
                        next_s += step
                    else:
                        next_s += length
                i += step
            else:
                i += length
        print("\r[{0}] ({1} out of {1}) 100% - {2:.2f} seconds".format('#' * pbTxtNo, nrow, time.time() - tic))
        result = pd.DataFrame(data={'idx': [i for i in range(len(mp))], 'mp': mp, 'mp_idx': mp_idx})
        return result

    def learnDraftDictionary_VM(self, activity_df, atom_length_sec, prc=10, sampling_rate=10, step_size_sec=None, dist_func=Utils.dtw):
        """
        A version of generic `learnDraftDictionary_VM` that Works for UF dataset.

        :param activity_df: A DataFrame with columns: <Participant: str, Activity: str, VM: float, timestamp: str>
        :param atom_length_sec: length of atoms in second.
        :param prc: top `prc` of recurring patterns to pick. (default 10%)
        :param dist_func: a distance function which takes two vectors and returns a float as their distances. (default: Utils.dtw)
        :param sampling_rate: how many data points exist within a second.
        :param step_size_sec: sliding window step size in seconds. (default 1/2 of atom_length_sec)
        :return: a DataFrame with columns: <participant, activity, atom_number, VM>
        """
        tic = time.time()

        atom_length = int(atom_length_sec * sampling_rate)
        if step_size_sec is None:
            step_size = int(atom_length / 2)
        else:
            step_size = int(step_size_sec * sampling_rate)

        # Converting timestamps to proper indeces to
        #     1. identify orders
        #     2. detect gaps between data points.
        ids = self.__convertAllTimestampsToInt(activity_df.Timestamp.values)

        # Calculating min similarities between each window and another window that comes after it.
        mp_df = self.__similarities(activity_df.VM.values, ids, atom_length, step_size, Utils.dtw)
        sorted_mp_df = mp_df.sort_values(by="mp")


        # Select a part to plot
        # TODO: add a module to plot a potential atom

        # Identifying non-overlapping atoms
        number_of_atoms = int(np.ceil((activity_df.shape[0] / atom_length) * (prc / 100)))
        atom_start_end = []
        c = 0 # number of selected atoms
        lastJ = 0 # point to consider in `sorted_mp_df`

        while c < number_of_atoms and lastJ < sorted_mp_df.shape[0]:
            curr_s = sorted_mp_df.idx.values[lastJ]
            curr_e = curr_s + atom_length
            mp_s = sorted_mp_df.mp_idx.values[lastJ]
            mp_e = mp_s + atom_length
            if not SupervisedLearner.hasOverlapWithPreviousOnes(curr_s, curr_e, mp_s, mp_e, atom_start_end):
                atom_start_end.append((curr_s, curr_e, mp_s, mp_e))
                c += 1
            lastJ += 1

        # Preparing the DataFrame for output
        atom_number = 0
        ppt_activity_df = None
        for start, end, mp_start, mp_end in atom_start_end:
            atom_number += 1
            atom_df = pd.DataFrame(data={"participant": activity_df.Participant.values[0],
                                         "activity": activity_df.Activity.values[0],
                                         "atom_number": atom_number,
                                         "VM": activity_df.VM.values[start:end]})
            if ppt_activity_df is None:
                ppt_activity_df = atom_df
            else:
                ppt_activity_df = ppt_activity_df.append(atom_df, ignore_index=True)

        toc = time.time()
        print("{} atoms found. (elapsed time: {:.2f} seconds)".format(number_of_atoms, toc - tic))
        return ppt_activity_df

    def __makeTabularDraftDictionary(self, rawDF, activity, last_atom_number=0, atom_length=None, verbose=True):
        """
        Converts a draft dictionary (vertical) into a feature vector matrix.
        Every atom will be placed on one row.

        :param rawDF: direct output of pd.read_csv(ppt_activity_atomLength.csv)
        :param activity: str
        :param last_atom_number: to merge with previously read and processed draft dictionaries, this number is used
                                 as offset to correct atom_numbers.
        :param atom_length: number of samples in an atom. For example, if sampling_rate=10 and atomLength=1sec, then
                            `atom_length` is 10.
        :param verbose: If true, logs are printed.
        :return: A pandas.DataFrame where each row is an atom.
        """
        nrow = rawDF.atom_number.max()
        if atom_length is None:
            atom_length = rawDF.loc[rawDF.atom_number == 1, :].shape[0]

        if verbose:
            print("\"{}\" has {} atoms for \"{}\" activity.".format(rawDF.participant.values[0], nrow, activity))
        dataMatrix = np.zeros(shape=(nrow, atom_length + 1))
        for an in rawDF.atom_number.unique():
            VM = [an + last_atom_number]
            VM.extend(rawDF.loc[rawDF.atom_number == an, "VM"].values)
            dataMatrix[an - 1][:] = VM
        result = pd.DataFrame(dataMatrix)
        result.loc[:, "participant"] = [rawDF.participant.values[0]]
        result.loc[:, "activity"] = activity
        colnames = ["atom_number"]
        colnamesVM = ["VM{}".format(i + 1) for i in range(atom_length)]
        colnames.extend(colnamesVM)
        colnames.extend(["participant", "activity"])
        result.columns = colnames
        resultColnames = ["participant", "activity", "atom_number"]
        resultColnames.extend(colnamesVM)
        result = result.loc[:, resultColnames]
        if verbose:
            print("Converted to tabular. (rows: {} to {})\n".format(result.atom_number.min(), result.atom_number.max()))
        return result

    def reshapeToTabular(self, input_folder, atom_length_sec, activity):
        """
        Reads draft dictionary files (.csv), converts them to tabular format (instead of the vector form) and merge
        them into one single file.

        :param input_folder: address of the folder where `atom_length` folders are kept.
        :param atom_length_sec: inside `input_folder`, there are folders for each atom_length_sec.
        :param activity: each draft dictionary is constructed for one activity. E.g., "locomotion", "sedentary", etc.
        :return: pandas.DataFrame which has the following columns:
                 <ppt, activity, atom_nuber, VM1, VM2, ..., VM-{atom_length}>
        """
        import os
        tic = time.time()
        filenames = os.listdir("{}{}/".format(input_folder, atom_length_sec))
        last_atom_number = 0
        allTabularDF = None
        for f in range(len(filenames)):
            filename = filenames[f]
            print("\r{} of {} ({:.2f} seconds)".format(f, len(filenames), time.time() - tic), end="")
            if filename.endswith(".csv") and activity in filename:
                ppt, activity, _ = filename.split("_")
                rawDF = pd.read_csv("{}{}/{}".format(input_folder, atom_length_sec, filename))
                atomLength = rawDF.loc[rawDF.atom_number == 1, :].shape[0]
                tabularDF = self.__makeTabularDraftDictionary(rawDF, activity, last_atom_number, atomLength, False)
                last_atom_number = tabularDF.atom_number.max()
                if allTabularDF is None:
                    allTabularDF = tabularDF
                else:
                    allTabularDF = allTabularDF.append(tabularDF, ignore_index=True)
        print("\"{}\" --- Done! (elapsed time: {:.2f} seconds)".format(activity, time.time() - tic))
        return allTabularDF

    def __calculateDistanceMatrix_while(self, X, dist_func=Utils.dtw):
        """
        **Deprecated: super slow for large datasets.**

        :param X: (n*m) matrix where n is nrow and m is ncol. Every row is an atom and ncol (m) is the number of samples within each atom.
        :param dist_func: a distance function which takes two vectors of the same size and returns a number.
        :return: A symmetric (n*n) matrix D where D[i][j] (=D[j][i]) represents the distance between rows i and j.
        """
        tic = time.time()
        D = np.zeros((len(X), len(X)))
        total = int((len(X)**2) / 2)
        c = 0
        for i in range(len(X)):
            for j in range(len(X)):
                if i != j and np.round(D[i][j], decimals=4) == 0:
                    c += 1
                    progress = c / total * 100
                    pbTxtNo = int(progress / 10)
                    toc = time.time()
                    print("\r[{0}] ({1} of {2}) {3:.2f}% - {4:.2f} seconds".format('#'*pbTxtNo, c, total, progress, toc - tic))
                    diss = dist_func(X[i], X[j])
                    D[i][j] = np.round(diss, decimals=4)
                    D[j][i] = D[i][j]
        toc = time.time()
        print("\r[{0}] ({1} of {1}) 100% - {2:.2f} seconds".format('#' * pbTxtNo, total, toc - tic))
        return D

    def calculateDistanceMatrix_comprehensive(self, X, dist_func=Utils.timedDTW):
        """

        :param X: (n*m) matrix where n is nrow and m is ncol. Every row is an atom and ncol (m) is the number of samples within each atom.
        :param dist_func: a distance function which takes two vectors of the same size and returns a number.
        :return: A symmetric (n*n) matrix D where D[i][j] (=D[j][i]) represents the distance between rows i and j.
        """
        tic = time.time()
        D = [[dist_func(X, i, j, tic) if i <= j else 0 for j in range(len(X))] for i in range(len(X))]
        for i in range(len(D)):
            for j in range(len(D[0])):
                D[j][i] = D[i][j]
        toc = time.time()
        print("\nDistance matrix is calculated for {} rows. (elapsed time: {:.2f} seconds)".format(len(X), toc - tic))
        return D

    def __hierarchicalClustering(self, X, cutoff=0.06, method='complete', dist_func=Utils.dtw):
        """
        Applies hierarchical clustering on rows of X (atoms) and return cluster_numbers.
        :param X: An m*n matrix where each row is a sample (atom).
        :param cutoff: cutoff applied to the dendrogram to find the right number of clusters.
                        This is the minimum distance between atoms to be considered as 'different'.
                        The default value is chosen based on experiments on UF ChoresXL dataset.
                        This value should be adjusted according to the dataset. See `plot_dendrogram`.
        :param method: `method` parameter passed to hierarchical clustering.
                        Default is 'complete' to find spherical clusters.
        :param dist_func: A function that receives two vectors of the same size (X rows) and returns the distance between them.
        :return: A vector of size m, where i^{th} value shows the cluster_number assigned to the i^{th} row of X.
        """
        lnkg = linkage(X, method=method, metric=dist_func)
        cluster_numbers = fcluster(lnkg, cutoff, criterion='distance')
        return cluster_numbers

    def __appendClusterNumberToX(self, X, cluster_numbers):
        """

        :param X: An m*n matrix of numeric values.
        :param cluster_number:  A numeric vector of size m.
        :return: Pandas.DataFrame(VM1, VM2, ..., VM_m_, cn)
        """
        VM_colnames = ["VM{}".format(i + 1) for i in range(len(X[0]))]
        df = pd.DataFrame(X, columns=VM_colnames)
        df.loc[:, "cn"] = cluster_numbers
        return df

    def __clusterCenters(self, X_df):
        """
        Calculates the center of each cluster by averaging column values.

        :param X_df: Pandas.DataFrame(VM1, ..., VM_n, cn) where VM_i_ are vector magnitude values and `cn` is the cluster number.
        :return: Pandas.DataFrame(VM1, ..., VM_n, cn), where there is one row for each `cn`.
        """
        cluster_numbers = X_df.cn.values
        result = None
        for cn in range(min(cluster_numbers), max(cluster_numbers) + 1, 1):
            cnDF = X_df.loc[X_df.cn == cn, :]
            newDF = pd.DataFrame(np.zeros(shape=(1, X_df.shape[1])))
            for j in range(X_df.shape[1]):
                newDF.iloc[0, j] = cnDF.iloc[:, j].mean() if X_df.columns.values[j] != 'cn' else int(cn)
            newDF.columns = X_df.columns.values
            if result is None:
                result = newDF
            else:
                result = result.append(newDF, ignore_index=True)
        return result

    def __findClosestToCenters(self, X_df, clusterCenter_df):
        """
        For each cluster, picks the sample in X_df (an atom) that is closest (euclidean distance) to the cluster center.

        :param X_df: Pandas.DataFrame(VM1, ..., VM_n, cn) where VM_i_ are vector magnitude values and `cn` is the cluster number.
        :param clusterCenter_df: Pandas.DataFrame(VM1, ..., VM_n, cn) where VM_i_ are vector magnitude values and `cn` is the cluster number. One row for each `cn`.
        :return: Pandas.DataFrame(VM1, ..., VM_n, cn) where VM_i_ are vector magnitude values and `cn` is the cluster number. One actual atom for each `cn`.
        """
        result = None
        for cn in clusterCenter_df.cn.values:
            c = clusterCenter_df.loc[clusterCenter_df.cn == cn, :].values
            X_c = X_df.loc[X_df.cn == cn, :].values

            dist = [np.linalg.norm(X - c) for X in X_c]
            minIdx = list(np.where(dist == min(dist))[0])[0]

            newData = np.reshape(X_c[minIdx], newshape=(1, len(X_c[0])))
            newDF = pd.DataFrame(newData, columns=X_df.columns)
            if result is None:
                result = newDF
            else:
                result = result.append(newDF, ignore_index=True)
        return result

    def shrinkDraftDictionaryPool(self, draftDF):
        """
        This is an additional step to remove similar atoms from draft dictionary to avoid time-consuming calculations for learning the final dictionary.
        The procedure is very similar to learning the final dictionary, but performed participant-wise.

        :param draftDF: Pandas.DataFrame(participant, activity, atom_number, VM1, ..., VM_n)
        :return: draftDF2: Pandas.DataFrame(participant, activity, atom_number, VM1, ..., VM_n)
        """
        tic = time.time()
        participants = draftDF.participant.unique()
        newDraftDF = None
        lastAtomNumber = 0
        for p in range(len(participants)):
            participant = participants[p]
            print("({} out of {}) {} ... ".format(p, len(participants), participant), end="")

            pptDF = draftDF.loc[draftDF.participant == participant, :].copy()
            X = pptDF.iloc[:, 3:].values # just vector magnitude values

            cluster_numbers = self.__hierarchicalClustering(X, cutoff=0.06)
            X_df = self.__appendClusterNumberToX(X, cluster_numbers)

            # Average (center) for each cluster
            clusterCentersDF = self.__clusterCenters(X_df)

            # Atom closest to each center
            atomCentroids = self.__findClosestToCenters(X_df, clusterCentersDF)

            newDF = atomCentroids.copy()
            newDF = newDF.drop(['cn'], axis=1) # Keeping only VM values

            newDF.loc[:, "atom_number"] = np.array(atomCentroids.cn.values) + lastAtomNumber
            newDF.loc[:, "activity"] = pptDF.activity.values[0]
            newDF.loc[:, "participant"] = pptDF.participant.values[0]
            VM_colnames = atomCentroids.columns[:-1]
            colnames = ["participant", "activity", "atom_number"]
            colnames.extend(VM_colnames)
            newDF = newDF.loc[:, colnames]
            if newDraftDF is None:
                newDraftDF = newDF
            else:
                newDraftDF = newDraftDF.append(newDF, ignore_index=True)
            lastAtomNumber = newDF.atom_number.max()
            toc = time.time()
            print("Done (elapsed time: {:.2f} seconds)".format(toc - tic))
        print("\nNew draft dictionary is prepared. ")
        return newDraftDF

    def tSNE_dimReduction(self, X, metric='precomputed'):
        """
        Performs t-Distributed Stochastic Neighbor Embedding (t-SNE) to obtain a 2D representation of the data.

        :param X: an m*n matrix, where m is the number of samples and n is original dimensions.
        :param metric: by default, we pass a distance matrix instead of the original data.
        :return: Pandas.DataFrame(x, y)
        """
        seed = 5855
        tsne = TSNE(n_components=2, metric=metric, random_state=seed)
        pos = tsne.fit(pd.DataFrame(X)).embedding_
        x = [pos[i][0] for i in range(pos.shape[0])]
        y = [pos[i][1] for i in range(pos.shape[0])]
        df = pd.DataFrame(data={'x':x, 'y':y})
        return df

    def scatter2D(self, plot_df, annotate=False):
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(plot_df.x.values, plot_df.y.values)  # original points
        if annotate:
            for i in range(plot_df.shape[0]):
                ax.annotate(str(i), (plot_df.x.values[i], plot_df.y.values[i]))
        plt.show()

    def scatter2DColored(self, plot_df):
        """

        :param plot_df: <x, y, cn>
        """
        plt.figure(figsize=(6, 6))
        markers = ['o', 'o', '*', 'v', 'D', ',', '^', '<', '>', '8', 's', 'p', 'P']
        colors = [(0, 0, 0), (0, 0, 0), (1, 0, 0), (0, 0.5, 0), (0, 0, 1), (1, 0.4, 0), (0.5, 0, 0.6), (0.1, 0.9, 0.3),
                  (0.5, 0.6, 0), (1, 0, 1), (0.8, 0.1, 0), (0, 1, 1), (1, 0.5, 1)]
        for cn in plot_df.cn.unique():
            cn_df = plot_df.loc[plot_df.cn == cn, :]
            cn_df.reset_index(inplace=True)
            plt.scatter(cn_df.x.values, cn_df.y.values, c=colors[cn], marker=markers[cn])
        plt.show()

    def dendrogram(self, lnkg, truncate_level=10):
        """
        Plots dendrogram of the given linkage.

        :param lnkg: Object returned by scipy.cluster.hierarchy.linkage
        :param truncate_level: plot dendrogram up to this level.
        """
        fig = plt.figure(figsize=(10, 10))
        dn = dendrogram(lnkg, truncate_mode='level', p=truncate_level)
        plt.show()

    def learnFinalDictionary(self, clustered_draft_df):
        """

        :param clustered_draft_df: <participant, activity, atom_number, VM1....VM_n, cn>
        :return: <atom_number (former cn), frequency, VM1...VM_n> -- where frequency is the number of atoms that belonged to the cn cluster.
        """
        tic = time.time()
        activity = clustered_draft_df.activity.values[0]
        colnames = clustered_draft_df.columns

        X_df = clustered_draft_df.loc[:, colnames[3:]]

        # finding the cluster centers (average of values)
        cluster_centers_df = self.__clusterCenters(X_df)

        # finding atoms that are closest to the centers
        atom_centroids_df = self.__findClosestToCenters(X_df, cluster_centers_df)

        # Preparing output: <atom_number, frequency, VM_1...VM_n>
        colnames = X_df.columns.values[:-1]
        new_df = atom_centroids_df.copy()
        new_df = new_df.drop(['cn'], axis=1)
        new_df.loc[:, 'atom_number'] = atom_centroids_df.cn.values
        for an in new_df.atom_number.unique():
            new_df.loc[new_df.atom_number == an, 'frequency'] = X_df.loc[X_df.cn == an, :].shape[0]

        newColumns = ['atom_number', 'frequency']
        newColumns.extend(colnames)
        new_df = new_df.loc[:, newColumns].copy()

        toc = time.time()
        print("Final dictionary learned for {}. (elapsed time: {:.2f} seconds)".format(activity, toc-tic))

        return new_df