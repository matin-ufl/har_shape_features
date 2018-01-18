"""
Created on Sun Jan  7 16:28:11 2018

@author: Matin Kheirkhahan (matinkheirkhahan@ufl.edu)
"""

import pandas as pd
import numpy as np
import time
from dictionary_learning.utils import Utils
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

class SupervisedLearner(object):
    """
    This class provides functions to learn dictionary of recurring patterns by examining acceleration patterns in the
    activity of interest.
    """

    def __hasOverlapWithPreviousOnes(self, atom_s, atom_e, mp_s, mp_e, previousAtoms: list):
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

    def learnAndAppendDraftDictionary(self, activity_df, draftDictionary_df, atom_length, prc: int, dist_func=Utils.dtw, samplingRate=10):
        """
        Finds recurring patterns by calculating "matrix profile" and selecting the top `prc`% of them. The selected
        acceleration patterns will be appended to the `draftDictionary_df` DataFrame.

        :param activity_df: pandas.DataFrame containing the accelerometer data for a specific participant and an activity.
        This DataFrame is populated by reading csv files produced by `UCI_Handler.downsampleAndSeparateSubjectData()`.
                       subject: int  |  activity: str  |  X: float  |  Y: float  |  Z: float  |  VM: float
                       ------------------------------------------------------------------------------------
                             1       |     SITTING     |   0.5670   |   0.1234   |   0.4567   |   0.7384
        :param draftDictionary_df: pandas.DataFrame containing previously found atoms from previous participants.
                       atom_number: int  |  VM: float
                       -------------------------------
                                 1       |   0.7301
        :param atom_length: length of the atoms (in seconds) to be found.
        :param prc: integer (0 to 100) showing what proportion of the accelerometer patterns should be considered.
        :param dist_func: a distance function which receives two vectors of the same length. Use distance functions implemented in the `Utils` class. (default: dtw(euclidean))
        :return: Appends the newly found atoms to `draftDictionary_df` and returns the updated `draftDictionary_df`.
        """
        tic = time.time()
        atomLen = samplingRate * atom_length
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