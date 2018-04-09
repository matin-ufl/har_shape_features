from dictionary_learning.supervised_learner import UF_SupervisedLearner
from dictionary_learning.utils import Utils
from scipy.cluster.hierarchy import linkage, fcluster
import os
import pandas as pd
import time


def learnDraftDictionaries_loc_sed():
    """
    Learning dictionary drafts from all participants in the ChoresXL training set and a variety of `atom_lengths`.
    :return:
    """
    learner = UF_SupervisedLearner()
    atom_lengths = [1]

    input_folder = r"/Users/matin/Dropbox/Work-Research/Current Directory/Shape Features/Data/UF dataset/Training_Set/"
    output_folder = r"/Users/matin/Dropbox/Work-Research/Current Directory/Shape Features/Data/UF dataset/draft_dictionaries/"
    activities = [('locomotion', ['LEISURE WALK', 'RAPID WALK', 'WALKING AT RPE 1', 'WALKING AT RPE 5', 'STAIR DESCENT', 'STAIR ASCENT']),
                  ('sedentary', ['STANDING STILL', 'TV WATCHING', 'COMPUTER WORK'])]


    #os.listdir(input_folder)
    filenames = ['FLTO098_downsampled_Data.csv', 'GRMC032_downsampled_Data.csv', 'CACO104_downsampled_Data.csv',
                 'VIDO036_downsampled_Data.csv', 'CAHU150_downsampled_Data.csv', 'MALI043_downsampled_Data.csv',
                 'SHWI059_downsampled_Data.csv', 'MASC007_downsampled_Data.csv', 'TAPA044_downsampled_Data.csv',
                 'JAJE025_downsampled_Data.csv', 'PAZE141_downsampled_Data.csv', 'PABE081_downsampled_Data.csv',
                 'KAWI016_downsampled_Data.csv', 'DOKI133_downsampled_Data.csv', 'DEFE072_downsampled_Data.csv',
                 'VIRE124_downsampled_Data.csv', 'OSWA105_downsampled_Data.csv', 'VIBU147_downsampled_Data.csv',
                 'DOCA084_downsampled_Data.csv', 'CAFI066_downsampled_Data.csv', 'WEHA020_downsampled_Data.csv',
                 'VIPU052_downsampled_Data.csv', 'KRRO062_downsampled_Data.csv', 'ALWE037_downsampled_Data.csv',
                 'ANAD152_downsampled_Data.csv', 'DIWI034_downsampled_Data.csv', 'ALSH094_downsampled_Data.csv',
                 'ERPU089_downsampled_Data.csv', 'ANHU024_downsampled_Data.csv', 'SUFA135_downsampled_Data.csv',
                 'FRTA067_downsampled_Data.csv', 'CLFE128_downsampled_Data.csv', 'JOBR004_downsampled_Data.csv',
                 'KALA048_downsampled_Data.csv', 'DOGI144_downsampled_Data.csv', 'WIPE026_downsampled_Data.csv',
                 'MAPE123_downsampled_Data.csv', 'MESA088_downsampled_Data.csv', 'JOGA122_downsampled_Data.csv',
                 'NABR076_downsampled_Data.csv', 'CYSA100_downsampled_Data.csv', 'FRHU012_downsampled_Data.csv',
                 'VIMI071_downsampled_Data.csv', 'JERO138_downsampled_Data.csv', 'JAJO014_downsampled_Data.csv',
                 'RILY031_downsampled_Data.csv', 'CAHA118_downsampled_Data.csv', 'BHTR097_downsampled_Data.csv',
                 'KECH079_downsampled_Data.csv', 'STCO087_downsampled_Data.csv', 'BEMC113_downsampled_Data.csv',
                 'ROOW064_downsampled_Data.csv', 'PAMA035_downsampled_Data.csv', 'KIWI041_downsampled_Data.csv',
                 'JOLO130_downsampled_Data.csv', 'MENA107_downsampled_Data.csv', 'KEMC006_downsampled_Data.csv',
                 'ROCA057_downsampled_Data.csv', 'DAVA131_downsampled_Data.csv', 'LURH050_downsampled_Data.csv',
                 'BELA125_downsampled_Data.csv', 'KENO149_downsampled_Data.csv', 'ADAD151_downsampled_Data.csv',
                 'MAPL063_downsampled_Data.csv', 'MALE001_downsampled_Data.csv', 'PEED040_downsampled_Data.csv',
                 'JATA096_downsampled_Data.csv', 'RIMI146_downsampled_Data.csv', 'KAPO074_downsampled_Data.csv',
                 'ANCR140_downsampled_Data.csv', 'DAGO106_downsampled_Data.csv', 'LYFA132_downsampled_Data.csv',
                 'ELSU103_downsampled_Data.csv', 'MAME139_downsampled_Data.csv', 'JOCH075_downsampled_Data.csv',
                 'KAFO065_downsampled_Data.csv', 'DIDI009_downsampled_Data.csv', 'PRTU030_downsampled_Data.csv',
                 'MIAT080_downsampled_Data.csv', 'GADE017_downsampled_Data.csv', 'JECA117_downsampled_Data.csv',
                 'MAGA003_downsampled_Data.csv', 'JASA047_downsampled_Data.csv', 'JIGA102_downsampled_Data.csv',
                 'JOHE069_downsampled_Data.csv', 'DAGO046_downsampled_Data.csv', 'TOBR033_downsampled_Data.csv',
                 'JICH056_downsampled_Data.csv', 'CEKE028_downsampled_Data.csv', 'BESM109_downsampled_Data.csv',
                 'CACL010_downsampled_Data.csv', 'STBA108_downsampled_Data.csv', 'FABU077_downsampled_Data.csv',
                 'ELNA005_downsampled_Data.csv', 'HUMC112_downsampled_Data.csv', 'JOLE086_downsampled_Data.csv',
                 'ROWE023_downsampled_Data.csv', 'LIRE090_downsampled_Data.csv', 'BRHO018_downsampled_Data.csv',
                 'HERH045_downsampled_Data.csv', 'AUBR116_downsampled_Data.csv', 'HEAH055_downsampled_Data.csv',
                 'NOWI051_downsampled_Data.csv', 'SAPH073_downsampled_Data.csv', 'MACA142_downsampled_Data.csv',
                 'JABU078_downsampled_Data.csv', 'DARA053_downsampled_Data.csv', 'WOBL134_downsampled_Data.csv',
                 'TALI002_downsampled_Data.csv', 'PECA093_downsampled_Data.csv', 'JOLO120_downsampled_Data.csv',
                 'DEMO129_downsampled_Data.csv', 'BAEL121_downsampled_Data.csv', 'VIJE127_downsampled_Data.csv',
                 'HEPA039_downsampled_Data.csv', 'KEMI042_downsampled_Data.csv', 'JUSA095_downsampled_Data.csv',
                 'DOMI119_downsampled_Data.csv', 'LIWI008_downsampled_Data.csv', 'CABO111_downsampled_Data.csv',
                 'CHDO148_downsampled_Data.csv', 'CHTH110_downsampled_Data.csv', 'SUPE091_downsampled_Data.csv',
                 'CAHO060_downsampled_Data.csv', 'PRTR049_downsampled_Data.csv', 'ELRE145_downsampled_Data.csv']
    for f in range(len(filenames)):
        filename = filenames[f]
        print("\n\n[{}]---> {}".format(f, filename))
        df = pd.read_csv(input_folder + filename)
        for i in range(len(activities)):

            # vvvvv   REMOVE THIS PART vvvvv
            if f < 5:
                continue
            # ^^^^^   REMOVE THIS PART ^^^^^

            # Stiching all parts of one activity category together and
            # preparing `activity_df`.
            activity = activities[i][0]
            activityLs = activities[i][1]
            activity_df = None
            for a in activityLs:
                curDF = df.loc[df.Activity == a, :].copy()
                if activity_df is None:
                    activity_df = curDF
                else:
                    activity_df = activity_df.append(curDF, ignore_index=True)
            for atom_length in atom_lengths:
                if activity_df is None or activity_df.shape[0] <= atom_length:
                    continue
                tic = time.time()
                print("Started mining {} for {} activities. Atom length ({} seconds)".format(filename, activity, atom_length))
                draft_dictionary_df = learner.learnDraftDictionary_VM(activity_df, atom_length)
                draft_dictionary_df.to_csv("{}{}/{}_{}_{}sec.csv".format(output_folder, atom_length, filename[:7], activity, atom_length), index=False)
                toc = time.time()
                print("Found draft dictionary for PPT:({})-Activity:({})-Atom:({}) (Elapsed time: {:.2f} seconds)\n".format(filename, activity, atom_length, toc - tic))

def convertDraftsToTabular(activity, atom_length_sec):
    sl = UF_SupervisedLearner()
    input_folder = r"/Users/matin/Dropbox/Work-Research/Current Directory/Shape Features/Data/UF dataset/draft_dictionaries/"
    allDrafts = sl.reshapeToTabular(input_folder, atom_length_sec, activity)
    allDrafts.to_csv("{}draft_{}sec_{}.csv".format(input_folder, atom_length_sec, activity), index=False)

def shrinkDraftToDraft2():
    atom_length_sec = 6
    activity = "locomotion" # locomotion
    input_filename = r"/Users/matin/Dropbox/Work-Research/Current Directory/Shape Features/Data/UF dataset/draft_dictionaries/draft_{}sec_{}.csv".format(atom_length_sec, activity)
    sl = UF_SupervisedLearner()
    newDraft = sl.shrinkDraftDictionaryPool(pd.read_csv(input_filename))
    output_filename = r"/Users/matin/Dropbox/Work-Research/Current Directory/Shape Features/Data/UF dataset/draft_dictionaries/draft02_{}sec_{}.csv".format(atom_length_sec, activity)
    newDraft.to_csv(output_filename, index=False)

def labelDraftDictionary():
    activity = 'locomotion'
    atom_length_second = 3
    draft_filename = r"/Users/matin/Dropbox/Work-Research/Current Directory/Shape Features/Data/UF dataset/draft_dictionaries/draft02//draft02_{}sec_{}.csv".format(atom_length_second, activity)
    draft_df = pd.read_csv(draft_filename)

    sl = UF_SupervisedLearner()

    X = draft_df.iloc[:, 3:].values # Discarding <participant, activity, atom_number> columns
    D = sl.calculateDistanceMatrix_comprehensive(X)

    # plot 2D scatter plot of points
    plot_df = sl.tSNE_dimReduction(D)
    sl.scatter2D(plot_df)

    # Deciding on the number of clusters
    lnkg = linkage(X, method='complete', metric=Utils.dtw)
    sl.dendrogram(lnkg) # Decide what should the cutoff be

    # Applying clustering and obtaining cluster numbers
    cutoff = 0.2
    cluster_numbers = fcluster(lnkg, cutoff, criterion='distance')
    plot_df.loc[:, 'cn'] = cluster_numbers
    sl.scatter2DColored(plot_df)

def learnFinalDictionary():
    atom_length_sec = 6
    activity = "sedentary"  # locomotion
    draft_folder_address = r"/Users/matin/Dropbox/Work-Research/Current Directory/Shape Features/Data/UF dataset/draft_dictionaries/draft02/"
    final_folder_address = r"/Users/matin/Dropbox/Work-Research/Current Directory/Shape Features/Data/UF dataset/final_dictionaries/"
    input_filename = r"{}clustered_draft02_{}sec_{}.csv".format(draft_folder_address, atom_length_sec, activity)
    sl = UF_SupervisedLearner()
    final_dictionary_df = sl.learnFinalDictionary(pd.read_csv(input_filename))
    output_filename = r"{}draft02_{}sec_{}.csv".format(final_folder_address, atom_length_sec, activity)
    final_dictionary_df.to_csv(output_filename, index=False)

def mergeLearnedDictionary():
    folder = r"/Users/matin/Dropbox/Work-Research/Current Directory/Shape Features/Data/UF dataset/final_dictionaries/"
    for atom_length in (3, 6):
        final_df = None
        input_folder = folder + str(atom_length) + "/"
        filenames = os.listdir(input_folder)
        for filename in filenames:
            if filename.endswith(".csv"):
                df = pd.read_csv(input_folder + filename)
                if "locomotion" in filename:
                    df.loc[:, "activity"] = "locomotion"
                else:
                    df.loc[:, "activity"] = "sedentary"
                if final_df is None:
                    final_df = df
                else:
                    final_df = final_df.append(df, ignore_index=True)
        final_df.to_csv(input_folder + "final_dictionary_{}sec.csv".format(atom_length), index=False)

if __name__ == "__main__":
    # Learning dictionary drafts from ChoresXL dataset
    # Activities: Locomotion & Sedentary
    #learnDraftDictionaries_loc_sed()

    # Reshape and Merge draft dictionaries
    #for sec in range(1, 7):
    #    for activity in ["locomotion", "sedentary"]:
    #        convertDraftsToTabular(activity, sec)

    # Too many atoms in the draft pool? No problem. Shrink the number of atoms for each participant.
    #shrinkDraftToDraft2()

    # Finding cluster numbers for
    #labelDraftDictionary()

    # Learning final dictionary
    #learnFinalDictionary()

    # Forming one dictionary for all activities
    mergeLearnedDictionary()
