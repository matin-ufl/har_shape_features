from dictionary_learning.supervised_learner import SupervisedLearner
from dictionary_learning.utils import Utils
import os
import pandas as pd
import time

def learnDictionaryDraft():
    tic = time.time()

    # Instantiating supervised learner class.
    sl = SupervisedLearner()

    # List of activities in csv files processed from UCI repository
    activities = ['STANDING', 'SITTING', 'LAYING', 'WALKING', 'WALKING_DOWNSTAIRS', 'WALKING_UPSTAIRS']
    standing_draft_dictionary_df = None
    sitting_draft_dictionary_df = None
    laying_draft_dictionary_df = None
    walking_draft_dictionary_df = None
    walkingDown_draft_dictionary_df = None
    walkingUp_draft_dictionary_df = None

    # Folder address where training csv files are located.
    training_folder = r"/Users/matin/Dropbox/Work-Research/Current Directory/Shape Features/Data/UCI dataset/UCI HAR Dataset/downsampled/train/"
    out_folder = r"/Users/matin/Dropbox/Work-Research/Current Directory/Shape Features/Data/UCI dataset/UCI HAR Dataset/downsampled/supervised/"

    filenames = os.listdir(training_folder)

    for filename in filenames:
        print("\n\n---------- {} ----------".format(filename))
        ppt_df = pd.read_csv("{}{}".format(training_folder, filename))

        # Standing
        activity_df = ppt_df.loc[ppt_df.activity == "STANDING", :]
        standing_draft_dictionary_df = sl.learnAndAppendDraftDictionary(activity_df, standing_draft_dictionary_df,
                                                                        atom_length=3, prc=20, dist_func=Utils.dtw,
                                                                        samplingRate=10)

        # Sitting
        activity_df = ppt_df.loc[ppt_df.activity == "SITTING", :]
        sitting_draft_dictionary_df = sl.learnAndAppendDraftDictionary(activity_df, sitting_draft_dictionary_df,
                                                                       atom_length=3, prc=20, dist_func=Utils.dtw,
                                                                       samplingRate=10)

        # Laying
        activity_df = ppt_df.loc[ppt_df.activity == "LAYING", :]
        laying_draft_dictionary_df = sl.learnAndAppendDraftDictionary(activity_df, laying_draft_dictionary_df,
                                                                      atom_length=3, prc=20, dist_func=Utils.dtw,
                                                                      samplingRate=10)

        # Walking
        activity_df = ppt_df.loc[ppt_df.activity == "WALKING", :]
        walking_draft_dictionary_df = sl.learnAndAppendDraftDictionary(activity_df, walking_draft_dictionary_df,
                                                                       atom_length=3, prc=20, dist_func=Utils.dtw,
                                                                       samplingRate=10)

        # Walking downstairs
        activity_df = ppt_df.loc[ppt_df.activity == "WALKING_DOWNSTAIRS", :]
        walkingDown_draft_dictionary_df = sl.learnAndAppendDraftDictionary(activity_df, walkingDown_draft_dictionary_df,
                                                                           atom_length=3, prc=20, dist_func=Utils.dtw,
                                                                           samplingRate=10)
        # Walking upstairs
        activity_df = ppt_df.loc[ppt_df.activity == "WALKING_UPSTAIRS", :]
        walkingUp_draft_dictionary_df = sl.learnAndAppendDraftDictionary(activity_df, walkingUp_draft_dictionary_df,
                                                                         atom_length=3, prc=20, dist_func=Utils.dtw,
                                                                         samplingRate=10)

    standing_draft_dictionary_df.to_csv("{}supervised_{}_draft_dictionary.csv".format(out_folder, "standing"),
                                        index=False)
    sitting_draft_dictionary_df.to_csv("{}supervised_{}_draft_dictionary.csv".format(out_folder, "sitting"),
                                       index=False)
    laying_draft_dictionary_df.to_csv("{}supervised_{}_draft_dictionary.csv".format(out_folder, "laying"), index=False)
    walking_draft_dictionary_df.to_csv("{}supervised_{}_draft_dictionary.csv".format(out_folder, "walking"),
                                       index=False)
    walkingDown_draft_dictionary_df.to_csv("{}supervised_{}_draft_dictionary.csv".format(out_folder, "walkingdown"),
                                           index=False)
    walkingUp_draft_dictionary_df.to_csv("{}supervised_{}_draft_dictionary.csv".format(out_folder, "walkingup"),
                                         index=False)

    toc = time.time()
    print("Learning dictionary is over. (elapsed time: {:.4f} seconds)".format(toc - tic))

    # Process took 24703.4936 seconds

def visualizeDendrogram(draftDictionary_filename):
    tic = time.time()
    sl = SupervisedLearner()
    # Reading draft dictionary
    draft_dictionary_df = pd.read_csv(draftDictionary_filename)
    # Reshaping to tabular for clustering
    dictionary_df = sl.reshapeToTabular(draft_dictionary_df)
    # Dendrogram visualization
    sl.atomVisualization_dendrogram(dictionary_df, dist_func=Utils.dtw)
    toc = time.time()
    print("elapsed time: {:.2f} seconds".format(toc - tic))

def learnFinalDictionary(draft_folder_address: str, cutoffs: dict, outfolder: str):
    # found cutoffs by visualizing the dendrograms (visualizeDendrogram) for each case and finding the desired number of clusters.
    tic = time.time()
    sl = SupervisedLearner()

    for k in cutoffs.keys():
        filename = "supervised_{}_draft_dictionary.csv".format(k)
        draft_dictionary_df = pd.read_csv("{}{}".format(draft_folder_address, filename))
        # Reshaping to tabular for clustering
        dictionary_df = sl.reshapeToTabular(draft_dictionary_df)
        # Finding final dictionary by picking an exemplar from each group of similar atoms.
        final_dictionary_df = sl.learnFinalDictionary(dictionary_df, cutoffs[k], dist_func=Utils.dtw)
        # Saving final dictionary to file.
        final_dictionary_df.to_csv("{}supervised_{}_final_dictionary.csv".format(outfolder, k), index=False)
    toc = time.time()
    print("Final dictionary learned for all cases. (elapsed time: {:.2f} seconds)".format(toc - tic))

def visualizeAtomPoints(draft_dictionary_filename: str, cutoff: float):
    # found cutoffs by visualizing the dendrograms (visualizeDendrogram) for each case and finding the desired number of clusters.
    tic = time.time()
    sl = SupervisedLearner()

    draft_dictionary_df = pd.read_csv(draft_dictionary_filename)
    # Reshaping to tabular for clustering
    dictionary_df = sl.reshapeToTabular(draft_dictionary_df)

    # Plotting original points
    sl.atomVisualization_2DScatterPoints(dictionary_df, Utils.dtw, None, False)

    # Finding final dictionary by picking an exemplar from each group of similar atoms.
    clst_df = sl.applyClustering(dictionary_df, cutoff, Utils.dtw)

    # Plotting points with cluster coloring
    sl.atomVisualization_2DScatterPoints(dictionary_df, Utils.dtw, clst_df.cluster_number.values, False)

    # Plotting clusters only
    sl.atomVisualization_2DScatterPoints(dictionary_df, Utils.dtw, clst_df.cluster_number.values, True)

    toc = time.time()
    print("All three figures plotted. (elapsed time: {:.2f} seconds)".format(toc - tic))

def visualizeAtomWaveforms(dictionary_folder, output_folder):
    sl = SupervisedLearner()
    activities = ['laying', 'sitting', 'standing', 'walking', 'walkingdown', 'walkingup']
    dictionary_df = None  # dataset to hold all atoms from every activity

    # Going over all the files in the folder and append all atoms to one DataFrame
    for activity in activities:
        df = pd.read_csv(
            r'{}supervised_{}_final_dictionary.csv'.format(dictionary_folder, activity))
        df["activity"] = activity
        if dictionary_df is None:
            dictionary_df = df
        else:
            dictionary_df = dictionary_df.append(df, ignore_index=True)

    sl.visualizeDictionaryWaveforms(dictionary_df, output_folder)

if __name__ == "__main__":
    # Learning draft dictionary
    #learnDictionaryDraft()

    # Checking dendrogram and finding the right number of clusters from draft dictionaries.
    #draftDictionary_filename = r"/Users/matin/Dropbox/Work-Research/Current Directory/Shape Features/Data/UCI dataset/UCI HAR Dataset/downsampled/supervised/supervised_walking_draft_dictionary.csv"
    #visualizeDendrogram(draftDictionary_filename)

    # Learning final dictionaries after carefully checking the dendrograms
    draft_folder_address = r"/Users/matin/Dropbox/Work-Research/Current Directory/Shape Features/Data/UCI dataset/UCI HAR Dataset/downsampled/supervised/"
    final_folder_address = r"/Users/matin/Dropbox/Work-Research/Current Directory/Shape Features/Data/UCI dataset/UCI HAR Dataset/downsampled/supervised/final_dictionaries/"
    cutoffs = {"laying": 0.020,
               "sitting": 0.008,
               "standing": 0.006,
               "walking": 0.042,
               "walkingdown": 0.072,
               "walkingup": 0.053}
    #learnFinalDictionary(draft_folder_address, cutoffs, outfolder=final_folder_address)

    #visualizeAtomPoints("{}supervised_{}_draft_dictionary.csv".format(draft_folder_address, "walking"), cutoffs["walking"])

    visualizeAtomWaveforms(final_folder_address, "/Users/matin/Desktop/dictionary.pdf")