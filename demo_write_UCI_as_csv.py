from data_cleaning.uci_handler import UCI_Handler

if __name__ == "__main__":
    uh = UCI_Handler()

    # Write training data as csv
    dataset_type = "train"
    data_folder = r"/Users/matin/Dropbox/Work-Research/Current Directory/Shape Features/Data/UCI dataset/UCI HAR Dataset/{}/".format(
        dataset_type)

    uh.saveUCIFilesAsCSV(data_folder, dataset_type)

    # Write test data as csv
    dataset_type = "test"
    data_folder = r"/Users/matin/Dropbox/Work-Research/Current Directory/Shape Features/Data/UCI dataset/UCI HAR Dataset/{}/".format(
        dataset_type)
    uh.saveUCIFilesAsCSV(data_folder, dataset_type)