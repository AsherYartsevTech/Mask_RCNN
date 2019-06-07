



## configuration file to be modified and adapt for arbitrary use:

augConfig = {
    # relevant paths:
    'folderOfDatasetToBeManipulated': '/home/simon/Documents/cucu_dataset/real/4000/cucumber/train/original',
    'outputFolderOfEquallyResizedDataset': '/home/simon/Documents/cucu_dataset//real/1024/cucumber/train/original',
    'outputFolderOfAugmentedAndResizedDataset': '/home/simon/Documents/cucu_dataset/real/1024/cucumber/train/augmented',

    #relevant algorithm options:
    'resizeWidth': 768,
    'resizeHeight': 512,
    'sizeOfExpectedAugmentedDataset': 2000

}