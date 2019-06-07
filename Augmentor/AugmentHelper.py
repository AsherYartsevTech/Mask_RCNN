from Augmentor.Augmentor_phase1 import *
from Augmentor.Augmentor_phase2 import *
from Augmentor.Augmentor_phase2_5_renamer import *
from Augmentor.Augmentor_phase3 import *
from Augmentor.augmentor_config import augConfig
import os.path
import os
import shutil
import re

annotation_file_name = "annotations.json"

def purge(dir, pattern):
    for f in os.listdir(dir):
        if re.search(pattern, f):
            os.remove(os.path.join(dir, f))

class AugmentHelper(object):
    def __init__(self, src_dataset_path, output_dataset_path):

        self.src_dataset_path = src_dataset_path
        self.output_dataset_path = output_dataset_path

        self.ground_truth_path = os.path.join(os.path.dirname(self.src_dataset_path), os.path.split(src_dataset_path)[1] + "_ground_truth")
        if not os.path.exists(self.ground_truth_path):
            os.mkdir(self.ground_truth_path)
            # todo: remove after augmentation finished
        
        if not os.path.exists(self.output_dataset_path):
            os.mkdir(self.output_dataset_path)
        
        self.annotation_file_path = os.path.join(src_dataset_path, annotation_file_name)  
        self.output_dataset_path = output_dataset_path
    
    def configure(self):
        pass

    def generate(self, action, augConfig):
        categories_map = augment_create_mask_files(self.src_dataset_path, self.annotation_file_path, self.ground_truth_path)
        if action is 'resize':
            aug_PerformResizePipe(self.src_dataset_path, self.ground_truth_path, self.output_dataset_path, augConfig)
        if action is 'augment':
            aug_AugmentingStage(self.src_dataset_path, self.ground_truth_path, self.output_dataset_path, augConfig)
        augment_batch_rename(self.output_dataset_path)
        augment_reconstruct_json(self.output_dataset_path, self.annotation_file_path, categories_map)
        shutil.rmtree(self.ground_truth_path)   
        purge(self.output_dataset_path, "ground_truth_")




aug_ResizeStage = AugmentHelper(augConfig['folderOfDatasetToBeManipulated'],
                                    augConfig['outputFolderOfEquallyResizedDataset'])
aug_ResizeStage.generate('resize', augConfig)
#just emphasing the uselessness of this object after this stage
del aug_ResizeStage

aug_AugmentingStage = AugmentHelper(augConfig['outputFolderOfEquallyResizedDataset'],
                                     augConfig['outputFolderOfAugmentedAndResizedDataset'])
aug_AugmentingStage.generate('augment', augConfig)

print("finish")


