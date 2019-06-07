import sys
import Augmentor as alteredAugmentor
# sys.path.append("/Users/orshemesh/Desktop/project repo/alteredAugmentor")  # To find local version of the library
# path = "/Users/orshemesh/Desktop/Project/augmented_leaves/"
# this dir must  contain input_dir (the name does not mather) + ground_truth dir

def aug_PerformResizePipe(src_path, ground_truth_path, destination_path, augConfig):

    # output dir will be in <path>/output
    p = alteredAugmentor.Pipeline(src_path, output_directory=destination_path, save_format="JPEG")

    p.ground_truth(ground_truth_path)

    p.flip_left_right(probability=0.4)
    p.flip_top_bottom(probability=0.2)
    p.rotate(probability=0.3, max_left_rotation=25, max_right_rotation=25)
    p.random_color(probability=0.3, min_factor=0.7, max_factor=1.3)
    p.random_contrast(probability=0.4, min_factor=0.7, max_factor=1.3)
    p.random_brightness(probability=0.3, min_factor=0.7, max_factor=1.3)
    p.skew(probability=0.1, magnitude=0.1)

    # randomly choose and augment from original dataset , augConfig['sizeOfExpectedAugmentedDataset']-times.
    p.sample(augConfig['sizeOfExpectedAugmentedDataset'])


def aug_PerformAugmentingPipe(src_path, ground_truth_path, destination_path, augConfig):
    # output dir will be in <path>/output
    p = alteredAugmentor.Pipeline(src_path, output_directory=destination_path, save_format="JPEG")

    p.ground_truth(ground_truth_path)
    p.resize(probability=1.0, width=augConfig['resizeWidth'], height=augConfig['resizeHeight'])
    p.process()

