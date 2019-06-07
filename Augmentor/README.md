# Hands on

## Modules to focus on:
* Augmentor_phase2.py
* AugmentorHelper.py

## Algorithm assumes that:
* There exists a folder which contains ({images}, json_annotations_for_{images}_in_COCO_format).
* we suggest first to resize all images to equal sizes.
* second stage is to perform all augmentation.
* package is installed: 'pip install git+https://github.com/simonlousky/alteredAugmentor.git'

## Working with algorithm:
* Go to augmentor_config.py and adjust paths and parameters of execution to your own evironment.
* Execute AugmentorHelper.py and result will wait for you in path you've configured in previous stage.
* if you wish to remove augmenting techniques:
    * Go to Augmentor_phase2.py --> function:aug_PerformAugmentingPipe
    and comment out some functionalities.
*if you wish to add augmenting techniques:
    *navigate to Augmentor.pipeline package pre-Installed by you in previous section,
    and add from there any funvtionality that suites you.
    
That's it, plain and simple