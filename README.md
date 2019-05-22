# Academic reference 
* [Essay of Project](ourEssay.pdf)


# Environment
* we worked over TensorFlow 1.4 and anaconda environment. a .yml file with dependencies will be uploaded soon.

# Hands on
## This bullets are consecutive steps we suggest to perform for easier diving-in:
* First, we are available for any question and support here and on asheryartsev@gmail.com - so don't be shy.
* focus on `cucu_train.py` only. read the code that deals with creating the model and relevant data. it is self explanatory.
 * we bring to notice that many sections are stiil to-be-extracted to methods etc. 
* Run it and handle all env. obstacles which will get to you:)
* now, once you are ready to play with all the parameters of our project open `cucu_config.py` - you got there anything you need to control NN hyper-parameters and data-generating parameters.
  * we suggest to nevigate to original config file of Mask RCNN where hyperparameters definitions are more elaborated.
* Next we suggest to explore our `project_assets` folder.
  * There, you'll get aqcuainted with our classes for generated dataset creation, real, and hybrid (`cucu_classes.py`).
  * In `cucu_utils.py`  we poured core-functions for generating synthetic images of crops.
* Now, you should be ready for `playground.py` where you can run different metrics on a small test-set to benchmark your trained NN.
 * Lot's of analysing functionalities where imported and upgraded from [here](https://github.com/matterport/Mask_RCNN/tree/master/samples/balloon)

  


