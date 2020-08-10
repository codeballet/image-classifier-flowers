# Deep Neural Network for classifying flowers
This repository contains a Deep Neural Network image classifier, built with PyTorch, operated through a command line application.

In order to use the classifier, you first need to train the network. The training process will generate a file called 'checkpoint.pth', which contains the trained network. That file will probably be around 700 MB in size.

Once the network is trained, it can classify the 102 most common flowers in the UK. In my testing, I  have managed to get the network up to around 80% accuracy, using the default settings of the command line app.


## Hardware requirements
Please note that unless you have access to a fairly powerful NVIDIA graphics card with CUDA, training the Network may take a very long time indeed (like weeks or months)!


## Software requirements
- Python 3.6 or higher.
- PyTorch, which could be installed with either pip3 or Anaconda / Conda.


## Organizing the training, testing, and validation data
In order to train the network, you first need to create a folder called 'flowers', containing images. The images may be aqcuired from [this dataset of flowers](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html), provided by Maria-Elena Nilsback and Andrew Zisserman.

The images should be divided into three datasets, for training, testing, and validation, according to the below structure:

```
flowers/train/[index]/[filename].jpg
flowers/test/[index]/[filename].jpg
flowers/valid/[index]/[filename].jpg
```

All of the indices for the flowers are listed as numbered categories in the file `cat_to_name.json`.

Please note that each of the three data sets (training, testing, and validation) must include instances of all of the 102 flower categories / indeces.

The data for the imageclassifier is loaded using [torchvision.datasets.ImageFolder](https://pytorch.org/docs/0.3.0/torchvision/datasets.html#imagefolder). Please consult that documentation in case you want further details about how to use 'ImageFolder' and the organization of the training data.


## Creating the Network
Once the images are organized, as describec above, you may run the command line application for training the Network with default settings by entering:
`python train.py`
or
`python3 train.py`
depending on your computer.

The help menu for using the command line applications can be accessed by entering the `-h` flag, as in:
`python train.py -h`


## Using the network for image classification
Once the network is trained and you have a `checkpoint.pth` file, you may use the network for image classification by running the command:
`python predict.py -f <filepath of flower>`

You may access the help menu of the command line application by typing:
`python predict.py -h`

The help menu has the following information.

```
usage: predict.py [-h] [-c CHECKPOINT] [-f FLOWER] [-g {on,off}] [-k K]
                  [-n NAMES]

get the class probability of an image using a checkpoint

optional arguments:
  -h, --help            show this help message and exit
  -c CHECKPOINT, --checkpoint CHECKPOINT
                        state checkpoint filepath
  -f FLOWER, --flower FLOWER
                        state flower filepath
  -g {on,off}, --gpu {on,off}
                        turn graphics processor on or off
  -k K                  define the amount of classes and probabilities to
                        print
  -n NAMES, --names NAMES
                        json file path with category names

```

Do note from the above that you may specify your own checkpoint file, as well as your own json file with different category names to the one supplied by default.

In case you do not give a filename with the `-f` flag, the program will simply grab a flower from the `flowers` directory (expecting a file on the path `flowers/test/1/image_06743.jpg`).


## Adapting the code for your own use
Please note that the file `workspace_utils.py` contains utilities for training the Network on Udacity's servers, where I first developed and trained the network.
In case you are training the network yourself, you may exclude that file, imports of that file, as well as removing line 32 in file `train.py`:
```
    with active_session():
```
The `active_sessins` utility only serves the purpose of keeping the Udacity server alive during training session.