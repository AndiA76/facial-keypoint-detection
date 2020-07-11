[//]: # (Image References)

[image1]: ./images/key_pts_example.png "Facial Keypoint Detection"


# Facial Keypoint Detection using CNNs built from scratch and Pytorch

## Overview

This project from Udacity's nano-degree course "Computer Vision" is about Facial Keypoint Detection using Convolutional Neural Networks (CNNs) and Pytorch. The task is to combine computer vision and deep-learning techniques to built a Facial Keypoint Detector based on a CNN model built from scratch, which is trained on a labeled dataset of color images to detect characteristic keypoints of human faces. Facial keypoints include points around the eyes, nose, mouth on a face and the outer facial contour. Potential applications are facial tracking, facial pose recognition, facial filters, or emotion recognition. 

![Facial Keypoint Detection][image1]

For training, model validation and testing of the facial keypoint detector a dataset has been extracted from the [YouTube Faces Dataset](https://www.cs.tau.ac.il/~wolf/ytfaces/), which includes videos of people in YouTube videos. These videos have been fed through some processing steps and turned into sets of image frames containing one face and the associated keypoints.

The project is broken down in the following subtasks, which are implemented using Python in individual Jupyter notebooks:

__Notebook 1__ : Loading and Visualizing the Facial Keypoint Data ![1. Load and Visualize Data](./1. Load and Visualize Data.ipynb)

__Notebook 2__ : Defining and Training a Convolutional Neural Network (CNN) to Predict Facial Keypoints ![2. Define the Network Architecture](./2. Define the Network Architecture.ipynb)

__Notebook 3__ : Facial Keypoint Detection Using Haar Cascades and your Trained CNN ![3. Facial Keypoint Detection, Complete Pipeline](./3. Facial Keypoint Detection, Complete Pipeline.ipynb)

__Notebook 4__ : Fun Filters and Keypoint Uses ![4. Fun with Keypoints](./4. Fun with Keypoints.ipynb)


### Further Project Instructions from Udadity

Starting code, installation instructions and resources for this project have been provided by Udacity. Please refer to  [exercise code](https://github.com/udacity/CVND_Exercises).


## Installation

### 1. Install Anaconda with Python

In order to run the notebook please download and install [Anaconda](https://docs.anaconda.com/anaconda/install/) with Python 3.6 on your machine. Further packages that are required to run the notebook are installed in a virtual environment using conda.


### 2. Create a Virtual Environment

In order to set up the prerequisites to run the project notebook you should create a virtual environment, e. g. using conda, Anaconda's package manager, and the following command

```
conda create -n computer-vision python=3.6
```

The virtual environment needs to be activated by

```
activate computer-vision
```


### 3. Download the project from github

You can download the project from github as a zip file to your Downloads folder from where you can unpack and move the files to your local project folder. Or you can clone from Github using the terminal window. Therefore, you need to prior install git on your local machine e. g. using

```
conda install -c anaconda git
```

When git is installed you can create your local project version. Clone the repository, and navigate to the download folder. This may take a minute or two to clone due to the included image data.

```
git clone https://github.com/AndiA76/facial-keypoint-detection.git
cd facial-keypoint-detection
```

### 4. Download the dataset for the project from Udacity's github repository

This project uses image data that has been extracted from the [YouTube Faces Dataset](https://www.cs.tau.ac.il/~wolf/ytfaces/), which includes videos of people in YouTube videos. These videos have been fed through some processing steps and turned into sets of image frames containing one face and the associated keypoints.

This preprocessed facial keypoints dataset extract consists of 5770 color images. All of these images are separated into either a training or a test set of data. The data for validation is split off the training data set. 

* 3462 of these images are training images, for you to use as you create a model to predict keypoints.
* 2308 are test images, which will be used to test the accuracy of your model.

The information about the images and the keypoints in (x, y) coordinates in this dataset are summarized in CSV files, which we can read in using `pandas`. 

All of the data you'll need to train, validate and test the CNN models in this repo are stored in a sub-directory `data`. In this folder are training and tests set of image/keypoint data, and their respective csv files. This will be further explored in Notebook 1: [1. Load and Visualize Data](./1. Load and Visualize Data.ipynb).

Please create a sub-directory `data` in your local project folder and download the image dataset directly from Udacity's github repository. This may take a few minutes to clone or download due to the size.

```
mkdir data
cd data
```
Download Udacity's original project repository from https://github.com/udacity/P1_Facial_Keypoints and look in the sub-directory `data` for the dataset. Copy all files and folders to your local project directory `facial-keypoint-detection/data`.


### 5. Install Pytorch with GPU support

In order to run the project notebook you need to install Pytorch. If you wish to install Pytorch with GPU support you also need to take care of your CUDA version and some [dependencies with Pytorch](https://pytorch.org/get-started/previous-versions/). I have used Ubuntu 18.04 LTS with CUDA 10.0 and Python 3.6 to run the project notebook. Therefore you need to enter the following installation command:

CUDA 10.0
```
conda install pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=10.0 -c pytorch
```

### 6. Further requirements 

Besides Pytorch you need to install a couple of further packages, which are required by the project notebook. The packages are specified in the [requirements.txt](requirements.txt) file (incl. OpenCV for Python). You can install them using pip resp. pip3:

```
pip install -r requirements.txt
```


### 7. Run the notebook

Now start a Jupyter notebook to run the project using following command

```
jupyter notebook
```

Navigate to your local project folder in the Jupyter notebook, open the notebooks 1...4

[1. Load and Visualize Data](./1. Load and Visualize Data.ipynb)  

[2. Define the Network Architecture](./2. Define the Network Architecture.ipynb)  

[3. Facial Keypoint Detection, Complete Pipeline](./3. Facial Keypoint Detection, Complete Pipeline.ipynb)  

[4. Fun with Keypoints](./4. Fun with Keypoints.ipynb)  

and run them one after another.
