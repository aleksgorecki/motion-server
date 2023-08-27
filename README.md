<h1> Motion Server </h1>

A repository containing the code of the HTTP server and tensorflow neural network model used in the 'Motion Gestures for Android' project.

## Table of Contents
- [Table of Contents](#table-of-contents)
- [Motions](#motions)
  - [Classes and Datasets](#classes-and-datasets)
- [Server](#server)
- [Models](#models)

## Motions

Motions are represented by 3 lists of floating point numbers. These lists represent the values from 3 possible axes (X, Y, Z) of motion recorded by the smartphone's accelerometer. This data may be also arranged in a matrix, since all the lists have the same length.

Motions can be visualized on a 2D plot:
![](readme_res/motionplot.jpg)

Their matrix form may be also saved as an image, in which the values of XYZ acceleration are converted to RGB color channels:  
<!-- ![](readme_res/motionimage.bmp) -->

<img src="readme_res/motionimage.bmp" width="500" height="30"/>

Motions are represented by a `Motion` class, defined in `motion.py`.

### Classes and Datasets
l
Recognized motions can be divided into 7 classes:
- nothing
- x_negative (XNEG)
- x_positive (XPOS)
- y_negative (YNEG)
- y_positive (YPOS)
- z_negative (ZNEG)
- z_positive (ZPOS)

The `nothing` class is a 'dump' class, created in order to combat the models overconfidence problem. 

The rest of the classes correspond to tilting the phone in one of the axes (X, Y, Z) and the direction of this tilting movement (positive/negative).

Movements are organised into datasets, which at the basic form are directories, with folders relating to each of the class. An example of such dataset is the `basic2` directory. Datasets may be converted to different forms, such as a HDF5 file or a Tensorflow Dataset. The conversion operation may be performed by creating an instance of the `MotionDataset` class and then calling the appropriate method.

## Server

The server is used in conjunction with the Android Motion Recorder application ([github.com/aleksgorecki/AndroidMotionRecorder](https://github.com/aleksgorecki/AndroidMotionRecorder)). 

Its most important use is to receive and save motions recorded by the application. The server may be run by executing the `main.py` script. 

The server has 3 endpoints:

- `/` (GET) - used to test if the server is running
- `/new` (POST) - used to send a recorded motion and save it as a json in a specified directory 
- `/predict` (POST) - used to get a prediction of the recorded motion's class

By default the server will run on the address assigned to the device by the local network.

## Models

The project uses small, 1D convolutional neural networks. The default model may be created by calling the `get_prototype_model` function inside `model.py`. It consists of two convolutional layers and three dropout layers. This model can be also loaded from the `model_latest.h5` file.