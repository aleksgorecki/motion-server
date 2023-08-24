# Motion Server

A repository containing the code of the HTTP server and tensorflow neural network model used in the 'Motion Gestures for Android' project.

## Motions

Motions are represented by 3 lists of floating point numbers. These lists represent the values from 3 possible axes (X, Y, Z) of motion recorded by the smartphone's accelerometer. This data may be also arranged in a matrix, since all the lists have the same length.

Motions can be visualized on a 2D plot:
![](readme_res/motionplot.jpg)

Their matrix form may be also saved as an image:
![](readme_res/motionimage.bmp)

## Server

The server is used in conjunction with the Android Motion Recorder motion application ([github.com/aleksgorecki/AndroidMotionRecorder](https://github.com/aleksgorecki/AndroidMotionRecorder)).

## Models

