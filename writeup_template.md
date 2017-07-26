#**Behavioral Cloning** 

##Udacity-Self Driving Car Project-3


---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

I implemented nVIDIA CNN model for training/prediction and used Pyhton/Keras to implement the model. 
1. Input size 66x200x3 shape.
2. Used Lambada to normalize the data
3. 24 filter of Convolution2D (5x5) kernal followed by ELU activation
4. 36 filter of Convolution2D (5x5) kernal followed by ELU activation
5. 48 filter of Convolution2D (5x5) kernal followed by ELU activation
6. 64 filter of Convolution2D (3x3) kernal followed by ELU activation
7. 64 filter of Convolution2D (3x3) kernal followed by ELU activation
8. flatten layer
9. Dense layer (100) followed by ELU activation and dropout (.5) to avoid overfitting
10. Dense layer (50) followed by ELU activation and dropout (.5) to avoid overfitting
11. Dense layer (10) followed by ELU activation and dropout (.5) to avoid overfitting
12. Output Dense layer (1)
13. Used mse loss and Adam optimizer
14. Used fit generator to optimize memory usage

####2. Attempts to reduce overfitting in the model

The model contains dropout (.5) layers after last 3 dense layers in order to reduce overfitting (model.py lines 124, 127 & 130). 

The model was trained and validated on different data sets, used sklearn to split the dataset into training and validation. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 134).

####4. Appropriate training data

I chose data provide by Udacity to train the model, used various data Augmentation method to make it more versatile. Details of data augmentation is discussed in next paragraph.

###Model Architecture and Training Strategy

####1. Solution Design Approach
Preparing data for Training was the crucial part of my project. As suggested by David in project description, the data should have variety for the model to train e.g. turning, recovering from left/right drifting, removing left bias, driving on the center of the road.
My aim was to have: 
1. Balance data as the correct data was skewed on zero (because car drives straight most of the time)
2. To have data recovering from left and right drift, so that if model can learn to recover from drifts

I had nVIDIA, VGG and comma.ai CNN model as option for training, I started with nVidia and it worked well for me.

I split the data (80-20) for training and validationm, generated the data on run time using fit_generator feature to optimize memory usage, avoided overfitting by having 3 layers of dropout. I tried 3,5, 7 epochs and 7 worked for me.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle drifted on the sides but it recovered immediately..

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes
1. Input size 66x200x3 shape.
2. Used Lambada to normalize the data
3. 24 filter of Convolution2D (5x5) kernal followed by ELU activation
4. 36 filter of Convolution2D (5x5) kernal followed by ELU activation
5. 48 filter of Convolution2D (5x5) kernal followed by ELU activation
6. 64 filter of Convolution2D (3x3) kernal followed by ELU activation
7. 64 filter of Convolution2D (3x3) kernal followed by ELU activation
8. flatten layer
9. Dense layer (100) followed by ELU activation and dropout (.5) to avoid overfitting
10. Dense layer (50) followed by ELU activation and dropout (.5) to avoid overfitting
11. Dense layer (10) followed by ELU activation and dropout (.5) to avoid overfitting
12. Output Dense layer (1)
13. Used mse loss and Adam optimizer
14. Used fit generator to optimize memory usage

![nVIDIA CNN Model](https://github.com/devksingh/Behavioural-Cloning-P3/blob/master/nVidia%20CNN%20Model.png)


####3. Creation of the Training Set & Training Process

Following was the strategy for data augmentation:
1. Correcting steering angle of left and right image (+0.23 for left and -0.23 for right) - this to help drift recovery
2. Moving the image by random number of pixels horizontly left and right by using cv2warpaffine funtion and using .004/pixel as steering angle correction - Again this is to simnulate drift recovery
3. Flip the image and multiply -1 to steering angle
Training Data included:
1. Center image 
2. Horizontly shifted center image by random pixel
3. Horizontly shifted left and right image by random pixel
Please note I did not use left and right image as it is with corrected steering angle since the original data was skewed to zero and thie augmentaion woould have made final data to be skewed on -0.23, 0 and +0.23
I did not use manipulating brightness or shadow on the picture and there is no variation of brightness and shadow on the given track so learning that would have been useless for project output.

After augmentation images were cropped by 50px top and 25 px bottom (to avoid insignificant data) and resized to 66x200

Histogram of data before and after augmentation
![Before Augmentation](https://github.com/devksingh/Behavioural-Cloning-P3/blob/master/Trainig_Data.png)

![After Augmentation](https://github.com/devksingh/Behavioural-Cloning-P3/blob/master/Training_Data_After_Augmentation.png)


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 7 as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
