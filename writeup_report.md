#**Behavioral Cloning** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/center_image.jpg "Center Image"
[image2]: ./examples/right_recover_1.jpg "Recovery Image 1"
[image3]: ./examples/right_recover_2.jpg "Recovery Image 2"
[image4]: ./examples/right_recover_3.jpg "Recovery Image 3"
[image5]: ./examples/before_flipped.jpg "Image before flipping"
[image6]: ./examples/after_flipped.jpg "Image after flipping"

## Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
Files Submitted & Code Quality

#1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model_2017_04_04/model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results
* run2.mp4 narating the autonomous driving of a full lap of track 1

#2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model_2017_04_04/model.h5
```

#3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.


Model Architecture and Training Strategy

#1. An appropriate model architecture has been employed

My model consists of 2 convolution neural networks with 5x5 filter sizes and depths of 8, with each network followed by an relu activation and a max pooling of size 2x2. Then the vector is fed into a multi layer perceptron of 2 hidden dimension of 128 and 64 each. Output is a regression score toward human steering angle (line 53-65).

The model includes RELU layers to introduce nonlinearity (code line 56 and 58), and the data is normalized in the model using a Keras lambda layer (code line 55). 

#2. Attempts to reduce overfitting in the model

I select small network size to prevent overfiting. Drop out is not neccessary since the network size is small enough; drop out will hammer the results driving performance according to my experiments.

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 72). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (line 70). Epoch is tuned to select lowest validation lost. Batch size is selected large enough to maximize GPU capacity and reduce training time.

#4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. 
Augmented images also added to the trainding data (line 40-44).
Steering=0 examples are down sample to increase the response rate of network when approaching curves. (line 36)


Model Architecture and Training Strategy

#1. Solution Design Approach

The overall strategy for deriving a model architecture was to cloning human steering control given set of images.

My first step was to use a convolution neural network model similar to the LeNet; I thought this model might be appropriate because the tasks are similar (classifying score given an image v.s. regression score given an image).

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a high mean squared error on both training and validation sets. This implied that the model was underfitting. I increase the model size (conv net output chanel size, and multi perceptron hidden layer dimension).

The the training and validation loss both coverge to almost 0, yet the autonomous driving still does not perform well in the simulator. This may indicate insufficient training data.

I generate more training data via the simulator, yet the car still heads of road in many cases.

I add trainng data for the failure cases, and down sampling the no-steering example to increase the response rate of the model.

At the end of the process, the vehicle is able to drive autonomously around the full track without leaving the road (run2.mp4 video attached).

#2. Final Model Architecture

Convolution network, with filter size of 5x5, output chanel size of 8
Relu activation
Max pooling over filter size of 2x2, strike 1x1
Convolution network, with filter size of 5x5, output chanel size of 8
Relu activation
Max pooling over filter size of 2x2, strike 1x1
Multilayer perceptron: 2 hidden dimensions of 128, 64
Output 1 singe score fiting steeting angle

#3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded few laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image1]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to steering away for the lane liens. These images show what a recovery looks like starting from right land line :

![alt text][image2]
![alt text][image3]
![alt text][image4]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would enable the learn from steering from the other angle. For example, here is an image that has then been flipped:

![alt text][image5]
![alt text][image6]

No-steering examples are down sample to 30% to enable the network to be more reactive to steering rather than driving straight.

After the collection process, I had 71,076 training samples and 17,770 validating samples. 

I then preprocessed this data by cropping the top and bottom of the image, and normalizing pixel color to [-1, 1] for faster loss convergence.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 4 as evidenced by observation selecting smallest validation loss across all epochs. I used an adam optimizer so that manually training the learning rate wasn't necessary.
