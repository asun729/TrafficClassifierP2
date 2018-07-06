# **Traffic Sign Recognition** 

## Writeup

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./DataSetExploration/BarChartTrainingData.png "Visualization"
[image2]: ./DataSetExploration/sample_images.png "Samples"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./NewImages/1.jpg "Traffic Sign 1"
[image5]: ./NewImages/2.jpg "Traffic Sign 2"
[image6]: ./NewImages/3.jpg "Traffic Sign 3"
[image7]: ./NewImages/4.jpg "Traffic Sign 4"
[image8]: ./NewImages/5.jpg "Traffic Sign 5"

[image11]: ./DataSetExploration/SideBySide1.png "Compare1"
[image12]: ./DataSetExploration/SideBySide2.png "Compare2"
[image13]: ./DataSetExploration/prob.jpg "Top Five"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  
Here is a link to my [project code].

### Data Set Summary & Exploration

#### 1.Basic summary of the data set.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

        Number of training examples = 34799
        Number of validation examples = 4410
        Number of testing examples = 12630
        Image data shape = (32, 32, 3)
        Number of classes = 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing training set data are distributed among 43 classes of traffic signs.

![alt text][image1]

Frequencies of classified traffic signs are not ditributed evenly though. The speed limit signs (classid 0-8) take around 30.7% of the total training data.

For an intuitive vision of the training images, a sample of the training data set is shown below. 
![alt text][image2]

### Design and Test a Model Architecture

#### 1. Preprocessing
Three techniques of preprocessing were implemented in this section. 
* Enhancement of contrast

From the inspection of training set, the difference in luminance was very low in many images. Some are extremely overexposure while others are underexposure. Simply adjusting the brightness in one way won't solve both scenario or help detecting edges with low differences. 

Therefore, an enhancement of contrast was conducted on training, validation and test sets, as indicated in the *contrast* function.This is a simplified implementation of Yang's paper on Robust Contrast Enhancement Forensics Using CNN.[1]
A gamma value of 0.55 was chosen to enhance the contrast of images using the following correlation


        Y = 255*(X/255)^gamma


Gamma = 0.5,0.55,0.6,0.7,0.8 were tested on the training set, where 0.55 resulted in highest accuracy in validation sets. 

* Normalization. The purpose of this step is to ensure that each input has similar distribution and ranges from 0 to 1. After enhancing the contrast of images, mean value and ranges of enhanced training set were calculated. Data sets were normalized around mean and divided by ranges(255). 


        Y = (X-mean)/range
Here is an example of a traffic sign image before and after contrast enhancement and normalization.

![alt text][image11]

* Grayscale. To reduce the computational cost, grayscale was applied using the average value over three channels of the original images. Same strategies applied to training, validation and test sets. 
Here is an example of a traffic sign image before and after contrast all the three steps stated above.

![alt text][image12]


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model is based on LeNet lab solution, with modification of dropout to reduce overfitting. It consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   							| 
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6		 		|
| Convolution 3x3	    | 1x1 stride, valid padding, output 10x10x16   	|
| RELU					|												|
| Dropout         		| keep_prob = 0.75   							| 
| Fully connected		| Input = 120. Output = 84        									|
| Relu				|        									|
| Dropout         		| keep_prob = 0.75   							|
| Fully connected		|	Input = 84. Output = 43.					|
|						|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.
The method described in previous section was applied to the preprocessed training data, with the following setup:

    Optimizer = AdamOptimizer
    Batch Size = 128
    Number of epochs = 15
    mu =0
    sigma =0.1
    rate = 0.001

Learning rate is recommended by the optimizer. Initially, 50 EPCHOS was run and accuracy trend of the validation model was plotted. Accuracy was converged to 0.94 after 15 epochs. Then I reduced the epcho number to 15 to get the following solution. 
#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set >99%
* validation set accuracy is 93.5%
* test set accuracy is 92.8%

The first chosen architechture was LeNet CNN, since it was designed for recognition of MNIST sized handwritten/machine generated images. The initial architechture had proper relu and pooling settings that give reasonable accuracy and computational efficiency. The initial architecture had an accuracy around 91% given the same preprocessing, but shown a tendency of overfitting. Therefore, a dropout algorithm was added to reduce the overfitting facts. By adding dropout twice after relu, with a keep prop of 0.75, the accuracy was increated by 2%. 


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]
Challenges:

* The first image might be difficult to classify because the brigtness is very low. 
* The second image has similar pattern as "End of no passing" and "End of no passing by vehicles over 3.5 metric tons"
* The third image has strippes in the background, which might affect the recognition.
* The fourth image is not complete, losing some print on figure "0". 
* The fifth image is blurry.
#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Go straight or right      		| Go straight or right   									| 
| No passing     			| No passing 										|
| Keep left					| Keep left											|
| speed limit 70	      		| speed limit 70					 				|
| No entry			| No entry      							|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set. 

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code and output printing top five probabilities are shown below. 
![alt text][image13]
For the first image, the model is relatively sure that it is "Go Straight or Right". The probability of other class id are all approaching zero. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1         			| Go straight or right   									| 
| 0     				| Speed limit (20km/h) 										|
| 0     				| Speed limit (30km/h) 										|
| 0     				| Speed limit (40km/h) 										|
| 0     				| Speed limit (50km/h) 										|
Similarly for the other four images, all other predictions have extremely low probability approaching zero. 


 