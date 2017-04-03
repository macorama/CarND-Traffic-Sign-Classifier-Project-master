# Traffic Sign Classifier

### Introduction

I did most of the development on my macbook with an NVIDIA GPU.
In the beginning I struggled to get the GPU working with tensorflow and realized
conda does not support GPU yet.

I used numpy to calculate the stats of the given dataset

Number of training examples = 34799
Number of testing examples = 12630
Image data shape = (32, 32)
Number of classes = 43

# Visualization

I used a barchart to show the different traffic signs and number of samples for each.
This helped later in the exercise when i noticed getting good prediction for signs with 
minimal training sets was hard.

# Algorithm
Converting to gray improved the speed of the process little bit though results were 
not drastically different when the dataset was color vs gray

I created additional data using the tricks mentioned here 
https://github.com/vxy10/ImageAugmentation

It required scaling, rotating and transforming the images in the inout data set. This is 
almost like taking a picture of a sign from different angles.

openCV helps achieve this a lot simply than other libraries. 

Once the additional data was generated, shuffle was used to ensure the learning was not 
converging to a local area. 

We took a fourth of the training data set and used it for validation data set.

I downloaded 7 images from the internet and saw couple were not recognized too well. I 
added more samples to that dataset for that class and it still did not recognize 20kmph too 
well. 

```sh
def skewedImage(img):
tx = random.randint(-2,2)
ty = random.randint(-2,2)
scale = random.uniform(.9, 1.1)
theta = random.uniform(-15,15)
rows,cols,_ = img.shape
M = cv2.getRotationMatrix2D((cols/2,rows/2),theta,scale)
img = cv2.warpAffine(img,M,(cols,rows))
N = np.float32([[1,0,tx],[0,1,ty]])
return cv2.warpAffine(img,N,(cols,rows))
```

# Experimentation

DATASET_TIMES = 5 was used to generate 5 times the input data.

At DATASET_TIMES = 1 or 2 the accuracy did get worse.

At the end of this we landed we 

Number of training examples = 130496
Number of validation examples = 43499


I made little modifications to the LeNet described in the class 
(https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/6df7ae49-c61c-4bb2-a23e-6527e69209ec/lessons/b516a270-8600-4f93-a0a3-20dfeabe5da6/concepts/83a3a2a2-a9bd-4b7b-95b0-eb924ab14432)

Branched connection of the first convolutional layer after pooling to the fully connected output layer (layer 5) 

Network belongs to the class of convolutional neural networks whose architecture is inspired by the structure of the animal visual cortex.

# Final model looked like:

- Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
- RELU Activation.
- Pooling. Input = 28x28x6. Output = 14x14x6.
- Layer 2: Convolutional. Output = 10x10x16.
- RELU Activation.
- Pooling. Input = 10x10x16. Output = 5x5x16.
- Adding a branched output of stage 1 to the output of stage 2, input from stage 2 = 10x10x16, input from stage 1 = 14x14x6
- Layer 3: Fully Connected. Input = 1576. Output = 120.
- RELU Activation.
- Layer 4: Fully Connected. Input = 120. Output = 84.
- RELU Activation.
- Layer 5: Fully Connected. Input = 84. Output = 43.


# Model was trained by

- Using graycolor images and expanding the dataset by 5-6 times. 

- Epochs were experimented from 20-25

- rate was .001 or variable based on the output (latter did not help much)

- batch size of 128 or 256


I started with the LeNet implementation from the class and google my way to modify it to get slightly better accuracy. I played with params in the LeNet till I could not improve the accuracy further. 20 kmph was always getting predicted as 30kmph and i was able ti improve the testing accuracy closer to .7 but not much further.




N ex  | Target               | 1st choice           | 2nd choice           | 3rd choice          
--------------------------------------------------------------------------------------------------
990 | No entry             | No entry             | End of all speed and passing limits | Speed limit (20km/h)
1980 | Speed limit (30km/h) | Speed limit (30km/h) | Speed limit (50km/h) | Go straight or left 
300 | Roundabout mandatory | Roundabout mandatory | Priority road        | Speed limit (30km/h)
1320 | No passing           | No passing           | Speed limit (20km/h) | Speed limit (30km/h)
1350 | Road work            | Road work            | Bumpy road           | Children crossing   



| Target               | 1st choice prob %   
------------------------------------------------------------------
| No entry             |                  1.0
| Speed limit (30km/h) |                  1.0
| Roundabout mandatory |                  1.0
| No passing           |                  1.0
| Road work            |                  1.0
In [29]:


I was testing with both speed limit 20 and speed limit 30.

Both these had difficulty getting probablity 1%. 

I enhanced the dataset by another factor and this improved speed limit 30.

I think if I had worked on the brightness of the images more this could have helped the speed limit signs as they tend to look similar looking at the results. 

Test set accuracy is around 92.6% but lower than the validation set accuracy of 96.4%










