# Hand-Gesture-Recognition

### About
Detect hand gesture using webcam. Design neural network (Convolution Neural Network) from scratch using Tensorflow.



### Project Structure 
.<br>
├── **data**:- contains training data<br>
├── **gesture**:- contains all the scripts<br />
&emspg;&emsp;├── **builder.py**:- create custom image dataset using webcam<br />
&emsp;&emsp;├── **helper.py**:-  load and read the dataset for training<br />
&emsp;&emsp;├── **model.py**:-  train & validate the model on the  train dataset <br />
&emsp;&emsp;├── **evaluate.py**:- test/evaluate the model on the unknown/test dataset<br />
&emsp;&emsp;├── **gesture.py**:- live detection using hand gesture<br />
├── **model**:- contains the trained model <br />
├── **output**:- contains images.

### Models Overview
Its is a 5 layers convolution neural network with 2 fully connected network.
max-pooling is done on every convolution layer in this case 5 layers
![Neural Network](./output/nn.svg "NN")

**Data parameter** 
images dimensions are 64*64

###  Environment Dependencies
1. Tensorflow 1.3
2. Numpy
3. OpenCV

### Resources

https://course17.fast.ai/lessons/lesson4.html
