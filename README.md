# CSC 736 - Machine Learning
## Perceptron Learning Algorithm
## Objectives:  
1. Learn and demonstrate knowledge of Perceptron classification.
2. Visualize the learning process of Perceptron.

A perceptron is able to classify linear separable dataset. In this assignment, you are
required to develop a program that is able to:  
1. Generate points in the training set.  
   - Arbitrarily define a line (eg. y = ax+b or ax+by+c=0);  
   - Generate 20 random data points on a 1000 by 1000 size canvas. Based on the line in the previous step, assign the class (1 or -1) to each points.  
   - Visualize the line in green and the points (circles filled or unfilled) on a graphic user interface similar to the attached figure.  
2. Implement a perceptron and perceptron learning algorithm.  
   - Randomly initialize the weights to a double within (0, 1).  
   - Set your learning rate to 0.0001;  
   - Train your perceptron by the perceptron learning algorithm with the provided training data generated from the previous step.  
   - Define “epoch” as one iteration of training all the training data one time.  
   - Visualize the line represented by the current weights at the end of each epoch on GUI. (like an animation.)  
   - Output the number of misclassification on the training data at the end of each epoch.  
   - Terminate the training process if all the training data are correctly classified by the perceptron.  
