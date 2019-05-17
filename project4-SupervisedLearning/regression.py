# regression.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to Clemson University and the authors.
# 
# Authors: Pei Xu (peix@g.clemson.edu) and Ioannis Karamouzas (ioannis@g.clemson.edu)

"""
In this assignment, you will implement linear and logistic regression
using the gradient descent method. To complete the assignment, please 
modify the linear_regression(), and logistic_regression() functions. 

The package `matplotlib` is needed for the program to run.
You should also try to use the 'numpy' library to vectorize 
your code, enabling a much more efficient implementation of 
linear and logistic regression. You are also free to use the 
native 'math' library of Python. 

All provided datasets are extracted from the scikit-learn machine learning library. 
These are called `toy datasets`, because they are quite simple and small. 
For more details about the datasets, please see https://scikit-learn.org/stable/datasets/index.html

Each dataset is randomly split into a training set and a testing set using a ratio of 8 : 2. 
You will use the training set to learn a regression model. Once the training is done, the code
will automatically validate the fitted model on the testing set.  
"""

# use math and/or numpy if needed
import math
import numpy as np
import random as rand

def linear_regression(x, y, logger=None):
    """
    Linear regression using full batch gradient descent.
    A 1D array w should be returned by this function such that given a
    sample x, a prediction can be obtained by x^T w, where x is a column vector. 
    The intercept term can be ignored due to that x has been augmented by adding '1' as an extra feature. 
    If you scale the cost function by 1/#samples, you should use as learning rate alpha=0.001, otherwise alpha=0.0001  

    Parameters
    ----------
    x: a 2D array of size [N, f+1]
       where N is the number of samples, f is the number of features
    y: a 1D array of size [N]
       It contains the target value for each sample in x
    logger: a logger instance for plotting the loss
       Usage: logger.log(i, loss) where i is the number of iterations
       Log updates can be performed every several iterations to improve performance.
    
    Returns
    -------
    w: a 1D array
       linear regression parameters
    """
    max_iterations = 1000
    alpha = 0.01
    N , f = np.shape(x)
    w = np.zeros((f))
    print('No of features is: ', f, ' No of samples is: ', N)
   
    #changing the values of X0 to 1 for all rows of X
    #x[:,0] = np.ones(N)
    
    for 𝑘 in range(max_iterations): 
        prevW = w
        prediction = np.dot(x,w)
        #print(np.shape(prediction))
        #print(np.shape(y))
        loss = prediction - y #np.subtract(prediction,y)
        #print(np.shape(loss))
        cost = np.dot(np.transpose(loss), loss) / (2 * N)
        #print(cost)
        gradientStep = np.dot(np.transpose(x),loss) / N
        #print(np.shape(gradientStep))
        w = w - (alpha * gradientStep)  
        #print(w)
        logger.log(k,cost)
        if abs(prevW - w).all() < 1e-6:
            print('converged at iteration: ', k)
            return w
    return w


def logistic_regression(x, y, logger=None):
    """
    Logistic regression using batch gradient descent.
    A 1D array w should be returned by this function such that given a
    sample x, a prediction can be obtained by p = sigmoid(x^T w)
    with the decision boundary:
        p >= 0.5 => x in class 1
        p < 0.5  => x in class 0
    where x is a column vector. 
    The intercept/bias term can be ignored due to that x has been augmented by adding '1' as an extra feature. 
    In gradient descent, you should use as learning rate alpha=0.001    

    Parameters
    ----------
    x: a 2D array of size [N, f+1]
       where N is the number of samples, f is the number of features
    y: a 1D array of size [N]
       It contains the ground truth label for each sample in x
    logger: a logger instance for plotting the loss
       Usage: logger.log(i, loss) where i is the number of iterations
       Log updates can be performed every several iterations to improve performance.
        
    Returns
    -------
    w: a 1D array
       logistic regression parameters
    """
    max_iterations = 1000
    alpha = 0.001
    N , f = np.shape(x)
    w = np.zeros((f))
    print('No of features is: ', f, ' No of samples is: ', N)
   
    #changing the values of X0 to 1 for all rows of X
    #x[:,0] = np.ones(N)
    
    for 𝑘 in range(max_iterations): 
        prevW = w
        scores = np.dot(x,w)
        prediction = sigmoid(scores)
        loss = prediction - y #np.subtract(prediction,y)  
        z1 = np.subtract(1,y)
        z2 = np.subtract(1,prediction)
        cost = (- np.dot(np.transpose(y), np.log(prediction)) - np.dot(np.transpose(z1), np.log(z2)) )  / N    
        gradientStep = np.dot(np.transpose(x),loss) / N
        w = w - (alpha * gradientStep)  
        logger.log(k,cost)
        
    return w

def sigmoid(scores):
    return 1 / (1 + np.exp(-scores))

def linear_regression_sgd(x, y, logger=None):
    """
    Linear regression using stochastic gradient descent.
    A 1D array w should be returned by this function such that given a
    sample x, a prediction can be obtained by x^T w, where x is a column vector. 
    The intercept term can be ignored due to that x has been augmented by adding '1' as an extra feature. 
    If you scale the cost function by 1/#samples, you should use as learning rate alpha=0.001, otherwise alpha=0.0001  

    Parameters
    ----------
    x: a 2D array of size [N, f+1]
       where N is the number of samples, f is the number of features
    y: a 1D array of size [N]
       It contains the target value for each sample in x
    logger: a logger instance for plotting the loss
       Usage: logger.log(i, loss) where i is the number of iterations
       Log updates can be performed every several iterations to improve performance.
        
    Returns
    -------
    w: a 1D array
       linear regression parameters
    """
    max_iterations = 1000
    alpha = 0.01
    N , f = np.shape(x)
    w = np.zeros((f))
    
    print('No of features is: ', f, ' No of samples is: ', N)
    
    for 𝑘 in range(max_iterations): 
        randSample = rand.randint(0, N-1)
        prediction = np.dot(x[randSample], w)
        loss = prediction - y[randSample] 
        cost = loss * loss / 2 
        gradientStep = np.dot(np.transpose(x[randSample]),loss) 
        print(gradientStep)
        w = w - (alpha * gradientStep)  
        logger.log(k,cost)
    return w


def logistic_regression_sgd(x, y, logger=None):
    """
    Logistic regression using stochastic gradient descent.
    A 1D array w should be returned by this function such that given a
    sample x, a prediction can be obtained by p = sigmoid(x^T w)
    with the decision boundary:
        p >= 0.5 => x in class 1
        p < 0.5  => x in class 0
    where x is a column vector. 
    The intercept/bias term can be ignored due to that x has been augmented by adding '1' as an extra feature. 
    In gradient descent, you should use as learning rate alpha=0.001    

    Parameters
    ----------
    x: a 2D array of size [N, f+1]
       where N is the number of samples, f is the number of features
    y: a 1D array of size [N]
       It contains the ground truth label for each sample in x
    logger: a logger instance for plotting the loss
       Usage: logger.log(i, loss) where i is the number of iterations
       Log updates can be performed every several iterations to improve performance.
    
    Returns
    -------
    w: a 1D array
       logistic regression parameters
    """
    max_iterations = 1000
    alpha = 0.001
    N , f = np.shape(x)
    w = np.zeros((f))
    
    print('No of features is: ', f, ' No of samples is: ', N)
    
    for 𝑘 in range(max_iterations): 
        randSample = rand.randint(0, N-1)
        score = np.dot(x[randSample], w)
        prediction = sigmoid(score)
        loss = prediction - y[randSample] 
        z1 = np.subtract(1,y[randSample])
        z2 = np.subtract(1,prediction)
        cost = - np.dot(np.transpose(y[randSample]), np.log(prediction)) - np.dot(np.transpose(z1), np.log(z2))       
        gradientStep = np.dot(np.transpose(x[randSample]),loss) 
        w = w - (alpha * gradientStep)  
        logger.log(k,cost)
    return w


if __name__ == "__main__":
    import os
    import tkinter as tk
    from app.regression import App

    import data.load
    dbs = {
        "Boston Housing": (
            lambda : data.load("boston_house_prices.csv"),
            App.TaskType.REGRESSION
        ),
        "Diabetes": (
            lambda : data.load("diabetes.csv", header=0),
            App.TaskType.REGRESSION
        ),
        "Handwritten Digits": (
            lambda : (data.load("digits.csv", header=0)[0][np.where(np.equal(data.load("digits.csv", header=0)[1], 0) | np.equal(data.load("digits.csv", header=0)[1], 1))],
                      data.load("digits.csv", header=0)[1][np.where(np.equal(data.load("digits.csv", header=0)[1], 0) | np.equal(data.load("digits.csv", header=0)[1], 1))]),
            App.TaskType.BINARY_CLASSIFICATION
        ),
        "Breast Cancer": (
            lambda : data.load("breast_cancer.csv"),
            App.TaskType.BINARY_CLASSIFICATION
        )
     }

    algs = {
       "Linear Regression (Batch Gradient Descent)": (
            linear_regression,
            App.TaskType.REGRESSION
        ),
        "Logistic Regression (Batch Gradient Descent)": (
            logistic_regression,
            App.TaskType.BINARY_CLASSIFICATION
        ),
        "Linear Regression (Stochastic Gradient Descent)": (
            linear_regression_sgd,
            App.TaskType.REGRESSION
        ),
        "Logistic Regression (Stochastic Gradient Descent)": (
            logistic_regression_sgd,
            App.TaskType.BINARY_CLASSIFICATION
        )
    }

    root = tk.Tk()
    App(dbs, algs, root)
    tk.mainloop()
