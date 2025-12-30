# Logistic Regression

## Model definition

Logistic Regression is a linear binary classifier that estimates the probability of class membership using the sigmoid function.

Given input features X with d dimensions, the model computes:

![Linear combination](https://latex.codecogs.com/svg.image?\color{white}z%20%3D%20w%5ET%20x%20%2B%20b)

![Sigmoid function](https://latex.codecogs.com/svg.image?\color{white}\hat{y}%20%3D%20\text{sigmoid}(z)%20%3D%20\frac{1}{1%20%2B%20e^{-z}})


where:
- w — weight vector (d,)
- b — bias (scalar)
- sigmoid(z) maps real values to (0, 1)

---

## Probabilistic interpretation

The model predicts probabilities:

![Probability y=1](https://latex.codecogs.com/svg.image?\color{white}P(y%20%3D%201%20%7C%20x)%20%3D%20\hat{y})

![Probability y=0](https://latex.codecogs.com/svg.image?\color{white}P(y%20%3D%200%20%7C%20x)%20%3D%201%20-%20\hat{y})

Each prediction represents model confidence.

---

## Maximum Likelihood Estimation (MLE)

Assuming independent samples, the likelihood of observing the dataset is:

L(w, b) = product over i of:
    ![Likelihood term](https://latex.codecogs.com/svg.image?\color{white}\hat{y}_i^{y_i}%20\cdot%20(1%20-%20\hat{y}_i)^{(1-y_i)})

To avoid numerical underflow and simplify optimization, we maximize the log-likelihood:

log L(w, b) = sum over i of:
    ![Log-likelihood term](https://latex.codecogs.com/svg.image?\color{white}y_i%20\cdot%20\log(\hat{y}_i)%20+%20(1-y_i)%20\cdot%20\log(1-\hat{y}_i))

---

## Loss function (Binary Cross-Entropy)

Minimizing the negative log-likelihood leads to the logistic loss:

J(w, b) = -(1 / n) * sum over i of:
    ![Log-likelihood](https://latex.codecogs.com/svg.image?\color{white}y_i%20\cdot%20\log(\hat{y}_i)%20+%20(1-y_i)%20\cdot%20\log(1-\hat{y}_i))


This loss heavily penalizes confident but incorrect predictions.

---

## Optimization

Logistic Regression has no closed-form solution.

Model parameters are optimized using gradient descent:

w = w - learning_rate * dw  
b = b - learning_rate * db

---

## Gradients

The gradients of the loss function are:

dw = (1 / n) * X^T (y_hat - y)  
db = (1 / n) * sum(y_hat - y)

These gradients are used to iteratively update the model parameters.

---
## Answers to my questions
Why do we use this particular sigmoid function? Its the most suitable function because:
1) it converts linear combination to probability
2) Compatible with Maximum Likelihood Estimation
3) Gives a convex loss function

What is the convexity of a function?
Function is convex when it has 1 the lowest point,if we go down, we definitely reach minimum, there is false(local) minimums
## Summary

- Logistic Regression is a linear model with a probabilistic output
- Training is based on Maximum Likelihood Estimation
- Binary Cross-Entropy loss comes directly from MLE
- Optimization is performed using gradient descent
