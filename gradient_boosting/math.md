# Gradient boosting
Nowadays gradient boosting is one of the main production-solutions for working with tabular data and inhomonegeous, because it has productivity and precision. Especially some of its modification, which will be mentioned, but wont be considered.
Gradient boosting machine adds basic models to ensemble consecutively, however instead of training model on loss weights of previous, in this case models trains on residual errors made by previous model.

## Principle of work of gradient boosting for regression
The algorithm is structured as follows:
  1) First prediction considers as arithmetic mean of all y_train
  2) Calculates remain of model based on antigradient of loss function
  3) Regression tree trains on x_train, then predicts x_train
  4) Received result adds to initial and steps 2-4 repeats for each tree
  5) After training all models, creates initial prediction from 1 step
  6) Next predicts x_test on trained trees and adds to initial
  7) Gained sum will be final prediction

## Formulas for calculation
![Loss function](https://latex.codecogs.com/svg.image?\color{white}%5Chat%7BC%7D_%7Brf%7D%5E%7BB%7D(x)%20=%20%5Cmathrm%7Bmajority%20vote%7D%5B%5Chat%7BC%7D_b(x)%5D_%7Bb=1%7D%5E%7BB%7D)

![Residuals](https://latex.codecogs.com/svg.image?\color{white}r_%7Bik%7D%20=%20-%20%5Cleft%5B%20%5Cfrac%7B%5Cpartial%20L(y_i,%20F(x_i))%7D%7B%5Cpartial%20F(x_i)%7D%20%5Cright%5D_%7BF(x)=F_%7Bm-1%7D(x)%7D%20=%20y_i%20-%20F_%7Bm-1%7D(x_i))

## Principle of work of gradient boosting for classification
In this case things are a little complicated, we must use apply one-hot-encoding for every class and transform to probability by softmax function, and adding coefficient y_gamma to trees's prediction, that regulate contribution of each trees

The algorithm is structured as follows:
  1) For y_train one-hot-encoding is used and first prediction are assigned to 0 for each class
  2) Predictions are transformed to probabilities using softmax
  3) Calculates remains on antigradient loss function and probabilities
  4) Regression tree trains on x_train and remains, then predicts x_train
  5) For every leaf calculates cofficient y_gamma based on remains taken from the positions of observations, which fell into a particular leaf node;
  6) Gained result and sum of coefficients y_gamma adds to initial
  7) steps 2-6 repeats for each tree in every class
  8) After training all tree creates initial prediction from step 1
  9) Next, predictions are made for X_test on trained trees per class and added to the initial ones;
  10) The classes with the maximum amount will be the final prediction.

## Formulas for calculation
![Loss function](https://latex.codecogs.com/svg.image?\color{white}L(%5C%7By_k,%20F_k(x)%5C%7D_%7B1%7D%5E%7BK%7D)%20=%20-%20%5Csum_%7Bk=1%7D%5E%7BK%7Dy_k%20%5Clog%20p_k(x))

![Probability of each class](https://latex.codecogs.com/svg.image?\color{white}p_k(x)%20=%20\mathrm{softmax}(F_k(x))%20=%20%5Cfrac%7Be%5E%7BF_k(x)%7D%7D%7B%5Csum_%7Bl=1%7D%5E%7BK%7De%5E%7BF_l(x)%7D%7D)

![Residuals](https://latex.codecogs.com/svg.image?\color{white}r_%7Bik%7D%20=%20-%20%5Cleft%5B%20%5Cfrac%7B%5Cpartial%20L(%5C%7By_%7Bil%7D,%20F_l(x_i)%5C%7D_%7Bl=1%7D%5E%7BK%7D)%7D%7B%5Cpartial%20F_k(x_i)%7D%20%5Cright%5D_%7B%5C%7BF(x)=F_%7Bm-1%7D(x)%5C%7D_%7B1%7D%5E%7BK%7D%7D%20=%20y_%7Bik%7D%20-%20p_%7Bk,m-1%7D(x_i))

![Gamma update](https://latex.codecogs.com/svg.image?\color{white}\gamma_%7Bjkm%7D%20=%20%5Cfrac%7BK-1%7D%7BK%7D%20%5Ccdot%20%5Cfrac%7B%5Csum_%7Bx_i%20%5Cin%20R_%7Bjkm%7D%7Dr_%7Bik%7D%7D%7B%5Csum_%7Bx_i%20%5Cin%20R_%7Bjkm%7D%7D%7Cr_%7Bik%7D%7C(1-%7Cr_%7Bik%7D%7C)%7D)
