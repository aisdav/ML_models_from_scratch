# Gradient boosting
Nowadays, gradient boosting is one of the main production solutions for tabular and heterogeneous data because of its strong performance and high accuracy.
Gradient boosting machine adds basic models to ensemble consecutively, however, instead of reweighting observations based on previous errors, gradient boosting fits each new model to the errors made by the current ensemble. In this case each new model is trained on the negative gradient of the loss function
and in the regression case with squared error this corresponds to residuals made by previous model.

## Principle of work of gradient boosting for regression
The algorithm is structured as follows:
  1) The initial prediction is the arithmetic mean of all values in y_train
  2) The residuals of the current model are computed based on antigradient of loss function
  3) A regression tree is trained on X_train using the computed residuals as targets, then it makes predictions on X_train
  4) Received result adds to initial and steps 2-4 repeats for each tree. The new tree prediction is usually multiplied by a learning rate before being added to the current model.
  5) After training all models, creates initial prediction from 1 step
  6) Next predicts x_test on trained trees and adds to initial
  7) The resulting sum is the final prediction.

## Formulas for calculation
![Loss function](https://latex.codecogs.com/svg.image?\color{white}L%28y_i%2CF%28x_i%29%29%3D%5Cfrac%7B1%7D%7B2%7D%28y_i-F%28x_i%29%29%5E2)

![Residuals](https://latex.codecogs.com/svg.image?\color{white}r_%7Bik%7D%20=%20-%20%5Cleft%5B%20%5Cfrac%7B%5Cpartial%20L(y_i,%20F(x_i))%7D%7B%5Cpartial%20F(x_i)%7D%20%5Cright%5D_%7BF(x)=F_%7Bm-1%7D(x)%7D%20=%20y_i%20-%20F_%7Bm-1%7D(x_i))

## Principle of work of gradient boosting for classification
In this case things are a little complicated, For multiclass classification, the target is usually represented using one-hot encoding and transform to probability by softmax function, and For each leaf, a coefficient 
𝛾 is computed to determine how much this leaf contributes to the class score.
The algorithm is structured as follows:
  1) For y_train one-hot-encoding is used and firstthe initial scores are set to 0 for each class
  2) Predictions are transformed to probabilities using softmax
  3) Residuals are computed from the negative gradient of the loss function using the current class probabilities
  4) Regression tree trains on x_train and remains, then predicts x_train
  5) For each leaf, a coefficient is calculated y_gamma based on remains taken from the positions of observations, which fell into a particular leaf node
  6) Gained result and sum of coefficients y_gamma adds to initial
  7) steps 2-6 repeats for each tree in every class
  8) After all trees are trained, the initial prediction from step 1 is recreated
  9) Next, predictions are made for X_test on trained trees per class and added to the initial ones;
  10) The class with the highest final score (or highest probability) is taken as the final prediction.

## Formulas for calculation
![Loss function](https://latex.codecogs.com/svg.image?\color{white}L(%5C%7By_k,%20F_k(x)%5C%7D_%7B1%7D%5E%7BK%7D)%20=%20-%20%5Csum_%7Bk=1%7D%5E%7BK%7Dy_k%20%5Clog%20p_k(x))

![Probability of each class](https://latex.codecogs.com/svg.image?\color{white}p_k(x)%20=%20\mathrm{softmax}(F_k(x))%20=%20%5Cfrac%7Be%5E%7BF_k(x)%7D%7D%7B%5Csum_%7Bl=1%7D%5E%7BK%7De%5E%7BF_l(x)%7D%7D)

![Residuals](https://latex.codecogs.com/svg.image?\color{white}r_%7Bik%7D%20=%20-%20%5Cleft%5B%20%5Cfrac%7B%5Cpartial%20L(%5C%7By_%7Bil%7D,%20F_l(x_i)%5C%7D_%7Bl=1%7D%5E%7BK%7D)%7D%7B%5Cpartial%20F_k(x_i)%7D%20%5Cright%5D_%7B%5C%7BF(x)=F_%7Bm-1%7D(x)%5C%7D_%7B1%7D%5E%7BK%7D%7D%20=%20y_%7Bik%7D%20-%20p_%7Bk,m-1%7D(x_i))

![Gamma update](https://latex.codecogs.com/svg.image?\color{white}\gamma_%7Bjkm%7D%20=%20%5Cfrac%7BK-1%7D%7BK%7D%20%5Ccdot%20%5Cfrac%7B%5Csum_%7Bx_i%20%5Cin%20R_%7Bjkm%7D%7Dr_%7Bik%7D%7D%7B%5Csum_%7Bx_i%20%5Cin%20R_%7Bjkm%7D%7D%7Cr_%7Bik%7D%7C(1-%7Cr_%7Bik%7D%7C)%7D)

## Some definitions 
Softmax function that turns some numbers into probabilities
