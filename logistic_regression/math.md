# Logistic Regression

## Model definition

Logistic Regression is a linear binary classifier that models the probability of class membership using the sigmoid function.

Given input features \( x \in \mathbb{R}^d \), the model computes:

\[
z = w^T x + b
\]

\[
\hat{y} = \sigma(z) = \frac{1}{1 + e^{-z}}
\]

where:
- \( w \) — weight vector
- \( b \) — bias
- \( \sigma(\cdot) \) — sigmoid function

---

## Probabilistic interpretation

The model estimates probabilities:

\[
P(y=1 \mid x) = \hat{y}
\]

\[
P(y=0 \mid x) = 1 - \hat{y}
\]

---

## Maximum Likelihood Estimation (MLE)

Assuming samples are independent, the likelihood function is:

\[
L(w, b) = \prod_{i=1}^{n} \hat{y}_i^{y_i}(1 - \hat{y}_i)^{1 - y_i}
\]

To simplify optimization, we maximize the log-likelihood:

\[
\log L(w, b) = \sum_{i=1}^{n} \left[
y_i \log \hat{y}_i + (1 - y_i) \log (1 - \hat{y}_i)
\right]
\]

---

## Loss function (Binary Cross-Entropy)

Minimizing the negative log-likelihood leads to the logistic loss:

\[
J(w, b) = -\frac{1}{n} \sum_{i=1}^{n}
\left[
y_i \log \hat{y}_i + (1 - y_i) \log (1 - \hat{y}_i)
\right]
\]

This loss penalizes confident but incorrect predictions.

---

## Optimization

There is no closed-form solution for logistic regression, so parameters are optimized using gradient descent.

---

## Gradients

Gradient of the loss with respect to the parameters:

\[
\frac{\partial J}{\partial w} = \frac{1}{n} X^T(\hat{y} - y)
\]

\[
\frac{\partial J}{\partial b} = \frac{1}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i)
\]

These gradients are used to update parameters during training.
