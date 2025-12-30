# Naive Bayes Classifier

## Description

Naive Bayes is a probabilistic classifier based on Bayes' theorem with a strong assumption that all features are conditionally independent given the class.  
Due to this assumption, we work with one-dimensional probability density functions instead of a full joint multidimensional distribution.

Bayes' theorem is defined as:

P(A | B) = P(B | A) · P(A) / P(B)

where:
- P(A | B) is the **posterior probability** of event A given that event B has occurred
- P(B | A) is the **likelihood**
- P(A) and P(B) are **prior probabilities**

---

## Bayes Theorem in Machine Learning

In the context of machine learning, Bayes' theorem is written as:

P(yₖ | X) = P(yₖ) · P(X | yₖ) / P(X)

where:
- P(yₖ | X) is the **posterior probability** that a sample with features X belongs to class yₖ
- P(X | yₖ) is the **likelihood**, i.e. the probability of observing features X given class yₖ
- P(yₖ) is the **prior probability** of class yₖ
- P(X) is the marginal probability of observing features X

---

## Naive Independence Assumption

Naive Bayes assumes that all features are conditionally independent given the class:

P(X | yₖ) = ∏ᵢ P(xᵢ | yₖ)

Therefore, the posterior probability can be written as:

P(yₖ | X) ∝ P(yₖ) · ∏ᵢ P(xᵢ | yₖ)

The denominator P(X) is the same for all classes and does not affect class comparison, so it is omitted.

The final prediction rule is:

ŷ = argmaxₖ P(yₖ) · ∏ᵢ P(xᵢ | yₖ)

---

## Types of Naive Bayes Classifiers

Different variants of Naive Bayes exist depending on the assumed distribution of features:

- Gaussian Naive Bayes
- Multinomial Naive Bayes
- Bernoulli Naive Bayes
- Complement Naive Bayes
- Categorical Naive Bayes

In this project, we focus on **Gaussian Naive Bayes**, which assumes that each feature follows a normal distribution.

---

## Principle of Work (Gaussian Naive Bayes)

1. Compute the **prior probabilities** of each class.
2. For each class and each feature, compute the **mean** and **standard deviation**.
3. For a test sample, compute the **probability density** of each feature using the Gaussian distribution.
4. Compute the **posterior probability** for each class as the product of the prior probability and feature likelihoods.
5. Select the class with the **maximum posterior probability** as the final prediction.

## Some definitions
**Prior probability:** The probability of an event before observing any new data.  

**Posterior probability:** The probability of an event updated based on new evidence or data.

**Likelihood:** A function of the parameters of a statistical model given observed data, measuring how probable the observed data is for different parameter values.
