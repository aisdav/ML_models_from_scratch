# Random Forest

Random Forest is one of the earliest and still widely used ensemble learning methods. It is based on constructing a large number of decision trees trained on bootstrap samples and aggregating their predictions.

---

## Classifiers and Regressors with Voting

Suppose we have several base classifiers trained on the same dataset. A simple way to build a stronger classifier is to aggregate their predictions by majority voting. This approach is known as a **hard voting classifier**, where the final prediction corresponds to the class receiving the most votes.

If the base classifiers are able to estimate class probabilities (i.e., implement the `predict_proba` method), a **soft voting classifier** can be used. In this case, class probabilities predicted by individual models are averaged (optionally weighted), and the class with the highest averaged probability is selected.

For regression tasks, predictions of base models are averaged, forming a **voting regressor**.

In scikit-learn, these methods are implemented as `VotingClassifier` and `VotingRegressor`. While voting ensembles often outperform individual models, they may still be prone to overfitting, since all base models are trained on the same dataset and are sensitive to outliers.

---

## Bagging and Pasting

A more advanced ensemble technique involves training the same base model on different random subsets of the training data.

- If sampling is performed **with replacement**, the method is called **bagging** (bootstrap aggregating).
- If sampling is performed **without replacement**, the method is called **pasting**.

A **bootstrap sample** is a sample drawn with replacement from the original dataset.

After training multiple base models on bootstrap samples, the ensemble prediction is obtained by aggregating their outputs: majority voting for classification and arithmetic mean for regression. Bagging primarily reduces model variance and is especially effective for high-variance models such as decision trees.

Bagging models can be trained in parallel, which makes the method scalable. In scikit-learn, parallelization is implemented via `joblib`, where the `n_jobs` parameter specifies the number of CPU cores used.

In practice, bagging usually performs better than pasting because bootstrap sampling produces more diverse training subsets.

---

## Evaluation on Unused Samples (Out-of-Bag)

When bagging with replacement (`bootstrap=True`) is used, each bootstrap sample contains, on average, about **63% unique training samples**. The remaining **37% of samples are not selected at all** and are called **out-of-bag (OOB)** samples.

This result follows from the limit:

![Limit to 1/e](https://latex.codecogs.com/svg.image?\color{white}\lim_{m%20\to%20\infty}(1-\frac{1}{m})^{m}%20=%20\frac{1}{e}%20\approx%200.37)

OOB samples can be used as a validation set to estimate the generalization performance of the ensemble without requiring a separate test set.

---

## Random Subspaces and Random Patches Methods

- The **random subspaces method** uses all training samples but selects a random subset of features for each base model.
- The **random patches method** selects both a random subset of training samples and a random subset of features.

Feature sampling increases the diversity of base models, slightly increasing bias while reducing variance, which often improves overall ensemble performance, especially for high-dimensional data.

---

## Random Forest and Extremely Randomized Trees

Random Forest is a specialized and optimized form of bagging based on decision trees. Its algorithm can be summarized as follows:

1. For each tree, a bootstrap sample of training objects is generated.
2. Each tree is trained independently and in parallel.
3. At each split of a tree, only a random subset of features is considered.
4. Predictions of all trees are aggregated by majority voting (classification) or averaging (regression).

For regression:

![Random Forest prediction](https://latex.codecogs.com/svg.image?\color{white}\hat{f}_{rf}^{B}(x)%20=%20\frac{1}{B}%20%5Csum_{b=1}^{B}T_b(x))

For classification:

![Random Forest classification](https://latex.codecogs.com/svg.image?\color{white}\hat{C}_{rf}^{B}(x)%20=%20\text{majority%20vote}[%5C{\hat{C}_b(x)%5C}_{b=1}^{B}])

Where:

- T_b(x) is the prediction of the b-th tree for regression,
- hat{C}_b(x) is the predicted class of the \(b\)-th tree,
- B is the number of trees.

Typical choices for the number of features considered at each split:

- Regression (classical recommendation): m = p/3 
- Regression (scikit-learn default): m = p  
- Classification: m =sqrt(p)

where p is the total number of features.

Trees can be made even more random by selecting split thresholds at random instead of optimizing them. This approach is known as the **Extremely Randomized Trees (ExtraTrees)** ensemble.

---

## Feature Importance

Random Forest also provides an estimate of feature importance, which is based on the average reduction of impurity (e.g., Gini or MSE) contributed by each feature across all trees.
