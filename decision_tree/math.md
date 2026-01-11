# Decision Tree (CART)

A decision tree is a model for classification and regression based on a binary tree structure. It is a fundamental component of gradient boosting and random forest algorithms and one of the most powerful tools in machine learning.

## Structure of a Decision Tree

A decision tree consists of a root node, branches (left and right), decision nodes, and terminal (leaf) nodes. Root and decision nodes represent questions with a threshold used to split the training data into parts. Leaf nodes represent the final prediction: the mean value of samples in the leaf for regression and the statistical mode (most frequent class) for classification.

Each terminal node corresponds to a rectangular region in the feature space on the decision boundary plot. If several adjacent regions produce the same prediction, they are automatically merged.

## Choosing the Best Partition

Choosing the best partition when creating a decision node resembles a game where you can ask only Yes/No questions to guess a celebrity. It is logical to choose questions that eliminate the largest number of incorrect answers.

For example, a question about sex may eliminate about 50 percent of candidates (assuming a balanced distribution), while a question about age is usually less informative.

A measure that indicates how well a question at a node separates correct answers from incorrect ones is called a node impurity (contamination) measure.

In the case of classification, the following criteria are used to assess the quality of node partitioning.

Gini impurity measures the diversity of the class probability distribution in a node. If all samples in a node belong to the same class, the Gini impurity is 0. For a binary classification problem with a uniform class distribution, the Gini impurity reaches its maximum value of 0.5.

![Gini impurity](https://latex.codecogs.com/svg.image?\color{white}G%20=%201%20-%20\sum_{i=1}^{K}p_i^2)

Shannon entropy measures the uncertainty or disorder of classes in a node. It represents the amount of information required to describe the system state: the higher the entropy, the higher the disorder.

![Shannon entropy](https://latex.codecogs.com/svg.image?\color{white}H%20=%20-%20\sum_{i=1}^{K}p_i%20\log%20p_i)

Here, p_i is the proportion of samples belonging to class i in the node.

In the case of regression, the mean squared error (MSE) is most commonly used to evaluate the quality of node partitioning, but Friedman MSE and MAE can also be applied.

## Loss Function

After choosing a quality criterion, all unique feature values are sorted, and candidate thresholds are computed as the midpoints between adjacent values.

The training data is then split into two subsets. Samples with feature values less than or equal to the threshold go to the left subset, and samples with feature values greater than the threshold go to the right subset.

For each subset, impurity is calculated using the selected criterion. The weighted sum of impurities represents the loss function associated with that threshold. The threshold that produces the minimum loss value on the training set (or current node subset) is selected as the best split.

Regression loss:

![CART regression loss](https://latex.codecogs.com/svg.image?\color{white}L_{reg}%20=%20\frac{1}{m}\sum_{i=1}^{m}(y_i-\hat{y})^2)

Classification impurity (Gini):

![CART classification loss Gini](https://latex.codecogs.com/svg.image?\color{white}L_{cls}%20=%201-%5Csum_{k=1}^{K}p_k^2)

Classification impurity (Entropy):

![CART classification loss entropy](https://latex.codecogs.com/svg.image?\color{white}L_{cls}%20=%20-%5Csum_{k=1}^{K}p_k%20\log%20p_k)

## Principle of Work

1. Create the root node using the best threshold.
2. Split the training data into two subsets: smaller values go to the left, and larger values go to the right.
3. Recursively repeat this process for all subsets until one of the stopping criteria is met: maximum depth, maximum number of leaf nodes, minimum number of samples in a node, or no further impurity reduction.

## Regularization of Decision Trees

There are two main regularization approaches: pre-pruning, which limits tree growth during construction, and post-pruning, which simplifies the tree after it has been fully built. Post-pruning is generally more effective because it allows the full tree structure to form and then removes unnecessary branches more precisely.

Types of Post-Pruning

Top-down pruning starts from the root node. This method is computationally cheap but may lead to underfitting, as potentially informative branches can be removed too early.

Bottom-up pruning starts from the leaf nodes and proceeds upward. This approach is more precise but requires more computational resources.

Minimal Cost-Complexity Pruning

In scikit-learn, a variant of cost-complexity pruning is used. First, a fully grown tree is built without any constraints. Then the loss is computed: weighted impurity for classification or weighted MSE for regression.

For each subtree, the loss of all its leaves is calculated. For each subtree, the complexity parameter alpha is computed as the increase in loss when the subtree is replaced by a single leaf. The subtree with the smallest alpha is pruned and replaced with a leaf node. The corresponding alpha value is stored in the cost_complexity_pruning_path array. This procedure is repeated recursively until pruning reaches the root node.
