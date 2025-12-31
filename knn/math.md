# K nearest neighbours (KNN)

It is an algorithm of classification and regression based on the compactness hypothesis. It states that objects located close to each other in the feature space have close target values or belong to the same class.

## Principle of work

1) Calculate distance between test and training samples  
2) Then choose k nearest samples, where k is predefined  
3) Final prediction among chosen k nearest samples is the mode when it is classification, and the arithmetic mean when it is regression  
4) Previous steps are repeated for all test samples  

## Metrics

There are a lot of metrics for calculating distance between objects. We use Euclidean distance.  
The Euclidean distance is the simplest and most commonly accepted metric, which is defined as the length of the segment between two objects *a* and *b* in a space with *n* features and is calculated using the formula:

![Euclidean distance](https://latex.codecogs.com/svg.image?\color{white}d(x,y)%20=%20\sqrt{\sum_{i=1}^{n}(x_i-y_i)^2})

Alternatives: Manhattan distance, cosine distance.

## Optimization method

The method described above is called Brute Force type KNN. It needs huge computing resources, but is simple in realisation. Therefore, there are methods like BallTree and KD-Tree based on tree structures.

BallTree is a tree-like structure that assumes dividing the initial space into embedded hyperspheres.

### Realisation of BallTree

1) In the set of points, one point is chosen and for it the furthest point is found  
2) Next, all points are divided into two hyperspheres (nodes) according to the closest location to the two points from step 1  
3) This process is then repeated recursively for each hypersphere until a certain number of points remain in it or the specified depth of the tree is reached  

When searching for the k-nearest neighbors for a new point, the algorithm compares the distance from a given point to the center of each child node and leaves only those where this distance is less than the radius of the node.

Information about intersecting hyperspheres (nodes) can be useful for evaluating the quality of the resulting tree structure and its further optimization. For two hyperspheres A and B in the metric M, the distance between them can be determined as follows:

![Distance between balls](https://latex.codecogs.com/svg.image?\color{white}d_M(A,B)%20=%20\max(0,%20d_M(c_A,c_B)%20-%20r_A%20-%20r_B))

where \( c_A \) and \( c_B \) are the centers of the spheres, and \( r_A \) and \( r_B \) are their radii.

## Realisation of KD-Tree

The KD-Tree (k-dimensional tree) is another tree structure that resembles BallTree, but in this case hyperplanes are used to split points instead of hyperspheres.

Building a KD-Tree consists of the following steps:

1) One of the coordinates is selected from the set of points (usually one at a time for each level of the tree, but it is also possible randomly) and the median is calculated  
2) Next, all points are divided into two nodes (subsets) with respect to the median: those with the value of the selected coordinate less than or equal to the median, and those with more  
3) This process is repeated recursively for each node until a certain number of points remain in it or the specified depth of the tree is reached  

When searching for the nearest neighbors for a new point, the algorithm compares the value of a given point with the median at each node, thus choosing the nearest subspace, which will be a leaf with the nearest neighbors. Going back to the root, the algorithm compares points in other nodes and updates the nearest neighbors if they are closer to the specified point.

The general pruning idea from BallTree is also applicable to KD-Tree, but it is based on distances to splitting hyperplanes rather than hyperspheres.
