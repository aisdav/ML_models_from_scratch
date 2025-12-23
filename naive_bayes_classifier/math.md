# Naive Bayes Classifier

## Naive Bayes Classifier 
It is probabilistic classifier based on Bayes formula with a strict assumption that all the features are independent among themselfes. So we work with one-dimensional probability densities, 
instead of many-dimensional when features depend on each other. Bayes formula P(A|B) =P(B|A)*P(A)/P(B) where
P(A|B) - aposterior probability of event A on condition event B is happened
P(B|A) - Conditional probability of B on condition event A is happened
P(A),P(B) - aprior probabilities of A and B
In context of Machine learning it looks:
P(yk|X) = P(yk)P(X|yk)/P(X)
where

P(yk|X) - aposterior probability of the sample belongs to the yk class taking into account its features X;
P(X|yk) - likelihood, that is, the probability of features X for a given class yk;
P(yk) is the a priori probability that a randomly selected observation belongs to the yk class.;
P(X) is the a priori probability of features of X.

In common view it looks:
P(yk|X1,X2,X3,X4.....) = P(yk)Product(from i=1 to n)P(Xi|yk)/P(X1,X2,X3,X4.....)
the denominator always same and its doesnt depends on class, so final formula looks:
yk = argmax P(yk) Product(from i=1 to n)P(Xi|yk)
There is also a number of versions of realisation differing assumptions about the distribution of features for a given class. They are Gaussian Naive Bayes, Multinomial Naive Bayes, Complement Naive Bayes, Bernoulli Naive Classifier, Categorical
