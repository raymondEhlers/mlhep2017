# Lecture 2- Decision Trees, Bagging, Ensembles, Random Forest, Stacked Generalization

## Decision Trees

### 5

- Composed of Internal nodes and leaves
    - Internal nodes have weights, $\Beta$: X->{0,1}
    - Leaf predicts

### 6

Greedy Tree:
- Split into subspaces
- Basically force into classifying into one or another category depending on a threshold value and the output

### 9

Overfitting is extremely easy

### 10

Decision tree design choices:
- Type of predicate in internal node
- Loss function
- Stopping criterion
- How to stop overfitting

- CART, C4.5, ID3 all specify design choices

### 11

- H(R) = Impurity criterion
    - We want to minimize this quantity

### 13

Summary of loss functions (?)
- Information for both regression and classification

### 15

Tree size = tree depth (in this case)

Data points:
- Yellow = test data
- Purple = training data

Overfitting shows when the two diverge

## Bagging and Random Forests

### 17

Bootstrapping procedure:

- Generate new samples by randomly selected from other samples
    - Note that selecting the same is allowed (ie. Selecting the same entry is possibled)

Bagging:
- Apply bootstraping to multiple models and average them
- It is a meta-learning technique
- (I think of it as putting the models into a bag)

---------
DEMO: Bagging neural nets

---------

### 19

- Random Forest: Bagging over decision trees
    - Reduces error via averaging over instances and features

Algorithm explained here.

## Learning Theory

### 24

Bias-variance decomposition is explained here

- Noise:
    -Error from ideal learner
- Bias:
    -Only an approximation of the ideal
        - More complex could lower the bias
- Variance
    - Can represent training data well, but can handle validation data less effectively
        - How does this relate to overfitting? Is this just a different way to look at it?

### 28 and 29

- Bias will not get worse due to bagging
- Variance can get better with bagging

## Stacking

- Another meta-learning technique

### 34

Description of algorithm is on this slide.

- Divide up test and train data, make predictions on some subset, and then use that model to predict on the part that was left out.
    - Save the result
    - Then use saved predictions to train on
- Basically a way to combine the models together

Those whole description seems to be derived from [here](http://blog.kaggle.com/2016/12/27/a-kagglers-guide-to-model-stacking-in-practice/), which seems to have more detail.
