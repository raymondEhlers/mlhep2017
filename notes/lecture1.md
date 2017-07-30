# Lecture 1 - Introduction

## 5

- Feature space = parameter space
    - Also described as feature vector
- Target space = desired output (ie variable, whatever)

## 6

Conditions for machine learning:
- No prior knowledge or theory of the problem
- Too complex for manual examination
- Sufficiently large dataset for training

ML = Inference of statistical dependency

## 11

- ML = Hypothesis h, which takes h: X->Y
    - ML proposes the hypothesis, and then we test it to select viability
    - Checked by minimizing some error function

## 12

ML pipeline image

## 14

- Select some linear function $y_{i}=w_{1}x_{i} + w_0 + \epsilon_{i}$
    - $\epsilon_{i}$ is some noise

## 16

Evaluation metrics for regression.

## 17

Optimization solution in the mean square sense is on the slide

## 18

- Multivariate linear regression leads to some matrix expression
    - Multiple regressors (explanatory variables, independent variables, etc, are all equivalent)
- Noise is just the deviation
    - we can never fit the equality without noise

$$ y = xw$$

## 19

MSE = Mean square error

## 20

Analytic solutions are useful, but often fine to just solve numerically
-> Gradient descent

## 21

Full explanation of gradient descent is on this slide
- Note that gradient points __away__ from the minimum
- For k steps, will be within 1/k of optimal solution
    - Gives some assurance of convergence

## 22

For a sufficiently large dataset, gradient descent is too inefficient
- Can use stochastic gradient descent
    - Where we take the gradient descent for some subsample that was selected randomly
    - Converges to $1/\sqrt{k}$
- There are also additional optimizations available

## 24

$\alpha{k}$ is the step size

## 26

- Linear regression can model, but it can also classify
- Classifier problem: select a plausible hypothesis (classifier) h:X->Y such that $y\element{-1,1}$

## 27

- For example, take the sign of the linear model (to get {-1,1})
    - Then optimize by optimizing the error of the classification
- Cannot use gradient descent here (not differentiable??)
    - Reason is unclear
- Instead establish an upper bound function and then optimize that

For upper bound function, -1 if opposite sign (failed to properly), +1 if the same sign (success)

## 28

- L = Logistic loss
- Can have multiple functions to bound the "indicator function"

## 29

Logistic regression <- Sigmoid

## 30

To get {-1, 1} -> sigmoid

## 31

- Useful example image
    - Colors are the sigmoid

## Figures of Merit

### 33

- Accuracy is one option
- Not very useful for imbalanced data
    - If 5% positive, then just predicting negative, we get 95% accuracy

### 340 - Confusion matrix

- Can mitigate some with confusion matrix

### 35 - Operating Curves:

Receiver Operating Curves (ROC):

- Compares false positive rate (FPR) and true positive rate (TPR)
    - Depends on choice of threshold t in a(x) = [h(x) > t]
        - h(x) is the hypothesis and is often more useful
- Ideal classifier has area under the curve = 1

## Overfitting

###  40 - How to detect overfitting

- Split training data into some subsets
- Train different models on all subsets but one (ie h_{i} on all samples but the subsample X_{i})
    - Access quality with cross validation
    - Example shown is called k-fold cross validation

## Regularization

- Continued briefly after lunch.
- Began with a jupyter notebook demonstration.

Pesuedoinverse: Invert a matrix that is not invertible by adding some small addition to make it invertible.

### 43

- Replace fit with fit + penalty
- Q = function to minimize $(y-Xw)^{2}$

### 45

- L2 = Regularize multivariate linear regression. Penalty function is the norm squalred
- L1 = regularized regression (LASSO). Penalty function is just the norm.
- Can also combine L1 and L2. Called ElasticNet

### 46

Image intuition

- L1 always has some weights set to 0 because it will end up on an axis (?)
- L2 shares weights

### 47

- Regularizers impose constraints
