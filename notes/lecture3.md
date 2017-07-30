# Lecture 3 - Ensembling: Adaboost, Stacking, Gradient Boasting

## Boosting

*Follow up here!*

Re-training trained decision tree emphasizing what the previous tree got wrong.
- This combines multiple weak learners into a strong learner

## Reweighting and Adaboost

### 7

- Adaptive Boosting for classification = Adaboost
- Explained on this slide.
- Re-weight based on fit quality metric

### (Slide before gradient boosting)

Nice animation. Learner is on the left and output is on the right

Boosting selects the next learners with some criteria to help improve, while bagging just averages.

## Gradient Boosting

NOTE: Cannot read slide numbers, so I will just note relative slides from sections

### GB + 1

- With some set of models already trained, how do we select how to add another one?
- Try to optimize the loss function via gradient descent in $\alpha_{N}$_

### GB + 2

Gradient boosting machine:
- Basically all of the ingredients needed to do gradient boosting, and then how to do it
- NOTE: Depends on hyperparameters like tree depth

### GB + 3

How to do each iteration

### GB + ?

Provides very nice demos

### GBM: Regularization via shrinkage

NOTE: Weak learners = learner which cannot cover the whole problem, but only a part of it (?)

Weak simple learners: GB works okay
Complex weak learners: Even a few steps of GB may over fit

- Use shrinkage to handle: Slow down the learning rate
    - This is equivalent to taking a smaller step, right?

### Regularization approaches

Train test gap = Gap that represents overfitting

### Stochastic Gradient Descent for GBM

Train each new learner only on a subset of the overall data

## XGBoost

### Extreme Gradient Boosting

Just a careful tuning of normal gradient boosting

Conditions:
1. Use second order derivatives in addition to the first order
2. Controls leaf counts
    - Penalizes large leaf counts, which would indicate overfitting
    - Also doesn't allow large coefficient norm inside a leaf
3. Modifies splitting condition
4. Stops if splitting at node causes negative change in loss (?)

## Conclusions

Very nice summary

# Non-trivial applications of boosting

## HEP Re-weighting

### MC vs Data

- There is disagreement, so the learners are biased.
    - We want to re-weight MC to match data

### Typical approach: example

Nice image showing what is desired

### Holdouts

- Holdout is in the tail where there are few entries.
- Re-weighting should be checked there, since it will clearly show the problem.

- Many bins can also be problematic, because the values may not be stable

### Re-weighting quality

- 1D test on multivariate distribution is not enough to assure quality
    - This means that 1D projections are not sufficient.
        - They could match in projections, but the multivariate dist could still be different

### Comparing using ML

- Attempt to distinguish between MC and data via ML.
    - If you cannot distinguish then they match

### ML to do re-weighting

Train decision tree to do re-weighting:

- Symmetrized Chi2 used as the splitting criteria
- A bin can be considered a leaf

There are a number of different options, which are listed in the slides

- They are available in the hep\_ml library (made by some people at Yandex School of Data Analysis)
