# Neural Networks 101

- It is "super overhyped", so we'll sort through what is actually useful
- "About 50% science and 50% art of getting what you want out of the neural net"

logistic loss = cross entropy = etc... (Lots of names)

### 4

- In the case of nonlinear dependencies, instead of linear model, use a polynomial, for example

### 5

- Techniques we've learned so far needed manual feature extraction

### 7

- Can we extract the features with some machine learning model?
    - What makes a good feature?

### 10

- Expression at the bottom is a formulation of extracting the features in a linear manner and then applying a classifier

### 11
- we note that it's just a different linear combination
    - The features **must** be non-linear

### 13

- Use sigmoid to extract non-linear features

### 16

- Additional options for non-linear functions

### 17

- Cannot initialize to 0, because after applying sigmoid, they all end up at sigmond(0)=1/2
    - All features will learn the same things!
        - All symmetric, so gradient descent doesn't help here!

### 18

- Need to break symmetry.
    - For example, initialize with random weights

### 20

He is pretty down on the biological inspiration of NNs
    -  NNs don't have a strong connection anymore
            - For example, no time dependence

### 21

- Backprop - a fancy word for "chain rule"
    - Still use gradient descent, just add on chain rule

### 25

- Training takes longer because it is much more complex with many more parameters
- Overfitting is a big problem
    - Either needs lots of training data or a strong regularization
- It's not clear how to select a good architecture
    - Too much freedom

### 27

- L2 prevents from learning large weights by putting weights squared into the loss function
- 1000 inputs and 1000 parameters will because cause the NN to just memorize the input
    - Drop some connections between neurons during training
        - This forces it to learn features from multiple inputs
- Dropping connections is called **dropout**

## Faster than SGD?

### 32 - Stochastic Gradient Descent Momentum

- $\nu$ = inertia
    - Usually some sort of exponential decay function
- Decreases noise during SGD
- **Demo**: https://distill.pub/2017/momentum/

### 34 - AdaGrad

- Slows the movement in directs that have already moved a larger amount
- Lots of "ada" (adaptive) algorithms in NN

### 35 - RMSProp

- Make sure all gradient steps are approximately the same magnitude to ensure it doesn't get stuck on a plateau
    - It can boost you out of the plateau
- ms = mean square (of the global gradient norm)
- Speeds up progress near optimum

### 36

- Nice demo slide
- This demonstration doesn't hold in all cases - it's just an example

### 37

His favoriate is Adam (RMSprop + momentum)

[Additional demo from tensor flow](http://playground.tensorflow.org/)

