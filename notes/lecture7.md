# Categorical Data

First and introduction of CatBoost, a new algorithm from Yandex

- One-hot encoding:
    [a, b, c] = {[1,0,0], [0,1,0], [0,0,1]}

# How you actually do deep learning

Hacks, tricks, etc..

### 6

Common solutions, but may not be desirable

### Feature Learning

- First few layers of NN are often fairly general and re-usable
    - Further layers into the NN become less useful

NOTE: Data goes in the tail and comes out of the head

### Pre-training

- Re-use NN and just replace the last (few) layers
- Thus we can use NN without a huge training set

### Fine-tuning

- Chop off "head" and then add any model you would like
- Freeze the weights in the body to train the head model that you added
    - Once done, you can unfreeze the body and train the entire model

# Unsupervised Learning

Supervised:
- Take x,y and learn their relationship

Unsupervised:
- Take x and find the hidden features

### Why Bother?

- Can use unsupervised to generate new samples

### Autoencoders

- Subset of NNs
    - Internal structure is generally smaller than input
- Decode(Encode(X)) ~ X

### Why autoencoders

- Reduce dimensionality

### Matrix Decomposition

PCA, SVD can be seen as autoencoders

### Convolutional Autoencoders

- Decode by
    - Up-sampling
    - Convolution with padding
        - Un-convolution is basically the same as convolution with padding, which can increase the number of pixels slightly

### Why Autoencoders: Learn Great Features

- Directly using expanding autoencoder is not useful, as it can just encode the data in the network.
    - ie: f(X) -> X is just a map from X-> X. (ie if 10 in input, but 100 in between, just move the 10 through to output)
    - First, we need to constrain the model -> Regularization

### Sparse Autoencoder

- Use L1 on activation, and end up with a sparse model

### Filter Images

- He **doesn't** think that filter images are useful or meaningful.
    - Says that they are very domain specific

# Model Zoo

- So many pre-trained models are now available.
    - Can use some via "import keras.applications as zoo"
