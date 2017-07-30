# Recurrent Neural Networks

- Convolution is spatially focused. 
- RNNs are temporally focused:
    - Time series
    - Text
    - Etc

### Language Model

- Predict what is coming next
- Actual prob dist is way too complicated to calculate directly
    - For example, the DT on slide 6 doesn't work

### 8

- n-gram: n is the number of words to look back

### Hidden Markov

- Words are the known variables
    - Brain contains hidden variables
- ie. For EUR<->USD rate, the ratio is the known variable
    - However, the underlying market contains many hidden variables

### Recurrent Neural Networks (RNNs)

- Model hidden variables via RNNs
- Slide 13 and 14 are both options
- **Same weight for matricies at every step**

### 22

"Inp" = input

### 23

- Good summary of the expressions
- Shows why it is recurrent

### 28

- Backprop through time
- NOTE: Nice image on this slide

### 29

- Truncate at some point so that it doesn't take forever to train
    - Done by making additional gradients = 0

### 31

Red is output at each node in RNN

### 43

- Vanishing derivative
    - Stops training -> Bad
- Exploding derivative
    - Jumps around randomly -> Bad

### Residual RNNs

- Sum previous output with current output to help preserve the current state
- Good at preserving information
- Difficult to forget information

### 52

- Can forget for an appropriate input

### 62 - Gradient Clipping

- If the gradient value gets too large, just cut it at a threshold

[Fantastic application](https://karpathy.github.io/2015/05/21/rnn-effectiveness/)
