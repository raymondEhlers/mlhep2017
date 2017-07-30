# Convolution Networks

### 6

- Network can learn features, but it won't work well if the learned object is translated.
    - We want to learn features that are position independent

# **Through 18 needs to be followed up closely**

- Pooling lets you reduce dimensionality

### 18

Good summary image

# **This whole lecture needs to be reviewed closely!!**

- I couldn't focus at the time...

# Seminar tips

Using 3x3 filters is a reasonable way to go unless you have a good reason

- Number of parameters:
    - (Previous number of inputs) * (area) * (number of filters) + (biases) See: (wx+b)
        - So, 256 * 9 * 512 + 512 = 1,179,648 for 512 3x3 filters

-----
# Noel Dawe's Talk

- [DeepJets](https://github.com/deepjets/deepjets). Each stage of pythia -> detector sim -> fastjet is implemented as a generator function which returns a numpy array of 4 vectors
    - Uses cython to interface with pythia, fastjet, etc
    - Minimizes root dependencies
- numpythia
    - Numpy + pythia
- pyjet -> Pythia + fastjet
    - Only depends on numpy
    - See [here](https://github.com/ndawe/pyjet)
