# Generative Adversarial Networks

**Need to review start of the lecture**

Basically one network is trained to help another one learn

# 2nd Half from Maxim (?)

- At early stages, gradients tend to vanish
    - Without gradients, it cannot learn

### How to avoid gradients of zero

- Use gradient free methods
    - Has large downsides
- Add strong regularization to the discriminator to make it less powerful
    - But hurts later when the generator gets better
        - Need to adjust regulator as the generator improves
- Early stopping
    - Similar to strong regularization
- Add noise to discriminator 
    - Gives worse final result (similar to the 2 above)

Note that there are nice plots after each idea

### Better solution

- Ensemble of discriminators
    - Takes more computation, as you have to train more discriminants

## Energy Based GANs

- Instead of thinking in terms of probability, think in terms of energy
    - Assign high energy to improbable distributions and low energy to probable ones
- (Looks like Boltzmann)

- Discrimination is based on minimizing the energy difference

### De-normalization

- Lack of normalization means that gradients stay non-zero

### Discussion

- Seems to generally be better than a standard GAN

### Wasserstein GAN

- Slightly different loss function

[amusing demo of art styling](http://likemo.net/#sketch__canvas)
[Useful range tool](https://github.com/noamraph/tqdm)
