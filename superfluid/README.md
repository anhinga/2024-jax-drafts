### JAX experiments with superfluid DMM architecture

Superfluid upgrade allows each neuron to have a linear combination of all available activation functions.

This eliminated the remaining difference between various types of neurons, the architecture is becoming truly type-free.

The initial files have been copied from https://github.com/anhinga/jax-pytree-example/tree/main/September-2023

We would like to do a variation of JuliaCon 2023 experiment, https://github.com/anhinga/DMM-synthesis-lab-journal/tree/main/JuliaCon2023-talk,
with the difference being that we would use a more narrow feedforward network with all skip connections
with fully polymophic neurons as a starting point (the sparsification should, among other things,
decrease this polymorphism by making the linear combinations of activation functions within neurons
more sparse).

This is the successful pre-sparsification run we need to learn to reproduce first (in the superfluid setting):

https://github.com/anhinga/julia-flux-drafts/tree/main/arxiv-1606-09470-section3/May-August-2022/v0-1/feedforward-run-3
