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

And, in particular, we'd start to start with implementing a (based on polymorphic neuron) more narrow version of

https://github.com/anhinga/julia-flux-drafts/blob/main/arxiv-1606-09470-section3/May-August-2022/v0-1/feedforward-run-3/feedforward_with_accums.jl

### A novel regularization formula

In our Julia implementation, locally recurrent connections for accumulators were implemented as hard links with weight 1.

In a superfluid version, we can't do that, because we don't know how much of an accumulator a neuron would be.

So what we should try is a new regularization term, the abs(weight_of_accum_activation - weight_of_recurrent connection)
or its square must be low.

(It's still fine to include these weights into usual regularization terms. This new regularization term should have
a larger coefficient than usual regularization terms, so that it impact on these particular weights dominate.)

---

[preliminary_run_notes.md](preliminary_run_notes.md) - first runs (no jax.jit)

[jit-attempt](jit-attempt) - making it work with jax.jit

[jit-phase2](jit-phase2) - next phase of jax.jit exploration

decided not to pursue using `jax.jit` right now, but to do some light exploration of a "no jit" setup

[no-jit-run-2.md](no-jit-run-2.md) - continue "no jit" exploration
