# trying to sketch a feedword network with fluid neurons and optional local recurrences

# fluid neurons will have linear combinations of these activation functions

soft_activations = ['accum_add_args', 'max_norm_dict', 'dot_product', 'compare_scalars', 'const_1', 'const_end']

# names of inputs and outputs of soft activations

soft_inputs = ['accum', 'delta', 'dict', 'x', 'y']

# do we want to include constants here, or do we handle them as in Julia?
soft_outputs = ['result', 'norm', 'dot', 'true', 'false', 'const_1', 'char']

# we'll try soft links for local recurrences, unlike 
# https://github.com/anhinga/julia-flux-drafts/blob/main/arxiv-1606-09470-section3/May-August-2022/v0-1/feedforward-run-3/feedforward_with_accums.jl

# although this is a delicate thing, in terms of good training behavior

# but we'll have hard links for all :function to :function recurrencies
# and there are also hard links like these in Julia:
# hard_link!(trainable, "timer", "timer", "timer", "timer")
# hard_link!(trainable, "input", "timer", "timer", "timer")

# the width of interlayer will be 4 neurons (instead of 8 neurons for our situation with different types)
# the depth will be 5 interlayers, like in Julia experiment.

# work in progress ...

# if we literally copy the Julia experiment, we are doing it without a Self neuron, and we handle
# the optimization procedure as external (we need to represent both soft link weights and soft activation weights,
# and optimize with respect to those)

# Later we might decide to upgrade to explicit use of a Self neuron 
# (which would open various interesting possibilities)


