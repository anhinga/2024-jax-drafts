# trying to sketch a feedword network with fluid neurons and optional local recurrences

# and we are going to optimize with respect to it after a run

# so the network is described by `initial_output` dictionary, just like in `immutable_machine2.py`

# this `initial_output` dictionary has somewhere in it the `init_matrix` and
# all the `:function` dictionaries

# and this `initial_output` dictionary has hard leaves (constant leaves)
# and soft leaves ("sentinel" leaves) which are subject to initialization
# and optimization

# if we, at some point, decide to implement a "continual learning", where
# the gradient change things while the network is running, we'll modify this accordingly

# this experiment is still somewhat static, run the network, compute loss and grad,
# apply gradient-based update to a masked `initial_output`, run the network again,
# and so on


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

# OBSOLETE BELOW THIS LINE ============================================================================

# work in progress ...

# if we literally copy the Julia experiment, we are doing it without a Self neuron, and we handle
# the optimization procedure as external (we need to represent both soft link weights and soft activation weights,
# and optimize with respect to those)

# DELETED Later we might decide to upgrade to explicit use of a Self neuron 
# DELETED (which would open various interesting possibilities)

# But we might as well do it with a Self neuron with hard links, following an example at `immutable_machine2.py`

# However, there is a caveat: in that example, `init_matrix` and `:function` dictionaries are
# not fixed, but are rotating with a machine using local recurrences.

# If we do it this way, we'll need to figure out, how to reconcile this procedure
# for external data with presence of Self:
#    # Apply mask to gradients
#    mask = mask_tree(tree_structure)
#    grads = tree_util.tree_map(lambda g, m: g if m else 0, grads, mask)
#    
#    updates, opt_state = optimizer.update(grads, opt_state)
#    new_tree = optax.apply_updates(tree, updates)
#    
#    return new_tree, opt_state, loss
# So, instead of a trivial action of Self and of local recurrences for :function dictionaries
# we want to have these updates (but when we are just running a trained network,
# we basically want to assume that grads are all zeros, the whole thing is a mask;
# that would be a "Self-frienly architecture").

