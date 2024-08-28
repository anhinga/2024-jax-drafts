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

soft_activations = ['accum_add_args', 'max_norm_dict', 'dot_product', 'compare_scalars']  # these two are not included: 'const_1', 'const_end'

# Define a unique sentinel value that won't appear elsewhere in our data
SENTINEL = -9999.0

fluid_activation_template = {key: SENTINEL for key in soft_activations}

# a neuron template is a hard link

def neuron_self_reference(name):
    return matrix_element(name, ":function", name, ":function")

self_neuron = neuron_self_reference("self")
timer_neuron = neuron_self_reference("timer")
input_neuron = neuron_self_reference("input")
const_1_neuron = neuron_self_reference("const_1")
const_end_neuron = neuron_self_reference("eos")
output_neuron = neuron_self_reference("output")

# the width of interlayer will be 4 neurons (instead of 8 neurons for our situation with different types)
# the depth will be 5 interlayers, like in Julia experiment.

n_layers = 5
n_per_layer = 4 # fluid neurons per layer

def interneuron_name(layer, index_within_layer):
    return "fluid-"+str(layer)+"-"+str(index_within_layer)

interneurons = [neuron_self_reference(interneuron_name(layer, k)) 
                for layer in range(n_layers) for k in range(n_per_layer)]

# other hard links

timer_accum = matrix_element("timer", "timer", "timer", "timer")
timer_connect = matrix_element("input", "timer", "timer", "timer")
self_accum = matrix_element("self", "accum", "self", "result")

# names of inputs and outputs of soft activations

soft_inputs = ['accum', 'delta', 'dict', 'x', 'y']

# we'll try to handle constant emitting neurons as in Julia version

soft_outputs = ['result', 'norm', 'dot', 'true', 'false'] # these are separate: 'const_1', 'char' for "eos" and "input"

# we'll try soft links for local recurrences, unlike 
# https://github.com/anhinga/julia-flux-drafts/blob/main/arxiv-1606-09470-section3/May-August-2022/v0-1/feedforward-run-3/feedforward_with_accums.jl

# although this is a delicate thing, in terms of good training behavior

soft_local_recurrences =  [matrix_element(interneuron_name(layer, k), "accum", interneuron_name(layer, k), "result", SENTINEL) 
                           for layer in range(n_layers) for k in range(n_per_layer)]


# we are porting this logic from Julia
#
# for input_layer in 1:n_layers+1
#    for (input_neuron, input_field) in inputs_next[input_layer]
#        for output_layer in 1:input_layer
#            for (output_neuron, output_field) in outputs_this[output_layer]
#                link!(trainable, input_neuron, input_field, output_neuron, output_field, Float32(0.01*rand(normal_dist_0_1)))
# end end end end

# we have output of initial layer as ("input", "char"), ("const_1", "const_1"), ("eos", "char")

# we have outputs of an intermediate layer as [(interneuron_name(layer, k), output) for output in soft_outputs]

# we have inputs of an intermediate layer as [(interneuron_name(layer, k), input) for input in soft_inputs]

# we have input of the final layer as [("output", "dict-1"), ("output", "dict-2")]

def form_layer_input(layer): # from 0 + n_layers is the final one
    if layer < n_layers:
        return [(interneuron_name(layer, k), input) for k in range(n_per_layer) for input in soft_inputs]
    else: # assume layer == n_layers
        return [("output", "dict-1"), ("output", "dict-2")]

all_layer_inputs = [form_layer_input(layer) for layer in range(n_layers+1)]

def form_previous_layer_output(layer): # 0 is a special case, otherwise the index is (layer - 1)
    if layer == 0:
        return [("input", "char"), ("const_1", "const_1"), ("eos", "char")]
    else:
        return [(interneuron_name(layer - 1, k), output) for k in range(n_per_layer) for output in soft_outputs]

all_layer_outputs = [form_previous_layer_output(layer) for layer in range(n_layers+1)]

feed_forward_connections = [matrix_element(*input_pair, *output_pair, SENTINEL)
                            for input_layer in range(n_layers+1) for output_layer in range(input_layer)
                            for input_pair in all_layer_inputs[input_layer] for output_pair in all_layer_outputs[output_layer]] 

init_matrix = {'result': add_v_values(self_neuron, timer_neuron, input_neuron, const_1_neuron, const_end_neuron, output_neuron,
                                      *interneurons, timer_accum, timer_connect, self_accum, *soft_local_recurrences, *feed_forward_connections)}

manual_fields_of_initial_output = {'self': add_v_values(init_matrix, {':function': {'accum_add_args': 1.0}}),
                                   'timer': add_v_values({'timer': {':number': 0.0}}, {':function': {'timer_add_one': 1.0}}),
                                   'input': {':function': {'input_dummy': 1.0}},
                                   'const_1': add_v_values({'const_1': {':number': 1.0}}, {':function': {'const_1': 1.0}}),
                                   'eos': add_v_values({'char': {'.': 1.0}}, {':function': {'const_end': 1.0}}),
                                   'output': {':function': {'output_dummy': 1.0}}}

auto_fields_of_initial_output = {interneuron_name(layer, k): add_v_values({'result': {}}, {':function': fluid_activation_template})
                                 for layer in range(n_layers) for k in range(n_per_layer)}

initial_output = {**manual_fields_of_initial_output, **auto_fields_of_initial_output}

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

