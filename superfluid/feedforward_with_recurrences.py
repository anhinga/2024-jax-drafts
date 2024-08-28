# trying to sketch a feedword network with fluid neurons and optional local recurrences

# fluid neurons will have linear combinations of these activation functions

soft_activations = ['accum_add_args', 'max_norm_dict', 'dot_product', 'compare_scalars', 'const_1', 'const_end']

# names of inputs and outputs of soft activations

soft_inputs = ['accum', 'delta', 'dict', 'x', 'y']

soft_outputs = ['result', 'norm', 'dot', 'true', 'false', 'const_1', 'char']



